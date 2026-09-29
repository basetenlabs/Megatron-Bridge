# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a rank-``r`` LoRA inside adapters allocated at a larger ``dim``.

``dim`` sizes the adapter weights and never changes. ``active_dim`` is the rank the LoRA
runs at: it occupies global rank indices ``[0, active_dim)``, the rest of ``linear_in``'s
rows and ``linear_out``'s columns stay zero, and the forward scale is ``alpha / active_dim``.
Both padded blocks being zero is a fixed point of training: each one's gradient is a
product with the other, so it stays exactly zero under an elementwise optimizer.

Every helper takes the LoRA's allocated rank, ``lora_dim`` (the PEFT config's ``dim``),
explicitly: a pipeline stage sees only its own adapters and may hold none. Adapters with
a smaller ``dim`` (``normalize_moe_lora`` experts at ``dim / topk``) always run at their
own full ``dim`` and cannot be padded.

The caller owns three things:

- zeroing the padding once, using :func:`lora_padding_masks`;
- persisting ``active_dim``: it is not in the model's state dict, so a restored model
  comes back at ``active_dim == dim`` until the caller sets it;
- applying the same call on every rank. Refusals are checked per rank, so a pipeline
  stage whose adapters cannot be padded refuses while other stages accept.
"""

from collections.abc import Iterator, Sequence

import torch
import torch.nn as nn

from megatron.bridge.models.transformer_config import cuda_graphs_are_enabled
from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.lora_layers import TEFusedLoRALinear
from megatron.bridge.peft.multi_lora_layers import _MULTI_LORA_TYPES
from megatron.bridge.peft.utils import rank_indices, rank_padding_masks


def _iter_adapters(model: nn.Module | Sequence[nn.Module]) -> Iterator[tuple[AdapterWrapper, nn.Module]]:
    """Yield ``(wrapper, adapter)`` for every LoRA adapter, expanding canonical LoRA's per-projection dicts."""
    for chunk in [model] if isinstance(model, nn.Module) else model:
        for module in chunk.modules():
            if not isinstance(module, AdapterWrapper):
                continue
            if isinstance(module, _MULTI_LORA_TYPES):
                raise NotImplementedError(f"{type(module).__name__} sets each slot's rank through its rank_values")
            adapters = module.adapter
            if isinstance(adapters, (nn.ModuleDict, nn.ModuleList)):
                for adapter in adapters.children():
                    yield module, adapter
            else:
                yield module, adapters


def _paddable_pairs(
    model: nn.Module | Sequence[nn.Module], active_dim: int, lora_dim: int
) -> list[tuple[AdapterWrapper, nn.Module]]:
    """Every adapter, after checking that all of them can run at ``active_dim``."""
    if not 0 < active_dim <= lora_dim:
        raise ValueError(f"active_dim={active_dim} must be in (0, {lora_dim}]")
    pairs = list(_iter_adapters(model))
    if active_dim == lora_dim:
        return pairs
    for wrapper, adapter in pairs:
        if adapter.dim != lora_dim:
            # normalize_moe_lora experts run at dim / topk, rounded to expert-TP granularity.
            raise NotImplementedError(
                f"{type(adapter).__name__} has dim={adapter.dim}, not lora_dim={lora_dim}; "
                "adapters with a normalized dim (normalize_moe_lora) cannot be rank padded"
            )
        if not adapter.supports_rank_padding:
            raise NotImplementedError(f"{type(adapter).__name__} cannot run below its allocated dim={adapter.dim}")
        # An activation f between the factors with f(0) != 0 makes the padded rows of
        # linear_in's output nonzero, so linear_out's padded columns would train.
        # LinearAdapter has no activation between its factors.
        activation = getattr(adapter, "activation", nn.Identity())
        if not isinstance(activation, nn.Identity):
            raise NotImplementedError(
                f"{adapter.base_linear_name}: zero padding needs an identity activation, "
                f"not {type(activation).__name__}"
            )
        # Megatron linears carry the TransformerConfig; a plain nn.Linear has none.
        if cuda_graphs_are_enabled(getattr(wrapper.to_wrap, "config", None)):
            raise NotImplementedError("a captured CUDA graph bakes in the LoRA scale; padding needs it to change")
    return pairs


def get_lora_active_dim(model: nn.Module | Sequence[nn.Module], *, lora_dim: int) -> int | None:
    """Return the rank the LoRA currently runs at.

    Args:
        model: The model, or its list of chunks, holding the LoRA adapters.
        lora_dim: The LoRA's allocated rank (the PEFT config's ``dim``).

    Returns:
        The ``active_dim`` shared by every adapter allocated at ``lora_dim``, or ``None``
        when this pipeline stage has none (it cannot tell which rank the LoRA runs at).
    """
    dims = {adapter.active_dim for _, adapter in _iter_adapters(model) if adapter.dim == lora_dim}
    if len(dims) > 1:
        raise ValueError(f"LoRA adapters run at {sorted(dims)}; expected one shared rank")
    return dims.pop() if dims else None


def set_lora_active_dim(model: nn.Module | Sequence[nn.Module], active_dim: int, *, lora_dim: int) -> None:
    """Run every LoRA adapter at ``active_dim``, scaling its delta by ``alpha / active_dim``.

    Every adapter is checked before any changes, so a refusal leaves the model untouched.

    Args:
        model: The model, or its list of chunks, holding the LoRA adapters.
        active_dim: The rank to run at, in ``(0, lora_dim]``. ``lora_dim`` restores
            every adapter to its own full ``dim``.
        lora_dim: The LoRA's allocated rank (the PEFT config's ``dim``).

    Raises:
        ValueError: ``active_dim`` is out of range.
        NotImplementedError: An adapter cannot run below its ``dim``.
    """
    for wrapper, adapter in _paddable_pairs(model, active_dim, lora_dim):
        # Restoring puts every adapter back at its own dim, which differs from lora_dim
        # for normalized experts; padding only reaches adapters allocated at lora_dim.
        target = adapter.dim if active_dim == lora_dim else active_dim
        if adapter.active_dim == target:
            continue
        adapter.active_dim = target
        if isinstance(wrapper, TEFusedLoRALinear):
            # The fused branch bakes the scale in when it is built.
            wrapper._fused_branches = None


def lora_padding_masks(
    model: nn.Module | Sequence[nn.Module], active_dim: int, *, lora_dim: int
) -> dict[torch.Tensor, torch.Tensor]:
    """Map each padded local LoRA weight to the mask of its entries past ``active_dim``.

    Args:
        model: The model, or its list of chunks, holding the LoRA adapters.
        active_dim: The rank the LoRA runs at, in ``(0, lora_dim]``.
        lora_dim: The LoRA's allocated rank (the PEFT config's ``dim``).

    Returns:
        ``{weight: mask}`` for this rank's local ``linear_in`` and ``linear_out``
        weights; empty when ``active_dim == lora_dim``.

    Raises:
        ValueError: ``active_dim`` is out of range.
        NotImplementedError: An adapter cannot run below its ``dim``.
    """
    masks: dict[torch.Tensor, torch.Tensor] = {}
    pairs = _paddable_pairs(model, active_dim, lora_dim)
    if active_dim == lora_dim:
        return masks
    for _, adapter in pairs:
        in_mask, out_mask = rank_padding_masks(adapter, active_dim)
        masks[adapter.linear_in.weight] = in_mask
        masks[adapter.linear_out.weight] = out_mask
    return masks


def lora_rank_index(model: nn.Module | Sequence[nn.Module], *, lora_dim: int) -> dict[torch.Tensor, torch.Tensor]:
    """Map each local LoRA weight to the global rank index of each of its entries.

    ``index >= r`` is then the padding mask for any run rank ``r``, so a caller that
    serves several ranks builds this once instead of one mask per rank.

    Args:
        model: The model, or its list of chunks, holding the LoRA adapters.
        lora_dim: The LoRA's allocated rank (the PEFT config's ``dim``).

    Returns:
        ``{weight: index}`` for this rank's local ``linear_in`` and ``linear_out``
        weights, as expanded integer views; empty when ``lora_dim == 1``.

    Raises:
        NotImplementedError: An adapter cannot run below its ``dim``.
    """
    if lora_dim == 1:
        return {}
    index: dict[torch.Tensor, torch.Tensor] = {}
    # Validate as if padding by one rank: the checks do not depend on how far.
    for _, adapter in _paddable_pairs(model, lora_dim - 1, lora_dim):
        in_index, out_index = rank_indices(adapter)
        index[adapter.linear_in.weight] = in_index
        index[adapter.linear_out.weight] = out_index
    return index
