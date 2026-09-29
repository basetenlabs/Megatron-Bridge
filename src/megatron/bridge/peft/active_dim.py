# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

The caller owns two things: zeroing the padding once, using :func:`lora_padding_masks`,
and persisting ``active_dim``. It is not part of the model's state dict, so a model
restored from a checkpoint comes back at ``active_dim == dim`` until the caller sets it.
"""

from collections.abc import Iterator, Sequence

import torch
import torch.nn as nn

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.lora_layers import TEFusedLoRALinear
from megatron.bridge.peft.multi_lora_layers import _MULTI_LORA_TYPES
from megatron.bridge.peft.utils import rank_padding_masks


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


def _check_paddable(adapter: nn.Module, active_dim: int) -> None:
    if active_dim == adapter.dim:
        return
    if not adapter.supports_rank_padding:
        raise NotImplementedError(f"{type(adapter).__name__} cannot run below its allocated dim={adapter.dim}")
    # An activation f between the factors with f(0) != 0 makes the padded rows of
    # linear_in's output nonzero, so linear_out's padded columns would train.
    # LinearAdapter has no activation between its factors.
    if not isinstance(getattr(adapter, "activation", nn.Identity()), nn.Identity):
        raise NotImplementedError(
            f"{adapter.base_linear_name}: zero padding needs an identity activation, "
            f"not {type(adapter.activation).__name__}"
        )
    if not 0 < active_dim < adapter.dim:
        raise ValueError(f"active_dim={active_dim} must be in (0, {adapter.dim}]")


def get_lora_active_dim(model: nn.Module | Sequence[nn.Module]) -> int:
    """The rank every LoRA adapter currently runs at."""
    dims = {adapter.active_dim for _, adapter in _iter_adapters(model)}
    if len(dims) != 1:
        raise ValueError(f"LoRA adapters run at {sorted(dims)}; expected one shared rank")
    return dims.pop()


def set_lora_active_dim(model: nn.Module | Sequence[nn.Module], active_dim: int) -> None:
    """Scale every LoRA adapter's delta by ``alpha / active_dim``.

    Every adapter is checked before any changes, so a refusal leaves the model untouched.
    """
    pairs = [(wrapper, adapter) for wrapper, adapter in _iter_adapters(model) if adapter.active_dim != active_dim]
    for _, adapter in pairs:
        _check_paddable(adapter, active_dim)
    for wrapper, adapter in pairs:
        adapter.active_dim = active_dim
        if isinstance(wrapper, TEFusedLoRALinear):
            # The fused branch bakes the scale in when it is built.
            wrapper._fused_branches = None


def lora_padding_masks(model: nn.Module | Sequence[nn.Module], active_dim: int) -> dict[torch.Tensor, torch.Tensor]:
    """Map each padded local LoRA weight to the mask of its entries past ``active_dim``."""
    masks: dict[torch.Tensor, torch.Tensor] = {}
    for _, adapter in _iter_adapters(model):
        _check_paddable(adapter, active_dim)
        if active_dim == adapter.dim:
            continue
        in_mask, out_mask = rank_padding_masks(adapter, active_dim)
        masks[adapter.linear_in.weight] = in_mask
        masks[adapter.linear_out.weight] = out_mask
    return masks
