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

The LoRA occupies global rank indices ``[0, r)``; the rest of ``linear_in``'s rows
and ``linear_out``'s columns stay zero, and the forward scale is ``alpha / r``.
Both padded blocks being zero is a fixed point of training: each one's gradient
is a product with the other, so it stays exactly zero under an elementwise
optimizer. The caller zeroes the padding once, using :func:`lora_padding_masks`.
"""

from collections.abc import Iterator

import torch
import torch.nn as nn

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.lora_layers import LinearAdapter, TEFusedLoRALinear
from megatron.bridge.peft.utils import ParallelLinearAdapter, rank_padding_masks


# Adapters whose rank-axis layout ``rank_padding_masks`` knows. Exact types:
# subclasses such as the DoRA adapter precompute their scale from ``dim``.
_PADDABLE_ADAPTERS = (ParallelLinearAdapter, LinearAdapter)


def _iter_lora_wrappers(model: nn.Module | list[nn.Module]) -> Iterator[AdapterWrapper]:
    for chunk in model if isinstance(model, list) else [model]:
        for module in chunk.modules():
            if isinstance(module, AdapterWrapper):
                yield module


def _check_paddable(adapter: nn.Module, active_dim: int) -> None:
    if active_dim == adapter.dim:
        return
    if type(adapter) not in _PADDABLE_ADAPTERS:
        raise NotImplementedError(f"{type(adapter).__name__} cannot run below its allocated dim={adapter.dim}")
    # An activation f between the factors with f(0) != 0 makes the padded rows of
    # linear_in's output nonzero, so linear_out's padded columns would train.
    if not isinstance(getattr(adapter, "activation", nn.Identity()), nn.Identity):
        raise NotImplementedError(
            f"{adapter.base_linear_name}: zero padding needs an identity activation, "
            f"not {type(adapter.activation).__name__}"
        )
    if not 0 < active_dim < adapter.dim:
        raise ValueError(f"active_dim={active_dim} must be in (0, {adapter.dim}]")


def get_lora_active_dim(model: nn.Module | list[nn.Module]) -> int:
    """The rank every LoRA adapter currently runs at."""
    dims = {wrapper.adapter.active_dim for wrapper in _iter_lora_wrappers(model)}
    if len(dims) != 1:
        raise ValueError(f"LoRA adapters run at {sorted(dims)}; expected one shared rank")
    return dims.pop()


def set_lora_active_dim(model: nn.Module | list[nn.Module], active_dim: int) -> None:
    """Scale every LoRA adapter's delta by ``alpha / active_dim``."""
    for wrapper in _iter_lora_wrappers(model):
        adapter = wrapper.adapter
        if adapter.active_dim == active_dim:
            continue
        _check_paddable(adapter, active_dim)
        adapter.active_dim = active_dim
        if isinstance(wrapper, TEFusedLoRALinear):
            # The fused branch bakes the scale in when it is built.
            wrapper._fused_branches = None


def lora_padding_masks(model: nn.Module | list[nn.Module], active_dim: int) -> dict[torch.Tensor, torch.Tensor]:
    """Map each padded local LoRA weight to the mask of its entries past ``active_dim``."""
    masks: dict[torch.Tensor, torch.Tensor] = {}
    for wrapper in _iter_lora_wrappers(model):
        adapter = wrapper.adapter
        _check_paddable(adapter, active_dim)
        if active_dim == adapter.dim:
            continue
        in_mask, out_mask = rank_padding_masks(adapter, active_dim)
        masks[adapter.linear_in.weight] = in_mask
        masks[adapter.linear_out.weight] = out_mask
    return masks
