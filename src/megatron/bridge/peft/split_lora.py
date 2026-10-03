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

"""Split LoRA adapters inside fused Megatron linears.

``LoRA(split_adapters=...)`` maps a fused linear (``linear_qkv``, ``linear_fc1`` or a
Gated DeltaNet ``in_proj``) to groups of its output components. Each group gets one
adapter (one LoRA A shared by the group); components in no group get no adapter.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from megatron.core import parallel_state

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.canonical_lora import LoRALinearSplitFC1UpGate, LoRALinearSplitQKV, ModuleDict


# Gated DeltaNet in_proj output order on every TP rank (Megatron-LM ``in_proj_split_names``).
GDN_IN_PROJ_COMPONENTS = ("q", "k", "v", "z", "b", "a")
GDN_ADAPTER_PREFIX = "adapter_gdn_"
QKV_COMPONENTS = ("q", "k", "v")
FC1_COMPONENTS = ("gate", "up")

SplitGroups = Tuple[Tuple[str, ...], ...]


def gdn_adapter_key(group: Sequence[str]) -> str:
    """ModuleDict key of the GDN in_proj adapter for ``group``, e.g. ``adapter_gdn_qkvz``."""
    return GDN_ADAPTER_PREFIX + "".join(group)


def gdn_components_from_adapter_key(adapter_key: str) -> Tuple[str, ...]:
    """Inverse of :func:`gdn_adapter_key`."""
    return tuple(adapter_key.removeprefix(GDN_ADAPTER_PREFIX))


def gdn_in_proj_component_sizes(config: Any, tp_size: int = 1) -> Dict[str, int]:
    """Output rows of each GDN in_proj component held by one of ``tp_size`` ranks."""
    qk = config.linear_key_head_dim * config.linear_num_key_heads
    v = config.linear_value_head_dim * config.linear_num_value_heads
    heads = config.linear_num_value_heads
    sizes = {"q": qk, "k": qk, "v": v, "z": v, "b": heads, "a": heads}
    for name, size in sizes.items():
        if size % tp_size:
            raise ValueError(f"GDN in_proj component {name} ({size} rows) does not split over TP={tp_size}")
    return {name: size // tp_size for name, size in sizes.items()}


def _validate_groups(groups: SplitGroups, components: Sequence[str], *, singletons_only: bool) -> None:
    seen = [c for group in groups for c in group]
    if not seen:
        raise ValueError("split_adapters needs at least one component group")
    unknown = sorted(set(seen) - set(components))
    if unknown:
        raise ValueError(f"unknown split components {unknown}; expected some of {list(components)}")
    if len(seen) != len(set(seen)):
        raise ValueError(f"a split component appears in more than one group: {groups}")
    if singletons_only and any(len(group) != 1 for group in groups):
        raise ValueError(f"only one component per adapter is supported here, got {groups}")


class LoRALinearSplitGatedQKV(LoRALinearSplitQKV):
    """``LoRALinearSplitQKV`` that also handles gated attention (``attention_output_gate``).

    The q adapter emits HF ``q_proj`` rows, which hold each head's query followed by its
    output gate. Megatron's ``linear_qkv`` instead holds, per query group, the group's
    query heads, then its gate heads, then k, then v. Head counts are taken from the
    local tensors, so this works on any TP rank.
    """

    def _interleave_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        config = self.to_wrap.config
        gated = getattr(config, "attention_output_gate", False)
        head_size = config.kv_channels
        groups = key.size(-1) // head_size
        heads = query.size(-1) // (head_size * (2 if gated else 1))
        per_group = heads // groups

        leading = query.shape[:-1]
        key = key.reshape(-1, groups, 1, head_size)
        value = value.reshape(-1, groups, 1, head_size)
        if gated:
            qz = query.reshape(-1, groups, per_group, 2, head_size)
            q, z = qz[:, :, :, 0], qz[:, :, :, 1]
            parts = [q, z, key, value]
        else:
            parts = [query.reshape(-1, groups, per_group, head_size), key, value]
        return torch.cat(parts, dim=2).reshape(*leading, -1)


class LoRALinearSplitGDNInProj(AdapterWrapper):
    """Gated DeltaNet ``in_proj`` with one adapter per component group.

    Each adapter's local output holds its components' local rows in group order. They
    are scattered into the rank's ``[q | k | v | z | b | a]`` layout; components without
    an adapter get a zero delta.
    """

    def __init__(self, to_wrap: nn.Module, adapter: ModuleDict, local_sizes: Dict[str, int]) -> None:
        super().__init__(to_wrap, adapter)
        self.local_sizes = local_sizes
        self.groups = tuple((key, gdn_components_from_adapter_key(key)) for key in adapter.keys())

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return linear_output, bias
        pieces: Dict[str, torch.Tensor] = {}
        for key, components in self.groups:
            out = self.adapter_forward(self.adapter[key], layernorm_output, *args, **kwargs)
            sizes = [self.local_sizes[c] for c in components]
            pieces.update(zip(components, torch.split(out, sizes, dim=-1)))
        leading = linear_output.shape[:-1]
        delta = torch.cat(
            [
                pieces[c] if c in pieces else linear_output.new_zeros(*leading, self.local_sizes[c])
                for c in GDN_IN_PROJ_COMPONENTS
            ],
            dim=-1,
        )
        return linear_output + delta, bias


def build_split_lora(
    module: nn.Module,
    name: str,
    groups: SplitGroups,
    in_features: int,
    out_features: int,
    make_adapter: Callable[[int, int], nn.Module],
) -> AdapterWrapper:
    """Wrap fused ``module`` with one adapter per component group.

    ``make_adapter(in_features, out_features)`` builds an adapter whose output dimension is
    split across TP like the base linear's.
    """
    if name == "linear_qkv":
        _validate_groups(groups, QKV_COMPONENTS, singletons_only=True)
        config = module.config
        q_rows = config.kv_channels * config.num_attention_heads
        if getattr(config, "attention_output_gate", False):
            q_rows *= 2
        kv_rows = config.kv_channels * config.num_query_groups
        rows = {"q": q_rows, "k": kv_rows, "v": kv_rows}
        wanted = {group[0] for group in groups}
        adapters = ModuleDict(
            {f"adapter_{c}": make_adapter(in_features, rows[c]) if c in wanted else None for c in QKV_COMPONENTS}
        )
        return LoRALinearSplitGatedQKV(module, adapters)

    if name == "linear_fc1":
        _validate_groups(groups, FC1_COMPONENTS, singletons_only=True)
        wanted = {group[0] for group in groups}
        adapters = ModuleDict(
            {
                f"adapter_{c}": make_adapter(in_features, out_features // 2) if c in wanted else None
                for c in FC1_COMPONENTS
            }
        )
        return LoRALinearSplitFC1UpGate(module, adapters)

    if name == "in_proj":
        _validate_groups(groups, GDN_IN_PROJ_COMPONENTS, singletons_only=False)
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        global_rows = gdn_in_proj_component_sizes(module.config)
        local_rows = gdn_in_proj_component_sizes(module.config, tp_size)
        adapters = ModuleDict(
            {gdn_adapter_key(group): make_adapter(in_features, sum(global_rows[c] for c in group)) for group in groups}
        )
        return LoRALinearSplitGDNInProj(module, adapters, local_rows)

    raise ValueError(f"split_adapters does not support module {name!r}; expected linear_qkv, linear_fc1 or in_proj")
