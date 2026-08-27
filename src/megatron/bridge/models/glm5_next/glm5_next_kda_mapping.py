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

"""Fuse GLM-5.3's separate KDA q/k/v tensors onto Megatron-Core's single one.

GLM-5.3 stores the KDA input projection and the short convolution as three
per-projection tensors, while Megatron-Core's ``KimiDeltaAttention`` keeps one fused
tensor for each. Verified against ``zai-org/GLM-5.3-Flash`` (and the ``-BF16`` repo,
which is byte-identical in naming)::

    HF                                        Megatron
    self_attn.q_proj.weight   [8192, 4096]
    self_attn.k_proj.weight   [8192, 4096]  -> self_attn.in_proj.weight  [24576, 4096]
    self_attn.v_proj.weight   [8192, 4096]

    self_attn.q_conv1d.weight [8192, 1, 4]
    self_attn.k_conv1d.weight [8192, 1, 4]  -> self_attn.conv1d.weight   [24576, 1, 4]
    self_attn.v_conv1d.weight [8192, 1, 4]

Why this is not a plain concatenation
-------------------------------------
Under tensor parallelism each rank owns a *head slice of every component*, so the
fused tensor a rank holds is ``[q_shard_i; k_shard_i; v_shard_i]`` -- not a contiguous
slice of ``cat([q, k, v])``. Concatenating first and letting the generic
column-parallel mapping shard along dim 0 would hand rank 0 all of ``q`` and rank 1
all of ``k``: every projection silently wrong, with no shape error to catch it. So the
split happens per component and the shards are reassembled per rank, the same way
``GatedMLPMapping`` handles ``[gate; up]``.

``ChunkedMapping`` / ``GDNConv1dMapping`` do not fit: they split *one* concatenated HF
tensor, which is Qwen3-Next's layout, the mirror image of GLM-5.3's.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    MegatronParamMapping,
    get_module_and_param_from_name,
)

try:  # DTensor-aware parameters, as elsewhere in param_mapping
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    DTensor = ()


class Glm5NextKdaFusedMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """Map GLM-5.3's separate q/k/v KDA tensors onto one Megatron tensor.

    Args:
        megatron_param: Megatron parameter name pattern (``in_proj.weight`` or
            ``conv1d.weight``).
        q: HF pattern for the query component.
        k: HF pattern for the key component.
        v: HF pattern for the value component.

    The component order ``(q, k, v)`` matches ``KimiDeltaAttention``'s
    ``in_proj_split_names`` in its low-rank layout, which is the layout GLM-5.3
    selects by setting both ``kda_f_lora_rank`` and ``kda_gate_lora_rank``.
    """

    _ORDER: Tuple[str, ...] = ("q", "k", "v")

    def __init__(self, megatron_param: str, q: str, k: str, v: str):
        super().__init__(megatron_param, {"q": q, "k": k, "v": v})
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Return a resolved copy, keeping the component keyword arguments.

        Required: the base implementation reconstructs with
        ``type(self)(megatron_param, hf_param)``, which drops q/k/v and raises a
        TypeError the moment a wildcard is resolved. Every multi-argument mapping in
        ``param_mapping`` overrides this for the same reason.
        """
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["q"],
            resolved_hf_param["k"],
            resolved_hf_param["v"],
        )

    def _component_sizes(self, megatron_module: nn.Module) -> List[int]:
        """Per-component dim-0 sizes of the *unsharded* fused tensor."""
        config = self._get_config(megatron_module)
        qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
        v_dim = config.linear_value_head_dim * config.linear_num_value_heads
        return [qk_dim, qk_dim, v_dim]

    def hf_to_megatron(
        self,
        hf_weights: Dict[str, torch.Tensor],
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Shard each component by head, then concatenate this rank's slices."""
        components = [hf_weights[name] for name in self._ORDER]

        expected = self._component_sizes(megatron_module)
        for name, tensor, size in zip(self._ORDER, components, expected):
            if tensor.shape[0] != size:
                raise ValueError(
                    f"KDA component {name!r} for {self.megatron_param} has dim-0 size "
                    f"{tensor.shape[0]}, expected {size} from the provider's KDA "
                    "geometry (linear_num_key_heads / linear_key_head_dim / "
                    "linear_num_value_heads / linear_value_head_dim). A mismatch here "
                    "means the provider and the checkpoint disagree on head layout."
                )

        if self.tp_size == 1:
            return torch.cat(components, dim=0)

        _, target_param = get_module_and_param_from_name(megatron_module, self.megatron_param)

        if self.tp_rank == 0:
            for name, tensor in zip(self._ORDER, components):
                if tensor.shape[0] % self.tp_size != 0:
                    raise ValueError(
                        f"KDA component {name!r} dim-0 size {tensor.shape[0]} is not "
                        f"divisible by tp_size={self.tp_size}"
                    )
            per_component = [torch.chunk(tensor, self.tp_size, dim=0) for tensor in components]
            splits = [
                torch.cat([component[rank] for component in per_component], dim=0)
                for rank in range(self.tp_size)
            ]
        else:
            splits = None

        output_shape = (
            target_param.orig_param.shape if isinstance(target_param, DTensor) else target_param.shape
        )
        return self.scatter_to_tp_ranks(
            splits, output_shape, target_param.dtype, target_param.device
        )

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Gather the fused shards and split them back into q, k and v."""
        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )
        if megatron_weights is None:
            return {}

        megatron_weights = self.maybe_dequantize(megatron_weights)
        sizes = self._component_sizes(megatron_module)

        if self.tp_size == 1:
            pieces = torch.split(megatron_weights, sizes, dim=0)
        else:
            local_sizes = [size // self.tp_size for size in sizes]
            gathered = self.gather_from_tp_ranks(megatron_weights)
            per_component: List[List[torch.Tensor]] = [[] for _ in self._ORDER]
            for shard in gathered:
                for index, piece in enumerate(torch.split(shard, local_sizes, dim=0)):
                    per_component[index].append(piece)
            pieces = [torch.cat(parts, dim=0) for parts in per_component]

        return {self.hf_param[name]: piece for name, piece in zip(self._ORDER, pieces)}
