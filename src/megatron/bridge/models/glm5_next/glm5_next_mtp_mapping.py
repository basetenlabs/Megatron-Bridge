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

"""Split GLM-5.3's fused MTP ``eh_proj`` into Megatron-Core's ``e_proj`` / ``h_proj``.

GLM-5.3 ships one MTP layer (``num_nextn_predict_layers: 1``, layer 45 of the
checkpoint) carrying ``eh_proj``, ``enorm``, ``hnorm`` and ``shared_head.norm``. That
layer is an **MLA+DSA** layer, not KDA -- the indexer tensors appear on 12 layers, the
11 sparse layers plus this one, which is also what ``index_share_for_mtp_iteration``
implies.

The projection needs splitting because Megatron-Core changes its MTP layout when mHC
is on, and GLM-5.3 has mHC on:

    mHC off (DeepSeek-V3 form, what GLM-5.3 ships):
        hidden = eh_proj(cat([decoder_input, hnorm(hidden)], dim=-1))
        eh_proj.weight : [hidden, 2 * hidden]      -> [4096, 8192]

    mHC on (Megatron-Core, DSv4):
        hidden = e_proj(decoder_input).unsqueeze(2) + h_proj(hnorm(hidden_per_stream))
        e_proj.weight  : [hidden, hidden]          -> [4096, 4096]
        h_proj.weight  : [hidden, hidden]          -> [4096, 4096]

These are the same function. A fused matmul over a concatenated input is the sum of
two matmuls over the halves, and the concatenation order is fixed by Megatron-Core's
own non-mHC path -- ``torch.cat((decoder_input, hidden), -1)`` -- so the embedding
half comes first::

    eh_proj.weight[:, :hidden]  -> e_proj.weight
    eh_proj.weight[:,  hidden:] -> h_proj.weight

Verification limit, stated rather than hidden: ``transformers`` ships **no** MTP
implementation for ``glm5_next`` (no ``eh_proj`` or ``nextn`` anywhere in
``modeling_glm5_next.py``), so unlike KDA, MLA and mHC there is no HF reference to
check MTP numerics against. The split above is justified by algebra and by
Megatron-Core's concat order, not by a parity test.
"""

from typing import Dict, Optional

import torch
from torch import nn

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    MegatronParamMapping,
    get_module_and_param_from_name,
)

# Slice index -> Megatron parameter name, in the order Megatron-Core concatenates them.
MTP_EH_ORDER = ("e_proj", "h_proj")


class Glm5NextMtpEhProjMapping(MegatronParamMapping[torch.Tensor]):
    """Map one half of GLM-5.3's fused ``eh_proj`` onto a Megatron MTP projection.

    Args:
        megatron_param: full dotted path of ``e_proj.weight`` or ``h_proj.weight``.
        hf_param: the fused ``eh_proj.weight``.
        half: 0 for the embedding half (leading columns), 1 for the hidden half.
    """

    def __init__(self, megatron_param: str, hf_param: str, half: int):
        if half not in (0, 1):
            raise ValueError(f"eh_proj half must be 0 or 1, got {half}")
        super().__init__(megatron_param, hf_param)
        self.half = half
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    def _hidden_size(self, megatron_module: nn.Module) -> int:
        return self._get_config(megatron_module).hidden_size

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Take this half's columns, then shard as an ordinary column-parallel weight."""
        hidden = self._hidden_size(megatron_module)
        if hf_weights.shape[1] != 2 * hidden:
            raise ValueError(
                f"{self.hf_param} has input width {hf_weights.shape[1]}, expected "
                f"{2 * hidden} (embedding half + hidden half). GLM-5.3 ships the fused "
                "DeepSeek-V3 form; a different width means the MTP layout changed."
            )
        start = self.half * hidden
        piece = hf_weights[:, start : start + hidden].contiguous()
        return self._tp_mapping.hf_to_megatron(piece, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Re-fuse both halves, from the ``e_proj`` mapping only.

        Two mappings target one HF tensor, so one must own the export or they would
        write conflicting values for the same key.
        """
        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )
        if megatron_weights is None or self.half != 0:
            return {}

        parent = self.megatron_param.rsplit(".", 1)[0].rsplit(".", 1)[0]
        halves = []
        for name in MTP_EH_ORDER:
            _, param = get_module_and_param_from_name(megatron_module, f"{parent}.{name}.weight")
            weight = self.maybe_dequantize(param.detach())
            if self.tp_size > 1:
                # Column-parallel: each rank owns a slice of the output dim.
                weight = torch.cat(self.gather_from_tp_ranks(weight), dim=0)
            halves.append(weight)
        return {self.hf_param: torch.cat(halves, dim=1)}


def mtp_eh_proj_mappings(megatron_mtp_layer: str, hf_mtp_layer: str) -> list:
    """Build both halves of the ``eh_proj`` split for one MTP layer."""
    return [
        Glm5NextMtpEhProjMapping(
            megatron_param=f"{megatron_mtp_layer}.{name}.weight",
            hf_param=f"{hf_mtp_layer}.eh_proj.weight",
            half=half,
        )
        for half, name in enumerate(MTP_EH_ORDER)
    ]
