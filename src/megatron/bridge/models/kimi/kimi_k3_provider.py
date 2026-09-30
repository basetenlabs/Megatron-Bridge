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

"""Model provider for Kimi K3."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.mla_provider import MLAModelProvider


if TYPE_CHECKING:
    from megatron.bridge.models.kimi.kimi_k3_vl_model import KimiK3VLModel


@dataclass
class KimiK3ModelProvider(MLAModelProvider):
    """Megatron configuration and provider for Kimi K3.

    With ``vision_config`` set, ``provide`` builds the vision-language model, whose
    backbone sits under ``language_model``; without it, the bare language model.
    """

    variable_seq_lengths: bool = True
    kimi_kda_layers: tuple[int, ...] = ()
    kimi_linear_num_heads: int = 96
    kimi_linear_head_dim: int = 128
    kimi_linear_conv_kernel_size: int = 4
    kimi_kda_gate_lower_bound: float = -5.0
    kimi_attn_res_block_size: int = 12

    # HF ``KimiK3VisionConfig``; None builds the text-only backbone.
    vision_config: object = None
    media_placeholder_token_id: int = 163605

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> "MCoreGPTModel | KimiK3VLModel":
        """Build the VL model when the checkpoint carries a vision tower."""
        if self.vision_config is None:
            return self.provide_language_model(pre_process, post_process, vp_stage)
        if self.scatter_embedding_sequence_parallel:
            # The splice needs whole rows; a scattering embedding would shard twice.
            raise ValueError("Kimi K3 with vision requires scatter_embedding_sequence_parallel=False")

        from megatron.bridge.models.kimi.kimi_k3_vl_model import KimiK3VLModel

        return KimiK3VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

    def provide_language_model(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> MCoreGPTModel:
        """Build only the Megatron language backbone."""
        return MLAModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
