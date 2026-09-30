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

"""Kimi K3 vision-language model: MoonViT3d tower in front of the Megatron K3 backbone.

K3 is NoPE, so like GLM-5.3 there are no vision-aware position ids to rebuild; image
features are written over the placeholder embeddings and position ids pass through.
"""

from typing import TYPE_CHECKING

import torch
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from megatron.bridge.models.common.vision_splice import (
    context_parallel_group,
    scatter_spliced_embeddings,
    splice_features,
)
from megatron.bridge.models.kimi.kimi_k3_vision import KimiK3VisionProjector, KimiK3VisionTower
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext
    from megatron.core.packed_seq_params import PackedSeqParams

    from megatron.bridge.models.kimi.kimi_k3_provider import KimiK3ModelProvider


class KimiK3VLModel(MegatronModule):
    """Kimi K3 with its vision tower; with no ``pixel_values`` it is the text model."""

    def __init__(
        self,
        config: "KimiK3ModelProvider",
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=config)
        self.language_model = config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
        # The backbone resolves None to "first/last pipeline stage".
        self.pre_process = self.language_model.pre_process
        self.post_process = self.language_model.post_process
        self.vp_stage = vp_stage

        if self.pre_process:
            self.vision_tower = KimiK3VisionTower(config.vision_config)
            self.mm_projector = KimiK3VisionProjector(config.vision_config)
            # Plain torch modules replicated on every TP rank: mark them for TP grad sync.
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)
            hook_hf_module_setattr_for_tp_grad_sync(self.mm_projector)
        # Megatron's finalize-grad path looks these up on the top-level module.
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

    def set_input_tensor(self, input_tensor: Tensor | list[Tensor]) -> None:
        """Set this model chunk's input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        labels: Tensor | None = None,
        runtime_gather_output: bool | None = None,
        packed_seq_params: "PackedSeqParams | None" = None,
        padding_mask: Tensor | None = None,
        *,
        loss_mask: Tensor | None = None,
        inference_context: "BaseInferenceContext | None" = None,
    ) -> Tensor:
        """Embed text, write image features over the placeholders, then run the backbone.

        ``image_grid_thw`` is the processor's ``grid_thws``: one ``[t, h, w]`` patch grid
        per image.
        """
        decoder_input = None
        if self.pre_process and pixel_values is not None:
            decoder_input, padding_mask = self._embed_with_images(
                input_ids, pixel_values, image_grid_thw, packed_seq_params, padding_mask
            )

        # input_ids stay for GPTModel's own embedding on text-only batches; with
        # decoder_input set it skips the embedding and only MTP would read them.
        return self.language_model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            inference_context=inference_context,
        )

    def freeze(
        self, *, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ) -> None:
        """Set ``requires_grad = False`` on whole submodules."""
        modules = []
        if freeze_language_model:
            modules.append(self.language_model)
        if self.pre_process and freeze_vision_model:
            modules.append(self.vision_tower)
        if self.pre_process and freeze_vision_projection:
            modules.append(self.mm_projector)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _embed_with_images(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        packed_seq_params: "PackedSeqParams | None",
        padding_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        # [seq, batch, hidden] -> [batch, seq, hidden] for the splice.
        embeds = self.language_model.embedding(input_ids=input_ids, position_ids=None).transpose(0, 1)
        tower_dtype = self.mm_projector.post_norm.weight.dtype
        features = self.mm_projector(self.vision_tower(pixel_values.to(tower_dtype), image_grid_thw))
        embeds = splice_features(
            embeds,
            input_ids=input_ids,
            features=features,
            token_id=self.config.media_placeholder_token_id,
            packed_seq_params=packed_seq_params,
            cp_group=context_parallel_group(self.config),
        )
        return scatter_spliced_embeddings(embeds.transpose(0, 1).contiguous(), padding_mask, self.config)
