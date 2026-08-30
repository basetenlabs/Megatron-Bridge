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

"""GLM-5.3-Flash vision-language model: HF vision tower + Megatron language backbone.

Follows the arrangement already used for GLM-4.5V (``models/glm_vl``): the vision tower
runs as HuggingFace's own module, and only the language stack is Megatron's. That keeps
the vision weight mapping to a single wildcard (``visual.**`` <- ``model.visual.**``),
because the module embedded here is the one the checkpoint was saved from.

``Glm5NextForConditionalGeneration`` is GLM-5.3's *only* architecture -- there is no
text-only variant -- so this one class serves both. With no ``pixel_values`` the vision
tower is simply never called, which is the text path.

Simpler than GLM-4.5V in one respect that matters. GLM-4.5V spends most of its forward
computing mRoPE position ids, including unpacking and repacking THD batches and aligning
``mm_token_type_ids`` against collate padding. GLM-5.3 is NoPE -- ``qk_rope_head_dim``
is 0, ``mla_use_nope`` is set, and HF's ``Glm5NextModel`` exposes no ``get_rope_index``
at all -- so none of that exists here. Position ids pass through untouched.

The vision tower's own rotary embedding (``Glm5NextVisionRotaryEmbedding``) is internal
to the HF module and unaffected.
"""

import types
from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


class Glm5NextVLModel(MegatronModule):
    """GLM-5.3-Flash: HF ``Glm5NextVisionModel`` in front of the Megatron backbone.

    Args:
        config: a ``Glm5NextVLModelProvider``.
        pre_process: build the vision tower and the embedding (first PP stage).
        post_process: build the output head (last PP stage).
        vp_stage: virtual pipeline stage, or None.
    """

    def __init__(
        self,
        config: GPTModelProvider,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        # Imported lazily: the language-only path must not require a transformers build
        # that carries glm5_next.
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextModel,
            Glm5NextVisionModel,
        )

        if pre_process:
            self.visual = Glm5NextVisionModel._from_config(config.vision_config)
            # The HF tower's parameters are not Megatron modules, so they need marking
            # for TP gradient sync explicitly.
            hook_hf_module_setattr_for_tp_grad_sync(self.visual)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Megatron's finalize-grad path looks these up on the top-level module.
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        # Reuse HF's own feature extraction and placeholder-masking rather than
        # reimplementing the grid arithmetic. Note these come from Glm5NextModel, which
        # is the class that owns `self.visual` upstream, so they expect exactly the
        # attribute layout above.
        self.get_image_features = types.MethodType(Glm5NextModel.get_image_features, self)
        self.get_video_features = types.MethodType(Glm5NextModel.get_video_features, self)
        self.get_placeholder_mask = types.MethodType(Glm5NextModel.get_placeholder_mask, self)

        self.config.spatial_merge_size = getattr(config.vision_config, "spatial_merge_size", 2)

        # The borrowed HF methods reach through ``self.config``, which here is the
        # Megatron provider rather than a transformers config, so anything they read has
        # to be present on it. ``spatial_merge_size`` above is the same arrangement
        # GLM-4.5V uses. ``return_dict`` is read by transformers' generic output wrapper
        # as of 5.16 (``generic.py`` consults ``self.config.return_dict``); older
        # versions did not, which is why the GLM-4.5V model never needed it.
        if not hasattr(self.config, "return_dict"):
            self.config.return_dict = True

    def set_input_tensor(self, input_tensor) -> None:
        """Set this model chunk's input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        padding_mask: Optional[Tensor] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Embed text, splice in vision features, then run the Megatron backbone."""
        if self.pre_process:
            if inputs_embeds is None:
                # Megatron emits [seq, batch, hidden]; HF's masking helpers and
                # masked_scatter below work in [batch, seq, hidden].
                inputs_embeds = self.language_model.embedding(input_ids=input_ids, position_ids=None)
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw).pooler_output
                image_embeds = torch.cat(image_embeds, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = self._splice(
                    inputs_embeds,
                    input_ids,
                    image_embeds,
                    self.config.image_token_id,
                    packed_seq_params,
                    kind="image",
                )

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
                video_embeds = torch.cat(video_embeds, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = self._splice(
                    inputs_embeds,
                    input_ids,
                    video_embeds,
                    self.config.video_token_id,
                    packed_seq_params,
                    kind="video",
                )

            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

            if self.config.sequence_parallel:
                tp_group = (
                    self.config._pg_collection.tp
                    if self.config._pg_collection is not None
                    else None
                )
                inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds, group=tp_group)
                if padding_mask is not None:
                    # GPTModel scatters this alongside its own embedding, but that
                    # branch is skipped whenever decoder_input is supplied -- which
                    # is always, here. It is the MoE router's THD-padding mask and
                    # reaches the decoder, so an unscattered one would disagree with
                    # the hidden states it is indexed against. Same transpose dance
                    # as upstream: the mask is [batch, seq].
                    padding_mask = (
                        scatter_to_sequence_parallel_region(
                            padding_mask.transpose(0, 1).contiguous(), group=tp_group
                        )
                        .transpose(0, 1)
                        .contiguous()
                    )

        # No mRoPE step: GLM-5.3 is NoPE, so position_ids need no vision-aware
        # reconstruction and are passed through as given.
        #
        # input_ids are forwarded even though the embeddings are already computed.
        # GPTModel skips its own embedding whenever decoder_input is given
        # (``if decoder_input is not None: pass``), so they are not used twice -- but MTP
        # needs them: _postprocess hands input_ids to self.mtp, which rolls them by one
        # position to build its next-next-token targets. Passing None there, as the
        # GLM-4.5V model does, raises
        # ``roll(): argument 'input' must be Tensor, not NoneType`` as soon as MTP is
        # enabled. GLM-4.5V never hits it because it does not enable MTP; GLM-5.3 ships
        # a predict layer, so vision and MTP have to coexist.
        return self.language_model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )

    def _cp_group(self):
        """The context-parallel process group, or None when CP is off."""
        collection = getattr(self.config, "_pg_collection", None)
        group = getattr(collection, "cp", None) if collection is not None else None
        return group if group is not None and group.size() > 1 else None

    def _local_feature_index(self, placeholders, packed_seq_params, cp_group, n_features=None):
        """Which multimodal feature row each local placeholder wants.

        Under context parallelism the sequence reaching this model is already one
        zigzag THD shard, but the vision tower produces every image's features in
        global document order. ``masked_scatter`` consumes that tensor from the
        front, so on any rank but 0 it would splice the wrong images -- silently,
        because the shapes still line up.

        The fix is to look each local placeholder up by its rank among *global*
        placeholders. Positions come from the same helper the DSA indexer uses for
        its own CP bookkeeping, so this makes no independent assumption about the
        zigzag layout; the global placeholder mask is rebuilt by all-gathering the
        local ones and scattering them into those positions.

        That all-gather is a collective, so every rank in the CP group has to reach
        it. It holds because the data path attaches the batch's vision tensors to
        every CP rank whole rather than sharding them, which makes ``pixel_values is
        not None`` -- the condition guarding this call -- uniform across the group.
        A path that gave images to only some CP ranks would deadlock here.

        Indexing also makes the tower's gradients come out right, which is a second
        reason not to reach for a cheaper trick. Every global placeholder belongs to
        exactly one rank's shard, so the index sets are disjoint; the write's
        backward hands each rank gradient only for the feature rows it read, and the
        gradient reduction across the CP group reassembles the whole thing with
        nothing double-counted. The tower runs redundantly per rank, but it is not
        trained redundantly.

        The redundant forward is a real cost on image-dense data, and worth naming
        rather than waving at. One image per row makes it a rounding error against a
        321B language stack. MMLongBench-128K's ICL split averages 566 images per
        row -- roughly half a million patches through a depth-24, hidden-1024 tower
        -- and paying that on all eight CP ranks is not free. The fix, when it
        matters, is to split the images across the CP group and all-gather the
        features before this lookup; the lookup itself is unaffected, since it
        addresses features by global row either way.
        """
        from megatron.core.transformer.experimental_attention_variant import dsa_layout

        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError(
                "GLM-5.3 vision under context parallelism requires packed THD sequences: "
                "the local-to-global position map is derived from cu_seqlens."
            )

        cp_size, cp_rank = cp_group.size(), cp_group.rank()
        local_rows = placeholders.numel()
        cu_seqlens_q, _ = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
        device = placeholders.device

        positions = [
            dsa_layout.build_packed_allgather_cp_local_positions(
                cu_seqlens_q.to(device=device, dtype=torch.int64),
                cp_size,
                rank,
                device,
                output_size=local_rows,
                cu_seqlens_cover_output=False,
            )
            for rank in range(cp_size)
        ]

        # Padded rows are given positions past the real tokens, so the scratch mask
        # is sized for that tail rather than for the sequence alone. Padding is never
        # a placeholder, so those rows contribute nothing to the running count.
        span = 2 * cp_size * local_rows
        global_placeholders = torch.zeros(span, dtype=torch.int32, device=device)
        gathered = torch.empty(cp_size * local_rows, dtype=torch.int32, device=device)
        torch.distributed.all_gather_into_tensor(
            gathered, placeholders.to(torch.int32).contiguous(), group=cp_group
        )
        flat_positions = torch.cat(positions, dim=0).clamp_(max=span - 1)
        global_placeholders[flat_positions] = gathered

        counts = global_placeholders.to(torch.int64)
        if n_features is not None:
            # The cp=1 path gets this check from HF's get_placeholder_mask, which
            # cannot run here (it counts against the local shard). Without it a
            # tower/token-count disagreement reads as a plausible index instead of
            # an error -- the exact failure mode this whole method exists to stop.
            total = int(counts.sum().item())
            if total != n_features:
                raise ValueError(
                    f"GLM-5.3 vision splice: {total} placeholder(s) across the "
                    f"context-parallel group but {n_features} feature row(s) from the "
                    "tower; the expected token count per image disagrees with the "
                    "processor's grid."
                )
        feature_index = torch.cumsum(counts, dim=0) - 1
        local_index = feature_index.index_select(0, positions[cp_rank].clamp_(max=span - 1))
        return local_index[placeholders]

    def _splice(self, inputs_embeds, input_ids, features, token_id, packed_seq_params, *, kind):
        """Write multimodal features over their placeholder tokens.

        ``inputs_embeds`` is [batch, seq, hidden] here, matching HuggingFace's own
        masking helpers.
        """
        cp_group = self._cp_group()
        if cp_group is None:
            mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                **{f"{kind}_features": features},
            )
            return inputs_embeds.masked_scatter(mask, features)

        if inputs_embeds.size(0) != 1:
            raise ValueError(
                "GLM-5.3 vision under context parallelism expects packed THD batches "
                f"(batch 1), got batch {inputs_embeds.size(0)}."
            )

        placeholders = input_ids.view(-1) == token_id
        index = self._local_feature_index(
            placeholders, packed_seq_params, cp_group, n_features=features.size(0)
        )
        rows = inputs_embeds.view(-1, inputs_embeds.size(-1)).clone()
        rows[placeholders] = features.index_select(0, index).to(rows.dtype)
        return rows.view_as(inputs_embeds)

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ) -> None:
        """Set ``requires_grad = False`` on whole submodules.

        ``freeze_vision_model`` covers the patch embedding and transformer blocks;
        ``freeze_vision_projection`` covers the merger, which is the part that projects
        into the language model's 4096-wide space.
        """
        modules = []
        if freeze_language_model:
            modules.append(self.language_model)
        if self.pre_process:
            visual = getattr(self, "visual", None)
            if freeze_vision_model and visual is not None:
                modules += [visual.patch_embed, visual.blocks]
            if freeze_vision_projection and visual is not None:
                modules.append(visual.merger)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
