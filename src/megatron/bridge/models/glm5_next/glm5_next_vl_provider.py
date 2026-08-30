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

"""Provider for GLM-5.3-Flash with its vision tower.

The text backbone is configured entirely by ``Glm5NextBridge``; this adds only what
the vision half needs. It subclasses ``MLAModelProvider`` directly, which is the
provider the text bridge already builds, so the two halves stay independent: dropping
vision means building ``MLAModelProvider`` instead of this class and changing nothing
else.
"""

from dataclasses import dataclass

from megatron.bridge.models.mla_provider import MLAModelProvider


@dataclass
class Glm5NextVLModelProvider(MLAModelProvider):
    """GLM-5.3-Flash including the vision tower.

    ``Glm5NextForConditionalGeneration`` is the model's only architecture, so every
    GLM-5.3 checkpoint carries a tower. Passing no ``pixel_values`` gives the text
    path; the tower is then constructed but never called.

    Vision geometry is not declared here. The tower is HuggingFace's own
    ``Glm5NextVisionModel``, built from ``vision_config``, so the config object is the
    single source of truth and there is no second copy to drift. For reference, the
    shipped values are depth 24, hidden 1024, 16 heads, image 448, patch 14, spatial
    merge 2, temporal patch 2, out_hidden 4096 -- the GLM-4.5V tower rescaled.
    """

    # Set by the bridge from the HF config; typed loosely because it is a transformers
    # config object, not a Megatron dataclass.
    vision_config: object = None

    # Multimodal token ids. HF's get_placeholder_mask -- which the model reuses rather
    # than reimplementing -- reads these off the *Megatron* config, so they have to live
    # here. Defaults are zai-org/GLM-5.3-Flash's; the bridge overwrites them from the
    # checkpoint.
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833

    scatter_embedding_sequence_parallel: bool = False
    """The vision-language wrapper owns the sequence-parallel scatter, not the embedding.

    The splice works over ``[batch, seq, hidden]`` with every image placeholder of the
    rank's own shard present, so the wrapper needs embeddings that the embedding layer
    has not already reduce-scattered, and scatters afterwards. Left at Megatron's
    ``True`` default the sequence is scattered twice: measured at TP=8 with seq 8192,
    the DSA indexer saw ``x`` of 128 positions where 1024 was correct.

    It cannot be fixed by flipping the flag after construction:
    ``LanguageModelEmbedding`` derives ``reduce_scatter_embeddings`` from it in
    ``__init__`` and passes that into ``VocabParallelEmbedding``, so a late flip leaves
    the reduce-scatter path active and the sequence still sharded twice.

    ``GPTModel`` reads the same flag to decide whether to scatter a supplied
    ``decoder_input`` itself, so this keeps exactly one scatter in the graph."""

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build the VL model, applying any requested freezes."""
        from megatron.bridge.models.glm5_next.glm5_next_vl_model import Glm5NextVLModel

        if self.vision_config is None:
            raise ValueError(
                "Glm5NextVLModelProvider requires vision_config; GLM-5.3-Flash ships a "
                "vision tower and its weights would otherwise have nowhere to load."
            )

        # MTP + sequence parallelism + the vision splice cannot all hold at once, and
        # the symptom is an opaque shape error deep in MTP rather than anything
        # nameable:
        #
        #   RuntimeError: The size of tensor a (65536) must match the size of tensor b
        #   (8192) at non-singleton dimension 0
        #   multi_token_prediction.py:2172 _concat_embeddings
        #
        # The splice needs an embedding that does not scatter, so this provider sets
        # scatter_embedding_sequence_parallel=False. MTP then re-embeds input_ids
        # through that same embedding while its hidden states are SP-sharded, and the
        # two no longer line up. Measured at TP=8 / EP=1 / ETP=8, seq 8192. The text
        # bridge sets mtp_num_layers=None, so this only fires if MTP is turned back on.
        if self.sequence_parallel and self.mtp_num_layers and self.tensor_model_parallel_size > 1:
            raise ValueError(
                "GLM-5.3 cannot combine MTP with sequence parallelism: the vision splice "
                "requires an embedding that does not scatter, and MTP's own embedding call "
                "then disagrees with its SP-sharded hidden states. Use "
                "tensor_model_parallel_size=1, or set mtp_num_layers=None."
            )

        model = Glm5NextVLModel(
            self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None):
        """Build only the Megatron language backbone, for the VL model to wrap."""
        return MLAModelProvider.provide(
            self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
