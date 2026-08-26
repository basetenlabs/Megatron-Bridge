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

"""Heterogeneous KDA / MLA+DSA block spec for GLM-5.3-Flash.

Why this exists
---------------
Megatron-Core selects an experimental attention variant with a single scalar,
``config.experimental_attention_variant``, and its per-layer dispatch is::

    if is_linear_attention_variant(v):   pattern = get_linear_attention_pattern(config)
    elif v is not None:                  pattern = [1] * num_layers

with ``GDN_ATTENTION_VARIANTS = ("gdn", "gdn2")`` and ``"dsa"`` excluded from
``is_linear_attention_variant``. That admits either *{linear attention mixed with
standard attention}* or *{every layer DSA}* -- never *{KDA mixed with DSA}*, which is
GLM-5.3's 34:11 schedule.

The limitation is in the *variant-selection helper*, not in the layer-spec machinery:
``TransformerBlockSubmodules`` is a list of per-layer ``ModuleSpec`` objects and is
already heterogeneous by construction. So we take the all-DSA block the variant builder
produces and overwrite ``self_attention`` on the KDA layers.

Choice of base builder
----------------------
This builds on ``get_transformer_block_with_experimental_attention_variant_spec``
(keeping ``experimental_attention_variant="dsa"``) rather than on
``get_gpt_decoder_block_spec``. Three reasons, and the third is the important one:

1. ``get_gpt_decoder_layer_specs`` opens with
   ``assert config.experimental_attention_variant is None``, so the standard builder
   cannot be used while advertising DSA at all.
2. The variant builder already resolves the dense/MoE MLP pattern and, on the
   production stack, the mHC residual construction -- so replacing only attention
   leaves the rest of the block exactly as Megatron-Core built it.
3. **The variant scalar stays ``"dsa"``.** Downstream consumers infer real behaviour
   from it -- most importantly the fp32 output projection, which is mirrored on the
   inference side. Building on the standard builder would force the scalar to ``None``
   and silently disable the fp32 head on the trainer *and* the sampler at once:
   consistent on both, so nothing raises and the only symptom is degraded numerics.

This mirrors the arrangement qualified on B200 in the GLM-5.3-Flash bring-up
(``dev_job/model_bringup_hooks/glm53_flash/trainer_block.py`` in the trainers repo),
which preserves production mHC/MoE construction and replaces only the DSA attention.
"""

import copy

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from megatron.bridge.models.glm5_next.glm5_next_kpool import Glm5NextKPoolIndexer
from megatron.bridge.models.glm5_next.glm5_next_layers import Glm5NextLinearAttention


def _install_kpool_indexer(attention_spec) -> None:
    """Point a DSA attention spec at GLM-5.3's k-pool indexer.

    Only the indexer *class* is replaced; its submodules (the two projections, the key
    norm and the weights projection) are what ``Glm5NextKPoolIndexer`` inherits and
    still needs, so they are left exactly as the variant builder created them.

    Mutating in place is safe because the caller works on a per-layer deep copy.
    """
    try:
        indexer_spec = attention_spec.submodules.core_attention.submodules.indexer
    except AttributeError as error:
        raise ValueError(
            "expected a DSA attention spec with core_attention.submodules.indexer; "
            "Megatron-Core's DSA spec shape has changed and the k-pool indexer can no "
            "longer be installed"
        ) from error

    indexer_spec.module = Glm5NextKPoolIndexer


def build_glm5_next_spec(config, vp_stage=None, pp_rank=None):
    """Build GLM-5.3's heterogeneous KDA / MLA+DSA decoder block.

    Every layer is first built as MLA+DSA by the variant builder; the KDA layers then
    have their ``self_attention`` replaced. The MLP schedule and residual construction
    are left exactly as Megatron-Core resolved them.

    Args:
        config: a ``Glm5NextModelProvider``.
        vp_stage: virtual pipeline stage, or None.
        pp_rank: pipeline rank, or None.

    Returns:
        ``TransformerBlockSubmodules`` whose per-layer ``self_attention`` is KDA or
        MLA+DSA according to ``config.glm5_next_kda_layers``.
    """
    if config.virtual_pipeline_model_parallel_size is not None:
        # Same restriction Kimi K3 carries. Lifting it means resolving how the per-layer
        # KDA/DSA pattern maps onto interleaved virtual stages, which also interacts
        # with DSA top-k sharing across pipeline boundaries.
        raise ValueError("GLM-5.3 does not support virtual pipeline parallelism yet")

    # Also enforced by the provider; repeated because this function is reachable with
    # any TransformerConfig, and because building on the wrong variant would silently
    # change which attention every layer gets.
    if config.experimental_attention_variant != "dsa":
        raise ValueError(
            "GLM-5.3 builds on the experimental-variant block builder with "
            "experimental_attention_variant='dsa'; the KDA layers then replace their "
            f"attention. Got {config.experimental_attention_variant!r}."
        )
    if not config.multi_latent_attention:
        raise ValueError("GLM-5.3 requires multi_latent_attention=True for its DSA layers")

    block_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config, vp_stage=vp_stage, pp_rank=pp_rank
    )

    # The schedule itself comes from the checkpoint, not from this rule, but for the
    # record: ref modular:171 Glm5NextTextConfig.__post_init__ derives the default
    # layer_types as `linear_attention` where `idx % 4 != 3` and
    # `deepseek_sparse_attention` otherwise, and normalizes a legacy `full_attention`
    # spelling to `deepseek_sparse_attention` -- which is why the bridge keys on the
    # latter string.
    kda_attention_spec = ModuleSpec(
        module=Glm5NextLinearAttention,
        # KDA does not fuse the input layernorm into its projections. The variant
        # builder chooses each layer's input_layernorm from this same key, so declaring
        # it lets the assertion below confirm the norm it already picked for the DSA
        # spec is equally valid for KDA -- rather than silently relying on it.
        metainfo={"fuse_input_layernorm": False},
    )

    layer_offset = get_transformer_layer_offset(config, vp_stage)

    layer_specs = []
    for local_idx, layer_spec in enumerate(block_spec.layer_specs):
        layer_spec = copy.deepcopy(layer_spec)

        # get_transformer_layer_offset gives this PP rank's 0-indexed global offset;
        # layer_number is 1-indexed, matching Megatron-Core and glm5_next_kda_layers.
        layer_number = layer_offset + local_idx + 1

        if config.is_kda_layer(layer_number):
            dsa_spec = layer_spec.submodules.self_attention
            # If DSA ever starts fusing its input layernorm, the norm the builder chose
            # for this layer would be an IdentityOp and swapping in KDA would drop the
            # pre-attention normalization -- a silent numerical change, not an error.
            if dsa_spec.metainfo["fuse_input_layernorm"] != kda_attention_spec.metainfo["fuse_input_layernorm"]:
                raise ValueError(
                    "DSA and KDA disagree on fuse_input_layernorm, so the input "
                    "layernorm chosen for this layer is not valid for KDA. Re-derive "
                    "input_layernorm here before swapping the attention module."
                )
            layer_spec.submodules.self_attention = kda_attention_spec
        else:
            _install_kpool_indexer(layer_spec.submodules.self_attention)

        layer_specs.append(layer_spec)

    block_spec.layer_specs = layer_specs
    return block_spec
