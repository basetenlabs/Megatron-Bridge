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

with ``GDN_ATTENTION_VARIANTS = ("gdn", "gdn2")`` and ``"dsa"`` explicitly excluded from
``is_linear_attention_variant``. That admits either *{linear attention mixed with
standard attention}* or *{every layer DSA}* -- never *{KDA mixed with DSA}*, which is
GLM-5.3's 34:11 schedule.

The limitation is in the *variant-selection helper*, not in the layer-spec machinery:
``TransformerBlockSubmodules`` is a list of per-layer ``ModuleSpec`` objects and is
already heterogeneous by construction. So we build the standard block and overwrite
``self_attention`` per layer, exactly as ``kimi_k3_spec.build_kimi_k3_spec`` does.

Where this differs from Kimi K3
-------------------------------
K3 hand-builds its MLA half inside a single attention module. We instead call
``get_dsa_module_spec_for_backend`` and use its spec unchanged: it returns a complete
``GlmAbsorbedMLASelfAttention`` + ``DSAttention`` + ``DSAIndexer`` tree, and its only
precondition is ``multi_latent_attention``. That keeps the entire sparse-attention half
on an upstream code path that GLM-5.2 already exercises, and confines new code to KDA.

The constraint that shapes everything
-------------------------------------
``get_gpt_decoder_layer_specs`` -- reached through ``get_gpt_decoder_block_spec`` --
opens with ``assert config.experimental_attention_variant is None``. So the provider
must leave that scalar unset, which is why the fp32 LM head cannot be inferred from it.
See ``Glm5NextModelProvider.glm5_next_requires_fp32_lm_head``.
"""

import copy

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    _get_backend_spec_provider,
    get_dsa_module_spec_for_backend,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from megatron.bridge.models.glm5_next.glm5_next_layers import Glm5NextLinearAttention


def build_glm5_next_spec(config, vp_stage=None):
    """Build GLM-5.3's heterogeneous KDA / MLA+DSA decoder block.

    The dense/MoE MLP schedule is left exactly as ``get_gpt_decoder_block_spec``
    resolves it from ``config.moe_layer_freq``; only attention is replaced.

    Args:
        config: a ``Glm5NextModelProvider``.
        vp_stage: virtual pipeline stage, or None.

    Returns:
        ``TransformerBlockSubmodules`` whose per-layer ``self_attention`` is KDA or
        MLA+DSA according to ``config.glm5_next_kda_layers``.
    """
    if config.virtual_pipeline_model_parallel_size is not None:
        # Same restriction Kimi K3 carries. Lifting it means resolving how the per-layer
        # KDA/DSA pattern maps onto interleaved virtual stages, which also interacts
        # with DSA top-k sharing across pipeline boundaries.
        raise ValueError("GLM-5.3 does not support virtual pipeline parallelism yet")

    # Also checked in the provider's __post_init__; repeated here because this function
    # is reachable with any TransformerConfig and the Megatron-Core assert it would
    # otherwise trip does not explain the cause.
    if config.experimental_attention_variant is not None:
        raise ValueError(
            "GLM-5.3 builds its block from get_gpt_decoder_block_spec, which requires "
            "experimental_attention_variant=None; DSA layers receive their module spec "
            f"directly instead. Got {config.experimental_attention_variant!r}."
        )
    if not config.multi_latent_attention:
        raise ValueError("GLM-5.3 requires multi_latent_attention=True for its DSA layers")

    # Standard homogeneous block: every layer starts as MLA, with the dense/MoE MLP
    # pattern already resolved.
    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)

    # No public accessor exists for this; the three call sites inside Megatron-Core all
    # use the private helper. Worth upstreaming a public one rather than each hybrid
    # model reaching in.
    backend = _get_backend_spec_provider(config=config)

    # Built once and shared across layers: a ModuleSpec is a description, not an
    # instance, so every DSA layer constructing from the same spec is correct.
    dsa_attention_spec = get_dsa_module_spec_for_backend(config=config, backend=backend)

    kda_attention_spec = ModuleSpec(
        module=Glm5NextLinearAttention,
        # KDA does not fuse the input layernorm into its projections, so the layer needs
        # a standalone one. Declared here so the loop below can treat both branches
        # uniformly, matching how Megatron-Core reads this key.
        metainfo={"fuse_input_layernorm": False},
    )

    rms_norm = config.normalization == "RMSNorm"
    layer_offset = get_transformer_layer_offset(config, vp_stage)

    layer_specs = []
    for local_idx, layer_spec in enumerate(block_spec.layer_specs):
        layer_spec = copy.deepcopy(layer_spec)

        # get_transformer_layer_offset gives this PP rank's 0-indexed global offset;
        # layer_number is 1-indexed, matching Megatron-Core and glm5_next_kda_layers.
        layer_number = layer_offset + local_idx + 1
        attention_spec = kda_attention_spec if config.is_kda_layer(layer_number) else dsa_attention_spec

        layer_spec.submodules.self_attention = attention_spec

        # Replacing self_attention can invalidate the input layernorm the standard block
        # chose. The MLA spec that get_gpt_decoder_block_spec builds may fuse the norm
        # into its first projection and leave input_layernorm as IdentityOp; neither
        # replacement spec fuses, so leaving it would drop the pre-attention norm
        # entirely -- a silent numerical change rather than an error. Re-derive it the
        # same way Megatron-Core's own heterogeneous builder does.
        layer_spec.submodules.input_layernorm = (
            IdentityOp
            if attention_spec.metainfo["fuse_input_layernorm"]
            else backend.layer_norm(rms_norm=rms_norm, for_qk=False)
        )

        layer_specs.append(layer_spec)

    block_spec.layer_specs = layer_specs
    return block_spec
