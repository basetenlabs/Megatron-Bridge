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

"""Megatron Bridge for the GLM-5.3-Flash (``glm5_next``) language backbone.

Registration is by architecture *string* because ``glm5_next`` may be resolved through
remote code.

``AutoBridge.from_hf_pretrained`` resolves the bridge from the checkpoint config, so
this decorator is the whole dispatch story: nothing else needs to know GLM-5.3 exists.

``Glm5NextForConditionalGeneration`` is GLM-5.3's only architecture -- there is no
text-only variant -- so this bridge builds the vision-language model and the text path
is simply the one that passes no images. The tower itself is HuggingFace's
``Glm5NextVisionModel``, embedded verbatim as for GLM-4.5V, which is why its weights map
with a single ``visual.**`` wildcard.

A checkpoint that stores its text fields at the top level has them promoted into
``text_config`` by ``Glm5NextConfig.__post_init__``, so reading ``text_config`` works
either way.
"""

import torch.nn.functional as F

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.glm5_next.glm5_next_kda_mapping import Glm5NextKdaFusedMapping
from megatron.bridge.models.glm5_next.glm5_next_mhc_mapping import mhc_mappings
from megatron.bridge.models.glm5_next.glm5_next_mtp_mapping import mtp_eh_proj_mappings
from megatron.bridge.models.glm5_next.glm5_next_vl_model import Glm5NextVLModel
from megatron.bridge.models.glm5_next.glm5_next_provider import (
    Glm5NextModelProvider,
    Glm5NextVLModelProvider,
)
from megatron.bridge.models.glm5_next.glm5_next_spec import build_glm5_next_spec
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


def _layer_mappings(megatron_layer: str, hf_layer: str) -> list:
    """Every per-layer mapping, for one layer-name prefix pair.

    Called once for the main decoder stack and again for each MTP layer, whose inner
    transformer layer has the same shape. Wildcards make this safe across the hybrid
    stack: a KDA-only name simply does not match on an MLA+DSA layer and vice versa, so
    one set covers both -- including GLM-5.3's MTP layer, which is MLA+DSA.
    """
    megatron_attention = f"{megatron_layer}.self_attention"
    hf_attention = f"{hf_layer}.self_attn"
    indexer = f"{megatron_attention}.core_attention.indexer"

    auto = [
        (f"{megatron_layer}.input_layernorm.weight", f"{hf_layer}.input_layernorm.weight"),
        (f"{megatron_layer}.pre_mlp_layernorm.weight", f"{hf_layer}.post_attention_layernorm.weight"),
        # MLA. These names come from Megatron-Core's absorbed-MLA modules, which the
        # DSA spec builds, so they match GLM-5.2's mapping.
        (f"{megatron_attention}.linear_q_down_proj.weight", f"{hf_attention}.q_a_proj.weight"),
        (f"{megatron_attention}.linear_q_up_proj.weight", f"{hf_attention}.q_b_proj.weight"),
        (f"{megatron_attention}.linear_q_up_proj.layer_norm_weight", f"{hf_attention}.q_a_layernorm.weight"),
        (f"{megatron_attention}.q_layernorm.weight", f"{hf_attention}.q_a_layernorm.weight"),
        (f"{megatron_attention}.linear_kv_down_proj.weight", f"{hf_attention}.kv_a_proj_with_mqa.weight"),
        (f"{megatron_attention}.linear_kv_up_proj.weight", f"{hf_attention}.kv_b_proj.weight"),
        (f"{megatron_attention}.linear_kv_up_proj.layer_norm_weight", f"{hf_attention}.kv_a_layernorm.weight"),
        (f"{megatron_attention}.kv_layernorm.weight", f"{hf_attention}.kv_a_layernorm.weight"),
        (f"{megatron_attention}.linear_proj.weight", f"{hf_attention}.o_proj.weight"),
        # DSA indexer, including GLM-5.3's two k-pool tensors.
        (f"{indexer}.linear_wq_b.weight", f"{hf_attention}.indexer.wq_b.weight"),
        (f"{indexer}.linear_wk.weight", f"{hf_attention}.indexer.wk.weight"),
        (f"{indexer}.k_norm.weight", f"{hf_attention}.indexer.k_norm.weight"),
        (f"{indexer}.k_norm.bias", f"{hf_attention}.indexer.k_norm.bias"),
        (f"{indexer}.linear_weights_proj.weight", f"{hf_attention}.indexer.weights_proj.weight"),

        # MoE router and experts.
        (f"{megatron_layer}.mlp.router.weight", f"{hf_layer}.mlp.gate.weight"),
        (f"{megatron_layer}.mlp.router.expert_bias", f"{hf_layer}.mlp.gate.e_score_correction_bias"),
        (f"{megatron_layer}.mlp.experts.linear_fc2.weight*", f"{hf_layer}.mlp.experts.*.down_proj.weight"),
        (
            f"{megatron_layer}.mlp.experts.local_experts.*.linear_fc2.weight",
            f"{hf_layer}.mlp.experts.*.down_proj.weight",
        ),
        (
            f"{megatron_layer}.mlp.shared_experts.linear_fc2.weight",
            f"{hf_layer}.mlp.shared_experts.down_proj.weight",
        ),
        (f"{megatron_layer}.mlp.linear_fc2.weight", f"{hf_layer}.mlp.down_proj.weight"),
        # The 3 dense layers fuse their pre-MLP norm into linear_fc1 under TE, so the
        # weight arrives as linear_fc1.layer_norm_weight rather than pre_mlp_layernorm.
        (
            f"{megatron_layer}.mlp.linear_fc1.layer_norm_weight",
            f"{hf_layer}.post_attention_layernorm.weight",
        ),
    ]

    # KDA. Split by parallelism explicitly: these modules are not in AutoMapping's
    # registry, so it cannot infer how they shard.
    column_parallel = [
        (f"{megatron_attention}.{name}", f"{hf_attention}.{name}")
        for name in ("g_b_proj.weight",)
    ]
    column_parallel += [
        (f"{megatron_attention}.f_b_proj.weight", f"{hf_attention}.f_b_proj.weight"),
        (f"{megatron_attention}.A_log", f"{hf_attention}.A_log"),
        (f"{megatron_attention}.dt_bias", f"{hf_attention}.dt_bias"),
    ]
    replicated = [
        # GLM-5.3's two k-pool tensors are raw nn.Parameters on the indexer, not parallel
        # linears, so AutoMapping cannot infer their sharding ("Cannot determine
        # parallelism type for module 'Glm5NextKPoolIndexer'"). They are replicated: the
        # indexer is frozen and every rank holds the same copy.
        (f"{indexer}.index_kpool_compress_ape", f"{hf_attention}.indexer.index_kpool_compress_ape"),
        (f"{indexer}.index_kpool_compress_gate", f"{hf_attention}.indexer.index_kpool_compress_gate"),
        (f"{megatron_attention}.f_a_proj.weight", f"{hf_attention}.f_a_proj.weight"),
        (f"{megatron_attention}.g_a_proj.weight", f"{hf_attention}.g_a_proj.weight"),
        # KDA gated output RMSNorm. MCore names it out_norm (see
        # KimiDeltaAttentionSubmodules); the checkpoint calls it o_norm.
        (f"{megatron_attention}.out_norm.weight", f"{hf_attention}.o_norm.weight"),
    ]
    # KDA output projection. MCore names it out_proj; the checkpoint calls it o_proj,
    # which on the MLA+DSA layers instead maps to linear_proj above. Both mappings
    # coexist because a given Megatron parameter matches only one of them.
    row_parallel = [(f"{megatron_attention}.out_proj.weight", f"{hf_attention}.o_proj.weight")]

    mappings = [
        *(AutoMapping(megatron_param=m, hf_param=h) for m, h in auto),
        *(ColumnParallelMapping(megatron_param=m, hf_param=h) for m, h in column_parallel),
        *(ReplicatedMapping(megatron_param=m, hf_param=h) for m, h in replicated),
        *(RowParallelMapping(megatron_param=m, hf_param=h) for m, h in row_parallel),
        GatedMLPMapping(
            megatron_param=f"{megatron_layer}.mlp.linear_fc1.weight",
            gate=f"{hf_layer}.mlp.gate_proj.weight",
            up=f"{hf_layer}.mlp.up_proj.weight",
        ),
        GatedMLPMapping(
            megatron_param=f"{megatron_layer}.mlp.shared_experts.linear_fc1.weight",
            gate=f"{hf_layer}.mlp.shared_experts.gate_proj.weight",
            up=f"{hf_layer}.mlp.shared_experts.up_proj.weight",
        ),
        GatedMLPMapping(
            megatron_param=f"{megatron_layer}.mlp.experts.linear_fc1.weight*",
            gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
            up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
        ),
        GatedMLPMapping(
            megatron_param=f"{megatron_layer}.mlp.experts.local_experts.*.linear_fc1.weight",
            gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
            up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
        ),
    ]

    # KDA's fused input projection and short convolution. Direction is the
    # opposite of Qwen3-Next's: GLM-5.3 ships three per-projection tensors and
    # Megatron-Core's KimiDeltaAttention keeps one fused tensor for each, so this
    # concatenates 3 -> 1 while preserving per-component TP head sharding.
    mappings += [
        Glm5NextKdaFusedMapping(
            megatron_param=f"{megatron_attention}.in_proj.weight",
            q=f"{hf_attention}.q_proj.weight",
            k=f"{hf_attention}.k_proj.weight",
            v=f"{hf_attention}.v_proj.weight",
        ),
        Glm5NextKdaFusedMapping(
            megatron_param=f"{megatron_attention}.conv1d.weight",
            q=f"{hf_attention}.q_conv1d.weight",
            k=f"{hf_attention}.k_conv1d.weight",
            v=f"{hf_attention}.v_conv1d.weight",
        ),
        # Beta is its own module in Megatron-Core, one value per key head
        # (b_proj [64, 4096] against beta_proj's num_key_heads).
        ColumnParallelMapping(
            megatron_param=f"{megatron_attention}.beta_proj.weight",
            hf_param=f"{hf_attention}.b_proj.weight",
        ),
    ]

    # mHC, one hyper-connection around attention and one around the MLP.
    mappings += mhc_mappings(megatron_layer, hf_layer)
    return mappings


@MegatronModelBridge.register_bridge(
    source="Glm5NextForConditionalGeneration",
    target=Glm5NextVLModel,
    provider=Glm5NextVLModelProvider,
    model_type="glm5_next",
)
class Glm5NextBridge(MegatronModelBridge):
    """HF <-> Megatron conversion for GLM-5.3-Flash, vision tower included."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Glm5NextVLModelProvider:
        hf_config = hf_pretrained.config
        text_config = getattr(hf_config, "text_config", hf_config)

        provider = self._base_provider(text_config)
        self._apply_layout(provider, text_config)
        self._apply_attention(provider, text_config)
        self._apply_dsa(provider, text_config)
        self._apply_moe(provider, text_config)
        self._apply_mhc(provider, text_config)

        # The tower is HF's own module built from this config object, so it is handed
        # over whole rather than copied field by field.
        provider.vision_config = getattr(hf_config, "vision_config", None)

        # These live on the top-level config, not text_config, and HF's
        # get_placeholder_mask reads them off the Megatron config at forward time.
        for field in (
            "image_token_id",
            "video_token_id",
            "image_start_token_id",
            "image_end_token_id",
            "video_start_token_id",
            "video_end_token_id",
        ):
            value = getattr(hf_config, field, None)
            if value is None:
                raise ValueError(
                    f"GLM-5.3 checkpoint config is missing {field}; the vision path needs "
                    "it to locate image placeholders in the token stream."
                )
            setattr(provider, field, value)
        if provider.vision_config is None:
            raise ValueError(
                "GLM-5.3 checkpoint has no vision_config; Glm5NextForConditionalGeneration "
                "is a vision-language architecture and its model.visual.* weights would "
                "have nowhere to load."
            )
        return provider

    # ------------------------------------------------------------------ config

    def _base_provider(self, cfg) -> Glm5NextVLModelProvider:
        provider = Glm5NextVLModelProvider(
            num_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            ffn_hidden_size=cfg.intermediate_size,
            num_attention_heads=cfg.num_attention_heads,
            num_query_groups=cfg.num_key_value_heads,
            vocab_size=cfg.vocab_size,
            layernorm_epsilon=cfg.rms_norm_eps,
            # Passed at construction, not assigned afterwards: the provider validates
            # its own layout in __post_init__, so a provider is never briefly alive
            # holding an empty schedule. HF indexes layers from 0 and Megatron-Core's
            # layer_number is 1-indexed; this is the single place that shift happens.
            glm5_next_kda_layers=tuple(
                index + 1
                for index, kind in enumerate(cfg.layer_types)
                if kind == "linear_attention"
            ),
        )
        provider.transformer_layer_spec = build_glm5_next_spec

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = cfg.tie_word_embeddings
        provider.qk_layernorm = True
        provider.multi_latent_attention = True
        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False
        # MTP. GLM-5.3 ships one predict layer (checkpoint layer 45), an MLA+DSA layer
        # with its own MoE, mHC pair and indexer -- consistent with
        # index_share_for_mtp_iteration. Enabled so those weights load and are carried
        # through export rather than dropped on conversion.
        #
        # Verification limit: transformers has no MTP implementation for glm5_next, so
        # unlike KDA / MLA / mHC there is no HF reference to check MTP numerics against.
        # mtp_loss_scaling_factor is therefore the knob to zero for any run whose
        # measured quantity must not be perturbed by an unverified auxiliary loss.
        provider.mtp_num_layers = cfg.num_nextn_predict_layers or None
        if provider.mtp_num_layers:
            # The MTP layer is MLA+DSA, but the decoder stack ends on a KDA layer
            # (index 44, since KDA is idx % 4 != 3), and Megatron-Core copies the last
            # layer's spec by default. Point it at a DSA layer instead. Take the *last*
            # one (44): any DSA layer has the right shape, but Megatron-Core resolves
            # the source layer within the pipeline stage that owns the MTP block, so the
            # latest one is the one most likely to live on that stage under PP > 1.
            dsa_layers = [
                number
                for number in range(1, cfg.num_hidden_layers + 1)
                if not provider.is_kda_layer(number)
            ]
            if not dsa_layers:
                raise ValueError(
                    "GLM-5.3 has no MLA+DSA layer to model the MTP layer on; the KDA "
                    "schedule covers the whole stack, which contradicts layer_types."
                )
            provider.mtp_source_layer_number = dsa_layers[-1]

        # SwiGLU. GLM-5.3's hidden_act is silu; Megatron's default is gelu, and nothing
        # else in this path sets it. KDA refuses to run on anything but SiLU
        # ("FLA causal convolution requires SiLU, got gelu"), which is how this was
        # caught -- but a stack without KDA layers would have trained GELU MLPs against
        # SiLU weights with nothing raised, so it is asserted rather than assumed.
        if cfg.hidden_act != "silu":
            raise ValueError(
                f"GLM-5.3 expects hidden_act='silu', got {cfg.hidden_act!r}. The MLPs and "
                "the KDA short convolution both depend on it."
            )
        provider.activation_func = F.silu

        # Clamped SwiGLU. NOTE: activation_func_clamp_value is documented as applying to
        # MoE layers; GLM-5.3's first three layers are dense and also clamp. Verify the
        # dense path is covered, or those layers diverge from HF.
        provider.activation_func_clamp_value = cfg.swiglu_limit

        return provider

    def _apply_layout(self, provider, cfg) -> None:
        """Translate HF's remaining per-layer schedules onto the provider.

        The KDA schedule itself is passed at construction (see ``_base_provider``)
        because the provider validates it in ``__post_init__``.
        """
        provider.glm5_next_indexer_full_layers = tuple(
            index + 1 for index, kind in enumerate(cfg.indexer_types) if kind == "full"
        )
        # HF gives an explicit dense/sparse list rather than a first_k_dense_replace count.
        provider.moe_layer_freq = [1 if kind == "sparse" else 0 for kind in cfg.mlp_layer_types]

    def _apply_attention(self, provider, cfg) -> None:
        provider.q_lora_rank = cfg.q_lora_rank
        provider.kv_lora_rank = cfg.kv_lora_rank
        provider.qk_head_dim = cfg.qk_nope_head_dim
        provider.v_head_dim = cfg.v_head_dim

        if cfg.qk_rope_head_dim != 0:
            # GLM-5.3 is NoPE and HF's validate_architecture enforces it. Re-checked here
            # so a checkpoint that reintroduces RoPE fails at load rather than training
            # without position information.
            raise ValueError(f"GLM-5.3 expects NoPE attention (qk_rope_head_dim=0), got {cfg.qk_rope_head_dim}")
        provider.qk_pos_emb_head_dim = 0

        # KDA geometry goes onto Megatron-Core's own KDA fields -- the layers are
        # MCore's KimiDeltaAttention, so there is no second set to keep in sync.
        provider.linear_num_key_heads = cfg.linear_num_heads
        provider.linear_num_value_heads = cfg.linear_num_heads
        provider.linear_key_head_dim = cfg.linear_head_dim
        provider.linear_value_head_dim = cfg.linear_head_dim
        provider.linear_conv_kernel_dim = cfg.linear_conv_kernel_dim

        # HF builds both bottlenecks as Linear(hidden_size, linear_head_dim), so the
        # rank is the head dim. Derived rather than defaulted: these two fields are also
        # what selects MCore's low-rank KDA layout over its legacy fused projection.
        provider.kda_f_lora_rank = cfg.linear_head_dim
        provider.kda_gate_lora_rank = cfg.linear_head_dim

        # linear_lower_bound is None only when HF's safe_gate is off. Carrying the
        # pair together keeps the bounded-sigmoid gate and the unbounded-softplus gate
        # from being mixed -- see Glm5NextModelProvider.kda_safe_gate.
        provider.kda_lower_bound = cfg.linear_lower_bound
        provider.kda_safe_gate = cfg.linear_lower_bound is not None

    def _apply_dsa(self, provider, cfg) -> None:
        # Kept at "dsa": the block is built as all-DSA and the KDA layers replace their
        # attention afterwards. See glm5_next_spec for why this matters beyond dispatch.
        provider.experimental_attention_variant = "dsa"

        provider.dsa_indexer_head_dim = cfg.index_head_dim
        provider.dsa_indexer_n_heads = cfg.index_n_heads
        provider.dsa_indexer_topk = cfg.index_topk
        provider.dsa_indexer_k_norm_epsilon = 1e-6
        provider.dsa_kernel_backend = "cudnn"
        provider.dsa_indexer_scoring_relu = True

        # The indexer is frozen -- HF marks its forward @torch.no_grad and GLM-5.2
        # freezes it for LoRA -- so there is no indexer auxiliary loss to weight.
        provider.dsa_indexer_loss_coeff = 0.0
        provider.dsa_indexer_use_sparse_loss = False

        provider.glm5_next_index_kpool = cfg.index_kpool
        provider.glm5_next_index_kpool_always_select_tail = cfg.index_kpool_always_select_tail

        # Cross-layer top-k sharing. Megatron-Core models this as a (frequency, offset)
        # pair; HF ships an explicit per-layer list that can be an arbitrary pattern.
        # Only a regular pattern is expressible, so reject anything else rather than
        # approximating it into a schedule that shares from the wrong layer.
        provider.dsa_indexer_topk_freq, provider.dsa_indexer_skip_topk_offset = _indexer_sharing(cfg.indexer_types)

    def _apply_moe(self, provider, cfg) -> None:
        provider.num_moe_experts = cfg.n_routed_experts
        provider.moe_router_topk = cfg.num_experts_per_tok
        provider.moe_ffn_hidden_size = cfg.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = cfg.moe_intermediate_size * cfg.n_shared_experts
        provider.moe_router_topk_scaling_factor = cfg.routed_scaling_factor

        # n_group == topk_group == 1: no group-limited routing, unlike DeepSeek-V3.
        provider.moe_router_num_groups = None
        provider.moe_router_group_topk = None

        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.moe_token_dispatcher_type = "alltoall"

    def _apply_mhc(self, provider, cfg) -> None:
        """Manifold-constrained hyper-connections.

        GLM-5.3's parameterization matches Megatron-Core's ``HyperConnectionModule``
        term by term: ``fn`` is ``mapping_proj.weight`` (both ``[(2+n)*n, n*hidden]``),
        ``base`` is ``bias``, ``scale`` holds the three alphas, and
        ``_sinkhorn_iterations`` is line-for-line HF's loop. ``_MHC_COMPUTE_H_EPS`` is
        1e-6, which is GLM-5.3's ``hc_eps``.

        The pin this repository now tracks exposes the production spelling
        (``enable_hyper_connections`` / ``num_residual_streams`` / ``use_fused_mhc``),
        where mHC over sparse MoE is qualified -- so the earlier ambiguity between that
        and ``enable_mhc_connections`` / ``mhc_num_residual_streams`` is resolved. The
        variant block builder constructs the hyper-connection modules itself once
        ``enable_hyper_connections`` is set.
        """
        if not getattr(cfg, "mhc", False):
            raise ValueError(
                "GLM-5.3-Flash is defined with mHC hyper-connections (config mhc=True); "
                "refusing to build it without them, since the residual stream shape "
                "differs throughout the stack."
            )

        provider.enable_hyper_connections = True
        provider.num_residual_streams = cfg.hc_mult
        provider.mhc_sinkhorn_iterations = cfg.hc_sinkhorn_iters

        # GLM-5.3 contracts the residual streams with an unweighted mean and ships no
        # hc_head_* weights. DeepSeek-V4's learned contraction is Megatron-Core's
        # default, and leaving it on would apply a randomly-initialized gated sum to the
        # final hidden states with nothing raised. See the flag's docstring in
        # TransformerConfig.
        provider.mhc_learned_output_contract = False

        # GLM-5.3's mHC input norm is Glm5NextTextUnweightedRMSNorm(eps=rms_norm_eps),
        # i.e. the standard rsqrt(mean + eps). Megatron-Core's historical form divides by
        # (RMS + 1e-6) instead, which only agrees while eps is negligible against the
        # RMS -- and GLM-5.3's layer-0 streams have RMS about 0.0079, so it is not.
        # Measured on the real checkpoint: the historical form leaves the mHC collapse
        # 3.46% too large (cos 0.996616 vs HF) and layer 0 at cos 0.9336; setting this
        # brings the collapse to cos 0.999999 and layers 0-2 to exact.
        provider.mhc_rms_norm_eps = cfg.rms_norm_eps

    # ----------------------------------------------------------------- weights

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map GLM-5.3's parameters between HF and Megatron layouts.

        Per-layer mappings come from ``_layer_mappings``, applied to the decoder stack
        and again to each MTP layer. Still open: quantization, noted at the end.
        """
        # Every Megatron-side name is prefixed with ``language_model.``: this bridge builds
        # Glm5NextVLModel, which nests the backbone under that attribute. Without it the
        # patterns resolve against nothing -- measured on the real 45-layer model, 4336 of
        # 4683 parameters went unmapped, the whole grouped-MoE expert set among them, and
        # a load would have silently left them at their initialised values.
        lm = "language_model"
        mappings = [
            AutoMapping(
                megatron_param=f"{lm}.embedding.word_embeddings.weight",
                hf_param="model.language_model.embed_tokens.weight",
            ),
            AutoMapping(megatron_param=f"{lm}.output_layer.weight", hf_param="lm_head.weight"),
            AutoMapping(
                megatron_param=f"{lm}.decoder.final_layernorm.weight",
                hf_param="model.language_model.norm.weight",
            ),
        ]
        mappings += _layer_mappings(
            f"{lm}.decoder.layers.*", "model.language_model.layers.*"
        )

        # MTP. GLM-5.3 appends its predict layers after the main stack, so MTP layer i
        # is checkpoint layer num_hidden_layers + i -- 45, for the single shipped layer.
        text_config = self.hf_config.text_config
        num_mtp_layers = getattr(text_config, "num_nextn_predict_layers", 0) or 0
        num_transformer_layers = text_config.num_hidden_layers
        for mtp_layer in range(num_mtp_layers):
            megatron_mtp = f"{lm}.mtp.layers.{mtp_layer}"
            hf_mtp = f"model.language_model.layers.{mtp_layer + num_transformer_layers}"
            mappings += [
                AutoMapping(
                    megatron_param=f"{megatron_mtp}.enorm.weight",
                    hf_param=f"{hf_mtp}.enorm.weight",
                ),
                AutoMapping(
                    megatron_param=f"{megatron_mtp}.hnorm.weight",
                    hf_param=f"{hf_mtp}.hnorm.weight",
                ),
                # Megatron-Core calls the MTP output norm final_layernorm; HF calls it
                # shared_head.norm, as for GLM-5.2.
                AutoMapping(
                    megatron_param=f"{megatron_mtp}.final_layernorm.weight",
                    hf_param=f"{hf_mtp}.shared_head.norm.weight",
                ),
                # mHC is on, so Megatron-Core wants e_proj/h_proj rather than the fused
                # eh_proj GLM-5.3 ships.
                *mtp_eh_proj_mappings(megatron_mtp, hf_mtp),
            ]
            # Megatron-Core has spelled the MTP layer's inner transformer layer both
            # ways across versions; register both and the absent one simply never
            # matches.
            for layer_prefix in ("transformer_layer", "mtp_model_layer"):
                mappings += _layer_mappings(f"{megatron_mtp}.{layer_prefix}", hf_mtp)

        # Vision tower. The module is HuggingFace's own, so the parameter names match
        # the checkpoint one for one and a single wildcard covers the whole tower --
        # patch_embed, the 24 blocks, downsample, post_layernorm and the merger.
        #
        # ReplicatedMapping, not AutoMapping: the tower is plain torch modules held
        # identically on every rank, and AutoMapping infers sharding from the Megatron
        # module class, so it raises on the first nn.Linear it meets
        # ("Cannot determine parallelism type for module 'Linear' at weight
        # visual.blocks.0.attn.proj.bias"). Same choice GLM-4.5V makes.
        mappings.append(ReplicatedMapping(megatron_param="visual.**", hf_param="model.visual.**"))

        # STILL OPEN -- deliberately absent rather than guessed:
        #
        # 1. Quantization. Resolved to *what* but not *wired*: zai-org/GLM-5.3-Flash
        #    is block FP8 e4m3, weight_block_size [128, 128], dynamic activations,
        #    1509 modules_to_not_convert, 328 GB. That is GLM-5.2's format, so the
        #    dequant-on-load path is reuse rather than new work. Until it is wired,
        #    point the trainer at zai-org/GLM-5.3-Flash-BF16 (642.6 GB, 321B params,
        #    no quantization_config, identical parameter names).
        return MegatronMappingRegistry(*mappings)


def _indexer_sharing(indexer_types: list[str]) -> tuple[int, int]:
    """Express HF's per-layer indexer schedule as Megatron-Core's (frequency, offset).

    ``"full"`` runs the indexer; ``"shared"`` reuses the previous full layer's top-k.
    Megatron-Core derives the same schedule from ``dsa_indexer_topk_freq`` and
    ``dsa_indexer_skip_topk_offset``, which can only express a regular pattern.

    Returns ``(1, 0)`` when every layer is full -- the default, and the case where
    sharing is simply off.
    """
    full = [index for index, kind in enumerate(indexer_types) if kind == "full"]
    if not full:
        raise ValueError("indexer_types has no 'full' layer; nothing would compute top-k indices")

    if len(full) == len(indexer_types):
        return 1, 0

    gaps = {b - a for a, b in zip(full, full[1:])}
    if len(gaps) != 1:
        # An irregular pattern would be silently rounded into a regular one that shares
        # from the wrong layer, so refuse it instead.
        raise ValueError(
            f"indexer_types is not a regular pattern (gaps between 'full' layers: {sorted(gaps)}); "
            "Megatron-Core can only express a fixed frequency and offset"
        )

    return gaps.pop(), full[0]
