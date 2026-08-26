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
remote code, and because GLM-5.3-Flash is a ``ForConditionalGeneration`` checkpoint
whose language backbone is bridged on its own -- the same arrangement Kimi K3 uses.

``AutoBridge.from_hf_pretrained`` resolves the bridge from the checkpoint config, so
this decorator is the whole dispatch story: nothing else needs to know GLM-5.3 exists.

The vision tower is out of scope. A text-only Flash checkpoint stores its text fields
at the top level and ``Glm5NextConfig.__post_init__`` promotes them into
``text_config``, so reading from ``text_config`` works either way.
"""

from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.glm5_next.glm5_next_provider import Glm5NextModelProvider
from megatron.bridge.models.glm5_next.glm5_next_spec import build_glm5_next_spec
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(
    source="Glm5NextForConditionalGeneration",
    target=GPTModel,
    provider=Glm5NextModelProvider,
    model_type="glm5_next",
)
class Glm5NextBridge(MegatronModelBridge):
    """HF <-> Megatron conversion for GLM-5.3-Flash (text backbone only)."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Glm5NextModelProvider:
        hf_config = hf_pretrained.config
        text_config = getattr(hf_config, "text_config", hf_config)

        provider = self._base_provider(text_config)
        self._apply_layout(provider, text_config)
        self._apply_attention(provider, text_config)
        self._apply_dsa(provider, text_config)
        self._apply_moe(provider, text_config)
        self._apply_mhc(provider, text_config)
        return provider

    # ------------------------------------------------------------------ config

    def _base_provider(self, cfg) -> Glm5NextModelProvider:
        provider = Glm5NextModelProvider(
            num_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            ffn_hidden_size=cfg.intermediate_size,
            num_attention_heads=cfg.num_attention_heads,
            num_query_groups=cfg.num_key_value_heads,
            vocab_size=cfg.vocab_size,
            layernorm_epsilon=cfg.rms_norm_eps,
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
        provider.mtp_num_layers = None  # MTP is unsupported, as for GLM-5.2.

        # Clamped SwiGLU. NOTE: activation_func_clamp_value is documented as applying to
        # MoE layers; GLM-5.3's first three layers are dense and also clamp. Verify the
        # dense path is covered, or those layers diverge from HF.
        provider.activation_func_clamp_value = cfg.swiglu_limit

        return provider

    def _apply_layout(self, provider, cfg) -> None:
        """Translate HF's per-layer schedules onto the provider.

        HF indexes layers from 0; the provider uses Megatron-Core's 1-indexed
        ``layer_number``. Converting here, once, is the only place that shift happens.
        """
        provider.glm5_next_kda_layers = tuple(
            index + 1 for index, kind in enumerate(cfg.layer_types) if kind == "linear_attention"
        )
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

        provider.glm5_next_linear_num_heads = cfg.linear_num_heads
        provider.glm5_next_linear_head_dim = cfg.linear_head_dim
        provider.glm5_next_linear_conv_kernel_size = cfg.linear_conv_kernel_dim
        provider.glm5_next_kda_gate_lower_bound = cfg.linear_lower_bound

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
        exactly -- ``fn`` is ``mapping_proj.weight`` (both ``[(2+n)*n, n*hidden]``),
        ``base`` is ``bias``, and ``scale`` holds the three alphas.

        NOT WIRED YET, deliberately. Two Megatron-Core versions disagree about how this
        is spelled and whether it works at all with MoE:

        * The production trainer image exposes ``enable_hyper_connections`` /
          ``num_residual_streams`` / ``use_fused_mhc``, and mHC over sparse MoE is
          qualified there.
        * The Megatron-Core pinned by this repository exposes
          ``enable_mhc_connections`` / ``mhc_num_residual_streams``, and its
          ``HyperConnectionTransformerLayer`` raises ``NotImplementedError`` on any MoE
          layer. GLM-5.3 is 42 MoE layers of 45, and the ``HyperConnectionHybridLayer``
          that error points to does not exist in this pin or upstream.

        Setting the production spelling against this pin would assign undeclared
        attributes: a non-frozen dataclass accepts them, nothing reads them, and mHC
        would silently never turn on. Leaving it unset is the honest state until the
        target Megatron-Core is settled.
        """
        del provider, cfg

    # ----------------------------------------------------------------- weights

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map GLM-5.3's parameters between HF and Megatron layouts.

        Wildcards are safe across the hybrid stack: a mapping resolves against the
        parameters that actually exist, so a KDA-only name simply does not match on an
        MLA+DSA layer and vice versa. Kimi K3 relies on the same property.

        NOT COMPLETE. The MLA, DSA-indexer, MoE and embedding groups below are the ones
        whose Megatron-side names are settled. Three groups are still open and are
        listed at the end rather than guessed at.
        """
        megatron_layer = "decoder.layers.*"
        hf_layer = "model.language_model.layers.*"
        megatron_attention = f"{megatron_layer}.self_attention"
        hf_attention = f"{hf_layer}.self_attn"
        indexer = f"{megatron_attention}.core_attention.indexer"

        auto = [
            ("embedding.word_embeddings.weight", "model.language_model.embed_tokens.weight"),
            ("output_layer.weight", "lm_head.weight"),
            ("decoder.final_layernorm.weight", "model.language_model.norm.weight"),
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
            (f"{indexer}.index_kpool_compress_ape", f"{hf_attention}.indexer.index_kpool_compress_ape"),
            (f"{indexer}.index_kpool_compress_gate", f"{hf_attention}.indexer.index_kpool_compress_gate"),
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
        ]

        # KDA. Split by parallelism explicitly: these modules are not in AutoMapping's
        # registry, so it cannot infer how they shard.
        column_parallel = [
            (f"{megatron_attention}.{name}", f"{hf_attention}.{name}")
            for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "b_proj.weight", "g_b_proj.weight")
        ]
        column_parallel += [
            (f"{megatron_attention}.f_b_proj.weight", f"{hf_attention}.forget_gate.f_b_proj.weight"),
            (f"{megatron_attention}.A_log", f"{hf_attention}.forget_gate.A_log"),
            (f"{megatron_attention}.dt_bias", f"{hf_attention}.forget_gate.dt_bias"),
        ]
        replicated = [
            (f"{megatron_attention}.f_a_proj.weight", f"{hf_attention}.forget_gate.f_a_proj.weight"),
            (f"{megatron_attention}.g_a_proj.weight", f"{hf_attention}.g_a_proj.weight"),
            (f"{megatron_attention}.o_norm.weight", f"{hf_attention}.o_norm.weight"),
        ]
        row_parallel = [(f"{megatron_attention}.o_proj.weight", f"{hf_attention}.o_proj.weight")]

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

        # STILL OPEN -- deliberately absent rather than guessed:
        #
        # 1. The KDA short convolution. HF holds one depthwise conv1d over 3*qkv_dim
        #    covering q, k and v; this layer keeps three per-projection convolutions
        #    because they shard by head under TP while one fused kernel does not. The
        #    mapping therefore has to split one HF tensor three ways, which needs a
        #    custom mapping rather than a name pair.
        # 2. mHC (hc_attn_fn / hc_attn_base / hc_attn_scale and the ffn equivalents).
        #    Blocked on the same question as _apply_mhc, and `scale` is one [3] tensor on
        #    the HF side against three scalar parameters on the Megatron side -- also a
        #    shape-changing mapping.
        # 3. Quantization. The checkpoint's format is not stated in the HF config, so
        #    whether a dequant-on-load step is needed is unresolved.
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
