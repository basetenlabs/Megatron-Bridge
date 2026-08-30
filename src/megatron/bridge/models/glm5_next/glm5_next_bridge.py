# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint and model configuration bridge for GLM-5.3-Flash."""

from typing import Mapping

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.conversion.quantization_utils import maybe_dequantize_fp8_blockwise
from megatron.bridge.models.glm5_next.glm5_next_vl_provider import Glm5NextVLModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class _HCAlphaMapping(MegatronParamMapping):
    """Map one checkpoint HC scale vector to Megatron's three scalar parameters."""

    def __init__(self, megatron_pre, megatron_post, megatron_res, hf_param):
        super().__init__(megatron_param=megatron_pre, hf_param=hf_param)
        self._megatron_post = megatron_post
        self._megatron_res = megatron_res

    @staticmethod
    def _resolve_single(pattern, captures):
        result = pattern
        for capture in captures:
            result = result.replace("*", capture, 1)
        return result

    def resolve(self, captures):
        megatron_pre, hf_param = self._resolve_names(captures)
        return type(self)(
            megatron_pre,
            self._resolve_single(self._megatron_post, captures),
            self._resolve_single(self._megatron_res, captures),
            hf_param,
        )

    def hf_to_megatron(self, hf_weights, megatron_module):
        return hf_weights.to(megatron_module.alpha_pre.device)[0:1]

    def megatron_to_hf(self, megatron_weights, megatron_module):
        post = megatron_module.alpha_post.detach() if megatron_module is not None else None
        residual = megatron_module.alpha_res.detach() if megatron_module is not None else None
        primary = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        post = self.broadcast_from_pp_rank(post, cache_key=f"{self.hf_param}_post")
        residual = self.broadcast_from_pp_rank(residual, cache_key=f"{self.hf_param}_res")
        if primary is None:
            return {}
        return {self.hf_param: torch.cat((primary.float(), post.float(), residual.float()))}


class _HCAlphaSecondaryMapping(MegatronParamMapping):
    def __init__(self, megatron_param, hf_param, index):
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self.index = index
        self.allow_hf_name_mismatch = True

    def resolve(self, captures):
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(megatron_param, hf_param, self.index)

    def hf_to_megatron(self, hf_weights, megatron_module):
        target = megatron_module.alpha_post if self.index == 1 else megatron_module.alpha_res
        return hf_weights.to(target.device)[self.index : self.index + 1]

    def megatron_to_hf(self, megatron_weights, megatron_module):
        return {}


@MegatronModelBridge.register_bridge(
    source="Glm5NextForConditionalGeneration",
    target=GPTModel,
    provider=Glm5NextVLModelProvider,
    model_type="glm5_next",
)
class Glm5NextBridge(MegatronModelBridge):
    """Bridge for ``zai-org/GLM-5.3-Flash``, vision tower included.

    Every text-side setting below is the validated text configuration; vision adds
    only the tower config, the placeholder token ids and one weight mapping. Dropping
    vision means building ``MLAModelProvider`` here instead and deleting those three.
    """

    MODEL_CONFIG_CLASS = None
    _vision_config = None

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Glm5NextVLModelProvider:
        from megatron.bridge.models.glm5_next.modeling_glm5_next.spec import get_glm5_next_layer_spec

        outer_config = hf_pretrained.config
        config = getattr(outer_config, "text_config", outer_config)
        kwargs = self.hf_config_to_provider_kwargs(config)
        valid_fields = Glm5NextVLModelProvider.__dataclass_fields__
        provider = Glm5NextVLModelProvider(**{key: value for key, value in kwargs.items() if key in valid_fields})

        provider.transformer_layer_spec = get_glm5_next_layer_spec
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True
        provider.position_embedding_type = "none"
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0

        provider.num_layers = config.num_hidden_layers
        provider.num_attention_heads = config.num_attention_heads
        # The released GLM-5.3 config deliberately reports ``head_dim=0`` because
        # its DSA layers use the MLA-specific Q/K dimensions below.  MCore's base
        # Attention constructor still creates the configured core-attention once
        # before AbsorbedMLA replaces it, and that path uses ``kv_channels`` to
        # initialize the softmax scale.  Keep it aligned with the actual Q/K head
        # width so construction cannot divide by the placeholder zero.
        provider.kv_channels = config.qk_head_dim
        provider.q_lora_rank = config.q_lora_rank
        provider.kv_lora_rank = config.kv_lora_rank
        provider.qk_head_dim = config.qk_nope_head_dim
        provider.qk_pos_emb_head_dim = config.qk_rope_head_dim
        provider.v_head_dim = config.v_head_dim
        provider.rotary_percent = 1.0
        provider.rotary_base = float(getattr(config, "rope_theta", 10_000.0))

        linear_config = config.linear_attn_config
        dsa_layers = set(linear_config["full_attn_layers"])
        provider.kimi_kda_layers = [i + 1 for i in linear_config["kda_layers"]]
        if dsa_layers | set(linear_config["kda_layers"]) != set(range(config.num_hidden_layers)):
            raise ValueError("GLM-5 Next attention schedule must cover every decoder layer exactly once")
        provider.kimi_linear_num_heads = linear_config["num_heads"]
        provider.kimi_linear_head_dim = linear_config["head_dim"]
        provider.kimi_linear_conv_kernel_size = linear_config["short_conv_kernel_size"]
        provider.kimi_kda_gate_lower_bound = linear_config["gate_lower_bound"]

        provider.experimental_attention_variant = "dsa"
        provider.dsa_indexer_n_heads = config.index_n_heads
        provider.dsa_indexer_head_dim = config.index_head_dim
        provider.dsa_indexer_topk = config.index_topk
        provider.dsa_indexer_topk_freq = 1
        provider.dsa_indexer_skip_topk_offset = 0
        provider.dsa_indexer_rope_interleaved = config.indexer_rope_interleave
        provider.dsa_indexer_rotate_activation = False
        provider.dsa_indexer_scoring_relu = True
        provider.dsa_indexer_k_norm_epsilon = 1e-6
        provider.dsa_indexer_loss_coeff = 0.0
        provider.dsa_indexer_use_sparse_loss = False
        provider.dsa_kernel_backend = "cudnn"
        provider.cp_comm_type = "allgather"
        # KPool derives sequence-local pool boundaries from packed cu_seqlens,
        # including when context parallelism is one.
        provider.requires_packed_sequence = True
        # Loss-masked EP synchronization rows still traverse KPool and the
        # fused sparse kernels. Use a kernel-safe row length with no THD tail.
        provider.packed_sequence_phantom_length = 64

        provider.enable_mhc_connections = True
        provider.mhc_num_residual_streams = config.hc_mult
        provider.mhc_sinkhorn_iterations = config.hc_sinkhorn_iters
        provider.mtp_num_layers = None

        first_dense = config.first_k_dense_replace
        provider.moe_layer_freq = [0] * first_dense + [1] * (config.num_hidden_layers - first_dense)
        provider.num_moe_experts = config.n_routed_experts
        provider.moe_router_topk = config.num_experts_per_tok
        provider.moe_ffn_hidden_size = config.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_pre_softmax = True
        provider.norm_topk_prob = config.norm_topk_prob
        provider.moe_router_topk_scaling_factor = config.routed_scaling_factor
        provider.moe_grouped_gemm = True
        # alltoall is the validated default (DSv4 line); flex/hybridep is the
        # production goal but currently hangs under investigation and is dead
        # on ali-apse7 (name-based GPU check vs the L20D masquerade). Callers
        # opt into flex by setting moe_token_dispatcher_type on the provider
        # (or via the trainer config's dispatcher knob); the tuned hybridep
        # settings below are inert until then, kept here so the flip is one
        # recorded field rather than a rediscovered recipe.
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_flex_dispatcher_backend = "hybridep"
        provider.moe_flex_dispatcher_num_sms = 16
        provider.moe_router_load_balancing_type = "noaux_tc"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_shared_expert_overlap = True
        provider.activation_func_clamp_value = config.swiglu_limit
        provider.make_vocab_size_divisible_by = 1280

        # The tower is HF's own module built from this config object, so it is handed
        # over whole rather than copied field by field.
        #
        # Left as None for a config that declares no tower. Every released GLM-5.3
        # checkpoint has one -- Glm5NextForConditionalGeneration is the architecture --
        # so in practice that means a text-only config assembled by hand, and this
        # provider then behaves exactly like the text one. Building the model is where
        # a missing tower is refused, because that is where its weights would go
        # missing; see Glm5NextVLModelProvider.provide.
        provider.vision_config = getattr(outer_config, "vision_config", None)

        # These live on the top-level config, not text_config, and HF's
        # get_placeholder_mask reads them off the Megatron config at forward time.
        # Only meaningful alongside a tower, so they follow the same condition.
        if provider.vision_config is not None:
            for field in (
                "image_token_id",
                "video_token_id",
                "image_start_token_id",
                "image_end_token_id",
                "video_start_token_id",
                "video_end_token_id",
            ):
                value = getattr(outer_config, field, None)
                if value is None:
                    raise ValueError(
                        f"GLM-5.3 checkpoint config is missing {field}; the vision path "
                        "needs it to locate image placeholders in the token stream."
                    )
                setattr(provider, field, value)

        self._text_config = config
        self._vision_config = provider.vision_config
        return provider

    def _has_vision_tower(self) -> bool:
        """Whether the checkpoint being converted carries a vision tower.

        Read off ``self.hf_config`` -- which the conversion machinery sets right
        before it calls ``mapping_registry`` -- rather than off state left by
        ``provider_bridge``. The weight-loading hook resolves its own bridge
        instance, so anything recorded during provider construction may simply not
        be there, and the whole registry then comes back with un-prefixed names
        that match nothing on the wrapper. That failure is silent apart from one
        warning per parameter, so it has to be impossible by construction.
        """
        config = getattr(self, "hf_config", None)
        if config is not None and getattr(config, "vision_config", None) is not None:
            return True
        return self._vision_config is not None

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict: Mapping[str, torch.Tensor]):
        hf_weights = super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
        if isinstance(hf_weights, dict):
            return {
                key: maybe_dequantize_fp8_blockwise(tensor, hf_state_dict.get(f"{hf_param[key]}_scale_inv"))
                for key, tensor in hf_weights.items()
            }
        return maybe_dequantize_fp8_blockwise(hf_weights, hf_state_dict.get(f"{hf_param}_scale_inv"))

    def mapping_registry(self) -> MegatronMappingRegistry:  # noqa: C901
        prefix = "model.language_model"
        mappings = [
            AutoMapping("embedding.word_embeddings.weight", f"{prefix}.embed_tokens.weight"),
            AutoMapping("decoder.final_layernorm.weight", f"{prefix}.norm.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
            AutoMapping("decoder.layers.*.input_layernorm.weight", f"{prefix}.layers.*.input_layernorm.weight"),
            AutoMapping(
                "decoder.layers.*.pre_mlp_layernorm.weight", f"{prefix}.layers.*.post_attention_layernorm.weight"
            ),
            AutoMapping(
                "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                f"{prefix}.layers.*.post_attention_layernorm.weight",
            ),
            AutoMapping("decoder.layers.*.mlp.linear_fc2.weight", f"{prefix}.layers.*.mlp.down_proj.weight"),
            AutoMapping("decoder.layers.*.mlp.router.weight", f"{prefix}.layers.*.mlp.gate.weight"),
            AutoMapping(
                "decoder.layers.*.mlp.router.expert_bias", f"{prefix}.layers.*.mlp.gate.e_score_correction_bias"
            ),
            AutoMapping(
                "decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                f"{prefix}.layers.*.mlp.shared_experts.down_proj.weight",
            ),
            AutoMapping(
                "decoder.layers.*.mlp.experts.linear_fc2.weight*", f"{prefix}.layers.*.mlp.experts.*.down_proj.weight"
            ),
            ReplicatedMapping(
                "decoder.layers.*.self_attention_hyper_connection.mapping_proj.weight", f"{prefix}.layers.*.hc_attn_fn"
            ),
            ReplicatedMapping(
                "decoder.layers.*.self_attention_hyper_connection.bias", f"{prefix}.layers.*.hc_attn_base"
            ),
            ReplicatedMapping(
                "decoder.layers.*.mlp_hyper_connection.mapping_proj.weight", f"{prefix}.layers.*.hc_ffn_fn"
            ),
            ReplicatedMapping("decoder.layers.*.mlp_hyper_connection.bias", f"{prefix}.layers.*.hc_ffn_base"),
        ]
        mappings += [
            GatedMLPMapping(
                "decoder.layers.*.mlp.linear_fc1.weight",
                gate=f"{prefix}.layers.*.mlp.gate_proj.weight",
                up=f"{prefix}.layers.*.mlp.up_proj.weight",
            ),
            GatedMLPMapping(
                "decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate=f"{prefix}.layers.*.mlp.shared_experts.gate_proj.weight",
                up=f"{prefix}.layers.*.mlp.shared_experts.up_proj.weight",
            ),
            GatedMLPMapping(
                "decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate=f"{prefix}.layers.*.mlp.experts.*.gate_proj.weight",
                up=f"{prefix}.layers.*.mlp.experts.*.up_proj.weight",
            ),
            GatedMLPMapping(
                "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                gate=f"{prefix}.layers.*.mlp.experts.*.gate_proj.weight",
                up=f"{prefix}.layers.*.mlp.experts.*.up_proj.weight",
            ),
            AutoMapping(
                "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                f"{prefix}.layers.*.mlp.experts.*.down_proj.weight",
            ),
        ]

        config = getattr(self, "_text_config", None)
        if config is None:
            hf_config = getattr(self, "hf_config", None)
            config = getattr(hf_config, "text_config", hf_config)
        if config is None:
            raise ValueError("GLM-5 Next mapping construction requires the HF text config")
        linear_config = config.linear_attn_config
        for layer in linear_config["kda_layers"]:
            megatron_attention = f"decoder.layers.{layer}.self_attention"
            hf_attention = f"{prefix}.layers.{layer}.self_attn"
            mappings.extend(
                [
                    *(
                        ColumnParallelMapping(f"{megatron_attention}.{name}", f"{hf_attention}.{name}")
                        for name in (
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                            "q_conv1d.weight",
                            "k_conv1d.weight",
                            "v_conv1d.weight",
                            "f_b_proj.weight",
                            "b_proj.weight",
                            "g_b_proj.weight",
                            "A_log",
                            "dt_bias",
                        )
                    ),
                    ReplicatedMapping(f"{megatron_attention}.f_a_proj.weight", f"{hf_attention}.f_a_proj.weight"),
                    ReplicatedMapping(f"{megatron_attention}.g_a_proj.weight", f"{hf_attention}.g_a_proj.weight"),
                    ReplicatedMapping(f"{megatron_attention}.o_norm.weight", f"{hf_attention}.o_norm.weight"),
                    RowParallelMapping(f"{megatron_attention}.o_proj.weight", f"{hf_attention}.o_proj.weight"),
                ]
            )

        for layer in linear_config["full_attn_layers"]:
            megatron_attention = f"decoder.layers.{layer}.self_attention"
            hf_attention = f"{prefix}.layers.{layer}.self_attn"
            indexer = f"{megatron_attention}.core_attention.indexer"
            hf_indexer = f"{hf_attention}.indexer"
            mappings.extend(
                [
                    AutoMapping(f"{megatron_attention}.linear_q_down_proj.weight", f"{hf_attention}.q_a_proj.weight"),
                    AutoMapping(f"{megatron_attention}.q_layernorm.weight", f"{hf_attention}.q_a_layernorm.weight"),
                    AutoMapping(f"{megatron_attention}.linear_q_up_proj.weight", f"{hf_attention}.q_b_proj.weight"),
                    AutoMapping(
                        f"{megatron_attention}.linear_kv_down_proj.weight", f"{hf_attention}.kv_a_proj_with_mqa.weight"
                    ),
                    AutoMapping(f"{megatron_attention}.kv_layernorm.weight", f"{hf_attention}.kv_a_layernorm.weight"),
                    AutoMapping(f"{megatron_attention}.linear_kv_up_proj.weight", f"{hf_attention}.kv_b_proj.weight"),
                    RowParallelMapping(f"{megatron_attention}.linear_proj.weight", f"{hf_attention}.o_proj.weight"),
                    ReplicatedMapping(f"{indexer}.linear_wq_b.weight", f"{hf_indexer}.wq_b.weight"),
                    ReplicatedMapping(f"{indexer}.linear_wk.weight", f"{hf_indexer}.wk.weight"),
                    ReplicatedMapping(f"{indexer}.k_norm.weight", f"{hf_indexer}.k_norm.weight"),
                    ReplicatedMapping(f"{indexer}.k_norm.bias", f"{hf_indexer}.k_norm.bias"),
                    ReplicatedMapping(f"{indexer}.linear_weights_proj.weight", f"{hf_indexer}.weights_proj.weight"),
                    ReplicatedMapping(f"{indexer}.index_kpool_compress_ape", f"{hf_indexer}.index_kpool_compress_ape"),
                    ReplicatedMapping(
                        f"{indexer}.index_kpool_compress_gate", f"{hf_indexer}.index_kpool_compress_gate"
                    ),
                ]
            )
        for kind in ("attn", "ffn"):
            base = "self_attention" if kind == "attn" else "mlp"
            scale = f"{prefix}.layers.*.hc_{kind}_scale"
            mappings.append(
                _HCAlphaMapping(
                    f"decoder.layers.*.{base}_hyper_connection.alpha_pre",
                    f"decoder.layers.*.{base}_hyper_connection.alpha_post",
                    f"decoder.layers.*.{base}_hyper_connection.alpha_res",
                    scale,
                )
            )
            mappings.append(_HCAlphaSecondaryMapping(f"decoder.layers.*.{base}_hyper_connection.alpha_post", scale, 1))
            mappings.append(_HCAlphaSecondaryMapping(f"decoder.layers.*.{base}_hyper_connection.alpha_res", scale, 2))
        # Everything above is the text bridge's own mapping list, naming Megatron
        # params as they sit on a bare GPTModel. The VL wrapper holds that backbone
        # as a submodule, so each of those names lives one level down. Rewriting
        # them here rather than at each construction site keeps the list itself
        # exactly as the text bridge wrote it: dropping vision means dropping this
        # block and nothing else.
        if not self._has_vision_tower():
            return MegatronMappingRegistry(*mappings)

        for mapping in mappings:
            mapping.megatron_param = f"language_model.{mapping.megatron_param}"
            # _HCAlphaMapping also carries the two sibling scalar names it rebuilds
            # itself from when a wildcard resolves.
            for attr in ("_megatron_post", "_megatron_res"):
                sibling = getattr(mapping, attr, None)
                if sibling is not None:
                    setattr(mapping, attr, f"language_model.{sibling}")

        # The tower is HuggingFace's own Glm5NextVisionModel, so its submodule names
        # match the checkpoint one for one and a single wildcard covers the whole tower --
        # patch_embed, the 24 blocks, downsample, post_layernorm and the merger.
        #
        # ReplicatedMapping, not AutoMapping: the tower is plain torch modules held
        # identically on every rank, and AutoMapping infers sharding from the Megatron
        # module class, so it raises on the first nn.Linear it meets
        # ("Cannot determine parallelism type for module 'Linear' at weight
        # visual.blocks.0.attn.proj.bias"). Same choice GLM-4.5V makes.
        mappings.append(ReplicatedMapping(megatron_param="visual.**", hf_param="model.visual.**"))
        return MegatronMappingRegistry(*mappings)
