# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer

from megatron.bridge.models.conversion.param_mapping import RowParallelMapping
from megatron.bridge.models.glm5_next.glm5_next_bridge import Glm5NextBridge
from megatron.bridge.models.glm5_next.modeling_glm5_next.kda import Glm5NextKDA
from megatron.bridge.models.glm5_next.modeling_glm5_next.kpool_indexer import Glm5NextKPoolIndexer
from megatron.bridge.models.glm5_next.modeling_glm5_next.spec import get_glm5_next_layer_spec
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


pytestmark = pytest.mark.unit


@pytest.fixture
def text_config():
    return SimpleNamespace(
        attention_bias=False,
        attention_dropout=0.0,
        dtype="bfloat16",
        first_k_dense_replace=3,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hidden_act="silu",
        hidden_size=4096,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        indexer_rope_interleave=True,
        intermediate_size=12288,
        kv_lora_rank=512,
        linear_attn_config={
            "num_heads": 64,
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "kda_layers": [0, 1, 2, 4],
            "full_attn_layers": [3],
        },
        max_position_embeddings=1_048_576,
        moe_intermediate_size=2048,
        n_routed_experts=288,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_attention_heads=64,
        num_experts_per_tok=8,
        num_hidden_layers=5,
        num_key_value_heads=64,
        q_lora_rank=1536,
        qk_head_dim=256,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        rms_norm_eps=1e-5,
        routed_scaling_factor=2.5,
        swiglu_limit=10.0,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
        v_head_dim=256,
        vocab_size=154880,
    )


@pytest.fixture
def pretrained(text_config):
    wrapper = Mock(spec=PreTrainedCausalLM)
    wrapper.config = SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"],
        model_type="glm5_next",
        text_config=text_config,
        dtype="bfloat16",
    )
    return wrapper


def test_provider_preserves_full_hybrid_sparse_moe_contract(pretrained):
    provider = Glm5NextBridge().provider_bridge(pretrained)

    assert provider.num_layers == 5
    assert provider.kimi_kda_layers == [1, 2, 3, 5]
    assert provider.moe_layer_freq == [0, 0, 0, 1, 1]
    assert provider.num_moe_experts == 288
    assert provider.moe_router_topk == 8
    assert provider.dsa_indexer_topk == 2048
    assert provider.enable_mhc_connections is True
    assert provider.mhc_num_residual_streams == 4
    assert provider.mtp_num_layers is None
    assert provider.position_embedding_type == "none"


def test_attention_output_mapping_is_layer_specific(pretrained):
    bridge = Glm5NextBridge()
    bridge.provider_bridge(pretrained)
    registry = bridge.mapping_registry()

    kda = registry.hf_to_megatron_lookup("model.language_model.layers.0.self_attn.o_proj.weight")
    dsa = registry.hf_to_megatron_lookup("model.language_model.layers.3.self_attn.o_proj.weight")
    assert isinstance(kda, RowParallelMapping)
    assert isinstance(dsa, RowParallelMapping)
    assert kda.megatron_param == "decoder.layers.0.self_attention.o_proj.weight"
    assert dsa.megatron_param == "decoder.layers.3.self_attention.linear_proj.weight"


def test_layer_spec_keeps_full_block_and_selects_kpool_only_for_dsa(pretrained):
    provider = Glm5NextBridge().provider_bridge(pretrained)
    block = get_glm5_next_layer_spec(provider, pp_rank=0)

    assert len(block.layer_specs) == 5
    assert all(spec.module is HyperConnectionTransformerLayer for spec in block.layer_specs)
    assert block.layer_specs[0].submodules.self_attention.module is Glm5NextKDA
    dsa_attention = block.layer_specs[3].submodules.self_attention
    assert dsa_attention.submodules.core_attention.submodules.indexer.module is Glm5NextKPoolIndexer


def test_kpool_expansion_appends_only_incomplete_tail():
    indexer = Glm5NextKPoolIndexer.__new__(Glm5NextKPoolIndexer)
    torch.nn.Module.__init__(indexer)
    indexer._pool_to_raw = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    indexer._raw_cu_seqlens = torch.tensor([0, 8])

    selected = torch.tensor([[[-1], [-1], [-1], [0], [0], [0], [0], [1]]])
    lengths = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    expanded, expanded_lengths = indexer.finalize_topk_indices(selected, lengths)

    assert expanded[0, 2].tolist() == [-1, -1, -1, -1, 0, 1, 2]
    assert expanded_lengths[0].tolist() == [1, 2, 3, 4, 5, 6, 7, 4]
    assert expanded[0, 7].tolist() == [4, 5, 6, 7, -1, -1, -1]
