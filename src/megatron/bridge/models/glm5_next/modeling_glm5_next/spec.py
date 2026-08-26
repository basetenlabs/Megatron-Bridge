# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous GLM-5 Next transformer specification."""

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer

from megatron.bridge.models.glm5_next.modeling_glm5_next.kda import Glm5NextKDA
from megatron.bridge.models.glm5_next.modeling_glm5_next.kpool_indexer import Glm5NextKPoolIndexer


def get_glm5_next_layer_spec(config, vp_stage=None, pp_rank=None):
    """Build exact KDA/DSA, dense/MoE, and mHC layer specs for GLM-5 Next."""
    if (config.pipeline_model_parallel_size or 1) != 1:
        raise ValueError("GLM-5 Next mHC currently requires pipeline parallel size 1")
    if config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("GLM-5 Next does not support virtual pipeline parallelism")

    block = get_transformer_block_with_experimental_attention_variant_spec(config, vp_stage=vp_stage, pp_rank=pp_rank)
    kda_layers = set(config.kimi_kda_layers)
    for layer_number, layer_spec in enumerate(block.layer_specs, start=1):
        layer_spec.module = HyperConnectionTransformerLayer
        layer_spec.submodules.self_attention_hyper_connection = HyperConnectionModule
        layer_spec.submodules.mlp_hyper_connection = HyperConnectionModule
        if layer_number in kda_layers:
            layer_spec.submodules.self_attention = ModuleSpec(module=Glm5NextKDA)
            continue

        core_attention = layer_spec.submodules.self_attention.submodules.core_attention
        indexer = core_attention.submodules.indexer
        indexer.module = Glm5NextKPoolIndexer
    return block
