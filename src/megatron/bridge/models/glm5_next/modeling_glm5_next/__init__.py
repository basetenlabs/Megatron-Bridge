# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron model components for GLM-5 Next."""

from megatron.bridge.models.glm5_next.modeling_glm5_next.kda import Glm5NextKDA
from megatron.bridge.models.glm5_next.modeling_glm5_next.kpool_indexer import Glm5NextKPoolIndexer
from megatron.bridge.models.glm5_next.modeling_glm5_next.spec import get_glm5_next_layer_spec


__all__ = ["Glm5NextKDA", "Glm5NextKPoolIndexer", "get_glm5_next_layer_spec"]
