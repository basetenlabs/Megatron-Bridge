# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GLM-5.3/GLM-5 Next model support."""

from megatron.bridge.models.glm5_next.glm5_next_bridge import Glm5NextBridge
from megatron.bridge.models.glm5_next.glm5_next_vl_provider import Glm5NextVLModelProvider


__all__ = ["Glm5NextBridge", "Glm5NextVLModelProvider"]
