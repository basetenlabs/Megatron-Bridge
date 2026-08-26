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

"""GLM-5.3-Flash (``glm5_next``): hybrid KDA/MLA-DSA backbone plus vision tower."""

from megatron.bridge.models.glm5_next.glm5_next_bridge import Glm5NextBridge
from megatron.bridge.models.glm5_next.glm5_next_kpool import Glm5NextKPoolIndexer
from megatron.bridge.models.glm5_next.glm5_next_provider import (
    Glm5NextModelProvider,
    Glm5NextVLModelProvider,
)
from megatron.bridge.models.glm5_next.glm5_next_spec import build_glm5_next_spec
from megatron.bridge.models.glm5_next.glm5_next_vl_model import Glm5NextVLModel


__all__ = [
    "Glm5NextBridge",
    "Glm5NextKPoolIndexer",
    "Glm5NextModelProvider",
    "Glm5NextVLModel",
    "Glm5NextVLModelProvider",
    "build_glm5_next_spec",
]
