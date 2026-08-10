# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import logging

import torch
from megatron.core.transformer import TransformerConfig

from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)

# CUDA compute-capability majors whose GPUs can run DeepEP / HybridEP:
#   8  = Ampere   (A100, sm_80/86)
#   9  = Hopper   (H100/H200, sm_90)
#   10 = Blackwell (B200 sm_100, B300 sm_103)
#
# Keyed on compute capability rather than on ``device_properties.name``. The name
# is a marketing string that does not reliably identify the SKU: our B300 hosts
# enumerate as "NVIDIA L20D", so the previous
# ``name.startswith(("NVIDIA B200", "NVIDIA B300"))`` test silently skipped DeepEP
# on every one of them and fell back to the alltoall dispatcher. Compute
# capability comes from the driver and cannot be relabelled.
_FLEX_DISPATCHER_CC_MAJORS = (8, 9, 10)


def apply_flex_dispatcher_backend(
    model_config: TransformerConfig,
    moe_flex_dispatcher_backend: str | None = None,
) -> None:
    """Apply DeepEP or HybridEP optimizations to the model config.

    DeepEP is applicable only for MoE models on Ampere, Hopper, B200 and B300 GPUs.
    HybridEP is applicable only for MoE models on GB200, GB300 with NVL72 and on Ampere, Hopper, B200 and B300 GPUs.
    """
    num_moe_experts = getattr(model_config, "num_moe_experts", None)
    if num_moe_experts is None or num_moe_experts == 0:
        if get_rank_safe() == 0:
            logger.warning(
                "DeepEP and HybridEP are only applicable to MoE models. "
                "Model config does not use MoE (num_moe_experts is not set or is 0). "
                "Skipping DeepEP configuration."
            )
        return

    device_properties = torch.cuda.get_device_properties(0)
    if moe_flex_dispatcher_backend == "deepep":
        if device_properties.major not in _FLEX_DISPATCHER_CC_MAJORS:
            if get_rank_safe() == 0:
                logger.warning(
                    f"DeepEP is only applicable to Ampere, Hopper, and Blackwell (B200/B300) GPUs. "
                    f"Current GPU: {device_properties.name}. Skipping DeepEP configuration."
                )
            return
    elif moe_flex_dispatcher_backend == "hybridep":
        if device_properties.major not in _FLEX_DISPATCHER_CC_MAJORS:
            if get_rank_safe() == 0:
                logger.warning(
                    f"HybridEP is only applicable for GB200, GB300 with NVL72 and for Ampere, Hopper, B200 and B300 GPUs. "
                    f"Current GPU: {device_properties.name}. Skipping HybridEP configuration."
                )
            return
    else:
        if get_rank_safe() == 0:
            logger.warning("Not a valid flex dispatcher backend. Skipping flex dispatcher backend configuration.")
        return

    model_config.moe_token_dispatcher_type = "flex"
    model_config.moe_flex_dispatcher_backend = moe_flex_dispatcher_backend
    model_config.moe_shared_expert_overlap = False


def validate_flex_dispatcher_backend(model_config: TransformerConfig) -> None:
    """Validate DeepEP or HybridEP is supported for the current GPU architecture."""
    if model_config.moe_token_dispatcher_type == "flex":
        device_properties = torch.cuda.get_device_properties(0)
        if model_config.moe_flex_dispatcher_backend == "deepep":
            if device_properties.major not in _FLEX_DISPATCHER_CC_MAJORS:
                raise ValueError(
                    f"DeepEP is supported for Ampere, Hopper, and Blackwell (B200/B300) GPUs. "
                    f"Current GPU: {device_properties.name}"
                )

        if model_config.moe_flex_dispatcher_backend == "hybridep":
            if device_properties.major not in _FLEX_DISPATCHER_CC_MAJORS:
                raise ValueError(
                    "HybridEP is supported for GB200, GB300 with NVL72 and for Ampere, Hopper, B200 and B300 GPUs"
                )
