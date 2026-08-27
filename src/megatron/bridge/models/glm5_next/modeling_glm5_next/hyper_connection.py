# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GLM-5 Next hyper-connection overrides."""

from typing import Tuple

import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from torch import Tensor


class Glm5NextHyperConnectionModule(HyperConnectionModule):
    """Use the Hugging Face GLM-5 Next RMS normalization semantics."""

    @torch.compile
    def _projection_and_get_norm(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        reciprocal_rms = torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.config.layernorm_epsilon
        ).to(x.dtype)
        return self.mapping_proj(x), reciprocal_rms
