# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GLM-5 Next hyper-connection overrides."""

from typing import Tuple

import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.utils import nvtx_decorator
from torch import Tensor


class Glm5NextHyperConnectionModule(HyperConnectionModule):
    """Use the Hugging Face GLM-5 Next mHC normalization semantics."""

    hc_eps = 1e-6

    @torch.compile
    def _projection_and_get_norm(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        reciprocal_rms = torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.config.layernorm_epsilon
        ).to(x.dtype)
        return self.mapping_proj(x), reciprocal_rms

    @torch.compile
    def _sinkhorn(self, logits: Tensor) -> Tensor:
        matrix = torch.softmax(logits, dim=-1) + self.hc_eps
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn_iterations - 1):
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + self.hc_eps)
            matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + self.hc_eps)
        return matrix

    @nvtx_decorator(message="HyperConnection::compute_mappings")
    def compute_mappings(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        s, b, _ = x.shape
        with torch.cuda.nvtx.range("HyperConnection::projection_and_get_norm"):
            proj, reciprocal_rms = self._projection_and_get_norm(x)
        with torch.cuda.nvtx.range("HyperConnection::compute_h"):
            h_pre, h_post, h_res = self._compute_h(proj, reciprocal_rms)
        h_res = self._sinkhorn(h_res.view(s, b, self.n, self.n))
        return h_pre, h_post, h_res
