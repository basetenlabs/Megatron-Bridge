# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GLM-5 Next's checkpoint-compatible manifold hyper-connections."""

import torch
import torch.nn.functional as F
from megatron.core.transformer.hyper_connection import HyperConnectionModule


GLM5_NEXT_HC_EPSILON = 1e-6


class Glm5NextHyperConnection(HyperConnectionModule):
    """Match the released GLM-5.3 mHC equations and numerical contract.

    The generic MCore module computes its mapping projection in the activation
    dtype and applies ``H_res`` without transposition.  GLM-5.3 instead computes
    the mapping in FP32 and applies ``H_res.T`` to the residual streams.  Both
    details are numerically material across many decoder layers.
    """

    def compute_mappings(self, x):
        n = self.n
        flat = x.float()
        flat = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + self.config.layernorm_epsilon)
        projected = F.linear(flat, self.mapping_proj.weight.float())
        pre_w, post_w, residual_w = projected.split([n, n, n * n], dim=-1)
        pre_b, post_b, residual_b = self.bias.float().split([n, n, n * n])

        # GLM has two distinct epsilon contracts: layernorm_epsilon belongs
        # inside rsqrt above, while hc_eps is the 1e-6 floor used by the gate
        # and Sinkhorn projection below.
        hc_eps = GLM5_NEXT_HC_EPSILON
        pre = torch.sigmoid(pre_w * self.alpha_pre.float() + pre_b) + hc_eps
        post = 2 * torch.sigmoid(post_w * self.alpha_post.float() + post_b)
        residual_logits = residual_w.view(*residual_w.shape[:-1], n, n) * self.alpha_res.float() + residual_b.view(
            n, n
        )
        residual = torch.softmax(residual_logits, dim=-1) + hc_eps
        residual = residual / (residual.sum(dim=-2, keepdim=True) + hc_eps)
        for _ in range(self.sinkhorn_iterations - 1):
            residual = residual / (residual.sum(dim=-1, keepdim=True) + hc_eps)
            residual = residual / (residual.sum(dim=-2, keepdim=True) + hc_eps)
        return pre, post.to(x.dtype), residual.to(x.dtype)

    def aggregate(self, x, h_pre):
        streams = x.view(*x.shape[:-1], self.n, self.hidden_size)
        return (streams.float() * h_pre.unsqueeze(-1)).sum(dim=2).to(x.dtype)

    def apply_h_res(self, h_res, residual):
        sequence, batch, _ = residual.shape
        streams = residual.view(sequence * batch, self.n, self.hidden_size)
        mixed = torch.bmm(
            h_res.transpose(-1, -2).reshape(sequence * batch, self.n, self.n),
            streams,
        )
        return mixed.view(sequence, batch, self.n * self.hidden_size)
