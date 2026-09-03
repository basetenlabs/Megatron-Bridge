# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GLM-5 Next's gated KDA layer."""

import torch

from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3Attention, _linear


class Glm5NextKDA(KimiK3Attention):
    """Kimi-style KDA with GLM-5 Next's two-stage output gate.

    Only the output gate differs from Kimi K3: a low-rank pair replaces K3's
    single projection. The dense and context-parallel KDA forwards, including
    the head-parallel all-to-all, are inherited unchanged.
    """

    def _init_kda(self, config) -> None:
        super()._init_kda(config)
        del self.g_proj
        self.g_a_proj = self._duplicated_linear(config.hidden_size, self.head_dim)
        self.g_b_proj = self._column_linear(self.head_dim, self.projection_size)

    def _gate_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Project the output gate through GLM-5 Next's low-rank pair."""
        return _linear(self.g_b_proj, _linear(self.g_a_proj, x))
