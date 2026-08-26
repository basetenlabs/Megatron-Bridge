# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GLM-5 Next's gated KDA layer."""

import torch
from einops import rearrange
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3Attention, _linear
from megatron.bridge.models.kimi.kimi_k3_ops import kda


class Glm5NextKDA(KimiK3Attention):
    """Kimi-style KDA with GLM-5 Next's two-stage output gate."""

    def _init_kda(self, config) -> None:
        super()._init_kda(config)
        del self.g_proj
        self.g_a_proj = self._duplicated_linear(config.hidden_size, self.head_dim)
        self.g_b_proj = self._column_linear(self.head_dim, self.projection_size)

    def _forward_kda(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        x = hidden_states.transpose(0, 1)
        cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        if packed_seq_params is not None and cu_seqlens is None:
            raise ValueError("Packed GLM-5 Next KDA input requires cu_seqlens_q")

        conv_kwargs = {"output_final_state": False, "cu_seqlens": cu_seqlens}
        q, _ = self.q_conv1d(x=_linear(self.q_proj, x), **conv_kwargs)
        k, _ = self.k_conv1d(x=_linear(self.k_proj, x), **conv_kwargs)
        v, _ = self.v_conv1d(x=_linear(self.v_proj, x), **conv_kwargs)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.local_num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.local_num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.local_num_heads)
        forget_gate = rearrange(
            _linear(self.f_b_proj, _linear(self.f_a_proj, x)),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        beta = _linear(self.b_proj, x).float().sigmoid()
        output = kda(
            q,
            k,
            v,
            forget_gate,
            beta,
            self.A_log,
            self.dt_bias,
            self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
        )
        gate = rearrange(
            _linear(self.g_b_proj, _linear(self.g_a_proj, x)),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        output = self.o_norm(output.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim))
        output = output.view(*gate.shape).flatten(-2)
        return _linear(self.o_proj, output.to(hidden_states.dtype)).transpose(0, 1)
