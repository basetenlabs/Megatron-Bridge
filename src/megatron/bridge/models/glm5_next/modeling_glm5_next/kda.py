# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GLM-5 Next's gated KDA layer."""

import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net.common import (
    a2a_cp_to_hp,
    a2a_hp_to_cp,
    get_parameter_local_cp,
)

from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3Attention, _linear
from megatron.bridge.models.kimi.kimi_k3_ops import kda


def _is_single_document(cu_seqlens: torch.Tensor | None) -> bool:
    return (
        cu_seqlens is not None and cu_seqlens.numel() == 2 and int(cu_seqlens[0]) == 0
    )


def _prepare_kda_inputs(
    tensors: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor | None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None, int]:
    """Use dense KDA for one document and remove any trailing sequence padding."""
    sequence_length = tensors[0].shape[1]
    if not _is_single_document(cu_seqlens):
        return tensors, cu_seqlens, sequence_length

    valid_length = int(cu_seqlens[-1])
    if not 0 < valid_length <= sequence_length:
        raise ValueError(
            f"single-document cu_seqlens ends at {valid_length}, expected 1..{sequence_length}"
        )
    # A batch dimension of one can hide a padded stride while still reporting
    # contiguous, including when valid_length == sequence_length. TileLang's
    # backward kernels require canonical input strides.
    valid_tensors = tuple(
        tensor[:, :valid_length].clone(memory_format=torch.contiguous_format) for tensor in tensors
    )
    return valid_tensors, None, valid_length


def _doc_aware_causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Depthwise causal conv + silu that resets at packed-document boundaries.

    Equivalent to ``ShortConvolution(activation="silu", bias=False)`` on each
    document independently. Used on the CP head-parallel path, where the conv
    weight is a per-CP-rank channel slice and FLA's module (sized for the full
    local channel count) cannot be called directly.

    Args:
        x: ``[s, b, C]`` full-sequence activations for this rank's head shard.
        weight: ``[C, 1, K]`` depthwise conv weight slice (fp32 per K3 policy).
        cu_seqlens: global cumulative document lengths; ``None`` = one document.
    """
    w = weight.squeeze(1).to(torch.float32)  # [C, K]
    ksize = w.shape[-1]
    seq_len = x.shape[0]
    xf = x.float()
    pos = torch.arange(seq_len, device=x.device)
    doc = None
    if cu_seqlens is not None:
        doc = torch.bucketize(pos, cu_seqlens.to(device=x.device)[1:], right=True)
    y = xf * w[:, ksize - 1]
    for offset in range(1, ksize):
        shifted = torch.zeros_like(xf)
        shifted[offset:] = xf[:-offset]
        keep = pos >= offset
        if doc is not None:
            same_doc = torch.zeros_like(keep)
            same_doc[offset:] = doc[offset:] == doc[:-offset]
            keep = keep & same_doc
        y = y + shifted * w[:, ksize - 1 - offset] * keep.view(-1, 1, 1)
    return F.silu(y).to(x.dtype)


class Glm5NextKDA(KimiK3Attention):
    """Kimi-style KDA with GLM-5 Next's two-stage output gate.

    Context parallel (cp>1) runs GDN-style head-parallel CP: an all-to-all
    converts the sequence-sharded activations into full-sequence shards over
    a head subset, the recurrence runs unchanged, and a second all-to-all
    restores sequence sharding. Per-head state (``A_log``/``dt_bias``) and
    the depthwise conv weights are sliced per CP rank; gradients flow into
    the full parameters through the slice views.
    """

    supports_kda_cp = True

    def _init_kda(self, config) -> None:
        super()._init_kda(config)
        del self.g_proj
        self.g_a_proj = self._duplicated_linear(config.hidden_size, self.head_dim)
        self.g_b_proj = self._column_linear(self.head_dim, self.projection_size)
        if self.cp_size > 1 and self.local_num_heads % self.cp_size:
            raise ValueError(
                f"GLM-5 Next KDA heads per TP rank ({self.local_num_heads}) must be "
                f"divisible by context parallel size ({self.cp_size})"
            )

    def _forward_kda(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        if self.cp_size > 1:
            return self._forward_kda_cp(hidden_states, packed_seq_params)
        x = hidden_states.transpose(0, 1)
        cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        if packed_seq_params is not None and cu_seqlens is None:
            raise ValueError("Packed GLM-5 Next KDA input requires cu_seqlens_q")
        cu_seqlens_padded = (
            packed_seq_params.cu_seqlens_q_padded if packed_seq_params is not None else None
        )
        if cu_seqlens_padded is None:
            cu_seqlens_padded = cu_seqlens

        # Convs must see the PHYSICAL (padded) boundaries: the THD packer pads
        # each document in place, so unpadded boundaries are misaligned with
        # the layout.
        conv_kwargs = {"output_final_state": False, "cu_seqlens": cu_seqlens_padded}
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
        kda_cu_seqlens = cu_seqlens
        kda_valid_length = q.shape[1]
        if _is_single_document(cu_seqlens):
            kda_valid_length = int(cu_seqlens[-1])
            if not 0 < kda_valid_length <= q.shape[1]:
                raise ValueError(
                    f"single-document cu_seqlens ends at {kda_valid_length}, "
                    f"expected 1..{q.shape[1]}"
                )
            # FLA's varlen KDA path mishandles a partial final chunk even for a
            # single packed document. Dense causal KDA is equivalent on the
            # valid prefix for [0, valid_length <= T] with trailing padding.
            kda_cu_seqlens = None
        elif cu_seqlens is not None:
            # Multi-document boundaries must also be the padded ones: a dangling
            # tail (cu_seqlens[-1] < T) is never written by FLA's varlen backward,
            # which corrupts valid-region gradients (uninitialized memory).
            kda_cu_seqlens = cu_seqlens_padded
            kda_valid_length = int(cu_seqlens_padded[-1])
            if not 0 < kda_valid_length <= q.shape[1]:
                raise ValueError(
                    f"padded cu_seqlens ends at {kda_valid_length}, expected 1..{q.shape[1]}"
                )
        output = kda(
            q[:, :kda_valid_length].clone(memory_format=torch.contiguous_format),
            k[:, :kda_valid_length].clone(memory_format=torch.contiguous_format),
            v[:, :kda_valid_length].clone(memory_format=torch.contiguous_format),
            forget_gate[:, :kda_valid_length].clone(memory_format=torch.contiguous_format),
            beta[:, :kda_valid_length].clone(memory_format=torch.contiguous_format),
            self.A_log,
            self.dt_bias,
            self.gate_lower_bound,
            cu_seqlens=kda_cu_seqlens,
        )
        if kda_valid_length < q.shape[1]:
            if output.requires_grad:
                output.register_hook(
                    lambda grad: grad.clone(memory_format=torch.contiguous_format)
                )
            output = F.pad(output, (0, 0, 0, 0, 0, q.shape[1] - kda_valid_length))
        gate = rearrange(
            _linear(self.g_b_proj, _linear(self.g_a_proj, x)),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        output = self.o_norm(output.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim))
        output = output.view(*gate.shape).flatten(-2)
        return _linear(self.o_proj, output.to(hidden_states.dtype)).transpose(0, 1)

    def _resolve_global_cu_seqlens(
        self, packed_seq_params: PackedSeqParams | None, seq_len_global: int
    ) -> torch.Tensor | None:
        if packed_seq_params is None:
            return None
        cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        if cu_seqlens is None:
            cu_seqlens = packed_seq_params.cu_seqlens_q
        if cu_seqlens is None:
            raise ValueError("Packed GLM-5 Next KDA input requires cu_seqlens_q")
        if int(cu_seqlens[-1]) != seq_len_global:
            raise ValueError(
                f"cu_seqlens must be global under context parallelism: got total "
                f"{int(cu_seqlens[-1])} for global sequence length {seq_len_global}"
            )
        return cu_seqlens

    def _forward_kda_cp(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        # hidden_states: [s_local, b, hidden], sequence-sharded over CP in the
        # attention load-balanced (zigzag) layout the THD packer emits.
        cp = self.cp_size
        seq_len_global = hidden_states.shape[0] * cp
        cu_seqlens = self._resolve_global_cu_seqlens(packed_seq_params, seq_len_global)

        proj = self.projection_size
        sections = (proj, proj, proj, proj, proj, self.local_num_heads)
        packed = torch.cat(
            [
                _linear(self.q_proj, hidden_states),
                _linear(self.k_proj, hidden_states),
                _linear(self.v_proj, hidden_states),
                _linear(self.f_b_proj, _linear(self.f_a_proj, hidden_states)),
                _linear(self.g_b_proj, _linear(self.g_a_proj, hidden_states)),
                _linear(self.b_proj, hidden_states),
            ],
            dim=-1,
        )
        # [s_local, b, sum(sections)] -> [s_global, b, sum(sections)/cp] in
        # natural token order (the THD perm folds the un-zigzag).
        packed, thd_cp_a2a_inv = a2a_cp_to_hp(
            packed,
            sections,
            cp,
            self.cp_group,
            cu_seqlens,
            seq_len_global,
            packed_seq_params,
        )
        q, k, v, forget_gate, gate, beta_logits = torch.split(
            packed, [section // cp for section in sections], dim=-1
        )

        conv_inputs = {"q": q, "k": k, "v": v}
        for name in conv_inputs:
            conv_weight = get_parameter_local_cp(
                getattr(self, f"{name}_conv1d").weight, dim=0, cp_group=self.cp_group
            )
            conv_inputs[name] = _doc_aware_causal_conv(conv_inputs[name], conv_weight, cu_seqlens)

        heads_cp = self.local_num_heads // cp
        q = rearrange(conv_inputs["q"].transpose(0, 1), "b s (h d) -> b s h d", h=heads_cp)
        k = rearrange(conv_inputs["k"].transpose(0, 1), "b s (h d) -> b s h d", h=heads_cp)
        v = rearrange(conv_inputs["v"].transpose(0, 1), "b s (h d) -> b s h d", h=heads_cp)
        forget_gate = rearrange(forget_gate.transpose(0, 1), "b s (h d) -> b s h d", h=heads_cp)
        beta = beta_logits.transpose(0, 1).float().sigmoid()
        (q, k, v, forget_gate, beta), kda_cu_seqlens, _ = _prepare_kda_inputs(
            (q, k, v, forget_gate, beta), cu_seqlens
        )

        output = kda(
            q,
            k,
            v,
            forget_gate,
            beta,
            get_parameter_local_cp(self.A_log, dim=0, cp_group=self.cp_group),
            get_parameter_local_cp(self.dt_bias, dim=0, cp_group=self.cp_group),
            self.gate_lower_bound,
            # The original metadata remains in use by the CP all-to-all and
            # document-aware convolution above.
            cu_seqlens=kda_cu_seqlens,
        )
        gate = rearrange(gate.transpose(0, 1), "b s (h d) -> b s h d", h=heads_cp)
        output = self.o_norm(output.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim))
        output = output.view(*gate.shape).flatten(-2).transpose(0, 1)
        # [s_global, b, proj/cp] -> [s_local, b, proj], back in zigzag layout.
        output = a2a_hp_to_cp(output, cp, self.cp_group, packed_seq_params, thd_cp_a2a_inv)
        return _linear(self.o_proj, output.to(hidden_states.dtype))
