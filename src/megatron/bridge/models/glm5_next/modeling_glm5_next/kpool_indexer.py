# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Complete-group key pooling for GLM-5 Next sparse attention."""

from typing import Optional, Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer
from torch import nn


class Glm5NextKPoolIndexer(DSAIndexer):
    """Select 4-token key pools, then expand them for sparse attention.

    Pooling reduces the candidate space by four while preserving the raw-token
    sparse-attention result. The split fused top-k and sparse-attention kernels
    remain mandatory; the quadratic score path is deliberately disabled.
    """

    supports_full_fused_attention = False
    pool_size = 4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.index_topk % self.pool_size:
            raise ValueError("GLM-5 Next index_topk must be divisible by index_kpool=4")
        if self.config.dsa_indexer_loss_coeff:
            raise ValueError("GLM-5 Next pooled DSA does not support the token-level indexer loss")
        device = torch.cuda.current_device()
        dtype = self.config.params_dtype
        self.index_kpool_compress_ape = nn.Parameter(
            torch.empty(self.pool_size, self.index_head_dim, device=device, dtype=dtype)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.index_head_dim, self.hidden_size, device=device, dtype=dtype)
        )
        if self.config.perform_initialization:
            torch.nn.init.normal_(self.index_kpool_compress_ape, mean=0.0, std=0.02)
            self.config.init_method(self.index_kpool_compress_gate)
        for parameter in (self.index_kpool_compress_ape, self.index_kpool_compress_gate):
            setattr(parameter, "average_gradients_across_tp_domain", True)
        self._pool_to_raw: Optional[torch.Tensor] = None
        self._pool_prefix: Optional[torch.Tensor] = None
        self._raw_cu_seqlens: Optional[torch.Tensor] = None

    def forward_before_topk(self, x, qr, packed_seq_params=None):
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("GLM-5 Next pooled DSA requires packed THD sequences")
        if self.pg_collection.cp.size() != 1:
            raise ValueError("GLM-5 Next pooled DSA currently requires context parallel size 1")

        q, raw_k, weights = super().forward_before_topk(x, qr, packed_seq_params)
        cu = packed_seq_params.cu_seqlens_kv
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        if cu is None:
            raise ValueError("GLM-5 Next pooled DSA requires packed cu_seqlens")
        cu = cu.to(device=raw_k.device, dtype=torch.int64)

        lengths = cu[1:] - cu[:-1]
        pools_per_sequence = torch.div(lengths, self.pool_size, rounding_mode="floor")
        pool_prefix = torch.cat((torch.zeros_like(pools_per_sequence[:1]), pools_per_sequence.cumsum(0)))
        raw_positions = torch.arange(raw_k.size(0), device=raw_k.device, dtype=torch.int64)
        raw_sequence_ids = torch.bucketize(raw_positions, cu[1:], right=True)
        local_positions = raw_positions - cu[raw_sequence_ids]
        complete_pool_start = (local_positions.remainder(self.pool_size) == 0) & (
            local_positions + self.pool_size <= lengths[raw_sequence_ids]
        )
        pool_bases = raw_positions[complete_pool_start]
        pool_to_raw = (
            pool_bases[:, None] + torch.arange(self.pool_size, device=raw_k.device, dtype=torch.int64)[None, :]
        )

        flat_x = x.squeeze(1)
        flat_k = raw_k.squeeze(1)
        gate = torch.nn.functional.linear(flat_x, self.index_kpool_compress_gate)
        pool_gate = gate[pool_to_raw] + self.index_kpool_compress_ape[None, :, :]
        pool_weights = pool_gate.float().softmax(dim=1).to(dtype=flat_k.dtype)
        pooled_k = (flat_k[pool_to_raw] * pool_weights).sum(dim=1).unsqueeze(1)

        self._pool_to_raw = pool_to_raw
        self._pool_prefix = pool_prefix
        self._raw_cu_seqlens = cu
        return q, pooled_k, weights

    def prepare_topk_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        index_topk: int,
        fused_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]],
        packed_seq_params: Optional[PackedSeqParams],
    ):
        del packed_seq_params
        if fused_bounds is None:
            raise RuntimeError("GLM-5 Next requires fused bounded top-k; quadratic fallback is forbidden")
        if self._pool_prefix is None or self._raw_cu_seqlens is None:
            raise RuntimeError("GLM-5 Next pool metadata was not prepared")
        starts, ends = fused_bounds
        cu = self._raw_cu_seqlens
        sequence_ids = torch.bucketize(starts.to(torch.int64), cu[1:], right=True)
        local_starts = starts.to(torch.int64) - cu[sequence_ids]
        local_ends = ends.to(torch.int64) - cu[sequence_ids]
        pool_base = self._pool_prefix[sequence_ids]
        pool_starts = pool_base + torch.div(local_starts, self.pool_size, rounding_mode="floor")
        pool_ends = pool_base + torch.div(local_ends, self.pool_size, rounding_mode="floor")
        return q, k, weights, index_topk // self.pool_size, (pool_starts, pool_ends)

    def finalize_topk_indices(self, topk_indices, topk_length, packed_seq_params=None):
        del packed_seq_params
        if self._pool_to_raw is None or self._raw_cu_seqlens is None:
            raise RuntimeError("GLM-5 Next pool metadata was not prepared")
        safe_indices = topk_indices.to(torch.int64).clamp(min=0, max=max(0, self._pool_to_raw.size(0) - 1))
        expanded = self._pool_to_raw[safe_indices].flatten(-2)
        valid_pools = topk_indices >= 0
        if topk_length is not None:
            positions = torch.arange(topk_indices.size(-1), device=topk_indices.device)
            valid_pools &= positions.view(1, 1, -1) < topk_length.unsqueeze(-1)
        expanded = expanded.masked_fill(
            ~valid_pools.unsqueeze(-1).expand(*valid_pools.shape, self.pool_size).flatten(-2), -1
        )

        # A query may see up to three tokens in the current incomplete pool.
        rows = torch.arange(expanded.size(-2), device=expanded.device, dtype=torch.int64)
        sequence_ids = torch.bucketize(rows, self._raw_cu_seqlens[1:], right=True)
        local_rows = rows - self._raw_cu_seqlens[sequence_ids]
        tail_size = (local_rows + 1).remainder(self.pool_size)
        tail_start = rows - tail_size + 1
        offsets = torch.arange(self.pool_size - 1, device=expanded.device, dtype=torch.int64)
        tail = tail_start[:, None] + offsets[None, :]
        tail = tail.masked_fill(offsets[None, :] >= tail_size[:, None], -1)
        tail = tail.unsqueeze(0).expand(expanded.size(0), -1, -1)
        result = torch.cat((expanded, tail), dim=-1)
        result_length = valid_pools.sum(dim=-1) * self.pool_size + tail_size.unsqueeze(0)
        return result, result_length
