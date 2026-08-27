# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Complete-group key pooling for GLM-5 Next sparse attention."""

from typing import Optional, Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer
from torch import nn


class Glm5NextKPoolIndexer(DSAIndexer):
    """Select 4-token key pools, then expand them for sparse attention.

    Pooling reduces the candidate space by four while preserving the raw-token
    sparse-attention result. The split fused top-k and sparse-attention kernels
    remain mandatory; the quadratic score path is deliberately disabled.
    """

    supports_full_fused_attention = False
    supports_local_indexer_varlen = False
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
        self._local_gate: Optional[torch.Tensor] = None
        self._tail_start: Optional[torch.Tensor] = None
        self._tail_size: Optional[torch.Tensor] = None

    def forward_before_topk(self, x, qr, packed_seq_params=None):
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("GLM-5 Next pooled DSA requires packed THD sequences")

        q, raw_k, weights = super().forward_before_topk(x, qr, packed_seq_params)
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            # super() gathered its own copy; the gate input must cover the same rows.
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
        _, cu = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
        self._raw_cu_seqlens = cu.to(device=raw_k.device, dtype=torch.int64)
        # Pooling is deferred to prepare_topk_inputs: under CP the caller
        # allgathers keys and restores global packed order between these hooks,
        # and pool groups must be formed over that global order.
        self._local_gate = torch.nn.functional.linear(
            x.squeeze(1), self.index_kpool_compress_gate
        )
        return q, raw_k, weights

    def prepare_topk_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        index_topk: int,
        fused_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]],
        packed_seq_params: Optional[PackedSeqParams],
    ):
        if fused_bounds is None:
            raise RuntimeError("GLM-5 Next requires fused bounded top-k; quadratic fallback is forbidden")
        if self._local_gate is None or self._raw_cu_seqlens is None:
            raise RuntimeError("GLM-5 Next pool metadata was not prepared")

        gate = self._local_gate
        cp_group = self.pg_collection.cp
        cp_size = cp_group.size()
        if cp_size > 1:
            # k arrives allgathered and restored to global packed order by
            # DSAttention; the gate rows must follow the same permutation.
            cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
            _, kv_reorder = dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cp_size=cp_size,
                cp_rank=cp_group.rank(),
                device=k.device,
                local_output_size=gate.size(0),
                key_local_output_size=gate.size(0),
                global_output_size=gate.size(0) * cp_size,
            )
            gate = gather_from_sequence_parallel_region(gate.unsqueeze(1), group=cp_group)
            gate = gate.squeeze(1).index_select(0, kv_reorder)
        if k.size(0) != gate.size(0):
            raise RuntimeError(
                "GLM-5 Next pooled DSA key/gate row mismatch: "
                f"k_rows={k.size(0)}, gate_rows={gate.size(0)}, cp_size={cp_size}"
            )

        cu = self._raw_cu_seqlens
        lengths = cu[1:] - cu[:-1]
        pools_per_sequence = torch.div(lengths, self.pool_size, rounding_mode="floor")
        pool_prefix = torch.cat((torch.zeros_like(pools_per_sequence[:1]), pools_per_sequence.cumsum(0)))
        raw_positions = torch.arange(k.size(0), device=k.device, dtype=torch.int64)
        raw_sequence_ids = torch.bucketize(raw_positions, cu[1:], right=True)
        local_positions = raw_positions - cu[raw_sequence_ids]
        complete_pool_start = (local_positions.remainder(self.pool_size) == 0) & (
            local_positions + self.pool_size <= lengths[raw_sequence_ids]
        )
        pool_bases = raw_positions[complete_pool_start]
        pool_to_raw = (
            pool_bases[:, None] + torch.arange(self.pool_size, device=k.device, dtype=torch.int64)[None, :]
        )

        flat_k = k.squeeze(1)
        pool_gate = gate[pool_to_raw] + self.index_kpool_compress_ape[None, :, :]
        pool_weights = pool_gate.float().softmax(dim=1).to(dtype=flat_k.dtype)
        pooled_k = (flat_k[pool_to_raw] * pool_weights).sum(dim=1).unsqueeze(1)
        self._pool_to_raw = pool_to_raw
        self._pool_prefix = pool_prefix

        starts, ends = fused_bounds
        sequence_ids = torch.bucketize(starts.to(torch.int64), cu[1:], right=True)
        local_starts = starts.to(torch.int64) - cu[sequence_ids]
        local_ends = ends.to(torch.int64) - cu[sequence_ids]
        self._tail_size = local_ends.remainder(self.pool_size)
        self._tail_start = ends.to(torch.int64) - self._tail_size
        pool_base = self._pool_prefix[sequence_ids]
        pool_starts = pool_base + torch.div(local_starts, self.pool_size, rounding_mode="floor")
        pool_ends = pool_base + torch.div(local_ends, self.pool_size, rounding_mode="floor")
        return q, pooled_k, weights, index_topk // self.pool_size, (pool_starts, pool_ends)

    def finalize_topk_indices(self, topk_indices, topk_length, packed_seq_params=None):
        del packed_seq_params
        if self._pool_to_raw is None or self._tail_start is None or self._tail_size is None:
            raise RuntimeError("GLM-5 Next pool metadata was not prepared")
        valid_pools = topk_indices >= 0
        if topk_length is not None:
            positions = torch.arange(topk_indices.size(-1), device=topk_indices.device)
            valid_pools &= positions.view(1, 1, -1) < topk_length.unsqueeze(-1)

        batch_size, query_rows, candidate_count = topk_indices.shape
        if self._tail_start.numel() != query_rows or self._tail_size.numel() != query_rows:
            raise RuntimeError(
                "GLM-5 Next pooled DSA tail metadata row mismatch: "
                f"topk_rows={query_rows}, tail_rows={self._tail_start.numel()}"
            )

        expanded_width = candidate_count * self.pool_size
        result = torch.full(
            (batch_size, query_rows, expanded_width + self.pool_size - 1),
            -1,
            dtype=torch.int64,
            device=topk_indices.device,
        )

        # Compact valid selected pools into the prefix consumed by FlashMLA.
        if self._pool_to_raw.size(0) > 0:
            safe_indices = topk_indices.to(torch.int64).clamp(
                min=0, max=self._pool_to_raw.size(0) - 1
            )
            expanded_pools = self._pool_to_raw[safe_indices]
            pool_slots = (valid_pools.cumsum(dim=-1) - 1).clamp_min(0) * self.pool_size
            token_offsets = torch.arange(
                self.pool_size, device=topk_indices.device, dtype=torch.int64
            )
            pool_targets = pool_slots.unsqueeze(-1) + token_offsets
            invalid_target = torch.full_like(pool_targets, expanded_width)
            pool_targets = torch.where(valid_pools.unsqueeze(-1), pool_targets, invalid_target)
            pool_values = expanded_pools.masked_fill(~valid_pools.unsqueeze(-1), -1)
            result.scatter_(
                -1,
                pool_targets.flatten(-2),
                pool_values.flatten(-2),
            )

        valid_token_count = valid_pools.sum(dim=-1) * self.pool_size

        # A query may always select the one-to-three raw tokens in its current
        # incomplete pool. Place that tail directly after the valid pool prefix.
        tail_offsets = torch.arange(
            self.pool_size - 1, device=topk_indices.device, dtype=torch.int64
        )
        tail = self._tail_start[:, None] + tail_offsets
        tail = tail.masked_fill(tail_offsets >= self._tail_size[:, None], -1)
        tail = tail.unsqueeze(0).expand(batch_size, -1, -1)
        tail_targets = valid_token_count.unsqueeze(-1) + tail_offsets
        result.scatter_(-1, tail_targets, tail)

        result_length = valid_token_count + self._tail_size.unsqueeze(0)
        return result, result_length
