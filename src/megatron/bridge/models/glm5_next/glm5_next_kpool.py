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

"""k-pool index selection for the GLM-5.3 DSA indexer.

GLM-5.2 scores every key position and selects ``index_topk`` of them. GLM-5.3 groups
keys into contiguous pools of ``index_kpool``, scores the *pools*, selects
``index_topk // index_kpool`` of them, and expands each winner back into its member
token indices.

Two properties keep this contained:

1. **The output contract is unchanged.** This still returns raw token indices with
   ``-1`` for unused slots, so everything downstream -- ``scatter_topk_into_index_mask``,
   ``_run_sparse_attention``, and the cuDNN/tilelang kernels -- is reused without
   modification. They assert only ``topk_indices.ndim == 3`` and
   ``shape[:2] == (batch, seqlen)``, and already treat negatives as invalid.

2. **There is no backward pass.** The indexer is frozen -- HF marks its forward
   ``@torch.no_grad()``, and GLM-5.2 already freezes it for LoRA -- so this is a
   forward-only selection stage.

It is also a deliberate reduction in work: scoring runs over ``ceil(seqlen / kpool)``
candidates instead of ``seqlen``.

The pooling and expansion are split across ``forward_before_topk`` and ``forward``
rather than reaching into the attention path, so ``DSAttention`` needs no changes.
"""

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer, fused_qk_topk_naive


class Glm5NextKPoolIndexer(DSAIndexer):
    """DSA indexer with learned key pooling, as GLM-5.3 uses it.

    Adds two parameters over the base indexer:

    ``index_kpool_compress_ape``
        ``[index_kpool, index_head_dim]`` -- a learned position embedding over the slots
        *within* a pool, so pooling is order-aware.
    ``index_kpool_compress_gate``
        ``[index_head_dim, hidden_size]`` -- projects the hidden state to per-slot gate
        logits; a softmax over them gives each pool a learned weighted mean of its keys.

    Both are frozen at training time but must still load from the checkpoint.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.index_kpool = self.config.glm5_next_index_kpool
        self.index_kpool_always_select_tail = self.config.glm5_next_index_kpool_always_select_tail

        if self.index_kpool < 1:
            raise ValueError(f"glm5_next_index_kpool must be positive, got {self.index_kpool}")
        if self.index_topk % self.index_kpool:
            # Selection takes index_topk // index_kpool pools and expands each by
            # index_kpool, so a non-multiple would quietly select fewer tokens than
            # index_topk implies.
            raise ValueError(
                f"dsa_indexer_topk ({self.index_topk}) must be divisible by glm5_next_index_kpool ({self.index_kpool})"
            )

        self.index_kpool_compress_ape = torch.nn.Parameter(
            torch.empty(self.index_kpool, self.index_head_dim, dtype=self.config.params_dtype)
        )
        self.index_kpool_compress_gate = torch.nn.Parameter(
            torch.empty(self.index_head_dim, self.hidden_size, dtype=self.config.params_dtype)
        )
        self.config.init_method(self.index_kpool_compress_ape)
        self.config.init_method(self.index_kpool_compress_gate)

    @property
    def pool_topk(self) -> int:
        """Number of pools selected per query."""
        return self.index_topk // self.index_kpool

    @property
    def output_width(self) -> int:
        """Width of the index tensor this indexer emits."""
        tail = self.index_kpool - 1 if self.index_kpool_always_select_tail else 0
        return self.index_topk + tail

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: PackedSeqParams | None = None,
    ):
        """Compute queries and *pooled* keys.

        The base implementation returns one key per position. Here contiguous groups of
        ``index_kpool`` keys are collapsed into a single pooled key, so the caller's
        top-k runs over pools.

        Shapes follow the base indexer: ``q`` is ``[seqlen, batch, heads, head_dim]``,
        ``k`` is ``[seqlen, batch, head_dim]``, ``weights`` is ``[seqlen, batch, heads]``.
        Only ``k`` changes length, to ``seqlen // index_kpool``.
        """
        if packed_seq_params is not None:
            # Pools are formed over contiguous positions, which would straddle document
            # boundaries in a packed batch and let a query select keys from a different
            # document. Refuse rather than silently mixing documents.
            raise RuntimeError("GLM-5.3 k-pool indexing does not support packed sequences yet")

        q, k, weights = super().forward_before_topk(x, qr, packed_seq_params)

        seqlen, batch, head_dim = k.shape
        if seqlen % self.index_kpool:
            raise RuntimeError(
                f"GLM-5.3 k-pool requires the sequence length to be divisible by {self.index_kpool}, got {seqlen}"
            )
        pools = seqlen // self.index_kpool

        # Per-slot gate logits, plus the learned within-pool position embedding, give a
        # softmax over the members of each pool. Computed in fp32: the softmax is over
        # only index_kpool entries, but it weights the keys the whole selection depends on.
        gate = torch.nn.functional.linear(x, self.index_kpool_compress_gate)
        gate = gate.reshape(pools, self.index_kpool, batch, head_dim)
        logits = gate.float() + self.index_kpool_compress_ape.float()[None, :, None, :]
        probabilities = logits.softmax(dim=1).to(k.dtype)

        pooled_k = (k.reshape(pools, self.index_kpool, batch, head_dim) * probabilities).sum(dim=1)
        return q, pooled_k, weights

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> torch.Tensor:
        """Select pools, then expand them into token indices.

        Returns:
            ``int32 [batch, seqlen, output_width]`` token indices, ``-1`` for unused
            slots -- the same contract as the base indexer, only wider.
        """
        q, pooled_k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        _, pool_indices = fused_qk_topk_naive(
            q,
            pooled_k,
            weights,
            min(self.pool_topk, pooled_k.size(0)),
            mask,
            use_relu=self.config.dsa_indexer_scoring_relu,
        )

        return self._expand_pools(pool_indices, seqlen=x.size(0))

    def _expand_pools(self, pool_indices: torch.Tensor, *, seqlen: int) -> torch.Tensor:
        """Turn selected pool ids into the token ids they cover.

        Pool ``p`` covers tokens ``p * index_kpool ... p * index_kpool + index_kpool - 1``.
        Invalid pools (``-1``, emitted where a candidate was masked out) stay invalid
        across all of their slots.
        """
        batch, queries, selected = pool_indices.shape
        device = pool_indices.device

        offsets = torch.arange(self.index_kpool, device=device, dtype=pool_indices.dtype)
        token_indices = pool_indices[..., None] * self.index_kpool + offsets
        token_indices = token_indices.masked_fill(pool_indices[..., None] < 0, -1)
        token_indices = token_indices.reshape(batch, queries, selected * self.index_kpool)

        if self.index_kpool_always_select_tail:
            token_indices = torch.cat(
                [token_indices, self._visible_tail(batch, queries, seqlen, device, token_indices.dtype)],
                dim=-1,
            )

        width = self.output_width
        if token_indices.size(-1) < width:
            padding = token_indices.new_full((batch, queries, width - token_indices.size(-1)), -1)
            token_indices = torch.cat([token_indices, padding], dim=-1)

        return token_indices[..., :width].to(torch.int32)

    def _visible_tail(self, batch: int, queries: int, seqlen: int, device, dtype) -> torch.Tensor:
        """Token ids in the query's own incomplete pool.

        A query at position ``t`` sits inside pool ``t // index_kpool``, whose later
        members are not yet causally visible and whose pooled key therefore cannot
        represent ``t`` itself. Those in-pool predecessors are appended directly so the
        most recent tokens are always reachable regardless of what the pool scores said.
        """
        positions = torch.arange(queries, device=device, dtype=dtype)
        pool_start = (positions // self.index_kpool) * self.index_kpool
        offsets = torch.arange(self.index_kpool - 1, device=device, dtype=dtype)

        tail = pool_start[:, None] + offsets[None, :]
        # Keep only causally visible members: strictly before the query, and in range.
        tail = tail.masked_fill(tail >= positions[:, None], -1)
        tail = tail.masked_fill(tail >= seqlen, -1)

        return tail[None].expand(batch, queries, self.index_kpool - 1).contiguous()
