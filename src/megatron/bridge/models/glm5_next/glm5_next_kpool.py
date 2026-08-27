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

**This is what now caps GLM-5.3's sequence length.** ``forward_before_topk`` refuses
packed sequences, and context parallelism in this stack *is* THD context parallelism
(``megatron_config`` routes ``context_parallel_size > 1`` through
``_validate_thd_context_parallelism``). So even though the KDA layers are now
Megatron-Core's CP-capable ``KimiDeltaAttention``, the 11 k-pool DSA layers still force
``cp=1`` for the model as a whole. Lifting it means forming pools per document from
``cu_seqlens`` rather than over the flat sequence, anchored at each document start, so a
pool never straddles a document boundary. Until then the ceiling moved, it did not go
away.

Integration point. ``DSAttention`` owns the top-k -- it calls ``forward_before_topk``
and selects itself, and never calls ``DSAIndexer.forward`` -- so pooling cannot live in
a ``forward`` override. It would be dead code, and the base class would top-k the pooled
keys and then treat the resulting *pool* indices as *token* indices.

Pooling is therefore expressed through three defaulted hooks on ``DSAIndexer``:
``topk_budget`` (select ``index_topk // index_kpool`` pools), ``indexer_key_positions``
(judge causality by each pool's last token) and ``postprocess_topk`` (expand pools back
to token indices, plus the visible tail). Only the top-k sees pool space; the attention
below is unchanged and still works in token space.
"""

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer, fused_qk_topk_naive


def visible_tail_indices(
    queries: int,
    index_kpool: int,
    seqlen: int,
    device=None,
    dtype=torch.long,
) -> torch.Tensor:
    """Per-query indices of the visible tokens no *complete* pool covers.

    Complete pools cover the first ``floor(visible / index_kpool) * index_kpool`` visible
    tokens, so the trailing ``visible % index_kpool`` are unreachable through pool
    selection. Mirrors HF's ``append_visible_tail``::

        tail_count = visible % index_kpool
        tail_start = visible - tail_count
        tail       = tail_start + [0 .. index_kpool - 2], while offset < tail_count

    With no left padding a query at position ``t`` sees ``t + 1`` tokens.

    The query's own token is included, which is the point: it is causally visible, and at
    ``t % index_kpool == 0`` it belongs to no complete pool. Masking to positions strictly
    before ``t`` instead leaves such a query with no guaranteed index at all -- every slot
    ``-1`` at ``t = 0`` -- so reaching itself would depend on the top-k happening to pick
    its own partly-future pool.

    Returns:
        ``[queries, index_kpool - 1]`` token indices, ``-1`` for unused slots.
    """
    positions = torch.arange(queries, device=device, dtype=dtype)
    visible = positions + 1
    tail_count = visible % index_kpool
    tail_start = visible - tail_count
    offsets = torch.arange(index_kpool - 1, device=device, dtype=dtype)

    tail = tail_start[:, None] + offsets[None, :]
    # Drop slots that exist only to pad the fixed width, and anything past the end.
    tail = tail.masked_fill(offsets[None, :] >= tail_count[:, None], -1)
    return tail.masked_fill(tail >= seqlen, -1)


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
    def topk_budget(self) -> int:
        """Select pools, not tokens: ``index_topk`` tokens is this many pools."""
        return self.index_topk // self.index_kpool

    @property
    def output_width(self) -> int:
        """Width of the index tensor this indexer emits."""
        tail = self.index_kpool - 1 if self.index_kpool_always_select_tail else 0
        return self.index_topk + tail

    def indexer_key_positions(self, seqlen: int, key_positions):
        """Judge causality in pool space, by each pool's last token.

        A candidate here is a pool of ``index_kpool`` consecutive tokens, so a query may
        select it only once the pool's final token is visible -- which is HF's rule. The
        base class compares query positions against per-token positions, and handing it
        pool indices unchanged would let a query select a pool whose members are all in
        its future.
        """
        pools = seqlen // self.index_kpool
        device = key_positions.device if key_positions is not None else None
        base = torch.arange(pools, device=device, dtype=torch.int32)
        return base * self.index_kpool + (self.index_kpool - 1)

    def postprocess_topk(self, topk_indices, seqlen: int):
        """Expand selected pools into token indices, then append the visible tail."""
        return self._expand_pools(topk_indices, seqlen=seqlen)

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
        # The gate must cover the same positions as the keys. The base implementation may
        # gather ``x`` out of the sequence-parallel region before building ``k`` -- the
        # indexer needs the whole sequence to pick a global top-k -- so ``x`` as handed to
        # this method can be shorter than ``k``.
        #
        # Decided on the observed shapes rather than by re-deriving the base's condition
        # from config flags. Duplicating that predicate is what produced two different
        # wrong answers here: a gate 1/tp as long as the keys when the base had gathered,
        # then one tp times too long when it had not.
        if x.shape[0] != seqlen:
            tp_size = self.pg_collection.tp.size()
            if x.shape[0] * tp_size != seqlen:
                raise RuntimeError(
                    f"k-pool cannot align the gate: x has {x.shape[0]} positions, the "
                    f"pooled keys {seqlen}, and tp={tp_size} does not bridge them."
                )
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)

        gate = torch.nn.functional.linear(x, self.index_kpool_compress_gate)
        gate = gate.reshape(pools, self.index_kpool, batch, head_dim)
        logits = gate.float() + self.index_kpool_compress_ape.float()[None, :, None, :]
        probabilities = logits.softmax(dim=1).to(k.dtype)

        pooled_k = (k.reshape(pools, self.index_kpool, batch, head_dim) * probabilities).sum(dim=1)
        return q, pooled_k, weights

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
        """Broadcast :func:`visible_tail_indices` across the batch."""
        tail = visible_tail_indices(queries, self.index_kpool, seqlen, device=device, dtype=dtype)
        return tail[None].expand(batch, queries, self.index_kpool - 1).contiguous()
