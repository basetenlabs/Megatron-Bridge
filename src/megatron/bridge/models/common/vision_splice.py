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

"""Write vision features over image placeholders, including under context parallelism.

Under CP the sequence reaching a VL model is one zigzag THD shard, but the vision
tower produces every image's features in global document order. Consuming that
tensor from the front would splice the wrong images on every rank but 0, silently,
because the shapes still line up. The helpers here look each local placeholder up by
its rank among *global* placeholders instead.
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


def context_parallel_group(config) -> torch.distributed.ProcessGroup | None:
    """The context-parallel process group, or None when CP is off."""
    collection = getattr(config, "_pg_collection", None)
    group = getattr(collection, "cp", None) if collection is not None else None
    return group if group is not None and group.size() > 1 else None


def local_feature_index(
    placeholders: Tensor,
    packed_seq_params: "PackedSeqParams | None",
    cp_group: torch.distributed.ProcessGroup,
    n_features: int | None = None,
) -> Tensor:
    """Which feature row each local placeholder wants, under context parallelism.

    Positions come from the same helper the DSA indexer uses for its own CP
    bookkeeping, so this makes no independent assumption about the zigzag layout; the
    global placeholder mask is rebuilt by all-gathering the local ones and scattering
    them into those positions.

    The all-gather is a collective, so every rank in the CP group has to reach it. It
    holds because the data path attaches the batch's vision tensors to every CP rank
    whole, which makes ``pixel_values is not None`` uniform across the group.

    Indexing also makes the tower's gradients come out right: every global placeholder
    belongs to exactly one rank's shard, so each rank gets gradient only for the rows
    it read, and the CP gradient reduction reassembles them with nothing double-counted.
    """
    from megatron.core.transformer.experimental_attention_variant import dsa_layout

    if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
        raise ValueError(
            "Vision under context parallelism requires packed THD sequences: "
            "the local-to-global position map is derived from cu_seqlens."
        )

    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    local_rows = placeholders.numel()
    cu_seqlens_q, _ = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
    device = placeholders.device

    positions = [
        dsa_layout.build_packed_allgather_cp_local_positions(
            cu_seqlens_q.to(device=device, dtype=torch.int64),
            cp_size,
            rank,
            device,
            output_size=local_rows,
            cu_seqlens_cover_output=False,
        )
        for rank in range(cp_size)
    ]

    # Padded rows get positions past the real tokens, so the scratch mask is sized for
    # that tail. Padding is never a placeholder, so those rows add nothing to the count.
    span = 2 * cp_size * local_rows
    global_placeholders = torch.zeros(span, dtype=torch.int32, device=device)
    gathered = torch.empty(cp_size * local_rows, dtype=torch.int32, device=device)
    torch.distributed.all_gather_into_tensor(gathered, placeholders.to(torch.int32).contiguous(), group=cp_group)
    flat_positions = torch.cat(positions, dim=0).clamp_(max=span - 1)
    global_placeholders[flat_positions] = gathered

    counts = global_placeholders.to(torch.int64)
    if n_features is not None:
        # The count is only checkable globally; on a local shard a tower/token-count
        # disagreement would read as a plausible index instead of an error.
        total = int(counts.sum().item())
        if total != n_features:
            raise ValueError(
                f"Vision splice: {total} placeholder(s) across the context-parallel group "
                f"but {n_features} feature row(s) from the tower; the expected token count "
                "per image disagrees with the processor's grid."
            )
    feature_index = torch.cumsum(counts, dim=0) - 1
    local_index = feature_index.index_select(0, positions[cp_rank].clamp_(max=span - 1))
    return local_index[placeholders]


def splice_features(
    inputs_embeds: Tensor,
    input_ids: Tensor,
    features: Tensor,
    token_id: int,
    packed_seq_params: "PackedSeqParams | None",
    cp_group: torch.distributed.ProcessGroup | None,
) -> Tensor:
    """Replace placeholder embeddings with vision features.

    ``inputs_embeds`` is [batch, seq, hidden]; ``features`` holds every image's rows in
    global document order.
    """
    placeholders = input_ids == token_id
    rows = inputs_embeds.clone()
    if cp_group is None:
        n_placeholders = int(placeholders.sum().item())
        if n_placeholders != features.size(0):
            raise ValueError(
                f"Vision splice: {n_placeholders} placeholder(s) but {features.size(0)} feature "
                "row(s) from the tower; the expected token count per image disagrees with the "
                "processor's grid."
            )
        rows[placeholders] = features.to(rows.dtype)
        return rows

    if inputs_embeds.size(0) != 1:
        raise ValueError(
            f"Vision under context parallelism expects packed THD batches (batch 1), got batch {inputs_embeds.size(0)}."
        )
    placeholders = placeholders.view(-1)
    index = local_feature_index(placeholders, packed_seq_params, cp_group, n_features=features.size(0))
    flat = rows.view(-1, rows.size(-1))
    flat[placeholders] = features.index_select(0, index).to(rows.dtype)
    return rows
