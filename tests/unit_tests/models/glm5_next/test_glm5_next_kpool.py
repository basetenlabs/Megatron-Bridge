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

"""k-pool tail selection, against a transcription of HF's ``append_visible_tail``.

These lock in a fix for a defect that produced no error and no shape mismatch: the tail
previously masked to positions *strictly before* the query, so at every
``t % index_kpool == 0`` it was entirely ``-1`` while the query's own token also belonged
to no complete pool. The query at ``t = 0`` had no valid index at all.

Pure index arithmetic, so no CUDA, no distributed init and no Megatron config.
"""

import pytest
import torch

from megatron.bridge.models.glm5_next.glm5_next_kpool import visible_tail_indices


INDEX_KPOOL_SHIPPED = 4  # zai-org/GLM-5.3-Flash


def hf_append_visible_tail(query: int, index_kpool: int, kv_length: int) -> list[int]:
    """Transcription of ``Glm5NextTextIndexer.append_visible_tail``.

    Specialised to the training case this implementation targets: no left padding, so
    ``first_key`` is 0 and a query at position ``t`` sees ``t + 1`` tokens.
    """
    visible_count = query + 1
    tail_count = visible_count % index_kpool
    tail_start = 0 + visible_count - tail_count

    out = []
    for offset in range(index_kpool - 1):
        index = tail_start + offset
        tail_valid = offset < tail_count and index < kv_length
        tail_visible = index <= query  # token_visible for this query
        out.append(index if (tail_valid and tail_visible) else -1)
    return out


def complete_pool_coverage(query: int, index_kpool: int) -> set[int]:
    """Tokens reachable through *complete* pools, which are anchored at multiples."""
    return {
        token
        for pool in range((query + 1) // index_kpool)
        for token in range(pool * index_kpool, pool * index_kpool + index_kpool)
    }


@pytest.mark.parametrize("index_kpool", [2, INDEX_KPOOL_SHIPPED, 8])
def test_matches_hf_over_a_full_period(index_kpool):
    """Every query in several full periods agrees with HF, slot for slot."""
    seqlen = 4 * index_kpool
    tail = visible_tail_indices(seqlen, index_kpool, seqlen)

    assert tail.shape == (seqlen, index_kpool - 1)
    for query in range(seqlen):
        assert tail[query].tolist() == hf_append_visible_tail(query, index_kpool, seqlen), (
            f"kpool={index_kpool} query={query}"
        )


@pytest.mark.parametrize("index_kpool", [2, INDEX_KPOOL_SHIPPED, 8])
def test_every_query_can_reach_its_own_token(index_kpool):
    """The regression itself: self must always be reachable without relying on top-k.

    ``index_kpool_always_select_tail`` exists to guarantee this. Before the fix it failed
    at every position that is a multiple of ``index_kpool``.
    """
    seqlen = 4 * index_kpool
    tail = visible_tail_indices(seqlen, index_kpool, seqlen)

    for query in range(seqlen):
        reachable = complete_pool_coverage(query, index_kpool) | {
            index for index in tail[query].tolist() if index >= 0
        }
        assert query in reachable, f"kpool={index_kpool} query={query} cannot attend to itself"


def test_first_query_has_a_valid_index():
    """At t=0 no complete pool exists, so the tail is the only source of indices."""
    tail = visible_tail_indices(8, INDEX_KPOOL_SHIPPED, 8)
    assert tail[0].tolist() == [0, -1, -1]


def test_tail_is_empty_when_pools_cover_everything():
    """visible % index_kpool == 0 means complete pools already cover the visible prefix."""
    tail = visible_tail_indices(8, INDEX_KPOOL_SHIPPED, 8)
    for query in (3, 7):
        assert tail[query].tolist() == [-1] * (INDEX_KPOOL_SHIPPED - 1)


def test_never_selects_a_future_or_out_of_range_token():
    """No emitted index may exceed the query position or the sequence length."""
    seqlen = 16
    tail = visible_tail_indices(seqlen, INDEX_KPOOL_SHIPPED, seqlen)
    for query in range(seqlen):
        for index in tail[query].tolist():
            if index >= 0:
                assert index <= query
                assert index < seqlen


def test_truncated_sequence_drops_out_of_range_slots():
    """seqlen shorter than the query range clamps the tail rather than pointing past it."""
    tail = visible_tail_indices(8, INDEX_KPOOL_SHIPPED, 5)
    for index in tail.flatten().tolist():
        assert index < 5


def test_dtype_and_device_are_honoured():
    tail = visible_tail_indices(8, INDEX_KPOOL_SHIPPED, 8, dtype=torch.int32)
    assert tail.dtype == torch.int32
