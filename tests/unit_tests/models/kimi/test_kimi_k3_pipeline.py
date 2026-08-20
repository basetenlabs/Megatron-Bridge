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

import pytest
import torch

from megatron.bridge.models.kimi.kimi_k3_pipeline import (
    bank_num_rows,
    pack_stage_boundary,
    unpack_stage_boundary,
)


@pytest.mark.parametrize(
    ("layer_idx", "block_size", "expected"),
    [(1, 12, 1), (12, 12, 1), (13, 12, 2), (24, 12, 2), (25, 12, 3), (5, 3, 2)],
)
def test_bank_num_rows_counts_snapshots_taken_before_the_layer(layer_idx, block_size, expected):
    # A wrong row count is not caught downstream: the payload width still
    # matches for the wrong split, so the bank silently shifts under PP.
    assert bank_num_rows(layer_idx, block_size) == expected


def test_bank_num_rows_rejects_a_boundary_before_the_first_snapshot():
    with pytest.raises(ValueError):
        bank_num_rows(0, 12)


def test_stage_boundary_round_trips():
    torch.manual_seed(0)
    seq, batch, hidden, rows = 3, 2, 8, 4
    prefix = torch.randn(seq, batch, hidden)
    bank = torch.randn(seq, batch, rows, hidden)

    packed = pack_stage_boundary(prefix, bank)
    assert packed.shape[-1] == (1 + rows) * hidden

    got_prefix, got_bank = unpack_stage_boundary(packed, hidden, rows)
    torch.testing.assert_close(got_prefix, prefix)
    torch.testing.assert_close(got_bank, bank)


def test_unpack_rejects_a_width_that_disagrees_with_the_row_count():
    packed = torch.zeros(2, 1, 5 * 8)
    with pytest.raises(ValueError):
        unpack_stage_boundary(packed, hidden_size=8, num_rows=3)


def test_pack_rejects_an_empty_bank():
    with pytest.raises(ValueError):
        pack_stage_boundary(torch.zeros(2, 1, 8), torch.zeros(2, 1, 0, 8))


@pytest.mark.parametrize(
    ("layout_str", "pp_size", "num_layers", "expected_starts", "expected_bank_rows"),
    [
        # Even split: 3 stages of 4 decoder layers.
        ("Et*4|t*4|t*4,L", 3, 12, {0, 4, 8}, [1, 2]),
        # Uneven custom layout: stage 0 carries the embedding and fewer layers.
        ("Et*2|t*5|t*5,L", 3, 12, {0, 2, 7}, [1, 2]),
    ],
)
def test_custom_layout_reports_one_start_per_stage(
    layout_str, pp_size, num_layers, expected_starts, expected_bank_rows
):
    """Every stage's first layer must be enumerable from a single process.

    ``KimiK3DecoderLayer`` decides whether it packs or unpacks the AttnRes bank
    by asking for each pipeline rank's first layer in turn. When the custom
    layout branch ignored the rank it was given, every rank answered with its own
    offset, the set collapsed to one element, and the sending and receiving
    stages disagreed about the payload width.
    """
    from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_attention_heads=8,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16,
        pipeline_model_parallel_layout=PipelineParallelLayerLayout.from_str(layout_str, pp_size),
    )
    starts = {get_transformer_layer_offset(config, None, rank) for rank in range(pp_size)}
    assert starts == expected_starts

    # The payload width each boundary carries follows from its start, so a
    # collapsed set of starts also means the wrong number of bank rows.
    boundaries = sorted(start for start in starts if start > 0)
    assert [bank_num_rows(start, 4) for start in boundaries] == expected_bank_rows
