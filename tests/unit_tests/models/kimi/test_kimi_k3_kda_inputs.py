# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the shared KDA input helpers in the Kimi K3 layer stack."""

import pytest
import torch


try:
    from megatron.bridge.models.kimi.kimi_k3_layers import (
        _is_single_document,
        _prepare_kda_inputs,
        kda_cp_split_sections,
    )

    HAVE_BRIDGE = True
except ImportError:
    HAVE_BRIDGE = False

requires_bridge = pytest.mark.skipif(not HAVE_BRIDGE, reason="megatron.bridge not importable")


def _inputs(sequence_length: int = 8) -> tuple[torch.Tensor, ...]:
    # The unused batch slot gives the batch-one view a noncanonical stride,
    # matching the layout that exposed the TileLang backward failure.
    return tuple(torch.randn(2, sequence_length, 2, 3)[:1] for _ in range(5))


@requires_bridge
def test_implicit_document_preserves_inputs():
    inputs = _inputs()
    got, cu_seqlens, valid_length = _prepare_kda_inputs(inputs, None)
    assert all(actual is expected for actual, expected in zip(got, inputs))
    assert cu_seqlens is None
    assert valid_length == 8


@requires_bridge
def test_full_single_document_uses_dense_kda_with_canonical_inputs():
    inputs = _inputs()
    cu = torch.tensor([0, 8], dtype=torch.int32)
    got, kda_cu_seqlens, valid_length = _prepare_kda_inputs(inputs, cu)
    assert kda_cu_seqlens is None
    assert valid_length == 8
    for actual, expected in zip(got, inputs):
        assert actual is not expected
        assert actual.shape == expected.shape
        assert actual.stride() == (48, 6, 3, 1)
        assert actual.is_contiguous()


@requires_bridge
def test_padded_single_document_trims_to_canonical_inputs():
    inputs = _inputs()
    cu = torch.tensor([0, 5], dtype=torch.int32)
    got, kda_cu_seqlens, valid_length = _prepare_kda_inputs(inputs, cu)
    assert kda_cu_seqlens is None
    assert valid_length == 5
    for tensor in got:
        assert tensor.shape == (1, 5, 2, 3)
        assert tensor.stride() == (30, 6, 3, 1)
        assert tensor.is_contiguous()


@requires_bridge
def test_multi_document_preserves_varlen_metadata():
    inputs = _inputs()
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    got, kda_cu_seqlens, valid_length = _prepare_kda_inputs(inputs, cu)
    assert all(actual is expected for actual, expected in zip(got, inputs))
    assert kda_cu_seqlens is cu
    assert valid_length == 8
    assert not _is_single_document(cu)


@requires_bridge
@pytest.mark.parametrize("end", [0, 9])
def test_invalid_single_document_length_is_rejected(end):
    with pytest.raises(ValueError, match="single-document cu_seqlens"):
        _prepare_kda_inputs(_inputs(), torch.tensor([0, end], dtype=torch.int32))


@requires_bridge
def test_cp_split_sections_are_tp_local():
    """The a2a sections must be this rank's widths, not the global ones.

    Global widths sum past the fused projection's actual last dimension at
    tp>1, so the head permutation runs off the end of the tensor and
    index_select dies with a device-side assert. It does not silently produce
    wrong numbers. tp>1 is the case Kimi K3 runs (tp=16/32) and GLM-5.3 Flash
    does not (tp=1), which is why this went unnoticed.
    """
    num_heads, head_dim, tp = 96, 128, 16
    local_num_heads = num_heads // tp
    local_projection_size = local_num_heads * head_dim

    sections = kda_cp_split_sections(local_projection_size, local_num_heads)

    assert sections == (local_projection_size,) * 5 + (local_num_heads,)
    # The width the fused projection actually produces on this rank.
    assert sum(sections) == 5 * local_projection_size + local_num_heads
    assert sum(sections) != 5 * num_heads * head_dim + local_num_heads


@requires_bridge
@pytest.mark.parametrize("cp", [2, 3, 6])
def test_cp_split_sections_divide_evenly_for_k3_b300_mesh(cp):
    """K3's B300 mesh (96 KDA heads, tp=16) leaves 6 heads per rank."""
    local_num_heads = 96 // 16
    sections = kda_cp_split_sections(local_num_heads * 128, local_num_heads)
    assert all(section % cp == 0 for section in sections)
