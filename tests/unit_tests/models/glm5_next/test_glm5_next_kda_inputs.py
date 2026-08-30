# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for preparing single-document inputs for FLA KDA."""

import pytest
import torch


try:
    from megatron.bridge.models.glm5_next.modeling_glm5_next.kda import (
        _is_single_document,
        _prepare_kda_inputs,
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
