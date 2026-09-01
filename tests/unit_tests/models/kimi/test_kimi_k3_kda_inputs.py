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


@pytest.mark.skipif(not HAVE_BRIDGE, reason="megatron.bridge not importable")
def test_head_perm_and_parameter_slice_agree_at_tp_gt_1():
    """The a2a permutation and the per-rank parameter slices must pick the same heads.

    The CP path permutes the fused projection with
    ``_build_head_perm_for_split_sections(sections, cp)`` and then slices
    ``A_log``/``dt_bias``/conv weights with ``get_parameter_local_cp``. If those
    two disagree about which heads rank r owns, every rank computes a correct
    recurrence on mismatched state and the output is silently wrong -- no shape
    error anywhere.

    Runs on CPU: this is index arithmetic, not a collective, so it covers the
    tp>1 case that the 2-GPU cp2-vs-cp1 parity test (tp=1 only) cannot reach.
    """
    torch = pytest.importorskip("torch")
    from megatron.core.ssm.gated_delta_net.common import _build_head_perm_for_split_sections

    tp, cp, num_heads, head_dim = 16, 2, 96, 128
    local_num_heads = num_heads // tp
    local_projection_size = local_num_heads * head_dim
    sections = kda_cp_split_sections(local_projection_size, local_num_heads)

    width = sum(sections)
    perm = _build_head_perm_for_split_sections(sections, cp, torch.device("cpu"))
    assert perm.numel() == width, "permutation must cover the fused projection exactly"
    assert sorted(perm.tolist()) == list(range(width)), "permutation must be a bijection"

    # After the permutation, the a2a hands rank r the r-th contiguous block.
    per_rank = width // cp
    for rank in range(cp):
        block = perm[rank * per_rank : (rank + 1) * per_rank].tolist()
        # Rebuild what that block should be: from each section in order, the
        # r-th even chunk -- exactly what get_parameter_local_cp slices out.
        expected, offset = [], 0
        for size in sections:
            chunk = size // cp
            expected.extend(range(offset + rank * chunk, offset + (rank + 1) * chunk))
            offset += size
        assert block == expected, f"rank {rank} a2a block does not match its parameter slice"
