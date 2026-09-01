# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for importing Kimi-K3's MXFP4 routed experts into NVFP4 storage.

The claim these tests defend is that the import is lossless: dequantizing the
imported NVFP4 parameter must return exactly what dequantizing the source MXFP4
tensor returns. If that ever stops holding, holding the experts in 4 bits stops
being a pure memory change and quietly becomes a numerical one.
"""

import pytest
import torch

from megatron.bridge.models.conversion.quantization_utils import dequantize_mxfp4_e2m1_packed
from megatron.bridge.models.kimi.native_nvfp4_import import (
    _build_expert_weight,
    _load_mxfp4_weight,
    copy_native_nvfp4_expert_weight,
    is_routed_expert_weight,
    prepare_native_nvfp4_expert_weight,
)


_MXFP4_GROUP_SIZE = 32


def _random_mxfp4(rows: int, columns: int, *, exponent_span: int = 10, seed: int = 0):
    """Build a packed MXFP4 tensor and its E8M0 scales."""
    generator = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8, generator=generator)
    # Biased E8M0 exponents around 2^-10, spanning the requested range.
    scale = torch.randint(
        127 - 10,
        127 - 10 + exponent_span + 1,
        (rows, columns // _MXFP4_GROUP_SIZE),
        dtype=torch.uint8,
        generator=generator,
    )
    return packed, scale


def test_is_routed_expert_weight_matches_only_expert_projections():
    assert is_routed_expert_weight("decoder.layers.3.mlp.experts.linear_fc1.weight7")
    assert is_routed_expert_weight("decoder.layers.0.mlp.experts.linear_fc2.weight0")
    assert not is_routed_expert_weight("decoder.layers.3.mlp.shared_experts.linear_fc1.weight")
    assert not is_routed_expert_weight("decoder.layers.3.self_attn.q_proj.weight")


def test_scale_regroup_preserves_values_exactly():
    """The regrouped NVFP4 scales must reconstruct the MXFP4 values bit for bit."""
    packed, scale = _random_mxfp4(64, 256)

    prepared = _build_expert_weight(rowwise_data=packed, exponents=scale)

    # Reconstruct the way TE does: element * block_scale * amax / (6 * 448).
    block_scale = prepared.scale_inv.view(torch.float8_e4m3fn).float()
    global_scale = prepared.amax.float() / (6.0 * 448.0)
    equivalent_scale = block_scale[:, ::2] * global_scale

    expected = dequantize_mxfp4_e2m1_packed(packed, scale)
    actual = dequantize_mxfp4_e2m1_packed(packed, equivalent_scale)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_scale_regroup_covers_two_blocks_per_source_scale():
    packed, scale = _random_mxfp4(8, 128)

    prepared = _build_expert_weight(rowwise_data=packed, exponents=scale)

    assert prepared.scale_inv.shape == (8, 128 // 16)
    block_scale = prepared.scale_inv.view(torch.float8_e4m3fn).float()
    # Each source scale spans 32 elements, so consecutive pairs must be equal.
    torch.testing.assert_close(block_scale[:, ::2], block_scale[:, 1::2], rtol=0, atol=0)


def test_zero_e8m0_exponent_preserves_nonzero_payload():
    """E8M0 byte zero means 2^-127, not a zero scale."""
    packed = torch.zeros((1, _MXFP4_GROUP_SIZE // 2), dtype=torch.uint8)
    packed[0, 0] = 1
    scale = torch.zeros((1, 1), dtype=torch.uint8)

    prepared = _build_expert_weight(rowwise_data=packed, exponents=scale)

    block_scale = prepared.scale_inv.view(torch.float8_e4m3fn).float()
    global_scale = prepared.amax.float() / (6.0 * 448.0)
    equivalent_scale = block_scale[:, ::2] * global_scale
    expected = dequantize_mxfp4_e2m1_packed(packed, scale)
    actual = dequantize_mxfp4_e2m1_packed(packed, equivalent_scale)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_signed_zero_payload_does_not_expand_exponent_span():
    packed = torch.zeros((1, _MXFP4_GROUP_SIZE), dtype=torch.uint8)
    packed[:, : _MXFP4_GROUP_SIZE // 2] = 0x88
    packed[0, _MXFP4_GROUP_SIZE // 2] = 1
    scale = torch.tensor([[0, 127]], dtype=torch.uint8)

    prepared = _build_expert_weight(rowwise_data=packed, exponents=scale)

    block_scale = prepared.scale_inv.view(torch.float8_e4m3fn).float()
    assert bool((block_scale[:, :2] == 0).all())


def test_exponent_span_wider_than_e4m3_is_rejected():
    """Silently rounding a scale would corrupt an expert, so this must raise."""
    packed, scale = _random_mxfp4(8, 128, exponent_span=20)

    with pytest.raises(ValueError, match="span at most 17 powers of two"):
        _build_expert_weight(rowwise_data=packed, exponents=scale)


def test_load_mxfp4_weight_rejects_a_mismatched_scale_grid():
    packed, scale = _random_mxfp4(8, 128)
    state_dict = {"w.weight_packed": packed, "w.weight_scale": scale[:, :-1]}

    with pytest.raises(ValueError, match="MXFP4 scale for"):
        _load_mxfp4_weight("w.weight", state_dict)


def test_load_mxfp4_weight_reports_a_missing_tensor_by_name():
    packed, _ = _random_mxfp4(8, 128)

    with pytest.raises(KeyError, match="w.weight_scale"):
        _load_mxfp4_weight("w.weight", {"w.weight_packed": packed})


def test_fc1_concatenates_gate_and_up_under_tensor_parallelism():
    gate_packed, gate_scale = _random_mxfp4(64, 128, seed=1)
    up_packed, up_scale = _random_mxfp4(64, 128, seed=2)
    state_dict = {
        "gate.weight_packed": gate_packed,
        "gate.weight_scale": gate_scale,
        "up.weight_packed": up_packed,
        "up.weight_scale": up_scale,
    }

    prepared = prepare_native_nvfp4_expert_weight(
        megatron_param="decoder.layers.1.mlp.experts.linear_fc1.weight0",
        hf_param={"gate": "gate.weight", "up": "up.weight"},
        hf_state_dict=state_dict,
        tp_size=2,
        tp_rank=1,
    )

    # Each half contributes its own upper shard of 32 rows.
    assert prepared.rowwise_data.shape == (64, 64)
    torch.testing.assert_close(prepared.rowwise_data[:32], gate_packed[32:], rtol=0, atol=0)
    torch.testing.assert_close(prepared.rowwise_data[32:], up_packed[32:], rtol=0, atol=0)


def test_fc2_shards_along_the_input_dimension():
    packed, scale = _random_mxfp4(64, 256, seed=3)
    state_dict = {"down.weight_packed": packed, "down.weight_scale": scale}

    prepared = prepare_native_nvfp4_expert_weight(
        megatron_param="decoder.layers.1.mlp.experts.linear_fc2.weight0",
        hf_param="down.weight",
        hf_state_dict=state_dict,
        tp_size=2,
        tp_rank=0,
    )

    assert prepared.rowwise_data.shape == (64, 64)
    torch.testing.assert_close(prepared.rowwise_data, packed[:, :64], rtol=0, atol=0)


class _FakeNVFP4Destination:
    """Stands in for a TE NVFP4Tensor, which cannot be built without a GPU."""

    def __init__(self, rows: int, columns: int, scale_columns: int):
        self._rowwise_data = torch.zeros((rows, columns // 2), dtype=torch.uint8)
        self._rowwise_scale_inv = torch.full((rows, scale_columns), 255, dtype=torch.uint8)
        self._amax_rowwise = torch.zeros(1, dtype=torch.float32)
        self._columnwise_data = None
        self._columnwise_scale_inv = None
        self._with_gemm_swizzled_scales = False


def test_copy_writes_payload_scales_and_amax_and_zeroes_scale_padding():
    packed, scale = _random_mxfp4(16, 128, seed=4)
    prepared = _build_expert_weight(rowwise_data=packed, exponents=scale)
    # A destination whose scale grid is padded past the real blocks.
    destination = _FakeNVFP4Destination(16, 128, scale_columns=12)

    copy_native_nvfp4_expert_weight(destination, prepared)

    torch.testing.assert_close(destination._rowwise_data, packed, rtol=0, atol=0)
    torch.testing.assert_close(destination._rowwise_scale_inv[:, :8], prepared.scale_inv, rtol=0, atol=0)
    assert bool((destination._rowwise_scale_inv[:, 8:] == 0).all())
    torch.testing.assert_close(destination._amax_rowwise, prepared.amax, rtol=0, atol=0)


def test_copy_refuses_a_destination_that_keeps_a_columnwise_copy():
    packed, scale = _random_mxfp4(16, 128, seed=5)
    prepared = _build_expert_weight(rowwise_data=packed, exponents=scale)
    destination = _FakeNVFP4Destination(16, 128, scale_columns=8)
    destination._columnwise_data = torch.zeros(1, dtype=torch.uint8)

    with pytest.raises(ValueError, match="rowwise-only storage"):
        copy_native_nvfp4_expert_weight(destination, prepared)


def test_copy_refuses_a_swizzled_scale_layout():
    packed, scale = _random_mxfp4(16, 128, seed=6)
    prepared = _build_expert_weight(rowwise_data=packed, exponents=scale)
    destination = _FakeNVFP4Destination(16, 128, scale_columns=8)
    destination._with_gemm_swizzled_scales = True

    with pytest.raises(ValueError, match="unswizzled scale layout"):
        copy_native_nvfp4_expert_weight(destination, prepared)


def test_copy_refuses_a_nonstandard_e4m3_maximum():
    packed, scale = _random_mxfp4(16, 128, seed=6)
    prepared = _build_expert_weight(rowwise_data=packed, exponents=scale)
    destination = _FakeNVFP4Destination(16, 128, scale_columns=8)
    destination._nvfp4_e4m3_max = 256.0

    with pytest.raises(ValueError, match="standard E4M3 maximum of 448"):
        copy_native_nvfp4_expert_weight(destination, prepared)


@pytest.mark.parametrize("scale_shape", [(15, 8), (16, 7)])
def test_copy_refuses_an_incompatible_destination_scale_grid(scale_shape):
    packed, scale = _random_mxfp4(16, 128, seed=7)
    prepared = _build_expert_weight(rowwise_data=packed, exponents=scale)
    destination = _FakeNVFP4Destination(16, 128, scale_columns=8)
    destination._rowwise_scale_inv = torch.zeros(scale_shape, dtype=torch.uint8)

    with pytest.raises(ValueError, match="scale grid shape"):
        copy_native_nvfp4_expert_weight(destination, prepared)


def test_copy_refuses_an_incompatible_destination_amax_shape():
    packed, scale = _random_mxfp4(16, 128, seed=8)
    prepared = _build_expert_weight(rowwise_data=packed, exponents=scale)
    destination = _FakeNVFP4Destination(16, 128, scale_columns=8)
    destination._amax_rowwise = torch.zeros(2, dtype=torch.float32)

    with pytest.raises(ValueError, match="amax shape"):
        copy_native_nvfp4_expert_weight(destination, prepared)
