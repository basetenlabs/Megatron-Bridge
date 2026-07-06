"""Unit tests for ModelOpt NVFP4/FP8 dequantization helpers.

Pure-CPU torch tests: no GPU, no distributed context, no model instantiation.
"""

import pytest
import torch

from megatron.bridge.models.conversion.quantization_utils import (
    _FP4_E2M1_VALUES,
    dequantize_fp8_per_tensor,
    dequantize_nvfp4,
    maybe_dequantize_modelopt_weight,
)


def _pack_nvfp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit FP4 codes into uint8 bytes (low nibble = even element)."""
    assert codes.dtype == torch.uint8 and codes.shape[-1] % 2 == 0
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _reference_dequant(
    codes: torch.Tensor, block_scale: torch.Tensor, global_scale: float
) -> torch.Tensor:
    """Straightforward per-element reference implementation."""
    out_features, in_features = codes.shape
    block_size = in_features // block_scale.shape[1]
    lut = torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float32)
    result = torch.empty(out_features, in_features, dtype=torch.float32)
    for i in range(out_features):
        for j in range(in_features):
            scale = block_scale[i, j // block_size].to(torch.float32) * global_scale
            result[i, j] = lut[int(codes[i, j])] * scale
    return result.to(torch.bfloat16)


class TestDequantizeNvfp4:
    def test_matches_elementwise_reference(self):
        torch.manual_seed(0)
        out_features, in_features = 4, 64
        codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.uint8)
        # Powers of two are exactly representable in float8_e4m3fn.
        block_scale = torch.tensor(
            [[0.5, 1.0, 2.0, 4.0]] * out_features, dtype=torch.float32
        ).repeat_interleave(in_features // 16 // 4, dim=1).to(torch.float8_e4m3fn)
        global_scale = torch.tensor(2.0, dtype=torch.float32)

        result = dequantize_nvfp4(_pack_nvfp4(codes), block_scale, global_scale)

        assert result.dtype == torch.bfloat16
        assert result.shape == (out_features, in_features)
        expected = _reference_dequant(codes, block_scale, global_scale.item())
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_nibble_order_low_is_even_element(self):
        # 0x21: low nibble = code 1 (+0.5) -> element 0; high nibble = code 2 (+1.0) -> element 1.
        # 0xA3: low nibble = code 3 (+1.5) -> element 2; high nibble = code 10 (-1.0) -> element 3.
        packed = torch.tensor([[0x21, 0xA3]], dtype=torch.uint8)
        scale = torch.ones(1, 1, dtype=torch.float8_e4m3fn)
        result = dequantize_nvfp4(packed, scale, torch.tensor(1.0))
        expected = torch.tensor([[0.5, 1.0, 1.5, -1.0]], dtype=torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_block_size_derived_from_scale_shape(self):
        # 16 elements with 2 scale blocks -> block size 8, not the NVFP4 default 16.
        codes = torch.full((2, 16), 2, dtype=torch.uint8)  # every element +1.0
        block_scale = torch.tensor(
            [[1.0, 2.0], [4.0, 0.5]], dtype=torch.float32
        ).to(torch.float8_e4m3fn)
        result = dequantize_nvfp4(_pack_nvfp4(codes), block_scale, torch.tensor(1.0))
        expected = torch.cat(
            [
                torch.cat([torch.full((1, 8), 1.0), torch.full((1, 8), 2.0)], dim=1),
                torch.cat([torch.full((1, 8), 4.0), torch.full((1, 8), 0.5)], dim=1),
            ],
            dim=0,
        ).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_global_scale_applied(self):
        packed = torch.tensor([[0x22]], dtype=torch.uint8)  # both elements +1.0
        scale = torch.full((1, 1), 2.0).to(torch.float8_e4m3fn)
        result = dequantize_nvfp4(packed, scale, torch.tensor(0.25))
        torch.testing.assert_close(
            result, torch.full((1, 2), 0.5, dtype=torch.bfloat16), rtol=0, atol=0
        )

    def test_rejects_indivisible_scale_blocks(self):
        packed = torch.zeros(2, 8, dtype=torch.uint8)  # in_features = 16
        bad_scale = torch.ones(2, 3, dtype=torch.float8_e4m3fn)  # 16 % 3 != 0
        with pytest.raises(ValueError, match="not divisible"):
            dequantize_nvfp4(packed, bad_scale, torch.tensor(1.0))

    def test_rejects_non_uint8_weight(self):
        with pytest.raises(ValueError, match="uint8"):
            dequantize_nvfp4(
                torch.zeros(2, 8, dtype=torch.int32),
                torch.ones(2, 1, dtype=torch.float8_e4m3fn),
                torch.tensor(1.0),
            )


class TestDequantizeFp8PerTensor:
    def test_scalar_scale(self):
        weight = torch.tensor([[1.0, -2.0], [0.5, 4.0]]).to(torch.float8_e4m3fn)
        result = dequantize_fp8_per_tensor(weight, torch.tensor(2.0))
        expected = torch.tensor([[2.0, -4.0], [1.0, 8.0]], dtype=torch.bfloat16)
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_rejects_non_fp8_weight(self):
        with pytest.raises(ValueError, match="FP8"):
            dequantize_fp8_per_tensor(torch.ones(2, 2), torch.tensor(1.0))


class TestMaybeDequantizeModeloptWeight:
    def test_nvfp4_dispatch(self):
        packed = torch.tensor([[0x21, 0xA3]], dtype=torch.uint8)
        state = {
            "backbone.layers.5.mixer.experts.0.up_proj.weight": packed,
            "backbone.layers.5.mixer.experts.0.up_proj.weight_scale": torch.ones(
                1, 1, dtype=torch.float8_e4m3fn
            ),
            "backbone.layers.5.mixer.experts.0.up_proj.weight_scale_2": torch.tensor(1.0),
        }
        result = maybe_dequantize_modelopt_weight(
            "backbone.layers.5.mixer.experts.0.up_proj.weight", state
        )
        expected = torch.tensor([[0.5, 1.0, 1.5, -1.0]], dtype=torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_fp8_dispatch_with_scale(self):
        state = {
            "backbone.layers.3.mixer.in_proj.weight": torch.tensor([[1.0, 2.0]]).to(
                torch.float8_e4m3fn
            ),
            "backbone.layers.3.mixer.in_proj.weight_scale": torch.tensor(4.0),
        }
        result = maybe_dequantize_modelopt_weight("backbone.layers.3.mixer.in_proj.weight", state)
        torch.testing.assert_close(
            result, torch.tensor([[4.0, 8.0]], dtype=torch.bfloat16), rtol=0, atol=0
        )

    def test_fp8_without_scale_is_plain_cast(self):
        state = {"w": torch.tensor([[1.0, -2.0]]).to(torch.float8_e4m3fn)}
        result = maybe_dequantize_modelopt_weight("w", state)
        torch.testing.assert_close(
            result, torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16), rtol=0, atol=0
        )

    def test_bf16_passthrough_is_identity(self):
        weight = torch.randn(4, 4, dtype=torch.bfloat16)
        state = {"backbone.layers.0.mixer.q_proj.weight": weight}
        result = maybe_dequantize_modelopt_weight("backbone.layers.0.mixer.q_proj.weight", state)
        assert result is weight

    def test_uint8_without_sidecars_passthrough(self):
        # Unknown uint8 packing: return unchanged so the loader's shape check fails loudly.
        weight = torch.zeros(2, 4, dtype=torch.uint8)
        state = {"w": weight}
        assert maybe_dequantize_modelopt_weight("w", state) is weight
