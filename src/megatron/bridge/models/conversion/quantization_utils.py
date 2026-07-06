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

"""Dequantization helpers for importing quantized HF checkpoints.

Currently covers the TensorRT Model Optimizer (ModelOpt) HF export formats
used by e.g. ``nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4``:

* NVFP4: ``weight`` packed as uint8 (two FP4-E2M1 nibbles per byte) with a
  ``weight_scale`` sidecar (float8_e4m3fn, one scale per block along the
  input dim) and a ``weight_scale_2`` sidecar (float32 per-tensor scale).
* FP8: ``weight`` stored as float8 with a ``weight_scale`` sidecar
  (float32 per-tensor scale).

Dispatch is gated purely on tensor dtype plus sidecar-key presence, so plain
BF16 checkpoints pass through untouched and no ``quantization_config``
parsing is required.
"""

from typing import Mapping

import torch


# FP4 E2M1 value table indexed by the 4-bit code (sign bit in the MSB).
_FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a ModelOpt NVFP4-packed weight to a dense tensor.

    The true weight is ``fp4_value * weight_scale[block] * weight_scale_2``
    where each scale block covers a contiguous group of elements along the
    input (last) dimension. Packing convention (ModelOpt/vLLM): the low
    nibble of each byte holds the even element, the high nibble the odd one.

    Args:
        weight_packed: uint8 tensor of shape ``[out_features, in_features // 2]``.
        weight_scale: float8_e4m3fn tensor of shape ``[out_features, num_blocks]``
            (unswizzled checkpoint layout).
        weight_scale_2: per-tensor float scale (scalar tensor).
        dtype: output dtype.

    Returns:
        Dense tensor of shape ``[out_features, in_features]`` in ``dtype``.
    """
    if weight_packed.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 packed NVFP4 weight, got {weight_packed.dtype}")
    if weight_packed.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError(
            f"Expected 2D packed weight and scale, got {tuple(weight_packed.shape)} / {tuple(weight_scale.shape)}"
        )

    out_features, packed_in = weight_packed.shape
    in_features = packed_in * 2
    if weight_scale.shape[0] != out_features:
        raise ValueError(f"weight_scale rows ({weight_scale.shape[0]}) do not match out_features ({out_features})")
    num_blocks = weight_scale.shape[1]
    if num_blocks == 0 or in_features % num_blocks != 0:
        raise ValueError(f"in_features ({in_features}) is not divisible by the number of scale blocks ({num_blocks})")
    block_size = in_features // num_blocks

    lut = torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float32, device=weight_packed.device)
    idx_lo = (weight_packed & 0x0F).to(torch.long)
    idx_hi = (weight_packed >> 4).to(torch.long)
    values = torch.empty(out_features, in_features, dtype=torch.float32, device=weight_packed.device)
    values[:, 0::2] = lut[idx_lo]
    values[:, 1::2] = lut[idx_hi]

    scales = weight_scale.to(torch.float32) * weight_scale_2.to(torch.float32)
    values = values.view(out_features, num_blocks, block_size) * scales.unsqueeze(-1)
    return values.view(out_features, in_features).to(dtype)


def dequantize_fp8_per_tensor(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 weight with a per-tensor (or broadcastable) scale."""
    if weight.dtype not in _FP8_DTYPES:
        raise ValueError(f"Expected an FP8 weight, got {weight.dtype}")
    return (weight.to(torch.float32) * weight_scale.to(torch.float32)).to(dtype)


def maybe_dequantize_modelopt_weight(
    param_name: str,
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Load ``param_name`` from ``hf_state_dict``, dequantizing if quantized.

    Dispatch:
    * uint8 weight with ``<name>_scale`` and ``<name>_scale_2`` sidecars ->
      NVFP4 dequant.
    * float8 weight with a ``<name>_scale`` sidecar -> FP8 dequant. Without a
      sidecar the weight is plainly cast to ``dtype``.
    * anything else -> returned unchanged.

    A uint8 weight without both NVFP4 sidecars is returned unchanged: it is
    not a packing we understand, and the loader's downstream shape check will
    fail loudly rather than load garbage.
    """
    weight = hf_state_dict[param_name]

    if weight.dtype == torch.uint8:
        scale_key = f"{param_name}_scale"
        global_scale_key = f"{param_name}_scale_2"
        if scale_key in hf_state_dict and global_scale_key in hf_state_dict:
            return dequantize_nvfp4(
                weight,
                hf_state_dict[scale_key],
                hf_state_dict[global_scale_key],
                dtype=dtype,
            )
        return weight

    if weight.dtype in _FP8_DTYPES:
        scale_key = f"{param_name}_scale"
        if scale_key in hf_state_dict:
            return dequantize_fp8_per_tensor(weight, hf_state_dict[scale_key], dtype=dtype)
        return weight.to(dtype)

    return weight
