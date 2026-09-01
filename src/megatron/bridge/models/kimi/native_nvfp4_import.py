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

"""Load Kimi-K3's MXFP4 routed experts straight into NVFP4 parameter storage.

Both formats carry the same E2M1 elements, packed two per byte, so the payload
moves across untouched. Only the scales are rearranged, and the rearrangement is
exact:

* MXFP4 carries one E8M0 scale, always a power of two, per 32 elements along a
  row. NVFP4 carries one E4M3 scale per 16 elements, so each source scale simply
  covers two destination blocks.
* NVFP4 then applies a second, per-tensor level: a value is reconstructed as
  ``element * block_scale * amax / (E2M1_MAX * E4M3_MAX)``. Choosing that
  per-tensor factor to be a power of two keeps every block scale a power of two
  too, and powers of two from 2^-9 to 2^8 are exact in E4M3.

The consequence worth stating plainly: dequantizing the imported parameter
returns bit-for-bit what dequantizing the source MXFP4 checkpoint would have
returned. Holding the experts in 4 bits is a memory change, not a numerical one.

The one real constraint is that a tensor's exponents must span at most the 17
powers of two E4M3 can represent exactly. That is checked per tensor and raises
rather than silently rounding.
"""

import re
from dataclasses import dataclass
from typing import Mapping, cast

import torch


# MXFP4 groups 32 elements per scale; NVFP4 groups 16.
_MXFP4_GROUP_SIZE = 32
_NVFP4_BLOCK_SIZE = 16
_NVFP4_E2M1_MAX = 6.0
_NVFP4_E4M3_MAX = 448.0

# E8M0 stores a biased exponent: the scale is 2 ** (stored - 127).
_E8M0_BIAS = 127

# E4M3 represents powers of two exactly from 2^-9 (subnormal) to 2^8.
_E4M3_MIN_EXACT_EXPONENT = -9
_E4M3_MAX_EXACT_EXPONENT = 8

_ROUTED_EXPERT_WEIGHT = re.compile(
    r"^decoder\.layers\.\d+\.mlp\.experts\.linear_fc(?P<projection>[12])\.weight(?P<expert>\d+)$"
)
_ROUTED_EXPERT_PREFIX = re.compile(r"^decoder\.layers\.\d+\.mlp\.experts\.linear_fc[12]\.weight")


@dataclass(frozen=True)
class NativeNVFP4ExpertWeight:
    """One ETP-local expert weight in TE rowwise NVFP4 layout."""

    rowwise_data: torch.Tensor
    scale_inv: torch.Tensor
    amax: torch.Tensor


def is_routed_expert_weight(param_name: str) -> bool:
    """Return whether a parameter belongs to a routed expert projection."""
    return _ROUTED_EXPERT_PREFIX.match(param_name) is not None


def prepare_native_nvfp4_expert_weight(
    *,
    megatron_param: str,
    hf_param: str | Mapping[str, str],
    hf_state_dict: Mapping[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> NativeNVFP4ExpertWeight:
    """Prepare one native NVFP4 expert shard without dequantization."""
    match = _ROUTED_EXPERT_WEIGHT.fullmatch(megatron_param)
    if match is None:
        raise ValueError(f"Native NVFP4 import does not support parameter {megatron_param!r}")

    # The mapping registry gives FC1 gate/up names and one FC2 down-projection name.
    if match.group("projection") == "1":
        return _prepare_fc1(
            hf_params=cast(Mapping[str, str], hf_param),
            hf_state_dict=hf_state_dict,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

    return _prepare_fc2(
        hf_param=cast(str, hf_param),
        hf_state_dict=hf_state_dict,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )


def _prepare_fc1(
    *,
    hf_params: Mapping[str, str],
    hf_state_dict: Mapping[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> NativeNVFP4ExpertWeight:
    gate_data, gate_exponent = _load_mxfp4_weight(hf_params["gate"], hf_state_dict)
    up_data, up_exponent = _load_mxfp4_weight(hf_params["up"], hf_state_dict)
    if gate_data.shape != up_data.shape or gate_exponent.shape != up_exponent.shape:
        raise ValueError("Native NVFP4 FC1 gate and up shapes must match")

    # FC1 is column parallel: gate and up are each sharded on the output dimension
    # and then concatenated, which is a plain row concatenation for the payload and
    # for the scale grid alike.
    gate_data = _shard(gate_data, dim=0, size=tp_size, rank=tp_rank, name="FC1 payload")
    up_data = _shard(up_data, dim=0, size=tp_size, rank=tp_rank, name="FC1 payload")
    gate_exponent = _shard(gate_exponent, dim=0, size=tp_size, rank=tp_rank, name="FC1 scale grid")
    up_exponent = _shard(up_exponent, dim=0, size=tp_size, rank=tp_rank, name="FC1 scale grid")

    return _build_expert_weight(
        rowwise_data=torch.cat((gate_data, up_data), dim=0),
        exponents=torch.cat((gate_exponent, up_exponent), dim=0),
    )


def _prepare_fc2(
    *,
    hf_param: str,
    hf_state_dict: Mapping[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> NativeNVFP4ExpertWeight:
    down_data, down_exponent = _load_mxfp4_weight(hf_param, hf_state_dict)
    # FC2 is row parallel, so the shard runs along the input dimension. The payload
    # holds two elements per byte and the scale grid one entry per 32 elements, so
    # both shard cleanly only when the split lands on those group boundaries.
    down_data = _shard(down_data, dim=1, size=tp_size, rank=tp_rank, name="FC2 payload")
    down_exponent = _shard(down_exponent, dim=1, size=tp_size, rank=tp_rank, name="FC2 scale grid")

    return _build_expert_weight(rowwise_data=down_data, exponents=down_exponent)


def _build_expert_weight(*, rowwise_data: torch.Tensor, exponents: torch.Tensor) -> NativeNVFP4ExpertWeight:
    """Regroup MXFP4 scales into NVFP4's two scaling levels, exactly.

    ``exponents`` holds the raw E8M0 bytes, so the source scale of a block is
    ``2 ** (exponent - 127)``. Anchoring the per-tensor level at the largest
    exponent puts every block scale at or below 2^8, and the span check below is
    what guarantees none falls under 2^-9.
    """
    bytes_per_group = _MXFP4_GROUP_SIZE // 2
    payload_groups = rowwise_data.reshape(*exponents.shape, bytes_per_group)
    empty_groups = torch.all(payload_groups == 0, dim=-1)
    present = exponents[~empty_groups]
    if present.numel() == 0:
        raise ValueError("Native NVFP4 import found an expert weight with no non-zero scales")

    max_exponent = int(present.max().item()) - _E8M0_BIAS
    min_exponent = int(present.min().item()) - _E8M0_BIAS
    span = max_exponent - min_exponent
    exact_span = _E4M3_MAX_EXACT_EXPONENT - _E4M3_MIN_EXACT_EXPONENT
    if span > exact_span:
        raise ValueError(
            f"Native NVFP4 import requires MXFP4 scale exponents to span at most "
            f"{exact_span} powers of two, got {span} (2^{min_exponent}..2^{max_exponent})"
        )

    global_exponent = max_exponent - _E4M3_MAX_EXACT_EXPONENT
    block_exponents = exponents.int() - _E8M0_BIAS - global_exponent
    # Empty payload groups need no scale. E8M0 byte zero is otherwise the valid
    # exponent -127 and must not be confused with a zero scale.
    block_scale = torch.where(
        ~empty_groups,
        torch.pow(2.0, block_exponents.float()),
        torch.zeros_like(block_exponents, dtype=torch.float32),
    )

    # Each source scale covers 32 elements, which is two NVFP4 blocks of 16.
    block_scale = block_scale.repeat_interleave(_MXFP4_GROUP_SIZE // _NVFP4_BLOCK_SIZE, dim=1)

    amax = torch.tensor([2.0**global_exponent * _NVFP4_E2M1_MAX * _NVFP4_E4M3_MAX], dtype=torch.float32)
    return NativeNVFP4ExpertWeight(
        rowwise_data=rowwise_data,
        scale_inv=block_scale.to(torch.float8_e4m3fn).view(torch.uint8),
        amax=amax,
    )


def copy_native_nvfp4_expert_weight(destination: torch.Tensor, source: NativeNVFP4ExpertWeight) -> None:
    """Copy native nibbles, regrouped scales, and the per-tensor amax into a TE tensor."""
    rowwise_data = destination._rowwise_data
    rowwise_scale_inv = destination._rowwise_scale_inv
    amax = destination._amax_rowwise
    if rowwise_data is None or rowwise_data.dtype is not torch.uint8:
        raise ValueError("Native NVFP4 destination requires uint8 rowwise payload")
    if rowwise_scale_inv is None or rowwise_scale_inv.dtype is not torch.uint8:
        raise ValueError("Native NVFP4 destination requires uint8 rowwise scales")
    if amax is None:
        raise ValueError("Native NVFP4 destination requires a rowwise amax")
    if destination._columnwise_data is not None or destination._columnwise_scale_inv is not None:
        raise ValueError("Native NVFP4 import requires rowwise-only storage")
    if destination._with_gemm_swizzled_scales:
        raise ValueError("Native NVFP4 import requires an unswizzled scale layout")
    if rowwise_data.shape != source.rowwise_data.shape:
        raise ValueError(
            f"Native NVFP4 payload shape {tuple(source.rowwise_data.shape)} does not match "
            f"destination {tuple(rowwise_data.shape)}"
        )
    source_scale_shape = tuple(source.scale_inv.shape)
    destination_scale_shape = tuple(rowwise_scale_inv.shape)
    if (
        len(destination_scale_shape) != 2
        or destination_scale_shape[0] != source_scale_shape[0]
        or destination_scale_shape[1] < source_scale_shape[1]
    ):
        raise ValueError(
            f"Native NVFP4 scale grid shape {source_scale_shape} does not fit destination {destination_scale_shape}"
        )
    if amax.shape != source.amax.shape:
        raise ValueError(
            f"Native NVFP4 amax shape {tuple(source.amax.shape)} does not match destination {tuple(amax.shape)}"
        )

    with torch.no_grad():
        rowwise_data.copy_(source.rowwise_data.to(device=rowwise_data.device))
        rowwise_scale_inv.zero_()
        rowwise_scale_inv[:, : source.scale_inv.shape[1]].copy_(source.scale_inv.to(device=rowwise_scale_inv.device))
        amax.copy_(source.amax.to(device=amax.device))


def _load_mxfp4_weight(
    weight_name: str, hf_state_dict: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the packed nibbles and raw E8M0 exponent bytes for one expert tensor."""
    packed_name = f"{weight_name}_packed"
    scale_name = f"{weight_name}_scale"
    try:
        packed = hf_state_dict[packed_name]
        scale = hf_state_dict[scale_name]
    except KeyError as error:
        raise KeyError(f"Native NVFP4 checkpoint is missing tensor {error.args[0]!r}") from None

    if packed.dtype is not torch.uint8:
        raise ValueError(f"MXFP4 payload {packed_name!r} must be uint8, got {packed.dtype}")
    if scale.dtype is not torch.uint8:
        raise ValueError(f"MXFP4 scale {scale_name!r} must be uint8 E8M0, got {scale.dtype}")
    if packed.ndim != 2 or scale.ndim != 2:
        raise ValueError("MXFP4 payloads and scales must be two-dimensional")

    columns = packed.shape[1] * 2
    if columns % _MXFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"MXFP4 weight {weight_name!r} needs whole {_MXFP4_GROUP_SIZE}-element groups, got {columns} columns"
        )
    expected_scale_shape = (packed.shape[0], columns // _MXFP4_GROUP_SIZE)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(f"MXFP4 scale for {weight_name!r} must be {expected_scale_shape}, got {tuple(scale.shape)}")
    return packed, scale


def _shard(tensor: torch.Tensor, *, dim: int, size: int, rank: int, name: str) -> torch.Tensor:
    if tensor.shape[dim] % size != 0:
        raise ValueError(f"Cannot shard {name} dimension {tensor.shape[dim]} across {size} ranks")
    shard_size = tensor.shape[dim] // size
    return tensor.narrow(dim, rank * shard_size, shard_size)
