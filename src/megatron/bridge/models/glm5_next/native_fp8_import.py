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

import re
from dataclasses import dataclass
from typing import Mapping

import torch


_FP8_BLOCK_SIZE = 128
_ROUTED_EXPERT_WEIGHT = re.compile(
    r"^decoder\.layers\.\d+\.mlp\.experts\.linear_fc(?P<projection>[12])\.weight(?P<expert>\d+)$"
)
_ROUTED_EXPERT_PREFIX = re.compile(r"^decoder\.layers\.\d+\.mlp\.experts\.linear_fc[12]\.weight")


@dataclass(frozen=True)
class NativeFP8ExpertWeight:
    """A local expert shard in TE's rowwise block-scaled storage layout."""

    rowwise_data: torch.Tensor
    scale_inv: torch.Tensor
    logical_shape: tuple[int, int]


def is_routed_expert_weight(param_name: str) -> bool:
    """Return whether a parameter belongs to a routed grouped expert projection."""
    return _ROUTED_EXPERT_PREFIX.match(param_name) is not None


def prepare_native_fp8_expert_weight(
    *,
    megatron_param: str,
    hf_param: str | Mapping[str, str],
    hf_state_dict: Mapping[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> NativeFP8ExpertWeight:
    """Prepare one ETP-local GLM expert shard without dequantizing its FP8 payload.

    FC1 shards gate and up independently along rows and then fuses the local
    shards. FC2 shards down projection columns. The compact scale grid follows
    the same operations in units of 128x128 blocks.

    Args:
        megatron_param: Resolved Megatron parameter name.
        hf_param: Resolved HF weight name, or gate/up names for FC1.
        hf_state_dict: Lazy HF checkpoint tensor mapping.
        tp_size: Expert tensor-parallel world size.
        tp_rank: Expert tensor-parallel rank.

    Returns:
        Native FP8 bytes and compact inverse scales for the local parameter.

    Raises:
        KeyError: If a required checkpoint weight or scale is absent.
        ValueError: If the layout, dtype, geometry, or ETP partition is unsupported.
    """
    match = _ROUTED_EXPERT_WEIGHT.fullmatch(megatron_param)
    if match is None:
        if is_routed_expert_weight(megatron_param):
            raise ValueError(f"Native FP8 import requires a numbered grouped-expert weight, got {megatron_param!r}")
        raise ValueError(f"Native FP8 import does not support Megatron parameter {megatron_param!r}")
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"Invalid expert tensor-parallel rank {tp_rank} for size {tp_size}")

    projection = match.group("projection")
    if projection == "1":
        if not isinstance(hf_param, Mapping) or set(hf_param) != {"gate", "up"}:
            raise ValueError("Native FP8 FC1 import requires exactly the HF gate and up weight names")
        gate_data, gate_scale = _load_and_validate_blockwise_weight(hf_param["gate"], hf_state_dict)
        up_data, up_scale = _load_and_validate_blockwise_weight(hf_param["up"], hf_state_dict)
        if gate_data.shape != up_data.shape or gate_scale.shape != up_scale.shape:
            raise ValueError("Native FP8 FC1 gate and up payload and scale shapes must match")

        gate_data = _shard(gate_data, dim=0, size=tp_size, rank=tp_rank, name="FC1 payload")
        up_data = _shard(up_data, dim=0, size=tp_size, rank=tp_rank, name="FC1 payload")
        gate_scale = _shard(gate_scale, dim=0, size=tp_size, rank=tp_rank, name="FC1 scale grid")
        up_scale = _shard(up_scale, dim=0, size=tp_size, rank=tp_rank, name="FC1 scale grid")
        rowwise_data = torch.cat((gate_data.view(torch.uint8), up_data.view(torch.uint8)), dim=0)
        scale_inv = torch.cat((gate_scale, up_scale), dim=0)
    else:
        if not isinstance(hf_param, str):
            raise ValueError("Native FP8 FC2 import requires one HF down-projection weight name")
        down_data, down_scale = _load_and_validate_blockwise_weight(hf_param, hf_state_dict)
        down_data = _shard(down_data, dim=1, size=tp_size, rank=tp_rank, name="FC2 payload")
        scale_inv = _shard(down_scale, dim=1, size=tp_size, rank=tp_rank, name="FC2 scale grid")
        rowwise_data = down_data.view(torch.uint8)

    return NativeFP8ExpertWeight(
        rowwise_data=rowwise_data,
        scale_inv=scale_inv,
        logical_shape=(rowwise_data.shape[0], rowwise_data.shape[1]),
    )


def copy_native_fp8_expert_weight(destination: torch.Tensor, source: NativeFP8ExpertWeight) -> None:
    """Copy native FP8 bytes and compact scales into a TE blockwise tensor."""
    rowwise_data = destination._rowwise_data
    rowwise_scale_inv = destination._rowwise_scale_inv
    if tuple(destination.shape) != source.logical_shape:
        raise ValueError(
            f"Native FP8 destination shape mismatch: expected {tuple(destination.shape)}, got {source.logical_shape}"
        )
    if rowwise_data is None or rowwise_data.dtype is not torch.uint8:
        raise ValueError("Native FP8 destination must have uint8 rowwise payload storage")
    if tuple(rowwise_data.shape) != source.logical_shape:
        raise ValueError(
            f"Native FP8 destination payload shape mismatch: expected {source.logical_shape}, got {rowwise_data.shape}"
        )
    if rowwise_scale_inv is None or rowwise_scale_inv.dtype is not torch.float32:
        raise ValueError("Native FP8 destination must have float32 rowwise inverse-scale storage")
    if (
        rowwise_scale_inv.shape[0] != source.scale_inv.shape[0]
        or rowwise_scale_inv.shape[1] < source.scale_inv.shape[1]
    ):
        raise ValueError(
            "Native FP8 destination scale storage cannot hold compact scale shape "
            f"{tuple(source.scale_inv.shape)} in {tuple(rowwise_scale_inv.shape)}"
        )
    if destination._columnwise_data is not None or destination._columnwise_scale_inv is not None:
        raise ValueError("Native FP8 expert import requires rowwise-only destination storage")

    with torch.no_grad():
        rowwise_data.copy_(source.rowwise_data.to(device=rowwise_data.device))
        rowwise_scale_inv.zero_()
        rowwise_scale_inv[:, : source.scale_inv.shape[1]].copy_(source.scale_inv.to(device=rowwise_scale_inv.device))


def _load_and_validate_blockwise_weight(
    weight_name: str, hf_state_dict: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_name = f"{weight_name}_scale_inv"
    try:
        weight = hf_state_dict[weight_name]
        scale_inv = hf_state_dict[scale_name]
    except KeyError as error:
        raise KeyError(f"Native FP8 checkpoint is missing required tensor {error.args[0]!r}") from None

    if weight.dtype is not torch.float8_e4m3fn:
        raise ValueError(f"Native FP8 checkpoint weight {weight_name!r} must use E4M3, got {weight.dtype}")
    if scale_inv.dtype is not torch.float32:
        raise ValueError(f"Native FP8 checkpoint scale {scale_name!r} must use float32, got {scale_inv.dtype}")
    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError("Native FP8 checkpoint weights and compact scale grids must both be two-dimensional")
    if any(dimension % _FP8_BLOCK_SIZE != 0 for dimension in weight.shape):
        raise ValueError(
            f"Native FP8 checkpoint weight {weight_name!r} must use complete 128x128 blocks, got {tuple(weight.shape)}"
        )
    expected_scale_shape = tuple(dimension // _FP8_BLOCK_SIZE for dimension in weight.shape)
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"Native FP8 checkpoint scale shape for {weight_name!r} must be {expected_scale_shape}, "
            f"got {tuple(scale_inv.shape)}"
        )
    return weight, scale_inv


def _shard(tensor: torch.Tensor, *, dim: int, size: int, rank: int, name: str) -> torch.Tensor:
    if tensor.shape[dim] % size != 0:
        raise ValueError(f"Cannot evenly shard {name} dimension {tensor.shape[dim]} across {size} ETP ranks")
    shard_size = tensor.shape[dim] // size
    return tensor.narrow(dim, rank * shard_size, shard_size)
