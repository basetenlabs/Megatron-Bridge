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

import re
from dataclasses import dataclass
from typing import Mapping, cast

import torch


_FP8_BLOCK_SIZE = 128
# VL wrappers (GLM-5.3-Flash) expose the language backbone on
# ``language_model.``, so every converted parameter carries that prefix;
# text-only checkpoints (GLM-5.2) resolve bare ``decoder.`` names. Both must
# gate into the native path — a bare-``decoder.``-anchored match silently
# routes the VL model through the dequantize-then-requantize fallback while
# the layout verifier stays green.
_ROUTED_EXPERT_WEIGHT = re.compile(
    r"^(?:language_model\.)?decoder\.layers\.\d+\.mlp\.experts\.linear_fc(?P<projection>[12])\.weight(?P<expert>\d+)$"
)
_ROUTED_EXPERT_PREFIX = re.compile(r"^(?:language_model\.)?decoder\.layers\.\d+\.mlp\.experts\.linear_fc[12]\.weight")


@dataclass(frozen=True)
class NativeFP8ExpertWeight:
    """One ETP-local expert weight in TE rowwise block-scaled layout."""

    rowwise_data: torch.Tensor
    scale_inv: torch.Tensor


def is_routed_expert_weight(param_name: str) -> bool:
    """Return whether a parameter belongs to a routed expert projection."""
    return _ROUTED_EXPERT_PREFIX.match(param_name) is not None


def prepare_native_fp8_expert_weight(
    *,
    megatron_param: str,
    hf_param: str | Mapping[str, str],
    hf_state_dict: Mapping[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> NativeFP8ExpertWeight:
    """Prepare one native E4M3 expert shard without dequantization."""
    match = _ROUTED_EXPERT_WEIGHT.fullmatch(megatron_param)
    if match is None:
        raise ValueError(f"Native FP8 import does not support parameter {megatron_param!r}")

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
) -> NativeFP8ExpertWeight:
    gate_data, gate_scale = _load_blockwise_weight(hf_params["gate"], hf_state_dict)
    up_data, up_scale = _load_blockwise_weight(hf_params["up"], hf_state_dict)
    if gate_data.shape != up_data.shape or gate_scale.shape != up_scale.shape:
        raise ValueError("Native FP8 FC1 gate and up shapes must match")

    gate_data = _shard(gate_data, dim=0, size=tp_size, rank=tp_rank, name="FC1 payload")
    up_data = _shard(up_data, dim=0, size=tp_size, rank=tp_rank, name="FC1 payload")
    gate_scale = _shard(gate_scale, dim=0, size=tp_size, rank=tp_rank, name="FC1 scale grid")
    up_scale = _shard(up_scale, dim=0, size=tp_size, rank=tp_rank, name="FC1 scale grid")

    return NativeFP8ExpertWeight(
        rowwise_data=torch.cat((gate_data.view(torch.uint8), up_data.view(torch.uint8)), dim=0),
        scale_inv=torch.cat((gate_scale, up_scale), dim=0),
    )


def _prepare_fc2(
    *,
    hf_param: str,
    hf_state_dict: Mapping[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> NativeFP8ExpertWeight:
    down_data, down_scale = _load_blockwise_weight(hf_param, hf_state_dict)
    down_data = _shard(down_data, dim=1, size=tp_size, rank=tp_rank, name="FC2 payload")
    down_scale = _shard(down_scale, dim=1, size=tp_size, rank=tp_rank, name="FC2 scale grid")

    return NativeFP8ExpertWeight(
        rowwise_data=down_data.view(torch.uint8),
        scale_inv=down_scale,
    )


def copy_native_fp8_expert_weight(destination: torch.Tensor, source: NativeFP8ExpertWeight) -> None:
    """Copy native bytes and compact scales into a TE blockwise tensor."""
    rowwise_data = destination._rowwise_data
    rowwise_scale_inv = destination._rowwise_scale_inv
    if rowwise_data is None or rowwise_data.dtype is not torch.uint8:
        raise ValueError("Native FP8 destination requires uint8 rowwise payload")
    if rowwise_scale_inv is None or rowwise_scale_inv.dtype is not torch.float32:
        raise ValueError("Native FP8 destination requires FP32 rowwise scales")
    if destination._columnwise_data is not None or destination._columnwise_scale_inv is not None:
        raise ValueError("Native FP8 import requires rowwise-only storage")

    with torch.no_grad():
        rowwise_data.copy_(source.rowwise_data.to(device=rowwise_data.device))
        rowwise_scale_inv.zero_()
        rowwise_scale_inv[:, : source.scale_inv.shape[1]].copy_(source.scale_inv.to(device=rowwise_scale_inv.device))


def _load_blockwise_weight(
    weight_name: str, hf_state_dict: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_name = f"{weight_name}_scale_inv"
    try:
        weight = hf_state_dict[weight_name]
        scale_inv = hf_state_dict[scale_name]
    except KeyError as error:
        raise KeyError(f"Native FP8 checkpoint is missing tensor {error.args[0]!r}") from None

    if weight.dtype is not torch.float8_e4m3fn:
        raise ValueError(f"Native FP8 weight {weight_name!r} must use E4M3, got {weight.dtype}")
    if scale_inv.dtype is not torch.float32:
        raise ValueError(f"Native FP8 scale {scale_name!r} must use FP32, got {scale_inv.dtype}")
    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError("Native FP8 weights and scales must be two-dimensional")
    if any(dimension % _FP8_BLOCK_SIZE != 0 for dimension in weight.shape):
        raise ValueError(
            f"Native FP8 weight {weight_name!r} requires complete 128x128 blocks, got {tuple(weight.shape)}"
        )
    expected_scale_shape = tuple(dimension // _FP8_BLOCK_SIZE for dimension in weight.shape)
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"Native FP8 scale for {weight_name!r} must be {expected_scale_shape}, got {tuple(scale_inv.shape)}"
        )
    return weight, scale_inv


def _shard(tensor: torch.Tensor, *, dim: int, size: int, rank: int, name: str) -> torch.Tensor:
    if tensor.shape[dim] % size != 0:
        raise ValueError(f"Cannot shard {name} dimension {tensor.shape[dim]} across {size} ranks")
    shard_size = tensor.shape[dim] // size
    return tensor.narrow(dim, rank * shard_size, shard_size)
