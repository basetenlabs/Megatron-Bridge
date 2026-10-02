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

"""Load ModelOpt NVFP4 routed experts straight into TE NVFP4 parameter storage.

ModelOpt and Transformer Engine share the NVFP4 encoding: E2M1 elements packed two
per byte and one E4M3 scale per 16 elements along a row. They differ only in how the
per-tensor level is stored. ModelOpt keeps it as ``weight_scale_2`` and reconstructs
``element * weight_scale * weight_scale_2``, while TE keeps the tensor amax and
reconstructs ``element * block_scale * amax / (E2M1_MAX * E4M3_MAX)``. The payload
and block scales therefore copy across unchanged, and
``amax = weight_scale_2 * E2M1_MAX * E4M3_MAX`` reproduces the checkpoint values up
to FP32 rounding of that one per-tensor factor.

Nemotron-H routed experts are not gated: FC1 is the up projection alone.
"""

import re
from typing import Mapping

import torch

from megatron.bridge.models.conversion.native_nvfp4 import NativeNVFP4ExpertWeight
from megatron.bridge.models.conversion.quantization_utils import (
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    NVFP4_BLOCK_SIZE,
)


# Matches the routed-expert glob the trainer's NVFP4 storage recipe uses, so every
# parameter that recipe quantizes is imported natively, including MTP experts.
_ROUTED_EXPERT_WEIGHT = re.compile(r"(?:^|\.)mlp\.experts\.linear_fc(?P<projection>[12])\.weight(?P<expert>\d+)$")


def is_routed_expert_weight(param_name: str) -> bool:
    """Return whether a parameter belongs to a routed expert projection."""
    return _ROUTED_EXPERT_WEIGHT.search(param_name) is not None


def prepare_native_nvfp4_expert_weight(
    *,
    megatron_param: str,
    hf_param: str | Mapping[str, str],
    hf_state_dict: Mapping[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> NativeNVFP4ExpertWeight:
    """Prepare one ETP-local expert shard from ModelOpt NVFP4 tensors, without dequantizing."""
    match = _ROUTED_EXPERT_WEIGHT.search(megatron_param)
    if match is None:
        raise ValueError(f"Native NVFP4 import does not support parameter {megatron_param!r}")
    if not isinstance(hf_param, str):
        raise ValueError(
            f"Native NVFP4 import supports non-gated experts only; {megatron_param!r} maps to {dict(hf_param)!r}"
        )

    payload, block_scale, global_scale = _load_modelopt_nvfp4_weight(hf_param, hf_state_dict)
    # FC1 is column parallel and FC2 row parallel. Requiring the scale grid to split
    # evenly too keeps an FC2 column shard on whole 16-element blocks.
    dim = 0 if match.group("projection") == "1" else 1
    payload = _shard(payload, dim=dim, size=tp_size, rank=tp_rank, name=f"{hf_param} payload")
    block_scale = _shard(block_scale, dim=dim, size=tp_size, rank=tp_rank, name=f"{hf_param} block scales")

    amax = (global_scale.to(torch.float64) * (FP4_E2M1_MAX * FP8_E4M3_MAX)).to(torch.float32).reshape(1)
    return NativeNVFP4ExpertWeight(rowwise_data=payload, scale_inv=block_scale.view(torch.uint8), amax=amax)


def _load_modelopt_nvfp4_weight(
    weight_name: str, hf_state_dict: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the packed payload, E4M3 block scales, and per-tensor scale of one weight."""
    try:
        payload = hf_state_dict[weight_name]
        block_scale = hf_state_dict[f"{weight_name}_scale"]
        global_scale = hf_state_dict[f"{weight_name}_scale_2"]
    except KeyError as error:
        raise KeyError(f"Native NVFP4 checkpoint is missing tensor {error.args[0]!r}") from None

    if payload.dtype is not torch.uint8 or payload.ndim != 2:
        raise ValueError(
            f"NVFP4 payload {weight_name!r} must be a 2-D uint8 tensor, got {payload.dtype} {tuple(payload.shape)}"
        )
    if block_scale.dtype is not torch.float8_e4m3fn:
        raise ValueError(f"NVFP4 block scales for {weight_name!r} must be float8_e4m3fn, got {block_scale.dtype}")
    columns = payload.shape[1] * 2
    expected_scale_shape = (payload.shape[0], columns // NVFP4_BLOCK_SIZE)
    if columns % NVFP4_BLOCK_SIZE != 0 or tuple(block_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"NVFP4 block scales for {weight_name!r} must be {expected_scale_shape}, got {tuple(block_scale.shape)}"
        )
    if global_scale.numel() != 1:
        raise ValueError(
            f"NVFP4 global scale for {weight_name!r} must hold one value, got {tuple(global_scale.shape)}"
        )
    # TE derives the per-tensor decode factor from amax; it must be a positive number.
    if not bool(torch.isfinite(global_scale).all()) or float(global_scale) <= 0:
        raise ValueError(
            f"NVFP4 global scale for {weight_name!r} must be positive and finite, got {float(global_scale)}"
        )
    return payload, block_scale, global_scale


def _shard(tensor: torch.Tensor, *, dim: int, size: int, rank: int, name: str) -> torch.Tensor:
    if tensor.shape[dim] % size != 0:
        raise ValueError(f"Cannot shard {name} dimension {tensor.shape[dim]} across {size} ranks")
    shard_size = tensor.shape[dim] // size
    return tensor.narrow(dim, rank * shard_size, shard_size)
