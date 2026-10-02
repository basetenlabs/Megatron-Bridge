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

"""Write checkpoint-derived NVFP4 payloads into Transformer Engine parameter storage.

Checkpoint importers convert their own format into a :class:`NativeNVFP4ExpertWeight`;
this module owns the destination side, which depends only on TE's rowwise NVFP4 layout.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeNVFP4ExpertWeight:
    """One ETP-local expert weight in TE rowwise NVFP4 layout."""

    rowwise_data: torch.Tensor
    scale_inv: torch.Tensor
    amax: torch.Tensor


def classify_te_quantized_tensor(tensor: torch.Tensor) -> tuple[bool, bool]:
    """Return whether ``tensor`` is a TE quantized tensor and whether it is NVFP4.

    Transformer Engine is imported lazily so bridges stay importable without it.
    """
    try:
        from transformer_engine.pytorch.tensor import QuantizedTensor
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor
    except (ImportError, ModuleNotFoundError):
        return False, False
    return isinstance(tensor, QuantizedTensor), isinstance(tensor, NVFP4Tensor)


def copy_native_nvfp4_expert_weight(destination: torch.Tensor, source: NativeNVFP4ExpertWeight) -> None:
    """Copy native nibbles, block scales, and the per-tensor amax into a TE tensor."""
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
