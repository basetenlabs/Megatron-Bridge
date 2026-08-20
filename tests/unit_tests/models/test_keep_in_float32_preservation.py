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

"""FP32 preservation across the mixed-precision wrapper.

Kimi-K3 marks numerically sensitive KDA state to stay FP32 via
``_keep_in_float32_parameter_names``: the short-convolution ``weight``
(``kimi_k3_layers.py:74``) and ``A_log`` / ``dt_bias`` (``kimi_k3_layers.py:205``).

This fork previously restored *only* ``expert_bias`` after the recursive
bfloat16 cast, so those three were silently truncated to BF16. Freezing a
parameter for LoRA prevents optimization, not dtype conversion, so every LoRA
forward ran against different numerics than intended -- with no crash and a
loss curve that still went down.
"""

import pytest
import torch
from torch import nn

from megatron.bridge.models.model_provider import _apply_mixed_precision_wrapper


class _FakeFloat16Module(nn.Module):
    """Stands in for MCore's Float16Module: recursively casts to bf16."""

    def __init__(self, config, module: nn.Module):
        super().__init__()
        self.config = config
        self.module = module.bfloat16()


class _KdaLike(nn.Module):
    """Mirrors KimiK3KimiDeltaAttention's FP32 marking."""

    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.other = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self._keep_in_float32_parameter_names = ("A_log", "dt_bias")


class _ExpertBiasLike(nn.Module):
    """Mirrors the pre-existing MCore expert-bias contract."""

    def __init__(self):
        super().__init__()
        self.expert_bias = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self._maintain_float32_expert_bias = True


class _Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.kda = _KdaLike()
        self.router = _ExpertBiasLike()


def test_marked_kda_parameters_survive_the_bf16_cast():
    root = _Root()
    wrapped = _apply_mixed_precision_wrapper([root], object(), _FakeFloat16Module)

    assert len(wrapped) == 1
    # The marked KDA state must still be FP32 -- this is the regression.
    assert root.kda.A_log.dtype is torch.float32
    assert root.kda.dt_bias.dtype is torch.float32
    # The pre-existing expert-bias contract must keep working.
    assert root.router.expert_bias.dtype is torch.float32
    # Unmarked parameters are expected to be cast, proving the wrapper ran.
    assert root.kda.other.dtype is torch.bfloat16


def test_conv_weight_marking_shape_is_supported():
    """K3 marks a plain ``weight`` on its short-convolution module."""

    class _ShortConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
            self._keep_in_float32_parameter_names = ("weight",)

    conv = _ShortConv()
    _apply_mixed_precision_wrapper([conv], object(), _FakeFloat16Module)
    assert conv.weight.dtype is torch.float32


def test_non_sequence_marking_is_rejected():
    class _Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(2))
            self._keep_in_float32_parameter_names = "p"  # str, not list/tuple

    with pytest.raises(TypeError, match="must be a list or tuple"):
        _apply_mixed_precision_wrapper([_Bad()], object(), _FakeFloat16Module)


def test_marking_a_non_parameter_is_rejected():
    class _Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.zeros(2)  # plain tensor, not a Parameter
            self._keep_in_float32_parameter_names = ("p",)

    with pytest.raises(TypeError, match="must be a Parameter to remain in FP32"):
        _apply_mixed_precision_wrapper([_Bad()], object(), _FakeFloat16Module)
