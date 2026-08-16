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

"""Coverage for ``HFWeightTuple.iter_finalized``.

Kimi-K3's bridge copies ``vision_tower.*`` / ``mm_projector.*`` through unchanged
via ``HFWeightTuple(name, state[name]).iter_finalized(cpu=cpu)``
(``kimi_k3_bridge.py``). This fork carried that call site without the method it
calls, so state-backed full HF export raised
``AttributeError: 'HFWeightTuple' object has no attribute 'iter_finalized'``.

The first two tests are restored verbatim from upstream
NVIDIA-NeMo/Megatron-Bridge PR #5130's ``tests/unit_tests/models/test_model_bridge.py``.
The remainder pin the export-hook semantics the K3 passthrough depends on.
"""

import torch

from megatron.bridge.models.conversion.model_bridge import HFWeightTuple


def test_hf_weight_tuple_iter_finalized_preserves_two_field_abi():
    tensor = torch.ones(2)
    weight = HFWeightTuple("hf.weight", tensor)

    name, unpacked_tensor = weight

    assert len(weight) == 2
    assert name == "hf.weight"
    assert unpacked_tensor is tensor
    finalized = list(weight.iter_finalized(cpu=False))
    assert finalized[0].param_name == "hf.weight"
    assert finalized[0].weight.data_ptr() == tensor.data_ptr()
    assert finalized[0].weight.requires_grad is False


def test_hf_weight_tuple_iter_finalized_allows_empty_export_hook():
    weight = HFWeightTuple("hf.weight", torch.ones(2))

    assert list(weight.iter_finalized(cpu=False, export_hook=lambda *_args: iter(()))) == []


def test_iter_finalized_is_the_k3_passthrough_contract():
    """The exact call K3 makes for vision/projector tensors must round-trip."""
    tensor = torch.arange(4, dtype=torch.float32)
    finalized = list(HFWeightTuple("vision_tower.blk.0.weight", tensor).iter_finalized(cpu=True))

    assert [w.param_name for w in finalized] == ["vision_tower.blk.0.weight"]
    torch.testing.assert_close(finalized[0].weight, tensor)
    assert finalized[0].weight.device.type == "cpu"


def test_iter_finalized_export_hook_may_fan_out_to_many_weights():
    tensor = torch.ones(2)
    hook = lambda name, t: ((f"{name}.a", t), (f"{name}.b", t * 2))  # noqa: E731
    finalized = list(HFWeightTuple("w", tensor).iter_finalized(cpu=False, export_hook=hook))

    assert [w.param_name for w in finalized] == ["w.a", "w.b"]
    torch.testing.assert_close(finalized[1].weight, tensor * 2)


def test_iter_finalized_clone_identity_output_breaks_shared_storage():
    """Tied weights exported under independent names must not share storage."""
    tensor = torch.ones(2)

    shared = list(HFWeightTuple("w", tensor).iter_finalized(cpu=False))
    assert shared[0].weight.data_ptr() == tensor.data_ptr()

    cloned = list(HFWeightTuple("w", tensor).iter_finalized(cpu=False, clone_identity_output=True))
    assert cloned[0].weight.data_ptr() != tensor.data_ptr()
    torch.testing.assert_close(cloned[0].weight, tensor)


def test_iter_finalized_detaches_from_autograd():
    tensor = torch.ones(2, requires_grad=True)
    finalized = list(HFWeightTuple("w", tensor).iter_finalized(cpu=False))

    assert finalized[0].weight.requires_grad is False
