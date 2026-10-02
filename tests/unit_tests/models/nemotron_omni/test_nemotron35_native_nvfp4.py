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

"""Tests for loading Nemotron 3.5 Super VL ModelOpt checkpoints.

Two loading paths must agree on the value of every routed-expert weight:
dequantizing to BF16, and importing the NVFP4 payload natively into TE storage.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.conversion.native_nvfp4 import copy_native_nvfp4_expert_weight
from megatron.bridge.models.conversion.quantization_utils import (
    dequantize_fp8_per_tensor,
    dequantize_nvfp4_e2m1_packed,
    maybe_dequantize_modelopt_weight,
)
from megatron.bridge.models.nemotron_omni import nemotron_omni_bridge
from megatron.bridge.models.nemotron_omni.native_nvfp4_import import (
    is_routed_expert_weight,
    prepare_native_nvfp4_expert_weight,
)
from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import Nemotron35SuperVLBridge


pytestmark = pytest.mark.unit

_E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_EXPERT = "language_model.backbone.layers.1.mixer.experts.3"


def _modelopt_nvfp4(weight: torch.Tensor):
    """Quantize like ModelOpt: per-tensor ``weight_scale_2``, E4M3 scales per 16 elements."""
    rows, columns = weight.shape
    weight_scale_2 = weight.abs().amax().float() / (6.0 * 448.0)
    blocks = weight.float().reshape(rows, columns // 16, 16)
    weight_scale = (blocks.abs().amax(dim=-1) / 6.0 / weight_scale_2).to(torch.float8_e4m3fn)
    scaled = blocks / (weight_scale.float() * weight_scale_2).unsqueeze(-1)
    magnitude = (scaled.abs().unsqueeze(-1) - _E2M1_MAGNITUDES).abs().argmin(dim=-1)
    codes = (magnitude | ((scaled < 0).long() << 3)).to(torch.uint8).reshape(rows, columns)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return payload, weight_scale, weight_scale_2


def _nvfp4_state_dict(name: str, rows: int, columns: int, *, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    payload, weight_scale, weight_scale_2 = _modelopt_nvfp4(torch.randn(rows, columns, generator=generator))
    return {name: payload, f"{name}_scale": weight_scale, f"{name}_scale_2": weight_scale_2}


def _te_reconstruction(prepared) -> torch.Tensor:
    """Dequantize a prepared weight the way TE does: element * block_scale * amax / (6 * 448)."""
    rows = prepared.rowwise_data.shape[0]
    blocks = prepared.scale_inv.shape[1]
    block_scale = prepared.scale_inv.view(torch.float8_e4m3fn)
    global_scale = prepared.amax.reshape(()) / (6.0 * 448.0)
    unit = dequantize_nvfp4_e2m1_packed(
        prepared.rowwise_data,
        block_scale=torch.ones(rows, blocks).to(torch.float8_e4m3fn),
        global_scale=torch.tensor(1.0),
        dtype=torch.float32,
    )
    return (unit.reshape(rows, blocks, 16) * (block_scale.float() * global_scale).unsqueeze(-1)).reshape(rows, -1)


class _FakeNVFP4Destination:
    """Stands in for a TE NVFP4Tensor, which cannot be built without a GPU."""

    def __init__(self, rows: int, columns: int):
        self._rowwise_data = torch.zeros((rows, columns // 2), dtype=torch.uint8)
        self._rowwise_scale_inv = torch.full((rows, columns // 16 + 4), 255, dtype=torch.uint8)
        self._amax_rowwise = torch.zeros(1, dtype=torch.float32)
        self._columnwise_data = None
        self._columnwise_scale_inv = None
        self._with_gemm_swizzled_scales = False


def _task(param_name: str, hf_param, *, destination, tp_size: int = 1, tp_rank: int = 0):
    mapping = SimpleNamespace(hf_param=hf_param, tp_size=tp_size, tp_rank=tp_rank)
    return SimpleNamespace(param_name=param_name, param_weight=destination, mapping=mapping)


def test_dequantize_nvfp4_reads_the_even_element_from_the_low_nibble():
    # 0x21 holds +0.5 (low nibble, even element) then +1.0; 0xF9 holds -0.5 then -6.0.
    payload = torch.tensor([[0x21, 0xF9] * 4], dtype=torch.uint8)
    block_scale = torch.tensor([[2.0]]).to(torch.float8_e4m3fn)

    actual = dequantize_nvfp4_e2m1_packed(
        payload, block_scale=block_scale, global_scale=torch.tensor(0.25), dtype=torch.float32
    )

    expected = torch.tensor([[0.5, 1.0, -0.5, -6.0] * 4]) * 0.5
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dequantize_nvfp4_applies_one_scale_per_sixteen_elements():
    payload = torch.full((2, 16), 0x22, dtype=torch.uint8)  # every element is 1.0
    block_scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]]).to(torch.float8_e4m3fn)

    actual = dequantize_nvfp4_e2m1_packed(
        payload, block_scale=block_scale, global_scale=torch.tensor([0.5]), dtype=torch.float32
    )

    expected = torch.tensor([[0.5] * 16 + [1.0] * 16, [2.0] * 16 + [4.0] * 16])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dequantize_nvfp4_round_trips_within_fp4_resolution():
    weight = torch.randn(64, 128, generator=torch.Generator().manual_seed(1))

    payload, weight_scale, weight_scale_2 = _modelopt_nvfp4(weight)

    restored = dequantize_nvfp4_e2m1_packed(
        payload, block_scale=weight_scale, global_scale=weight_scale_2, dtype=torch.float32
    )

    # E2M1 spacing is at most a third of a block's maximum, so half of it bounds the error.
    block_max = weight.abs().reshape(64, 8, 16).amax(dim=-1, keepdim=True)
    error = (restored - weight).abs().reshape(64, 8, 16)
    assert bool((error <= block_max / 6 + 1e-6).all())


@pytest.mark.parametrize(
    ("block_scale", "match"),
    [
        (torch.ones(4, 2).to(torch.float8_e4m3fn), r"must be \(4, 4\)"),
        (torch.ones(4, 4, dtype=torch.uint8), "float8_e4m3fn"),
    ],
)
def test_dequantize_nvfp4_rejects_a_mismatched_scale_grid(block_scale, match):
    with pytest.raises(ValueError, match=match):
        dequantize_nvfp4_e2m1_packed(
            torch.zeros(4, 32, dtype=torch.uint8), block_scale=block_scale, global_scale=torch.tensor(1.0)
        )


def test_modelopt_loader_dequantizes_nvfp4_weights():
    name = f"{_EXPERT}.up_proj.weight"
    state_dict = _nvfp4_state_dict(name, 32, 64)

    actual = maybe_dequantize_modelopt_weight(name, state_dict)

    expected = dequantize_nvfp4_e2m1_packed(
        state_dict[name], block_scale=state_dict[f"{name}_scale"], global_scale=state_dict[f"{name}_scale_2"]
    )
    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_modelopt_loader_dequantizes_per_tensor_fp8_weights():
    name = "language_model.backbone.layers.0.mixer.in_proj.weight"
    weight = torch.randn(8, 16, generator=torch.Generator().manual_seed(2)).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.0123, dtype=torch.float32)

    actual = maybe_dequantize_modelopt_weight(name, {name: weight, f"{name}_scale": scale})

    torch.testing.assert_close(actual, (weight.float() * scale).to(torch.bfloat16), rtol=0, atol=0)


def test_modelopt_loader_passes_unquantized_tensors_through():
    name = "language_model.backbone.layers.1.mixer.fc1_latent_proj.weight"
    weight = torch.randn(4, 8, dtype=torch.bfloat16)

    assert maybe_dequantize_modelopt_weight(name, {name: weight}) is weight


def test_modelopt_loader_refuses_a_quantized_weight_without_its_scales():
    name = f"{_EXPERT}.down_proj.weight"
    state_dict = _nvfp4_state_dict(name, 16, 32)
    del state_dict[f"{name}_scale_2"]

    with pytest.raises(KeyError, match=r"down_proj\.weight_scale_2"):
        maybe_dequantize_modelopt_weight(name, state_dict)


def test_per_tensor_fp8_dequant_rejects_a_per_channel_scale():
    with pytest.raises(ValueError, match="one value"):
        dequantize_fp8_per_tensor(torch.zeros(4, 4).to(torch.float8_e4m3fn), torch.ones(4))


@pytest.mark.parametrize(
    "param_name",
    [
        "language_model.decoder.layers.3.mlp.experts.linear_fc1.weight7",
        "decoder.layers.0.mlp.experts.linear_fc2.weight0",
        "language_model.mtp.layers.0.mtp_model_layer.layers.0.mlp.experts.linear_fc1.weight5",
    ],
)
def test_routed_expert_weights_match_with_or_without_a_wrapper_prefix(param_name):
    assert is_routed_expert_weight(param_name)


@pytest.mark.parametrize(
    "param_name",
    [
        "language_model.decoder.layers.3.mlp.shared_experts.linear_fc1.weight",
        "language_model.decoder.layers.3.mlp.experts.linear_fc1.weight",
        "language_model.decoder.layers.3.mlp.fc1_latent_proj.weight",
        "language_model.decoder.layers.3.mixer.in_proj.weight",
    ],
)
def test_other_weights_are_not_routed_experts(param_name):
    assert not is_routed_expert_weight(param_name)


def test_native_fc1_keeps_its_row_shard_and_reproduces_the_checkpoint_values():
    name = f"{_EXPERT}.up_proj.weight"
    state_dict = _nvfp4_state_dict(name, 64, 128, seed=3)

    prepared = prepare_native_nvfp4_expert_weight(
        megatron_param="language_model.decoder.layers.1.mlp.experts.linear_fc1.weight3",
        hf_param=name,
        hf_state_dict=state_dict,
        tp_size=2,
        tp_rank=1,
    )

    torch.testing.assert_close(prepared.rowwise_data, state_dict[name][32:], rtol=0, atol=0)
    torch.testing.assert_close(prepared.scale_inv, state_dict[f"{name}_scale"][32:].view(torch.uint8), rtol=0, atol=0)
    expected = maybe_dequantize_modelopt_weight(name, state_dict, dtype=torch.float32)[32:]
    torch.testing.assert_close(_te_reconstruction(prepared), expected, rtol=1e-6, atol=0)


def test_native_fc2_shards_along_the_input_dimension():
    name = f"{_EXPERT}.down_proj.weight"
    state_dict = _nvfp4_state_dict(name, 32, 128, seed=4)

    prepared = prepare_native_nvfp4_expert_weight(
        megatron_param="language_model.decoder.layers.1.mlp.experts.linear_fc2.weight3",
        hf_param=name,
        hf_state_dict=state_dict,
        tp_size=2,
        tp_rank=0,
    )

    torch.testing.assert_close(prepared.rowwise_data, state_dict[name][:, :32], rtol=0, atol=0)
    expected = maybe_dequantize_modelopt_weight(name, state_dict, dtype=torch.float32)[:, :64]
    torch.testing.assert_close(_te_reconstruction(prepared), expected, rtol=1e-6, atol=0)


def test_native_amax_encodes_the_modelopt_global_scale():
    name = f"{_EXPERT}.up_proj.weight"
    state_dict = _nvfp4_state_dict(name, 16, 32, seed=5)

    prepared = prepare_native_nvfp4_expert_weight(
        megatron_param="language_model.decoder.layers.1.mlp.experts.linear_fc1.weight3",
        hf_param=name,
        hf_state_dict=state_dict,
        tp_size=1,
        tp_rank=0,
    )

    assert prepared.amax.shape == (1,) and prepared.amax.dtype is torch.float32
    torch.testing.assert_close(prepared.amax / (6.0 * 448.0), state_dict[f"{name}_scale_2"].reshape(1))


def test_native_fc2_refuses_a_shard_that_splits_a_scale_block():
    name = f"{_EXPERT}.down_proj.weight"
    state_dict = _nvfp4_state_dict(name, 16, 32, seed=6)

    with pytest.raises(ValueError, match="block scales dimension 2 across 4 ranks"):
        prepare_native_nvfp4_expert_weight(
            megatron_param="language_model.decoder.layers.1.mlp.experts.linear_fc2.weight3",
            hf_param=name,
            hf_state_dict=state_dict,
            tp_size=4,
            tp_rank=0,
        )


def test_native_import_refuses_a_gated_expert():
    with pytest.raises(ValueError, match="non-gated experts only"):
        prepare_native_nvfp4_expert_weight(
            megatron_param="language_model.decoder.layers.1.mlp.experts.linear_fc1.weight3",
            hf_param={"gate": "gate.weight", "up": "up.weight"},
            hf_state_dict={},
            tp_size=1,
            tp_rank=0,
        )


@pytest.mark.parametrize("global_scale", [0.0, float("nan")])
def test_native_import_refuses_an_unusable_global_scale(global_scale):
    name = f"{_EXPERT}.up_proj.weight"
    state_dict = _nvfp4_state_dict(name, 16, 32, seed=7)
    state_dict[f"{name}_scale_2"] = torch.tensor(global_scale)

    with pytest.raises(ValueError, match="positive and finite"):
        prepare_native_nvfp4_expert_weight(
            megatron_param="language_model.decoder.layers.1.mlp.experts.linear_fc1.weight3",
            hf_param=name,
            hf_state_dict=state_dict,
            tp_size=1,
            tp_rank=0,
        )


def test_native_import_refuses_a_bf16_checkpoint():
    name = f"{_EXPERT}.up_proj.weight"
    state_dict = {
        name: torch.zeros(16, 32, dtype=torch.bfloat16),
        f"{name}_scale": torch.ones(16, 2).to(torch.float8_e4m3fn),
        f"{name}_scale_2": torch.tensor(1.0),
    }

    with pytest.raises(ValueError, match="must be a 2-D uint8 tensor"):
        prepare_native_nvfp4_expert_weight(
            megatron_param="language_model.decoder.layers.1.mlp.experts.linear_fc1.weight3",
            hf_param=name,
            hf_state_dict=state_dict,
            tp_size=1,
            tp_rank=0,
        )


def test_bridge_dequantizes_every_projection_of_a_mapping():
    names = {role: f"language_model.backbone.layers.7.mixer.{role}_proj.weight" for role in ("q", "k", "v")}
    state_dict = {}
    for index, name in enumerate(names.values()):
        state_dict[name] = torch.full((4, 8), 1.0 + index).to(torch.float8_e4m3fn)
        state_dict[f"{name}_scale"] = torch.tensor(0.5)

    loaded = Nemotron35SuperVLBridge().maybe_modify_loaded_hf_weight(names, state_dict)

    assert set(loaded) == {"q", "k", "v"}
    torch.testing.assert_close(loaded["v"], torch.full((4, 8), 1.5, dtype=torch.bfloat16), rtol=0, atol=0)


def test_bridge_loads_routed_experts_natively_into_nvfp4_storage(monkeypatch):
    monkeypatch.setattr(nemotron_omni_bridge, "classify_te_quantized_tensor", lambda tensor: (True, True))
    name = f"{_EXPERT}.up_proj.weight"
    state_dict = _nvfp4_state_dict(name, 32, 64, seed=8)
    destination = _FakeNVFP4Destination(32, 64)
    task = _task("language_model.decoder.layers.1.mlp.experts.linear_fc1.weight3", name, destination=destination)

    assert Nemotron35SuperVLBridge().maybe_load_native_hf_weight(task, state_dict)

    torch.testing.assert_close(destination._rowwise_data, state_dict[name], rtol=0, atol=0)
    scale_bits = state_dict[f"{name}_scale"].view(torch.uint8)
    torch.testing.assert_close(destination._rowwise_scale_inv[:, :4], scale_bits, rtol=0, atol=0)
    assert bool((destination._rowwise_scale_inv[:, 4:] == 0).all())


def test_bridge_leaves_bf16_and_non_expert_destinations_to_the_normal_path(monkeypatch):
    bridge = Nemotron35SuperVLBridge()
    expert = "language_model.decoder.layers.1.mlp.experts.linear_fc1.weight3"
    assert not bridge.maybe_load_native_hf_weight(_task(expert, "w", destination=torch.zeros(2, 2)), {})

    monkeypatch.setattr(nemotron_omni_bridge, "classify_te_quantized_tensor", lambda tensor: (True, True))
    shared = "language_model.decoder.layers.1.mlp.shared_experts.linear_fc1.weight"
    assert not bridge.maybe_load_native_hf_weight(_task(shared, "w", destination=object()), {})


def test_bridge_refuses_a_routed_expert_quantized_to_another_format(monkeypatch):
    monkeypatch.setattr(nemotron_omni_bridge, "classify_te_quantized_tensor", lambda tensor: (True, False))
    task = _task("language_model.decoder.layers.1.mlp.experts.linear_fc2.weight0", "w", destination=object())

    with pytest.raises(ValueError, match="requires TE NVFP4Tensor parameters"):
        Nemotron35SuperVLBridge().maybe_load_native_hf_weight(task, {})


def test_bridge_declares_the_modelopt_scale_tensors_it_reads():
    name = f"{_EXPERT}.up_proj.weight"
    available = {name, f"{name}_scale", f"{name}_scale_2", f"{name[: -len('weight')]}input_scale"}

    assert Nemotron35SuperVLBridge.get_hf_import_param_names(name, available) == (
        name,
        f"{name}_scale",
        f"{name}_scale_2",
    )
    assert Nemotron35SuperVLBridge.get_hf_import_param_names(name, {name}) == (name,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_imported_weight_dequantizes_identically_in_transformer_engine():
    """TE must read the copied nibbles, block scales and amax as ModelOpt wrote them."""
    nvfp4_tensor = pytest.importorskip("transformer_engine.pytorch.tensor.nvfp4_tensor")
    name = f"{_EXPERT}.down_proj.weight"
    rows, columns = 128, 256
    state_dict = _nvfp4_state_dict(name, rows, columns, seed=9)
    quantizer = nvfp4_tensor.NVFP4Quantizer(rowwise=True, columnwise=False, with_2d_quantization=False)
    destination = quantizer(torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16))
    prepared = prepare_native_nvfp4_expert_weight(
        megatron_param="language_model.decoder.layers.1.mlp.experts.linear_fc2.weight3",
        hf_param=name,
        hf_state_dict=state_dict,
        tp_size=1,
        tp_rank=0,
    )

    copy_native_nvfp4_expert_weight(destination, prepared)

    expected = maybe_dequantize_modelopt_weight(name, state_dict, dtype=torch.float32).cuda()
    torch.testing.assert_close(destination.dequantize(dtype=torch.float32), expected, rtol=1e-6, atol=0)
