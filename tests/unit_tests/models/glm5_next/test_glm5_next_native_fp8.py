# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.glm5_next.glm5_next_bridge import Glm5NextBridge
from megatron.bridge.models.glm5_next.native_fp8_import import prepare_native_fp8_expert_weight


pytestmark = pytest.mark.unit


def _fp8_tensor(rows: int, columns: int, offset: int = 0) -> torch.Tensor:
    values = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    return ((values + offset) % 31 - 15).to(torch.float8_e4m3fn)


def _scale_tensor(rows: int, columns: int, offset: int = 0) -> torch.Tensor:
    values = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    return values / 10 + 1 + offset


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_prepare_fc1_shards_gate_and_up_before_concatenating(tp_rank):
    gate_name = "model.language_model.layers.3.mlp.experts.7.gate_proj.weight"
    up_name = "model.language_model.layers.3.mlp.experts.7.up_proj.weight"
    gate = _fp8_tensor(256, 128)
    up = _fp8_tensor(256, 128, offset=7)
    gate_scale = _scale_tensor(2, 1)
    up_scale = _scale_tensor(2, 1, offset=10)
    state = {
        gate_name: gate,
        up_name: up,
        f"{gate_name}_scale_inv": gate_scale,
        f"{up_name}_scale_inv": up_scale,
    }

    prepared = prepare_native_fp8_expert_weight(
        megatron_param="decoder.layers.3.mlp.experts.linear_fc1.weight7",
        hf_param={"gate": gate_name, "up": up_name},
        hf_state_dict=state,
        tp_size=2,
        tp_rank=tp_rank,
    )

    row = slice(tp_rank * 128, (tp_rank + 1) * 128)
    expected_payload = torch.cat((gate[row].view(torch.uint8), up[row].view(torch.uint8)), dim=0)
    expected_scale = torch.cat((gate_scale[tp_rank : tp_rank + 1], up_scale[tp_rank : tp_rank + 1]), dim=0)
    assert torch.equal(prepared.rowwise_data, expected_payload)
    assert torch.equal(prepared.scale_inv, expected_scale)
    assert prepared.logical_shape == (256, 128)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_prepare_fc2_shards_payload_and_scale_columns(tp_rank):
    down_name = "model.language_model.layers.3.mlp.experts.7.down_proj.weight"
    down = _fp8_tensor(128, 256)
    down_scale = _scale_tensor(1, 2)
    state = {down_name: down, f"{down_name}_scale_inv": down_scale}

    prepared = prepare_native_fp8_expert_weight(
        megatron_param="decoder.layers.3.mlp.experts.linear_fc2.weight7",
        hf_param=down_name,
        hf_state_dict=state,
        tp_size=2,
        tp_rank=tp_rank,
    )

    column = slice(tp_rank * 128, (tp_rank + 1) * 128)
    assert torch.equal(prepared.rowwise_data, down[:, column].view(torch.uint8))
    assert torch.equal(prepared.scale_inv, down_scale[:, tp_rank : tp_rank + 1])
    assert prepared.logical_shape == (128, 128)


@pytest.mark.parametrize(
    ("weight_transform", "scale_transform", "error"),
    [
        (lambda tensor: tensor.float(), lambda tensor: tensor, "E4M3"),
        (lambda tensor: tensor, lambda tensor: tensor.bfloat16(), "float32"),
        (lambda tensor: tensor[:127], lambda tensor: tensor, "128x128"),
        (lambda tensor: tensor, lambda tensor: tensor[:, :0], "scale shape"),
    ],
)
def test_prepare_rejects_malformed_checkpoint_tensors(weight_transform, scale_transform, error):
    down_name = "model.language_model.layers.3.mlp.experts.7.down_proj.weight"
    state = {
        down_name: weight_transform(_fp8_tensor(128, 128)),
        f"{down_name}_scale_inv": scale_transform(_scale_tensor(1, 1)),
    }

    with pytest.raises(ValueError, match=error):
        prepare_native_fp8_expert_weight(
            megatron_param="decoder.layers.3.mlp.experts.linear_fc2.weight7",
            hf_param=down_name,
            hf_state_dict=state,
            tp_size=1,
            tp_rank=0,
        )


def test_prepare_rejects_missing_scale_and_unsupported_grouped_layout():
    down_name = "model.language_model.layers.3.mlp.experts.7.down_proj.weight"
    state = {down_name: _fp8_tensor(128, 128)}

    with pytest.raises(KeyError, match="scale_inv"):
        prepare_native_fp8_expert_weight(
            megatron_param="decoder.layers.3.mlp.experts.linear_fc2.weight7",
            hf_param=down_name,
            hf_state_dict=state,
            tp_size=1,
            tp_rank=0,
        )

    state[f"{down_name}_scale_inv"] = _scale_tensor(1, 1)
    with pytest.raises(ValueError, match="numbered grouped-expert"):
        prepare_native_fp8_expert_weight(
            megatron_param="decoder.layers.3.mlp.experts.linear_fc2.weight",
            hf_param=down_name,
            hf_state_dict=state,
            tp_size=1,
            tp_rank=0,
        )


@pytest.mark.gpu
@pytest.mark.run_only_on("GPU")
def test_glm_bridge_loads_native_fp8_bytes_and_scales_without_dequantizing(monkeypatch, record_property):
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("TE blockwise FP8 requires compute capability 9.0 or newer")
    if torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 9):
        pytest.skip("TE blockwise FP8 requires CUDA 12.9 or newer")

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

    import megatron.bridge.models.glm5_next.glm5_next_bridge as bridge_module

    def fail_if_dequantized(*args, **kwargs):
        raise AssertionError("native FP8 load fell back to the BF16 dequantizer")

    monkeypatch.setattr(bridge_module, "maybe_dequantize_fp8_blockwise", fail_if_dequantized)

    device = torch.device("cuda")
    down_name = "model.language_model.layers.3.mlp.experts.7.down_proj.weight"
    source = _fp8_tensor(256, 256).to(device)
    source_scale = torch.tensor([[0.13, 0.37], [0.91, 1.73]], dtype=torch.float32, device=device)
    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=False,
        block_scaling_dim=2,
    )
    destination = quantizer.make_empty((256, 128), dtype=torch.bfloat16, device=device)
    mapping = SimpleNamespace(hf_param=down_name, tp_size=2, tp_rank=1)
    task = WeightConversionTask(
        param_name="decoder.layers.3.mlp.experts.linear_fc2.weight7",
        global_param_name="decoder.layers.3.mlp.experts.linear_fc2.weight7",
        mapping=mapping,
        megatron_module=SimpleNamespace(),
        param_weight=destination,
    )

    bridge = Glm5NextBridge()
    model = torch.nn.Module()
    hf_pretrained = SimpleNamespace(
        model_name_or_path="synthetic-glm5-next",
        state={down_name: source, f"{down_name}_scale_inv": source_scale},
    )
    monkeypatch.setattr(bridge, "build_conversion_tasks", lambda _hf, _models: [task])
    monkeypatch.setattr(bridge, "_broadcast_shared_embeddings", lambda _models: None)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    assert bridge.load_weights_hf_to_megatron(hf_pretrained, model) == [model]
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline

    expected_payload = source[:, 128:].view(torch.uint8)
    expected_scale = source_scale[:, 1:]
    assert torch.equal(destination._rowwise_data, expected_payload)
    assert torch.equal(destination._rowwise_scale_inv[:, :1], expected_scale)
    assert torch.count_nonzero(destination._rowwise_scale_inv[:, 1:]) == 0
    assert destination._columnwise_data is None
    assert destination._columnwise_scale_inv is None

    reference = source[:, 128:].float() * expected_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    assert torch.equal(destination.dequantize(dtype=torch.bfloat16), reference.bfloat16())

    full_bf16_temporary_bytes = source.numel() * torch.bfloat16.itemsize
    assert peak_delta < full_bf16_temporary_bytes
    record_property("peak_native_load_bytes", peak_delta)
    record_property("avoided_full_bf16_temporary_bytes", full_bf16_temporary_bytes)


def test_base_bridge_native_load_hook_falls_through():
    task = WeightConversionTask(
        param_name="weight",
        global_param_name="weight",
        mapping=SimpleNamespace(),
        megatron_module=SimpleNamespace(),
        param_weight=torch.empty(1),
    )
    assert not MegatronModelBridge().maybe_load_native_hf_weight(task, {})


def test_glm_bridge_native_load_hook_falls_through_for_bf16_expert():
    task = WeightConversionTask(
        param_name="decoder.layers.3.mlp.experts.linear_fc2.weight7",
        global_param_name="decoder.layers.3.mlp.experts.linear_fc2.weight7",
        mapping=SimpleNamespace(),
        megatron_module=SimpleNamespace(),
        param_weight=torch.empty((128, 128), dtype=torch.bfloat16),
    )
    assert not Glm5NextBridge().maybe_load_native_hf_weight(task, {})
