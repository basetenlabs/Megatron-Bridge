# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.glm5_next.glm5_next_bridge import Glm5NextBridge
from megatron.bridge.models.glm5_next.native_fp8_import import (
    prepare_native_fp8_expert_weight,
)

pytestmark = pytest.mark.unit


def _fp8_tensor(rows: int, columns: int, offset: int = 0) -> torch.Tensor:
    values = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    return ((values + offset) % 31 - 15).to(torch.float8_e4m3fn)


def _scale_tensor(rows: int, columns: int, offset: int = 0) -> torch.Tensor:
    values = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    return values / 10 + 1 + offset


def test_prepare_fc1_shards_then_fuses_gate_and_up() -> None:
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
        tp_rank=1,
    )

    assert torch.equal(
        prepared.rowwise_data,
        torch.cat((gate[128:].view(torch.uint8), up[128:].view(torch.uint8)), dim=0),
    )
    assert torch.equal(
        prepared.scale_inv,
        torch.cat((gate_scale[1:], up_scale[1:]), dim=0),
    )


def test_prepare_fc2_shards_payload_and_scale_columns() -> None:
    down_name = "model.language_model.layers.3.mlp.experts.7.down_proj.weight"
    down = _fp8_tensor(128, 256)
    down_scale = _scale_tensor(1, 2)

    prepared = prepare_native_fp8_expert_weight(
        megatron_param="decoder.layers.3.mlp.experts.linear_fc2.weight7",
        hf_param=down_name,
        hf_state_dict={down_name: down, f"{down_name}_scale_inv": down_scale},
        tp_size=2,
        tp_rank=1,
    )

    assert torch.equal(prepared.rowwise_data, down[:, 128:].view(torch.uint8))
    assert torch.equal(prepared.scale_inv, down_scale[:, 1:])


def test_prepare_rejects_non_native_checkpoint_dtypes() -> None:
    down_name = "model.language_model.layers.3.mlp.experts.7.down_proj.weight"

    with pytest.raises(ValueError, match="E4M3"):
        prepare_native_fp8_expert_weight(
            megatron_param="decoder.layers.3.mlp.experts.linear_fc2.weight7",
            hf_param=down_name,
            hf_state_dict={
                down_name: torch.zeros((128, 128), dtype=torch.bfloat16),
                f"{down_name}_scale_inv": torch.ones((1, 1), dtype=torch.float32),
            },
            tp_size=1,
            tp_rank=0,
        )


@pytest.mark.gpu
@pytest.mark.run_only_on("GPU")
def test_glm_bridge_copies_native_bytes_and_scales_without_dequantizing(
    monkeypatch,
) -> None:
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("TE blockwise FP8 requires compute capability 9.0 or newer")
    if torch.version.cuda is None or tuple(
        map(int, torch.version.cuda.split(".")[:2])
    ) < (
        12,
        9,
    ):
        pytest.skip("TE blockwise FP8 requires CUDA 12.9 or newer")

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
    )

    import megatron.bridge.models.glm5_next.glm5_next_bridge as bridge_module

    monkeypatch.setattr(
        bridge_module,
        "maybe_dequantize_fp8_blockwise",
        lambda *args, **kwargs: pytest.fail("native load used the BF16 dequantizer"),
    )
    device = torch.device("cuda")
    down_name = "model.language_model.layers.3.mlp.experts.7.down_proj.weight"
    source = _fp8_tensor(256, 256).to(device)
    source_scale = torch.tensor(
        [[0.13, 0.37], [0.91, 1.73]], dtype=torch.float32, device=device
    )
    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=False,
        block_scaling_dim=2,
    )
    destination = quantizer.make_empty((256, 128), dtype=torch.bfloat16, device=device)
    task = WeightConversionTask(
        param_name="decoder.layers.3.mlp.experts.linear_fc2.weight7",
        global_param_name="decoder.layers.3.mlp.experts.linear_fc2.weight7",
        mapping=SimpleNamespace(hf_param=down_name, tp_size=2, tp_rank=1),
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

    assert bridge.load_weights_hf_to_megatron(hf_pretrained, model) == [model]
    assert torch.equal(destination._rowwise_data, source[:, 128:].view(torch.uint8))
    assert torch.equal(destination._rowwise_scale_inv[:, :1], source_scale[:, 1:])
    assert destination._columnwise_data is None
    assert destination._columnwise_scale_inv is None


def test_glm_native_hook_leaves_bf16_experts_on_the_standard_path() -> None:
    task = WeightConversionTask(
        param_name="decoder.layers.3.mlp.experts.linear_fc2.weight7",
        global_param_name="decoder.layers.3.mlp.experts.linear_fc2.weight7",
        mapping=SimpleNamespace(),
        megatron_module=SimpleNamespace(),
        param_weight=torch.empty((128, 128), dtype=torch.bfloat16),
    )
    assert not Glm5NextBridge().maybe_load_native_hf_weight(task, {})
