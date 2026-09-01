# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.glm5_next import glm5_next_bridge as glm5_next_bridge_module
from megatron.bridge.models.glm5_next.glm5_next_bridge import Glm5NextBridge


pytestmark = pytest.mark.unit


def _fp8(shape: tuple[int, int], offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + shape[0] * shape[1], dtype=torch.float32)
    return ((values.remainder(31) - 15) / 8).reshape(shape).to(torch.float8_e4m3fn)


def _expert_task(destination: object) -> SimpleNamespace:
    # GLM-5.3-Flash ships a vision tower, so the bridge wraps the backbone and
    # every converted parameter carries the ``language_model.`` prefix — the
    # exact name shape the native gate must accept (an anchored bare-decoder
    # match silently fell through to the dequantize/requantize path).
    return SimpleNamespace(
        param_weight=destination,
        param_name="language_model.decoder.layers.0.mlp.experts.linear_fc2.weight0",
        mapping=SimpleNamespace(
            hf_param="weight",
            tp_size=1,
            tp_rank=0,
        ),
    )


def test_glm5_next_bridge_direct_load_bypasses_logical_weight_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = _fp8((256, 256))
    scale = torch.tensor([[1.25, 2.5], [3.75, 5.0]], dtype=torch.float32)
    destination = SimpleNamespace(
        shape=torch.Size((256, 256)),
        _rowwise_data=torch.zeros((256, 256), dtype=torch.uint8),
        _rowwise_scale_inv=torch.zeros((2, 4), dtype=torch.float32),
        _columnwise_data=None,
        _columnwise_scale_inv=None,
    )
    monkeypatch.setattr(
        glm5_next_bridge_module,
        "_classify_te_quantized_tensor",
        lambda _tensor: (True, True),
    )

    loaded = Glm5NextBridge().maybe_load_native_hf_weight(
        _expert_task(destination),
        {"weight": weight, "weight_scale_inv": scale},
    )

    assert loaded is True
    assert torch.equal(destination._rowwise_data, weight.view(torch.uint8))
    assert torch.equal(destination._rowwise_scale_inv[:, :2], scale)


def test_glm5_next_bridge_falls_back_to_dequantizing_load_for_bf16_destination() -> None:
    destination = torch.zeros((256, 256), dtype=torch.bfloat16)

    loaded = Glm5NextBridge().maybe_load_native_hf_weight(
        _expert_task(destination),
        {},
    )

    assert loaded is False


def test_glm5_next_bridge_ignores_non_expert_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        glm5_next_bridge_module,
        "_classify_te_quantized_tensor",
        lambda _tensor: (True, True),
    )
    task = SimpleNamespace(
        param_weight=torch.zeros((4, 4)),
        param_name="language_model.decoder.layers.0.mlp.shared_experts.linear_fc2.weight",
        mapping=SimpleNamespace(hf_param="weight", tp_size=1, tp_rank=0),
    )

    assert Glm5NextBridge().maybe_load_native_hf_weight(task, {}) is False
