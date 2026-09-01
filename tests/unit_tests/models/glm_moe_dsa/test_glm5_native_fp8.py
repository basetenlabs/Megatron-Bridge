# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.glm_moe_dsa import glm5_bridge as glm5_bridge_module
from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge
from megatron.bridge.models.glm_moe_dsa.native_fp8_import import (
    copy_native_fp8_expert_weight,
    prepare_native_fp8_expert_weight,
)


pytestmark = pytest.mark.unit


def _fp8(shape: tuple[int, int], offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + shape[0] * shape[1], dtype=torch.float32)
    return ((values.remainder(31) - 15) / 8).reshape(shape).to(torch.float8_e4m3fn)


def test_prepare_fc1_preserves_payload_scales_and_etp_order() -> None:
    gate = _fp8((256, 256))
    up = _fp8((256, 256), offset=7)
    gate_scale = torch.tensor([[1.25, 2.5], [3.75, 5.0]], dtype=torch.float32)
    up_scale = torch.tensor([[6.25, 7.5], [8.75, 10.0]], dtype=torch.float32)
    state = {
        "gate.weight": gate,
        "gate.weight_scale_inv": gate_scale,
        "up.weight": up,
        "up.weight_scale_inv": up_scale,
    }

    result = prepare_native_fp8_expert_weight(
        megatron_param="decoder.layers.2.mlp.experts.linear_fc1.weight3",
        hf_param={"gate": "gate.weight", "up": "up.weight"},
        hf_state_dict=state,
        tp_size=2,
        tp_rank=1,
    )

    assert torch.equal(
        result.rowwise_data,
        torch.cat((gate[128:].view(torch.uint8), up[128:].view(torch.uint8))),
    )
    assert torch.equal(
        result.scale_inv,
        torch.cat((gate_scale[1:], up_scale[1:])),
    )


def test_prepare_fc2_shards_payload_and_scales_on_columns() -> None:
    down = _fp8((256, 256))
    scale = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)
    state = {
        "down.weight": down,
        "down.weight_scale_inv": scale,
    }

    result = prepare_native_fp8_expert_weight(
        megatron_param="decoder.layers.2.mlp.experts.linear_fc2.weight3",
        hf_param="down.weight",
        hf_state_dict=state,
        tp_size=2,
        tp_rank=1,
    )

    assert torch.equal(result.rowwise_data, down[:, 128:].view(torch.uint8))
    assert torch.equal(result.scale_inv, scale[:, 1:])


def test_copy_native_weight_preserves_compact_scales_and_zeros_padding() -> None:
    weight = _fp8((256, 256))
    scale = torch.tensor([[1.25, 2.5], [3.75, 5.0]], dtype=torch.float32)
    source = prepare_native_fp8_expert_weight(
        megatron_param="decoder.layers.0.mlp.experts.linear_fc2.weight0",
        hf_param="weight",
        hf_state_dict={"weight": weight, "weight_scale_inv": scale},
        tp_size=1,
        tp_rank=0,
    )
    destination = SimpleNamespace(
        shape=torch.Size((256, 256)),
        _rowwise_data=torch.zeros((256, 256), dtype=torch.uint8),
        _rowwise_scale_inv=torch.full((2, 4), -1.0, dtype=torch.float32),
        _columnwise_data=None,
        _columnwise_scale_inv=None,
    )

    copy_native_fp8_expert_weight(destination, source)

    assert torch.equal(destination._rowwise_data, weight.view(torch.uint8))
    assert torch.equal(destination._rowwise_scale_inv[:, :2], scale)
    assert torch.count_nonzero(destination._rowwise_scale_inv[:, 2:]) == 0


def test_prepare_rejects_incomplete_blocks() -> None:
    weight = _fp8((128, 129))

    with pytest.raises(ValueError, match="complete 128x128 blocks"):
        prepare_native_fp8_expert_weight(
            megatron_param="decoder.layers.0.mlp.experts.linear_fc2.weight0",
            hf_param="weight",
            hf_state_dict={
                "weight": weight,
                "weight_scale_inv": torch.ones((1, 2), dtype=torch.float32),
            },
            tp_size=1,
            tp_rank=0,
        )


def test_glm_bridge_direct_load_bypasses_logical_weight_conversion(
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
    task = SimpleNamespace(
        param_weight=destination,
        param_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
        mapping=SimpleNamespace(
            hf_param="weight",
            tp_size=1,
            tp_rank=0,
        ),
    )
    monkeypatch.setattr(
        glm5_bridge_module,
        "_classify_te_quantized_tensor",
        lambda _tensor: (True, True),
    )

    loaded = GLM5Bridge().maybe_load_native_hf_weight(
        task,
        {"weight": weight, "weight_scale_inv": scale},
    )

    assert loaded is True
    assert torch.equal(destination._rowwise_data, weight.view(torch.uint8))
    assert torch.equal(destination._rowwise_scale_inv[:, :2], scale)


def test_prepare_accepts_vl_language_model_prefixed_names() -> None:
    """GLM-5.3-Flash's VL wrapper prefixes every parameter with language_model."""
    down = _fp8((256, 256))
    scale = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)

    result = prepare_native_fp8_expert_weight(
        megatron_param="language_model.decoder.layers.2.mlp.experts.linear_fc2.weight3",
        hf_param="down.weight",
        hf_state_dict={"down.weight": down, "down.weight_scale_inv": scale},
        tp_size=1,
        tp_rank=0,
    )

    assert torch.equal(result.rowwise_data, down.view(torch.uint8))
