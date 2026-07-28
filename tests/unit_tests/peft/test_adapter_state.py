"""Unit tests for the PEFT adapter-state snapshot layer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.peft.adapter_state import (
    DistributedAdapterState,
    MegatronMixedPrecisionAdapterState,
    MixedPrecisionAdapterState,
    flatten,
    select_adapter_state,
    unflatten_into,
)


def _fake_optimizer(
    model_params,
    master_params,
    *,
    state_keys=("exp_avg", "exp_avg_sq"),
    step=0,
):
    state = {}
    for master_param in master_params:
        opt_state: dict = {key: torch.zeros_like(master_param) for key in state_keys}
        opt_state["step"] = step
        state[master_param] = opt_state
    inner = SimpleNamespace(
        state=state,
        param_groups=[{"lr": 1e-4, "step": step, "params": list(master_params)}],
    )
    return SimpleNamespace(
        float16_groups=[list(model_params)],
        fp32_from_float16_groups=[list(master_params)],
        optimizer=inner,
    )


def test_flatten_then_unflatten_roundtrips():
    src = [torch.randn(2, 3), torch.randn(4), torch.randn(1, 5)]
    flat = flatten(src)
    assert flat.numel() == 6 + 4 + 5

    dst = [torch.zeros_like(tensor) for tensor in src]
    unflatten_into(flat, dst)

    for expected, actual in zip(src, dst):
        assert torch.equal(expected, actual)


def test_clone_is_a_deep_copy():
    generator = torch.Generator().manual_seed(1)
    state = MegatronMixedPrecisionAdapterState(
        lora=torch.randn(8, generator=generator),
        opt_state={
            "exp_avg": torch.randn(8, generator=generator),
            "exp_avg_sq": torch.randn(8, generator=generator).abs(),
        },
        step=42,
        version=5,
    )

    clone = state.clone()

    assert clone.step == 42
    assert clone.version == 5
    assert torch.equal(clone.lora, state.lora)
    assert clone.lora is not state.lora
    clone.lora.add_(1.0)
    assert not torch.equal(clone.lora, state.lora)
    assert set(clone.opt_state) == {"exp_avg", "exp_avg_sq"}
    assert clone.opt_state["exp_avg"] is not state.opt_state["exp_avg"]
    clone.opt_state["exp_avg"].add_(1.0)
    assert not torch.equal(clone.opt_state["exp_avg"], state.opt_state["exp_avg"])


def test_nbytes_sums_master_and_opt_state():
    state = MegatronMixedPrecisionAdapterState(
        lora=torch.zeros(8, dtype=torch.float32),
        opt_state={
            "exp_avg": torch.zeros(8, dtype=torch.float32),
            "exp_avg_sq": torch.zeros(4, dtype=torch.float32),
        },
        step=0,
        version=0,
    )

    assert state.nbytes == 32 + 32 + 16


def test_nbytes_is_generic_across_opt_state_keys():
    state = MegatronMixedPrecisionAdapterState(
        lora=torch.zeros(8, dtype=torch.float32),
        opt_state={"momentum_buffer": torch.zeros(8, dtype=torch.float32)},
        step=0,
        version=0,
    )

    assert state.nbytes == 32 + 32


def test_capture_then_restore_roundtrips_all_state():
    torch.manual_seed(0)
    model = [torch.randn(4, dtype=torch.bfloat16), torch.randn(6, dtype=torch.bfloat16)]
    master = [torch.randn(4), torch.randn(6)]
    optimizer = _fake_optimizer(model, master, step=3)
    for master_param in master:
        optimizer.optimizer.state[master_param]["exp_avg"].fill_(0.5)
        optimizer.optimizer.state[master_param]["exp_avg_sq"].fill_(0.25)
    orig_master = [master_param.clone() for master_param in master]

    snapshot = MegatronMixedPrecisionAdapterState.capture(optimizer, version=7)

    assert snapshot.version == 7
    assert snapshot.step == 3
    assert set(snapshot.opt_state) == {"exp_avg", "exp_avg_sq"}

    for master_param in master:
        master_param.zero_()
        optimizer.optimizer.state[master_param]["exp_avg"].zero_()
        optimizer.optimizer.state[master_param]["exp_avg_sq"].zero_()
    optimizer.optimizer.param_groups[0]["step"] = 99

    snapshot.restore_into(optimizer)

    assert MegatronMixedPrecisionAdapterState._get_step(
        optimizer.optimizer.param_groups
    ) == 3
    for master_param, expected in zip(master, orig_master):
        assert torch.equal(master_param, expected)
    for model_param, master_param in zip(model, master):
        assert torch.equal(model_param, master_param.to(torch.bfloat16))
    for master_param in master:
        assert torch.equal(
            optimizer.optimizer.state[master_param]["exp_avg"],
            torch.full((master_param.numel(),), 0.5),
        )


def test_capture_is_generic_over_optimizer_state_keys():
    model = [torch.randn(4, dtype=torch.bfloat16)]
    master = [torch.randn(4)]
    optimizer = _fake_optimizer(model, master, state_keys=("momentum_buffer",))
    optimizer.optimizer.state[master[0]]["momentum_buffer"].fill_(0.125)

    snapshot = MegatronMixedPrecisionAdapterState.capture(optimizer, version=1)

    assert set(snapshot.opt_state) == {"momentum_buffer"}
    master[0].zero_()
    optimizer.optimizer.state[master[0]]["momentum_buffer"].zero_()
    snapshot.restore_into(optimizer)
    assert not torch.equal(master[0], torch.zeros_like(master[0]))
    assert not torch.equal(
        optimizer.optimizer.state[master[0]]["momentum_buffer"],
        torch.zeros_like(master[0]),
    )


def test_nonuniform_optimizer_state_keys_raise():
    model = [torch.randn(4, dtype=torch.bfloat16), torch.randn(4, dtype=torch.bfloat16)]
    master = [torch.randn(4), torch.randn(4)]
    optimizer = _fake_optimizer(model, master)
    optimizer.optimizer.state[master[1]] = {
        "exp_avg": torch.zeros(4),
        "step": 0,
    }

    with pytest.raises(RuntimeError, match="state keys differ"):
        MegatronMixedPrecisionAdapterState.capture(optimizer, version=1)


def test_genesis_uses_pristine_master_and_zeros_state():
    model = [torch.randn(4, dtype=torch.bfloat16)]
    master = [torch.randn(4)]
    optimizer = _fake_optimizer(model, master, step=5)
    for master_param in master:
        optimizer.optimizer.state[master_param]["exp_avg"].fill_(9.0)
    pristine = MegatronMixedPrecisionAdapterState.snapshot_master(optimizer)

    genesis = MegatronMixedPrecisionAdapterState.genesis(optimizer, pristine)

    assert genesis.version == 0
    assert genesis.step == 0
    assert torch.equal(genesis.lora, pristine)
    assert set(genesis.opt_state) == {"exp_avg", "exp_avg_sq"}
    for tensor in genesis.opt_state.values():
        assert torch.count_nonzero(tensor) == 0


def test_reset_optimizer_state_rebuilds_master_and_zeros():
    model = [torch.full((4,), 2.0, dtype=torch.bfloat16)]
    master = [torch.zeros(4)]
    optimizer = _fake_optimizer(model, master, step=8)
    for master_param in master:
        optimizer.optimizer.state[master_param]["exp_avg"].fill_(3.0)

    MegatronMixedPrecisionAdapterState.reset_optimizer_state(optimizer)

    assert torch.equal(master[0], torch.full((4,), 2.0))
    for master_param in master:
        assert torch.count_nonzero(optimizer.optimizer.state[master_param]["exp_avg"]) == 0
    assert MegatronMixedPrecisionAdapterState._get_step(
        optimizer.optimizer.param_groups
    ) == 0


def test_tensor_valued_step_mutates_in_place():
    model = [torch.randn(4, dtype=torch.bfloat16)]
    master = [torch.randn(4)]
    optimizer = _fake_optimizer(model, master, step=torch.tensor(11))
    step_tensor = optimizer.optimizer.param_groups[0]["step"]
    snapshot = MegatronMixedPrecisionAdapterState.capture(optimizer, version=1)
    snapshot.step = 4

    snapshot.restore_into(optimizer)

    assert optimizer.optimizer.param_groups[0]["step"] is step_tensor
    assert int(step_tensor.item()) == 4


def test_select_returns_mixed_for_non_distributed():
    assert (
        select_adapter_state(object(), is_distributed=False)
        is MegatronMixedPrecisionAdapterState
    )


def test_compatibility_alias_points_to_preferred_name():
    assert MixedPrecisionAdapterState is MegatronMixedPrecisionAdapterState


def test_distributed_optimizer_is_not_supported():
    state_cls = select_adapter_state(object(), is_distributed=True)

    assert state_cls is DistributedAdapterState
    with pytest.raises(NotImplementedError, match="distributed optimizer"):
        state_cls.assert_supported(object())


def test_assert_supported_rejects_optimizer_without_fp32_masters():
    optimizer = _fake_optimizer([], [])

    with pytest.raises(NotImplementedError, match="fp32 master"):
        MegatronMixedPrecisionAdapterState.assert_supported(optimizer)
