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

"""Two-rank tests for ``SharedOuterGroupedExpertAdapter``.

Each test runs at EP=1 (two data-parallel copies of every expert) and EP=2
(each rank owns half the experts), so the expert axis and the expert-data axis
are exercised separately.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_shared_outer_expert_lora_distributed.py
"""

import os
from collections.abc import Iterator

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
from megatron.core.dist_checkpointing.optimizer import get_param_id_to_sharded_param_map
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.peft.utils import ParallelLinearAdapter, SharedOuterGroupedExpertAdapter


_WORLD_SIZE = 2
_HIDDEN = 64
_MOE_FFN = 128
_NUM_EXPERTS = 4
_DIM = 8
_ALPHA = 16
_FC_IDS = ["fc1", "fc2"]


@pytest.fixture(scope="module")
def _process_group() -> Iterator[None]:
    if int(os.environ.get("WORLD_SIZE", "1")) != _WORLD_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="nccl")
    yield
    if owns_process_group:
        dist.destroy_process_group()


@pytest.fixture(params=[1, 2], ids=["ep1", "ep2"])
def ep_size(request: pytest.FixtureRequest, _process_group: None) -> Iterator[int]:
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=request.param,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(2026, force_reset_rng=True)
    try:
        yield request.param
    finally:
        parallel_state.destroy_model_parallel()


def _config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN,
        num_attention_heads=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
    )


def _make_adapter(*, is_fc1: bool, ep_size: int) -> SharedOuterGroupedExpertAdapter:
    in_features, out_features = (_HIDDEN, 2 * _MOE_FFN) if is_fc1 else (_MOE_FFN, _HIDDEN)
    return SharedOuterGroupedExpertAdapter(
        in_features,
        out_features,
        _DIM,
        num_local_experts=_NUM_EXPERTS // ep_size,
        base_linear_name="linear_fc1" if is_fc1 else "linear_fc2",
        activation="identity",
        column_init_method="xavier",
        # LoRA-B defaults to zero, which would make every comparison pass trivially.
        row_init_method="normal",
        input_is_parallel=not is_fc1,
        model_parallel_config=_config(),
        alpha=_ALPHA,
        params_device=torch.device("cuda"),
        params_dtype=torch.bfloat16,
    ).cuda()


def _shared_and_per_expert_names(is_fc1: bool) -> tuple[str, str]:
    return ("linear_in", "linear_out") if is_fc1 else ("linear_out", "linear_in")


def _reference_forward(
    x: torch.Tensor,
    m_splits: list[int],
    linear_in: torch.Tensor,
    linear_out: torch.Tensor,
    *,
    is_fc1: bool,
) -> torch.Tensor:
    """Apply the adapter one expert at a time in fp32."""
    outputs = []
    for expert, x_expert in enumerate(x.float().split(m_splits)):
        a = linear_in.float() if is_fc1 else linear_in[expert].float()
        b = linear_out[expert].float() if is_fc1 else linear_out.float()
        outputs.append(x_expert @ a.T @ b.T)
    return torch.cat(outputs) * (_ALPHA / _DIM)


def _all_gather(tensor: torch.Tensor) -> list[torch.Tensor]:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return gathered


def _sum_over(tensor: torch.Tensor, group: object) -> torch.Tensor:
    reduced = tensor.float().clone()
    dist.all_reduce(reduced, group=group)
    return reduced


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.float().abs().max().clamp_min(1e-6)
    return ((actual.float() - expected.float()).abs().max() / scale).item()


def _sharded_tensors(entry: object) -> list:
    """Flatten a sharded-state-dict entry; fc1's per-expert side is a gate/up split factory."""
    if hasattr(entry, "build"):
        entry = entry.build()
    if isinstance(entry, dict):
        entry = list(entry.values())
    if isinstance(entry, (list, tuple)):
        return [tensor for value in entry for tensor in _sharded_tensors(value)]
    return [entry] if hasattr(entry, "axis_fragmentations") else []


@pytest.mark.gpu
@pytest.mark.parametrize("is_fc1", [True, False], ids=_FC_IDS)
def test_forward_routes_each_token_to_its_expert(ep_size: int, is_fc1: bool) -> None:
    """The grouped GEMM must match a per-expert loop, including around empty experts."""
    adapter = _make_adapter(is_fc1=is_fc1, ep_size=ep_size)
    m_splits = [0, 7, 0, 25][: adapter.num_local_experts]
    in_features = _HIDDEN if is_fc1 else _MOE_FFN
    x = torch.randn(sum(m_splits), in_features, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        actual = adapter(x, m_splits)
        expected = _reference_forward(x, m_splits, adapter.linear_in.weight, adapter.linear_out.weight, is_fc1=is_fc1)

    assert expected.abs().max() > 0
    assert _rel_err(actual, expected) < 2e-2


@pytest.mark.gpu
@pytest.mark.parametrize("is_fc1", [True, False], ids=_FC_IDS)
def test_shared_side_starts_identical_on_every_rank(ep_size: int, is_fc1: bool) -> None:
    """Ranks are seeded differently, so identical values can only come from the init broadcast."""
    model_parallel_cuda_manual_seed(1000 + dist.get_rank(), force_reset_rng=True)
    adapter = _make_adapter(is_fc1=is_fc1, ep_size=ep_size)
    shared_name, per_expert_name = _shared_and_per_expert_names(is_fc1)

    shared = _all_gather(getattr(adapter, shared_name).weight.detach())
    assert torch.equal(shared[0], shared[1])
    per_expert = _all_gather(getattr(adapter, per_expert_name).weight.detach())
    assert not torch.equal(per_expert[0], per_expert[1]), "ranks drew identical init; the check above is vacuous"


@pytest.mark.gpu
@pytest.mark.parametrize("is_fc1", [True, False], ids=_FC_IDS)
def test_sharded_state_dict_is_checkpointable(ep_size: int, is_fc1: bool) -> None:
    """Optimizer state must map onto every weight, and each rank must save only its own experts."""
    adapter = _make_adapter(is_fc1=is_fc1, ep_size=ep_size)
    metadata = {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}
    sharded = adapter.sharded_state_dict(prefix="adapter.", metadata=metadata)

    # Megatron matches optimizer state to shards by tensor identity; a shard holding a copy
    # (e.g. ``weight.data``) is dropped here and optimizer save later raises KeyError.
    params = list(adapter.parameters())
    assert set(get_param_id_to_sharded_param_map(sharded, params)) == set(range(len(params)))

    shared_name, per_expert_name = _shared_and_per_expert_names(is_fc1)
    per_expert = _sharded_tensors(sharded[f"adapter.{per_expert_name}.weight"])
    shared = _sharded_tensors(sharded[f"adapter.{shared_name}.weight"])
    assert per_expert and shared
    expert_offset = parallel_state.get_expert_model_parallel_rank() * adapter.num_local_experts
    for tensor in per_expert:
        assert tensor.axis_fragmentations[0] == ep_size
        assert tensor.global_offset[0] == expert_offset
    for tensor in shared:
        assert tensor.axis_fragmentations[0] == 1


class _SharedOuterWithControl(torch.nn.Module):
    """An fc2 shared-outer adapter next to an ordinary LoRA adapter, so one DDP buckets both."""

    def __init__(self, ep_size: int) -> None:
        super().__init__()
        self.config = _config()
        self.shared_outer = _make_adapter(is_fc1=False, ep_size=ep_size)
        self.control = ParallelLinearAdapter(
            _HIDDEN,
            _HIDDEN,
            _DIM,
            base_linear_name="linear_proj",
            activation="identity",
            column_init_method="xavier",
            row_init_method="normal",
            input_is_parallel=False,
            model_parallel_config=self.config,
            alpha=_ALPHA,
            is_expert=False,
        )

    def forward(
        self, x_expert: torch.Tensor, m_splits: list[int], x_dense: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.shared_outer(x_expert, m_splits), self.control(x_dense)


@pytest.mark.gpu
def test_ddp_gradient_scaling_matches_ordinary_lora(ep_size: int) -> None:
    """After real DDP, the shared side must be SUM over experts and MEAN over expert-data copies.

    EP=1 catches a hook that also sums over data-parallel copies; EP=2 catches a missing
    sum over experts. The ordinary adapter checks the harness itself.
    """
    model = _SharedOuterWithControl(ep_size).cuda()
    ddp = DistributedDataParallel(
        config=model.config,
        ddp_config=DistributedDataParallelConfig(overlap_grad_reduce=False),
        module=model,
    )
    ddp.zero_grad_buffer()

    m_splits = [16] * model.shared_outer.num_local_experts
    n_tokens = sum(m_splits)
    torch.manual_seed(4321 + dist.get_rank())
    x_expert = torch.randn(n_tokens, _MOE_FFN, device="cuda", dtype=torch.bfloat16)
    x_dense = torch.randn(n_tokens, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(99)
    loss_weights = torch.randn(n_tokens, _HIDDEN, device="cuda", dtype=torch.bfloat16)

    out_expert, out_dense = ddp(x_expert, m_splits, x_dense)
    ((out_expert * loss_weights).sum() + (out_dense * loss_weights).sum()).backward()
    ddp.finish_grad_sync()

    # This rank's local gradients, recomputed without the adapter's hooks or DDP.
    local = {
        name: getattr(module, side).weight.detach().float().requires_grad_()
        for name, module, side in (
            ("per_expert", model.shared_outer, "linear_in"),
            ("shared", model.shared_outer, "linear_out"),
            ("control_in", model.control, "linear_in"),
            ("control", model.control, "linear_out"),
        )
    }
    expert_out = _reference_forward(x_expert, m_splits, local["per_expert"], local["shared"], is_fc1=False)
    dense_out = x_dense.float() @ local["control_in"].T @ local["control"].T * (_ALPHA / _DIM)
    ((expert_out + dense_out) * loss_weights.float()).sum().backward()

    dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    expt_dp = parallel_state.get_expert_data_parallel_group()
    dp_size = dist.get_world_size(group=dp_cp)
    expt_dp_size = dist.get_world_size(group=expt_dp)
    expected = {
        "control": (model.control.linear_out.weight, _sum_over(local["control"].grad, dp_cp) / dp_size),
        "shared": (model.shared_outer.linear_out.weight, _sum_over(local["shared"].grad, dp_cp) / expt_dp_size),
        # Megatron scales every expert parameter by 1/dp_cp, not 1/expt_dp.
        "per_expert": (
            model.shared_outer.linear_in.weight,
            _sum_over(local["per_expert"].grad, expt_dp) / dp_size,
        ),
    }
    for name, (param, want) in expected.items():
        err = _rel_err(param.main_grad, want)
        assert err < 5e-2, f"{name}: main_grad rel err {err:.3e}"
