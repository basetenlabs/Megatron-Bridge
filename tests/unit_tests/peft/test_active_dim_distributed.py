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

"""Two-rank tests for a real ``ParallelLinearAdapter`` run below its allocated dim.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_active_dim_distributed.py
"""

import os

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.peft.active_dim import lora_padding_masks, set_lora_active_dim
from megatron.bridge.peft.lora_merge import LoRAMerge
from megatron.bridge.peft.utils import ParallelLinearAdapter


_TP_SIZE = 2
_DIM = 8
_RANK = 3
_IN = 16
_OUT = 12


@pytest.fixture(scope="module")
def pg_collection():
    if int(os.environ.get("WORLD_SIZE", "1")) != _TP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="nccl")
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    if owns_model_parallel:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=_TP_SIZE)
    model_parallel_cuda_manual_seed(2026, force_reset_rng=True)
    yield ProcessGroupCollection.use_mpu_process_groups()
    if owns_model_parallel:
        parallel_state.destroy_model_parallel()
    if owns_process_group:
        dist.destroy_process_group()


def _padded_adapter(pg_collection, base_linear_name: str, *, input_is_parallel: bool) -> "_AdapterModel":
    config = ModelParallelConfig(tensor_model_parallel_size=_TP_SIZE, params_dtype=torch.float32)
    adapter = ParallelLinearAdapter(
        in_features=_IN,
        out_features=_OUT,
        dim=_DIM,
        base_linear_name=base_linear_name,
        activation="identity",
        input_is_parallel=input_is_parallel,
        model_parallel_config=config,
        alpha=16,
        pg_collection=pg_collection,
    ).cuda()
    model = _AdapterModel(adapter)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.normal_()
        for weight, mask in lora_padding_masks(model, _RANK).items():
            weight.masked_fill_(mask, 0)
    set_lora_active_dim(model, _RANK)
    return model


class _AdapterModel(nn.Module):
    def __init__(self, adapter: ParallelLinearAdapter) -> None:
        super().__init__()
        from megatron.bridge.peft.adapter_wrapper import AdapterWrapper

        class _Wrapper(AdapterWrapper):
            def forward(self, x):
                return self.adapter(x)

        self.wrapper = _Wrapper(nn.Identity(), adapter)

    @property
    def adapter(self) -> ParallelLinearAdapter:
        return self.wrapper.adapter


def _global_factors(adapter: ParallelLinearAdapter) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather each factor along whichever axis TP shards it."""
    factors = []
    for weight, full_shape in (
        (adapter.linear_in.weight, (_DIM, _IN)),
        (adapter.linear_out.weight, (_OUT, _DIM)),
    ):
        weight = weight.detach()
        for axis in (0, 1):
            if weight.shape[axis] * _TP_SIZE == full_shape[axis]:
                shards = [torch.empty_like(weight) for _ in range(_TP_SIZE)]
                dist.all_gather(shards, weight.contiguous())
                weight = torch.cat(shards, dim=axis)
                break
        assert tuple(weight.shape) == full_shape
        factors.append(weight)
    return factors[0], factors[1]


_LAYOUTS = [
    pytest.param("decoder.layers.0.self_attention.linear_qkv", False, id="column-parallel"),
    pytest.param("decoder.layers.0.self_attention.linear_proj", True, id="row-parallel"),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("base_linear_name", "input_is_parallel"), _LAYOUTS)
def test_padding_is_global_rank_indices_past_the_active_dim(pg_collection, base_linear_name, input_is_parallel):
    model = _padded_adapter(pg_collection, base_linear_name, input_is_parallel=input_is_parallel)
    linear_in, linear_out = _global_factors(model.adapter)
    assert torch.all(linear_in[_RANK:] == 0) and torch.all(linear_in[:_RANK] != 0)
    assert torch.all(linear_out[:, _RANK:] == 0) and torch.all(linear_out[:, :_RANK] != 0)


@pytest.mark.gpu
@pytest.mark.parametrize(("base_linear_name", "input_is_parallel"), _LAYOUTS)
def test_forward_scales_by_the_active_dim_and_padding_gets_no_gradient(
    pg_collection, base_linear_name, input_is_parallel
):
    model = _padded_adapter(pg_collection, base_linear_name, input_is_parallel=input_is_parallel)
    adapter = model.adapter
    linear_in, linear_out = _global_factors(adapter)

    x_full = torch.randn(5, 1, _IN, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    local_in = _IN // _TP_SIZE
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    x = x_full[..., tp_rank * local_in : (tp_rank + 1) * local_in] if input_is_parallel else x_full
    output = adapter(x.contiguous())

    expected = (16 / _RANK) * (x_full @ linear_in[:_RANK].T @ linear_out[:, :_RANK].T)
    if not input_is_parallel:
        local_out = _OUT // _TP_SIZE
        expected = expected[..., tp_rank * local_out : (tp_rank + 1) * local_out]
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    output.square().sum().backward()
    for weight, mask in lora_padding_masks(model, _RANK).items():
        assert weight.grad is not None
        assert torch.all(weight.grad[mask] == 0)
        # At TP=2 a column-parallel linear_in shard on rank 1 holds only padding rows.
        if not mask.all():
            assert torch.any(weight.grad[~mask] != 0)


@pytest.mark.gpu
def test_merge_at_tp2_detects_the_layout_from_the_allocated_dim(pg_collection):
    """A column-parallel adapter shards linear_in on the rank axis; merging must gather it."""
    adapter = _padded_adapter(
        pg_collection, "decoder.layers.0.self_attention.linear_qkv", input_is_parallel=False
    ).adapter
    linear_in, linear_out = _global_factors(adapter)
    base = torch.zeros(_OUT // _TP_SIZE, _IN, device="cuda")

    merged = LoRAMerge().merge(
        base,
        adapter.linear_out.weight.detach(),
        adapter.linear_in.weight.detach(),
        adapter.alpha,
        adapter.dim,
        tp_group=adapter.tp_group,
        scale=adapter.scale,
    )

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    rows = slice(tp_rank * base.shape[0], (tp_rank + 1) * base.shape[0])
    expected = (16 / _RANK) * linear_out[rows, :_RANK] @ linear_in[:_RANK]
    torch.testing.assert_close(merged, expected, rtol=1e-5, atol=1e-5)
