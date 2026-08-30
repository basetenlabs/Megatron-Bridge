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

from __future__ import annotations

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_replicated_base_lora(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TELinear
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.peft.lora_layers import LoRALinear

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    model_parallel_cuda_manual_seed(42)

    hidden_size = 8
    output_size = 4
    lora_rank = 2
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=2,
        tensor_model_parallel_size=world_size,
        sequence_parallel=True,
        params_dtype=torch.float32,
    )
    base = TELinear(
        hidden_size,
        output_size,
        parallel_mode="duplicated",
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        tp_group=None,
        name="linear_q_down_proj",
    )
    wrapped = LoRA(
        target_modules=["linear_q_down_proj"],
        dim=lora_rank,
        alpha=lora_rank,
        lora_A_init_method="xavier",
    ).transform(base, name="linear_q_down_proj")
    assert isinstance(wrapped, LoRALinear)
    adapter = wrapped.adapter

    assert isinstance(adapter.linear_in, TELinear)
    assert isinstance(adapter.linear_out, TELinear)
    assert adapter.linear_in.weight.shape == (lora_rank, hidden_size)
    assert adapter.linear_out.weight.shape == (output_size, lora_rank)
    assert not getattr(adapter.linear_in.weight, "tensor_model_parallel", False)
    assert not getattr(adapter.linear_out.weight, "tensor_model_parallel", False)
    assert getattr(adapter.linear_in.weight, "sequence_parallel", False)
    assert getattr(adapter.linear_out.weight, "sequence_parallel", False)

    with torch.no_grad():
        adapter.linear_in.weight.copy_(
            torch.arange(1, lora_rank * hidden_size + 1, device="cuda", dtype=torch.float32).reshape(
                lora_rank, hidden_size
            )
            / 10
        )
        adapter.linear_out.weight.copy_(
            torch.arange(1, output_size * lora_rank + 1, device="cuda", dtype=torch.float32).reshape(
                output_size, lora_rank
            )
            / 10
        )

    local_tokens = 2
    x_local = (
        torch.arange(
            rank * local_tokens * hidden_size + 1,
            (rank + 1) * local_tokens * hidden_size + 1,
            device="cuda",
            dtype=torch.float32,
        ).reshape(local_tokens, 1, hidden_size)
        / 10
    )
    actual = adapter(x_local)
    expected = torch.matmul(
        torch.matmul(x_local, adapter.linear_in.weight.T),
        adapter.linear_out.weight.T,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    grad_local = (
        torch.arange(
            rank * actual.numel() + 1,
            (rank + 1) * actual.numel() + 1,
            device="cuda",
            dtype=torch.float32,
        ).reshape_as(actual)
        / 100
    )
    (actual * grad_local).sum().backward()
    assert adapter.linear_in.weight.grad is not None
    assert adapter.linear_out.weight.grad is not None

    # Megatron's sequence-parallel parameter reduction sums these local-token
    # gradients across TP. Reproduce that reduction and compare with one dense
    # autograd graph over the full sequence.
    actual_a_grad = adapter.linear_in.weight.grad.detach().clone()
    actual_b_grad = adapter.linear_out.weight.grad.detach().clone()
    dist.all_reduce(actual_a_grad)
    dist.all_reduce(actual_b_grad)

    x_parts = [torch.empty_like(x_local) for _ in range(world_size)]
    grad_parts = [torch.empty_like(grad_local) for _ in range(world_size)]
    dist.all_gather(x_parts, x_local)
    dist.all_gather(grad_parts, grad_local)
    x_full = torch.cat(x_parts, dim=0)
    grad_full = torch.cat(grad_parts, dim=0)

    a_reference = adapter.linear_in.weight.detach().clone().requires_grad_()
    b_reference = adapter.linear_out.weight.detach().clone().requires_grad_()
    reference = torch.matmul(torch.matmul(x_full, a_reference.T), b_reference.T)
    reference_a_grad, reference_b_grad = torch.autograd.grad(
        (reference * grad_full).sum(),
        (a_reference, b_reference),
    )
    torch.testing.assert_close(actual_a_grad, reference_a_grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_b_grad, reference_b_grad, rtol=1e-6, atol=1e-6)

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_replicated_base_lora_matches_local_reference() -> None:
    mp.spawn(_run_replicated_base_lora, args=(2, _free_port()), nprocs=2, join=True)
