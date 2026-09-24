# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""A rank-r LoRA padded into a larger adapter trains exactly like a native rank-r LoRA."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.bridge.peft.active_dim import get_lora_active_dim, lora_padding_masks, set_lora_active_dim
from megatron.bridge.peft.lora_layers import LinearAdapter, LoRALinear
from megatron.bridge.peft.utils import rank_padding_masks


RANK = 4
DIM = 16
STEPS = 6


def _lora(base: nn.Linear, dim: int) -> LoRALinear:
    lora = LoRALinear(copy.deepcopy(base), LinearAdapter(base, dim=dim, alpha=32))
    lora.to_wrap.weight.requires_grad_(False)
    lora.to_wrap.bias.requires_grad_(False)
    return lora


def _trained_pair():
    """Train a native rank-RANK LoRA and the same LoRA padded to DIM on identical data."""
    torch.manual_seed(0)
    base = nn.Linear(24, 20)
    native = _lora(base, RANK)
    padded = _lora(base, DIM)
    masks = lora_padding_masks(padded, RANK)
    with torch.no_grad():
        for weight, mask in masks.items():
            weight.masked_fill_(mask, 0)
        padded.adapter.linear_in.weight[:RANK] = native.adapter.linear_in.weight
        padded.adapter.linear_out.weight[:, :RANK] = native.adapter.linear_out.weight
    set_lora_active_dim(padded, RANK)

    optimizers = [
        torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.1)
        for m in (native, padded)
    ]
    history = []
    for step in range(STEPS):
        x = torch.randn(8, 24, generator=torch.Generator().manual_seed(step))
        outputs = []
        for model, optimizer in zip((native, padded), optimizers):
            optimizer.zero_grad()
            out = model(x)
            out.square().mean().backward()
            outputs.append(out.detach())
            optimizer.step()
        grads = {w: w.grad.clone() for w in masks}
        state = {w: {k: v.clone() for k, v in optimizers[1].state[w].items() if k != "step"} for w in masks}
        history.append((outputs, grads, state))
    return native, padded, masks, history


def test_padded_forward_and_weights_match_native():
    native, padded, _, history = _trained_pair()
    for (native_out, padded_out), _, _ in history:
        torch.testing.assert_close(padded_out, native_out)
    torch.testing.assert_close(padded.adapter.linear_in.weight[:RANK], native.adapter.linear_in.weight)
    torch.testing.assert_close(padded.adapter.linear_out.weight[:, :RANK], native.adapter.linear_out.weight)
    assert padded.adapter.linear_out.weight[:, :RANK].abs().sum() > 0, "B never trained"


def test_padding_stays_exactly_zero_in_weights_grads_and_moments():
    _, _, masks, history = _trained_pair()
    for weight, mask in masks.items():
        assert torch.all(weight.detach()[mask] == 0)
    for _, grads, state in history:
        for weight, mask in masks.items():
            assert torch.all(grads[weight][mask] == 0)
            for moment in state[weight].values():
                assert torch.all(moment[mask] == 0)


def test_active_dim_sets_the_scale():
    base = nn.Linear(8, 8)
    lora = _lora(base, DIM)
    set_lora_active_dim(lora, RANK)
    assert lora.adapter.scale == 32 / RANK
    assert get_lora_active_dim(lora) == RANK
    set_lora_active_dim(lora, DIM)
    assert lora.adapter.scale == 32 / DIM


def test_rejects_unknown_adapters_and_out_of_range_ranks():
    class _Subclass(LinearAdapter):
        pass

    base = nn.Linear(8, 8)
    with pytest.raises(NotImplementedError):
        set_lora_active_dim(LoRALinear(base, _Subclass(base, dim=DIM)), RANK)
    with pytest.raises(ValueError):
        set_lora_active_dim(_lora(base, DIM), DIM + 1)


def test_rejects_an_activation_between_the_factors():
    """SiLU is zero at zero but sigmoid is not; any non-identity activation is refused."""
    lora = _lora(nn.Linear(8, 8), DIM)
    lora.adapter.activation = nn.Sigmoid()
    lora.adapter.base_linear_name = "linear_fc1"
    with pytest.raises(NotImplementedError, match="identity activation"):
        set_lora_active_dim(lora, RANK)
    with pytest.raises(NotImplementedError, match="identity activation"):
        lora_padding_masks(lora, RANK)
    lora.adapter.activation = nn.Identity()
    set_lora_active_dim(lora, RANK)


class _Group:
    def __init__(self, size: int, rank: int) -> None:
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


@pytest.mark.parametrize(
    ("tp_rank", "padded_rows"), [(0, []), (1, []), (2, [4, 5, 6, 7]), (3, [0, 1, 2, 3, 4, 5, 6, 7])]
)
def test_rank_sharded_linear_in_maps_local_rows_to_global_rank(tp_rank, padded_rows):
    """dim=32 over TP=4 gives each rank 8 rows; rank 20 pads global rows 20..31."""
    adapter = SimpleNamespace(
        dim=32,
        base_linear_name="linear_qkv",
        tp_group=_Group(4, tp_rank),
        linear_in=SimpleNamespace(weight=torch.empty(8, 5)),
        linear_out=SimpleNamespace(weight=torch.empty(6, 32)),
    )
    in_mask, out_mask = rank_padding_masks(adapter, 20)
    assert in_mask[:, 0].nonzero().flatten().tolist() == padded_rows
    assert in_mask.all(dim=1).tolist() == in_mask.any(dim=1).tolist()
    assert out_mask[0].nonzero().flatten().tolist() == list(range(20, 32))
