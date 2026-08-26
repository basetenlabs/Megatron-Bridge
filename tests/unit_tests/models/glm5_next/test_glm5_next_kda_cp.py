# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""cp2-vs-cp1 parity for the GLM-5 Next KDA context-parallel path.

Two ranks build the same ``Glm5NextKDA`` twice: once with a 2-rank CP group
(sequence-sharded zigzag input) and once with a singleton CP group (the cp=1
reference on the full stream). Identical weights, identical documents. The
gathered cp=2 output must match the cp=1 output, and the cp=2 parameter
gradients — summed across CP ranks — must match the reference gradients.

Run standalone: ``python test_glm5_next_kda_cp.py`` (needs 2 GPUs + FLA).
"""

import os
from dataclasses import dataclass

import pytest
import torch

try:
    import fla  # noqa: F401

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

requires_2gpu_fla = pytest.mark.skipif(
    not HAVE_FLA or not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs FLA and 2 CUDA devices",
)

CP = 2
HIDDEN = 512
NUM_HEADS = 8
HEAD_DIM = 64
CONV_KERNEL = 4
# Two documents; every length divides 2*CP as the zigzag partition requires.
DOC_LENS = (128, 64)
SEQ_LEN = sum(DOC_LENS)
DTYPE = torch.bfloat16


@dataclass
class _PGs:
    tp: torch.distributed.ProcessGroup
    cp: torch.distributed.ProcessGroup


def _build_config():
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=NUM_HEADS,
        use_cpu_initialization=False,
        bf16=True,
        params_dtype=DTYPE,
        sequence_parallel=False,
    )
    config.kimi_kda_layers = (1,)
    config.kimi_linear_num_heads = NUM_HEADS
    config.kimi_linear_head_dim = HEAD_DIM
    config.kimi_linear_conv_kernel_size = CONV_KERNEL
    config.kimi_kda_gate_lower_bound = -5.0
    return config


def _cu_seqlens(device):
    cu = torch.zeros(len(DOC_LENS) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(DOC_LENS, device=device), 0)
    return cu


def _natural_chunk(lb_chunk: int, cp: int) -> int:
    """Invert the load-balanced chunk ordering (mirrors GDN's convention)."""
    if lb_chunk % 2 == 0:
        return lb_chunk // 2
    return (4 * cp - 1 - lb_chunk) // 2


def _zigzag_doc_slices(doc_len: int, doc_start: int, rank: int, cp: int):
    """Global index ranges this rank owns for one document, in local order."""
    half = doc_len // (2 * cp)
    slices = []
    for lb in (2 * rank, 2 * rank + 1):
        nat = _natural_chunk(lb, cp)
        start = doc_start + nat * half
        slices.append((start, start + half))
    return slices


def _local_shard(full: torch.Tensor, rank: int) -> torch.Tensor:
    """Zigzag-shard [s, b, h] along dim 0 for `rank`."""
    parts, doc_start = [], 0
    for doc_len in DOC_LENS:
        for start, stop in _zigzag_doc_slices(doc_len, doc_start, rank, CP):
            parts.append(full[start:stop])
        doc_start += doc_len
    return torch.cat(parts, dim=0)


def _unshard(gathered: list[torch.Tensor]) -> torch.Tensor:
    """Rebuild the natural-order [s, b, h] stream from per-rank zigzag shards."""
    out = torch.empty(
        (SEQ_LEN, *gathered[0].shape[1:]), dtype=gathered[0].dtype, device=gathered[0].device
    )
    for rank, shard in enumerate(gathered):
        cursor, doc_start = 0, 0
        for doc_len in DOC_LENS:
            for start, stop in _zigzag_doc_slices(doc_len, doc_start, rank, CP):
                out[start:stop] = shard[cursor : cursor + (stop - start)]
                cursor += stop - start
            doc_start += doc_len
    return out


def _run_rank(rank: int, world: int, rdv_file: str, result_file: str):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{rdv_file}", world_size=world, rank=rank
    )
    from megatron.core.packed_seq_params import PackedSeqParams

    from megatron.bridge.models.glm5_next.modeling_glm5_next.kda import Glm5NextKDA

    cp_group = torch.distributed.new_group(list(range(world)))
    singletons = [torch.distributed.new_group([r]) for r in range(world)]
    self_group = singletons[rank]

    config = _build_config()
    device = torch.device("cuda", rank)
    cu = _cu_seqlens(device)
    psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)

    torch.manual_seed(1234)
    layer_cp = Glm5NextKDA(
        config, layer_number=1, pg_collection=_PGs(tp=self_group, cp=cp_group)
    ).to(device)
    layer_ref = Glm5NextKDA(
        config, layer_number=1, pg_collection=_PGs(tp=self_group, cp=self_group)
    ).to(device)
    layer_ref.load_state_dict(layer_cp.state_dict())

    torch.manual_seed(7)  # identical stream on both ranks
    full = torch.randn(SEQ_LEN, 1, HIDDEN, dtype=DTYPE, device=device, requires_grad=False)

    ref_out = layer_ref._forward_kda(full, psp)
    ref_out.float().sum().backward()

    local = _local_shard(full, rank).contiguous()
    cp_out = layer_cp._forward_kda(local, psp)
    cp_out.float().sum().backward()

    gathered = [torch.empty_like(cp_out) for _ in range(world)]
    torch.distributed.all_gather(gathered, cp_out.contiguous(), group=cp_group)
    merged = _unshard([g.detach() for g in gathered])

    fwd_max_err = (merged.float() - ref_out.detach().float()).abs().max().item()

    grad_max_err = 0.0
    for name, param in layer_cp.named_parameters():
        grad = param.grad
        if grad is None:
            grad = torch.zeros_like(param, dtype=torch.float32)
        grad = grad.detach().float().contiguous()
        torch.distributed.all_reduce(grad, group=cp_group)
        ref_grad = dict(layer_ref.named_parameters())[name].grad
        assert ref_grad is not None, f"reference grad missing for {name}"
        err = (grad - ref_grad.detach().float()).abs().max().item()
        denom = ref_grad.detach().float().abs().max().item() or 1.0
        grad_max_err = max(grad_max_err, err / denom)

    if rank == 0:
        with open(result_file, "w") as f:
            f.write(f"{fwd_max_err} {grad_max_err}")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@requires_2gpu_fla
def test_glm5_next_kda_cp2_matches_cp1(tmp_path):
    rdv = tmp_path / "rdv"
    result = tmp_path / "result"
    torch.multiprocessing.spawn(
        _run_rank, args=(CP, str(rdv), str(result)), nprocs=CP, join=True
    )
    fwd_err, grad_rel_err = (float(x) for x in result.read_text().split())
    # bf16 activations with fp32 state; the a2a itself is exact so tolerances
    # only absorb reduction-order noise.
    assert fwd_err < 5e-2, f"cp2 forward diverges from cp1: max abs err {fwd_err}"
    assert grad_rel_err < 5e-2, f"cp2 grads diverge from cp1: max rel err {grad_rel_err}"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        rdv, result = os.path.join(d, "rdv"), os.path.join(d, "result")
        torch.multiprocessing.spawn(_run_rank, args=(CP, rdv, result), nprocs=CP, join=True)
        print("fwd_max_err grad_rel_err:", open(result).read())
