# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""cp2-vs-cp1 parity for the GLM-5 Next kpool indexer hooks.

Two ranks drive ``Glm5NextKPoolIndexer`` through the same hook sequence
``DSAttention`` uses — ``forward_before_topk`` → CP allgather + global-order
restore of the returned raw keys → ``prepare_topk_inputs`` →
``finalize_topk_indices`` — once with a 2-rank CP group (zigzag-sharded
input) and once with a singleton CP group on the full stream. The base
``DSAIndexer.forward_before_topk`` is stubbed to position-deterministic
projections of the input so the test isolates the subclass's pooling/CP
logic. Pooled keys must match exactly (permutation-only data movement), and
each local query row's finalized raw indices and lengths must match the
cp=1 reference at the same global position.

Run standalone: ``python test_glm5_next_kpool_cp.py`` (needs 2 GPUs).
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

CP = int(os.environ.get("BT_KPOOL_CP_SIZE", "2"))

requires_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < CP,
    reason=f"needs {CP} CUDA devices",
)
HIDDEN = 256
INDEX_HEAD_DIM = 64
INDEX_TOPK = 16
# Doc lengths must divide 2*CP (zigzag) and exercise incomplete tail pools.
DOC_LENS = (128, 64)
SEQ_LEN = sum(DOC_LENS)
DTYPE = torch.bfloat16


@dataclass
class _PGs:
    tp: torch.distributed.ProcessGroup
    cp: torch.distributed.ProcessGroup


def _cu_seqlens(device):
    cu = torch.zeros(len(DOC_LENS) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(DOC_LENS, device=device), 0)
    return cu


def _stub_base_forward_before_topk(self, x, qr, packed_seq_params=None):
    """Position-deterministic stand-in for DSAIndexer.forward_before_topk."""
    del qr, packed_seq_params
    rows = x.size(0)
    q = x[..., :INDEX_HEAD_DIM].unsqueeze(2).contiguous()
    k = x[..., :INDEX_HEAD_DIM].contiguous()
    weights = x[..., :1].reshape(rows, 1, 1).squeeze(-1).contiguous()
    return q, k, weights


def _build_indexer(pgs, device):
    from megatron.bridge.models.glm5_next.modeling_glm5_next.kpool_indexer import (
        Glm5NextKPoolIndexer,
    )

    indexer = Glm5NextKPoolIndexer.__new__(Glm5NextKPoolIndexer)
    nn.Module.__init__(indexer)
    indexer.pg_collection = pgs
    indexer.config = SimpleNamespace(sequence_parallel=False)
    indexer.index_topk = INDEX_TOPK
    indexer.index_head_dim = INDEX_HEAD_DIM
    indexer.hidden_size = HIDDEN
    torch.manual_seed(42)  # identical pool params on every rank and instance
    indexer.index_kpool_compress_ape = nn.Parameter(
        torch.randn(indexer.pool_size, INDEX_HEAD_DIM, device=device, dtype=DTYPE)
    )
    indexer.index_kpool_compress_gate = nn.Parameter(
        torch.randn(INDEX_HEAD_DIM, HIDDEN, device=device, dtype=DTYPE)
    )
    indexer._pool_to_raw = None
    indexer._pool_prefix = None
    indexer._raw_cu_seqlens = None
    indexer._local_gate = None
    return indexer


def _drive_hooks(indexer, x, psp, cp_group, cp_rank, cp_size):
    """Replicate DSAttention's hook glue around the indexer."""
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
    from megatron.core.transformer.experimental_attention_variant import dsa_layout
    from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
        generate_varlen_mask_params_for_positions,
    )

    q, k, weights = indexer.forward_before_topk(x, x, psp)

    cu_q, cu_kv = dsa_layout.get_packed_qk_cu_seqlens(psp)
    t_local = k.size(0)
    if cp_size > 1:
        query_positions, kv_reorder = (
            dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_kv,
                cp_size=cp_size,
                cp_rank=cp_rank,
                device=k.device,
                local_output_size=t_local,
                key_local_output_size=t_local,
                global_output_size=t_local * cp_size,
            )
        )
        k = gather_from_sequence_parallel_region(k, group=cp_group)
        k = k.index_select(0, kv_reorder)
    else:
        query_positions = torch.arange(t_local, dtype=torch.int64, device=k.device)

    starts, ends = generate_varlen_mask_params_for_positions(
        cu_q.to(device=k.device, dtype=torch.int64), query_positions
    )
    bounds = (starts.to(torch.int32), ends.to(torch.int32))

    q, pooled_k, weights, eff_topk, pool_bounds = indexer.prepare_topk_inputs(
        q, k, weights, indexer.index_topk, bounds, psp
    )
    pool_starts, pool_ends = pool_bounds

    # Deterministic stand-in for the fused bounded top-k: every row selects
    # its first ``eff_topk`` visible pools.
    visible = (pool_ends - pool_starts).clamp(min=0)
    take = visible.clamp(max=eff_topk)
    offsets = torch.arange(eff_topk, device=k.device, dtype=torch.int64)
    topk_indices = pool_starts.unsqueeze(-1) + offsets.unsqueeze(0)
    topk_indices = topk_indices.masked_fill(offsets.unsqueeze(0) >= take.unsqueeze(-1), -1)
    topk_indices = topk_indices.unsqueeze(0)
    topk_length = take.unsqueeze(0).to(torch.int32)

    result, result_length = indexer.finalize_topk_indices(topk_indices, topk_length, psp)
    return query_positions, pooled_k, result.squeeze(0), result_length.squeeze(0)


def _run_rank(rank: int, world: int, rdv_file: str, result_file: str):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{rdv_file}", world_size=world, rank=rank
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer

    DSAIndexer.forward_before_topk = _stub_base_forward_before_topk

    cp_group = torch.distributed.new_group(list(range(world)))
    singletons = [torch.distributed.new_group([r]) for r in range(world)]
    self_group = singletons[rank]

    device = torch.device("cuda", rank)
    cu = _cu_seqlens(device)
    psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)

    torch.manual_seed(7)  # identical stream on both ranks
    full = torch.randn(SEQ_LEN, 1, HIDDEN, dtype=DTYPE, device=device)

    from megatron.core.transformer.experimental_attention_variant import dsa_layout

    local_positions = dsa_layout.build_packed_allgather_cp_local_positions(
        cu, world, rank, device, output_size=SEQ_LEN // world
    )
    local = full.index_select(0, local_positions).contiguous()

    ref = _build_indexer(_PGs(tp=self_group, cp=self_group), device)
    ref_positions, ref_pooled_k, ref_result, ref_length = _drive_hooks(
        ref, full, psp, self_group, 0, 1
    )

    cp = _build_indexer(_PGs(tp=self_group, cp=cp_group), device)
    cp_positions, cp_pooled_k, cp_result, cp_length = _drive_hooks(
        cp, local, psp, cp_group, rank, world
    )

    pooled_err = (cp_pooled_k.float() - ref_pooled_k.float()).abs().max().item()

    # Compare each local row against the reference row at its global position.
    ref_rows = ref_result.index_select(0, cp_positions)
    ref_lens = ref_length.index_select(0, cp_positions)
    rows_equal = bool(torch.equal(cp_result, ref_rows))
    lens_equal = bool(torch.equal(cp_length.to(ref_lens.dtype), ref_lens))

    verdicts = torch.tensor(
        [pooled_err, 0.0 if rows_equal else 1.0, 0.0 if lens_equal else 1.0],
        device=device,
    )
    torch.distributed.all_reduce(verdicts, op=torch.distributed.ReduceOp.MAX, group=cp_group)

    if rank == 0:
        with open(result_file, "w") as f:
            f.write(f"{verdicts[0].item()} {verdicts[1].item()} {verdicts[2].item()}")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _server_venv_python() -> str:
    """The interpreter whose megatron resolves from the build clone."""
    clone = os.environ.get("BT_BUILD_CLONE", "/root/trainers")
    for component in ("server-megatron-bridge", "server"):
        candidate = os.path.join(clone, component, ".venv", "bin", "python")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"no server venv python under {clone}")


@requires_gpus
def test_glm5_next_kpool_cp_matches_cp1():
    import re
    import subprocess

    proc = subprocess.run(
        [_server_venv_python(), os.path.abspath(__file__)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"standalone parity run failed:\n{output[-4000:]}"
    match = re.search(r"RESULT pooled=([0-9.eE+-]+) rows=([01].?[0-9]*) lens=([01].?[0-9]*)", output)
    assert match, f"no RESULT line in output:\n{output[-4000:]}"
    pooled_err = float(match.group(1))
    rows_bad, lens_bad = float(match.group(2)), float(match.group(3))
    # Gate gather + reorder are pure permutations; pooling runs in identical
    # global order on both sides, so pooled keys must match exactly.
    assert pooled_err == 0.0, f"cp pooled keys diverge from cp1: max abs err {pooled_err}"
    assert rows_bad == 0.0, "cp finalized top-k raw indices diverge from cp1"
    assert lens_bad == 0.0, "cp finalized top-k lengths diverge from cp1"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        rdv, result = os.path.join(d, "rdv"), os.path.join(d, "result")
        torch.multiprocessing.spawn(_run_rank, args=(CP, rdv, result), nprocs=CP, join=True)
        pooled, rows_bad, lens_bad = open(result).read().split()
        print(f"RESULT pooled={pooled} rows={rows_bad} lens={lens_bad}")
