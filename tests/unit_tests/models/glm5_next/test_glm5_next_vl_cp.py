# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The vision splice picks the right image features on every CP rank.

Under context parallelism each rank sees one zigzag THD shard of the sequence, but
the vision tower produces every image's features in global document order. The
plain ``masked_scatter`` the cp=1 path uses consumes that tensor from the front, so
on any rank but 0 it splices the wrong images -- and the shapes still line up, so
nothing raises. ``_local_feature_index`` is what prevents that.

The property checked here is the whole contract: for every local placeholder, the
returned feature row equals the count of image tokens before it in the *global*
sequence. The zigzag shard itself is built with the same layout helper the DSA
indexer uses, which is what the model does at runtime -- this test covers the index
arithmetic layered on top of it, not the helper.

The arithmetic is device-independent, so this runs on CPU over gloo and needs no
GPU. Run standalone: ``python test_glm5_next_vl_cp.py``.
"""

import os

import torch

CP = int(os.environ.get("BT_VL_CP_SIZE", "2"))

# Physical lengths divide 2*CP for zigzag sharding; real lengths leave padding at
# each document end, so the padded-row positions the helper emits are exercised.
REAL_DOC_LENS = (7, 9)
PADDED_DOC_LENS = (8, 12)
SEQ_LEN = sum(PADDED_DOC_LENS)
IMAGE_TOKEN = 154854
# Placeholder runs inside the real span of each document: one image in doc 0,
# two in doc 1, so a rank holding only doc-1 rows must not start from feature 0.
IMAGE_POSITIONS = (2, 3, 4, 9, 10, 13, 14, 15)


def _cu_seqlens(lengths, device):
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(lengths, device=device), 0)
    return cu


def _run_rank(rank: int, world: int, rdv_file: str, result_file: str):
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{rdv_file}", world_size=world, rank=rank
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant import dsa_layout

    from megatron.bridge.models.glm5_next.glm5_next_vl_model import Glm5NextVLModel

    cp_group = torch.distributed.new_group(list(range(world)))
    device = torch.device("cpu")
    cu = _cu_seqlens(REAL_DOC_LENS, device)
    cu_padded = _cu_seqlens(PADDED_DOC_LENS, device)
    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu_padded,
        cu_seqlens_kv_padded=cu_padded,
        max_seqlen_q=max(PADDED_DOC_LENS),
        max_seqlen_kv=max(PADDED_DOC_LENS),
    )

    global_ids = torch.zeros(SEQ_LEN, dtype=torch.int64, device=device)
    global_ids[list(IMAGE_POSITIONS)] = IMAGE_TOKEN

    local_rows = SEQ_LEN // world
    positions = dsa_layout.build_packed_allgather_cp_local_positions(
        cu_padded.to(torch.int64),
        world,
        rank,
        device,
        output_size=local_rows,
        cu_seqlens_cover_output=False,
    )
    in_range = positions < SEQ_LEN
    local_ids = torch.zeros(local_rows, dtype=torch.int64, device=device)
    local_ids[in_range] = global_ids.index_select(0, positions[in_range])
    placeholders = local_ids == IMAGE_TOKEN

    got = Glm5NextVLModel._local_feature_index(
        None, placeholders, psp, cp_group, n_features=len(IMAGE_POSITIONS)
    )

    # A tower/token-count disagreement must raise rather than resolve to a
    # plausible-looking index.
    mismatch_raised = False
    try:
        Glm5NextVLModel._local_feature_index(
            None, placeholders, psp, cp_group, n_features=len(IMAGE_POSITIONS) - 1
        )
    except ValueError:
        mismatch_raised = True

    # Reference: rank among global image tokens, read straight off the global mask.
    global_rank = torch.cumsum((global_ids == IMAGE_TOKEN).to(torch.int64), dim=0) - 1
    want = global_rank.index_select(0, positions[placeholders])

    mismatch = int((got != want).sum().item()) + (0 if mismatch_raised else 1000)
    counted = int(placeholders.sum().item())
    gathered = [torch.zeros(2, dtype=torch.int64, device=device) for _ in range(world)]
    torch.distributed.all_gather(
        gathered, torch.tensor([mismatch, counted], dtype=torch.int64, device=device)
    )
    if rank == 0:
        bad = sum(int(t[0].item()) for t in gathered)
        seen = sum(int(t[1].item()) for t in gathered)
        with open(result_file, "w") as handle:
            handle.write(f"{bad} {seen}")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def test_glm5_next_vision_splice_cp_indices_match_global_order():
    import re
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"standalone parity run failed:\n{output[-4000:]}"
    match = re.search(r"RESULT mismatched=(\d+) placeholders=(\d+)", output)
    assert match, f"no RESULT line:\n{output[-4000:]}"
    assert int(match.group(1)) == 0, f"wrong feature rows on some CP rank:\n{output[-4000:]}"
    # Every placeholder must be accounted for across the ranks, or a pass would
    # only mean the shards happened to hold no images.
    assert int(match.group(2)) == len(IMAGE_POSITIONS)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        rdv = os.path.join(directory, "rdv")
        result = os.path.join(directory, "result")
        torch.multiprocessing.spawn(_run_rank, args=(CP, rdv, result), nprocs=CP, join=True)
        bad, seen = open(result).read().split()
        print(f"RESULT mismatched={bad} placeholders={seen}")
