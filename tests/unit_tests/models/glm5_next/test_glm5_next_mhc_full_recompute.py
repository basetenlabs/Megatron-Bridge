# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Full-vs-none activation-recompute grad parity for GLM-5 Next mHC.

Builds one tiny GLM-5.3-shaped hybrid (KDA + DSA/kpool + dense + MoE, mHC
4-stream hyper-connections) through the real bridge provider, runs one
forward+backward without activation recompute and one with
``recompute_granularity="full"`` on the same weights and data, and compares
logits and every parameter gradient. Under full recompute the mHC selective
hook is dormant (requires granularity=="selective"), so the block wraps the
deterministic ``_forward_normal`` in the generic layer checkpoint — grads
must match to kernel-noise tolerance.

Per-head dims are production GLM-5.3 (kv_lora 512, qk_nope 256, KDA head
128, indexer head 128) so the fused kernels see familiar shapes; counts
(layers, heads, experts, hidden) are shrunk.

The noaux_tc router updates its expert-bias buffer during a training
forward, so the model state is snapshotted before the first pass and
restored before the second.

Run standalone: ``python test_glm5_next_mhc_full_recompute.py`` (1 GPU).
"""

import os
from types import SimpleNamespace

import pytest
import torch

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs 1 CUDA device"
)

SEQ_DOCS = (128, 64, 64)
SEQ_LEN = sum(SEQ_DOCS)
VOCAB = 2560


def _tiny_text_config():
    return SimpleNamespace(
        attention_bias=False,
        attention_dropout=0.0,
        dtype="bfloat16",
        first_k_dense_replace=1,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hidden_act="silu",
        hidden_size=512,
        head_dim=0,
        index_head_dim=128,
        index_n_heads=2,
        index_topk=64,
        indexer_rope_interleave=True,
        intermediate_size=768,
        kv_lora_rank=512,
        linear_attn_config={
            "num_heads": 4,
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "kda_layers": [0, 2],
            "full_attn_layers": [1],
        },
        max_position_embeddings=1_048_576,
        moe_intermediate_size=256,
        n_routed_experts=8,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_experts_per_tok=2,
        num_hidden_layers=3,
        num_key_value_heads=4,
        q_lora_rank=384,
        qk_head_dim=256,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        rms_norm_eps=1e-5,
        routed_scaling_factor=2.5,
        swiglu_limit=10.0,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
        vocab_size=VOCAB,
        v_head_dim=256,
    )


def _build_model():
    from unittest.mock import Mock

    from megatron.bridge.models.glm5_next.glm5_next_bridge import Glm5NextBridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

    wrapper = Mock(spec=PreTrainedCausalLM)
    wrapper.config = SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"],
        model_type="glm5_next",
        text_config=_tiny_text_config(),
        dtype="bfloat16",
    )
    provider = Glm5NextBridge().provider_bridge(wrapper)
    provider.seq_length = SEQ_LEN
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    model = provider.provide(pre_process=True, post_process=True)
    return model.cuda().train()


def _packed_inputs(device):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = torch.zeros(len(SEQ_DOCS) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(SEQ_DOCS, device=device), 0)
    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(SEQ_DOCS),
        max_seqlen_kv=max(SEQ_DOCS),
    )
    torch.manual_seed(7)
    input_ids = torch.randint(0, VOCAB, (1, SEQ_LEN), device=device)
    position_ids = torch.cat(
        [torch.arange(n, device=device) for n in SEQ_DOCS]
    ).unsqueeze(0)
    return input_ids, position_ids, psp


def _forward_backward(model, input_ids, position_ids, psp):
    model.zero_grad(set_to_none=True)
    logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        packed_seq_params=psp,
    )
    loss = (logits.float() ** 2).mean()
    loss.backward()
    grads = {
        name: p.grad.detach().float().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return logits.detach().float().clone(), loss.item(), grads


def _run():
    os.environ.setdefault("BT_GLM5_NEXT_MOE_DISPATCHER", "alltoall")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29711",
        world_size=1,
        rank=0,
    )
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)

    device = torch.device("cuda", 0)
    model = _build_model()
    input_ids, position_ids, psp = _packed_inputs(device)

    # The noaux_tc router mutates its expert-bias buffer during a training
    # forward; both passes must start from identical state.
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    assert model.config.recompute_granularity is None
    logits_ref, loss_ref, grads_ref = _forward_backward(
        model, input_ids, position_ids, psp
    )

    model.load_state_dict(initial_state)
    model.config.recompute_granularity = "full"
    model.config.recompute_method = "uniform"
    model.config.recompute_num_layers = 1
    logits_full, loss_full, grads_full = _forward_backward(
        model, input_ids, position_ids, psp
    )

    assert grads_ref.keys() == grads_full.keys(), (
        "parameter coverage differs between recompute modes: "
        f"{sorted(set(grads_ref) ^ set(grads_full))}"
    )

    fwd_err = (logits_full - logits_ref).abs().max().item()
    grad_rel_err, worst = 0.0, ""
    for name, g_ref in grads_ref.items():
        err = (grads_full[name] - g_ref).abs().max().item()
        denom = g_ref.abs().max().item() or 1.0
        if err / denom > grad_rel_err:
            grad_rel_err, worst = err / denom, name
    print(
        f"RESULT fwd={fwd_err} grad={grad_rel_err} worst={worst or 'none'} "
        f"loss_ref={loss_ref} loss_full={loss_full} nparams={len(grads_ref)}"
    )
    torch.distributed.destroy_process_group()


def _server_venv_python() -> str:
    """The interpreter whose megatron resolves from the build clone."""
    clone = os.environ.get("BT_BUILD_CLONE", "/root/trainers")
    for component in ("server-megatron-bridge", "server"):
        candidate = os.path.join(clone, component, ".venv", "bin", "python")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"no server venv python under {clone}")


@requires_gpu
def test_glm5_next_mhc_full_recompute_grad_parity():
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
    match = re.search(r"RESULT fwd=([0-9.eE+-]+) grad=([0-9.eE+-]+)", output)
    assert match, f"no RESULT line in output:\n{output[-4000:]}"
    fwd_err, grad_rel_err = float(match.group(1)), float(match.group(2))
    # The checkpoint replay re-runs identical deterministic ops; tolerance
    # only absorbs kernel-atomic noise (MoE grouped GEMM, alltoall).
    assert fwd_err < 5e-2, f"full-recompute logits diverge: max abs err {fwd_err}"
    assert grad_rel_err < 5e-2, f"full-recompute grads diverge: max rel err {grad_rel_err}"


if __name__ == "__main__":
    _run()
