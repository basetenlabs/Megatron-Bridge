# Kimi K3

[Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) is a large sparse MoE model from Moonshot AI. Megatron Bridge supports the **language backbone** of the published multimodal checkpoint through the `KimiK3Bridge`.

```{note}
Support for this model is in progress. Conversion (HF → Megatron) and Megatron greedy inference are verified; strict full-checkpoint export, exact round-trip parity, and every training workflow are not. See [Known Limitations](#known-limitations) and, for the numbers NVIDIA recorded, their machine-readable [verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/kimi-k3/card.yaml) — it lives upstream and is not vendored into this fork.
```

## Supported Variants

Megatron Bridge supports checkpoints with the `KimiK3ForConditionalGeneration` architecture and the `kimi_k3` model type:

| Variant | HF Path |
|---------|---------|
| Kimi-K3 | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) |

Requires `transformers >= 4.56.2` and `--trust-remote-code`.

## Architecture Notes

K3 uses a heterogeneous attention schedule rather than a single attention type:

- **KDA (Kimi Delta Attention)** on the layers listed in the HF config's `linear_attn_config.kda_layers` — a gated delta-rule linear-attention block with short depthwise convolutions over Q/K/V, a low-rank forget gate, and a per-head beta projection. In the published 93-layer checkpoint, 69 layers are KDA.
- **No-RoPE MLA** on the remaining layers (`full_attn_layers`, 24 layers in the published checkpoint).

Both layer lists hold **1-indexed** global layer numbers and together partition `1..num_hidden_layers`.

Other notable properties:

- Latent MoE (`moe_latent_size`) with shared experts, grouped GEMM, and all-to-all token dispatch.
- AttnRes residual banks with a configurable block size (`attn_res_block_size`), which the pipeline payload carries between stages.
- Published routed-expert weights are MXFP4 (`uint8`-packed E2M1 values with UE8M0 scales) and are dequantized to BF16 on import.
- The published KDA `A_log` tensors carry 96 active entries plus 32 zero-only padding entries. Import validates the padding is all-zero and removes it; export restores it.
- Export of the language backbone passes the checkpoint's `vision_tower.*` and `mm_projector.*` tensors through unchanged.

## Conversion

```bash
# HF → Megatron
srun --ntasks-per-node=8 python \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model moonshotai/Kimi-K3 \
    --megatron-path /workspace/kimi-k3 \
    --torch-dtype bfloat16 \
    --tp 2 --pp 3 --ep 8 --etp 2 \
    --distributed-timeout-minutes 180 \
    --trust-remote-code
```

The full checkpoint needs a multi-node allocation — import was validated on 48 GB200 GPUs at TP2/PP3/EP8/ETP2.

For a fast local iteration loop, build a truncated proxy checkpoint with
[`examples/conversion/create_hf_toy_model.py`](../../../examples/conversion/create_hf_toy_model.py), which truncates the heterogeneous layer schedule and downloads only the safetensor shards the selected layers need.

## Inference

Megatron greedy generation was validated on 24 GB300 GPUs at TP1/PP3/EP8/ETP1. NVIDIA's [verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/kimi-k3/card.yaml) carries the exact command and the recorded deterministic completion; this fork does not vendor it.

## Training

No training recipe ships for K3 yet. Pretraining, SFT, and PEFT configs, checkpoint-resume validation, and performance tuning are pending.

## Known Limitations

- Strict full-checkpoint Megatron → HF export, HF reload, and exact round-trip parity are unverified.
- Full HF/Megatron forward-logit correlation is unverified. A four-layer proxy reached cosine similarity `0.9998` and Pearson correlation `0.9998`.
- Virtual pipeline parallelism (VPP) is not supported.
- KDA layers do not support context parallelism (`CP > 1`).
- Only the language backbone is covered. Native K3 vision/video modeling and multimodal inference are not implemented; export only preserves the published vision and projector tensors unchanged.
- The model has not been performance-tuned. Reported timings are sanity checks, not optimized throughput results.
- Full-weight training with `TP > 1` is not supported. K3 marks its replicated norms and AttnRes projections with `sum_gradients_across_tp_domain`, and Megatron-LM's gradient finalization has no consumer for that marker, so those replicas would drift apart across tensor-parallel ranks. Only the frozen-base LoRA path, which does not train them, is supported today.
- Export to a plain dtype (`--weight-dtype`) is unverified for the routed experts. The published experts are MXFP4, and the source shard map only carries `.weight_packed` / `.weight_scale` keys, so a dense `.weight` output key may have nowhere to go.
- Config-only export is rejected with an explicit error. Restoring the MXFP4 expert layout and the `A_log` padding needs the source checkpoint, which the config-only path does not open.
- Cached incremental inference is rejected with an explicit error. MLA never writes the KV cache and KDA never carries its recurrent state, so the engines would silently forget the prefix. Decoding that recomputes the full prefix each step still works.

## Related Implementation

- Bridge: [`src/megatron/bridge/models/kimi/kimi_k3_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/kimi/kimi_k3_bridge.py)
- Provider: [`src/megatron/bridge/models/kimi/kimi_k3_provider.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/kimi/kimi_k3_provider.py)
- Layer spec and KDA/MLA modules: [`src/megatron/bridge/models/kimi/kimi_k3_layers.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/kimi/kimi_k3_layers.py)
- Verification card, upstream only: [`examples/model_verification_cards/kimi-k3/card.yaml` in NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/kimi-k3/card.yaml)
