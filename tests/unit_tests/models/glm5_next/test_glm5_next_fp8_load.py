# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import weakref

import pytest
import torch

from megatron.bridge.models.glm5_next.glm5_next_bridge import Glm5NextBridge


pytestmark = [pytest.mark.unit, pytest.mark.gpu, pytest.mark.run_only_on("GPU")]


def test_existing_fp8_expert_load_dequantizes_then_requantizes_with_bounded_memory(record_property):
    """Characterize the current GLM5-next FP8 expert checkpoint load path."""
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("TE blockwise FP8 requires compute capability 9.0 or newer")
    if torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 9):
        pytest.skip("TE blockwise FP8 requires CUDA 12.9 or newer")

    # Import TE only after the GPU requirements are known. Importing the public
    # package first registers the extension module used by the quantizer.
    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )

    device = torch.device("cuda")
    block_size = 128
    weight_shape = (2 * block_size, 2 * block_size)
    temporary_nbytes = weight_shape[0] * weight_shape[1] * torch.bfloat16.itemsize

    # Each block spans the E4M3 range, while deliberately non-power-of-two
    # inverse scales expose the scale drift introduced by Blackwell's current
    # power-of-two block-scaling emulation.
    fp8_block = torch.linspace(
        -448,
        448,
        block_size * block_size,
        dtype=torch.float32,
        device=device,
    ).reshape(block_size, block_size)
    source_weight = fp8_block.to(torch.float8_e4m3fn).repeat(2, 2)
    source_scale_inv = torch.tensor(
        [[0.13, 0.37], [0.91, 1.73]],
        dtype=torch.float32,
        device=device,
    )

    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=True,
        block_scaling_dim=2,
    )
    destination = quantizer.make_empty(weight_shape, dtype=torch.bfloat16, device=device)
    assert isinstance(destination, Float8BlockwiseQTensor)

    # Warm up the copy kernel so the measured peak describes the load rather
    # than one-time CUDA module initialization.
    warmup = torch.zeros(weight_shape, dtype=torch.bfloat16, device=device)
    destination.copy_(warmup)
    del warmup
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    param_name = "model.language_model.layers.3.mlp.experts.0.down_proj.weight"
    state_dict = {
        param_name: source_weight,
        f"{param_name}_scale_inv": source_scale_inv,
    }
    source_dequant_reference = Glm5NextBridge().maybe_modify_loaded_hf_weight(param_name, state_dict)
    torch.cuda.synchronize()
    allocated_with_temporary = torch.cuda.memory_allocated()

    assert source_dequant_reference.dtype is torch.bfloat16
    assert source_dequant_reference.shape == weight_shape
    assert allocated_with_temporary - baseline_allocated == temporary_nbytes

    with torch.no_grad():
        # This is the same copy operation used by ModelBridge.load_weights_hf_to_megatron.
        destination.copy_(source_dequant_reference)
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()

    assert isinstance(destination, Float8BlockwiseQTensor)
    assert destination._rowwise_data.dtype is torch.uint8
    assert destination._rowwise_scale_inv.dtype is torch.float32
    assert destination._columnwise_data is None
    assert destination._columnwise_scale_inv is None
    assert getattr(destination, "_high_precision_init_val", None) is None

    destination_dequantized = destination.dequantize(dtype=torch.bfloat16)
    absolute_error = (destination_dequantized.float() - source_dequant_reference.float()).abs()
    normalized_error = absolute_error / source_dequant_reference.float().abs().clamp_min(1.0)
    active_destination_scales = destination._rowwise_scale_inv[:2, :2]
    relative_scale_drift = (active_destination_scales - source_scale_inv).abs() / source_scale_inv

    max_absolute_error = absolute_error.max().item()
    rms_error = absolute_error.square().mean().sqrt().item()
    max_normalized_error = normalized_error.max().item()
    max_relative_scale_drift = relative_scale_drift.max().item()
    peak_load_bytes = peak_allocated - baseline_allocated

    assert max_absolute_error <= 32.0
    assert rms_error <= 8.0
    assert max_normalized_error <= 0.06
    assert 0.0 < max_relative_scale_drift <= 1.0
    assert temporary_nbytes <= peak_load_bytes <= 2 * 1024 * 1024

    record_property("cuda_device_name", torch.cuda.get_device_name())
    record_property("cuda_compute_capability", ".".join(map(str, torch.cuda.get_device_capability())))
    record_property("bf16_temporary_bytes", temporary_nbytes)
    record_property("peak_load_bytes", peak_load_bytes)
    record_property("max_absolute_error", max_absolute_error)
    record_property("rms_error", rms_error)
    record_property("max_normalized_error", max_normalized_error)
    record_property("max_relative_scale_drift", max_relative_scale_drift)

    temporary_ref = weakref.ref(source_dequant_reference)
    del (
        source_dequant_reference,
        destination_dequantized,
        absolute_error,
        normalized_error,
        active_destination_scales,
        relative_scale_drift,
    )
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    assert temporary_ref() is None
    assert torch.cuda.memory_allocated() == baseline_allocated
