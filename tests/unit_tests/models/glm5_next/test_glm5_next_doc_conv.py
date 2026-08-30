# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for ``_doc_aware_causal_conv`` (the CP-path conv).

The public helper dispatches to FLA's fused varlen conv on CUDA (the same
Triton kernel the cp=1 ``ShortConvolution`` path runs) with a pure-torch
fallback elsewhere. Ground truth is a naive per-document depthwise causal
conv built from ``F.conv1d`` (authoritative semantics, runs anywhere).

On CPU these tests exercise the fallback; on a CUDA box the same tests
exercise the FLA path (batch=1 for varlen cases — an FLA contract the
helper enforces), plus FLA-specific pins: guard errors, bitwise agreement
with ``ShortConvolution``, fp32-weight fidelity, and gradient flow through
a CP channel-slice view of the full parameter.
"""

import pytest
import torch
import torch.nn.functional as F


try:
    from megatron.bridge.models.glm5_next.modeling_glm5_next.kda import (
        _doc_aware_causal_conv,
        _doc_aware_causal_conv_torch,
    )

    HAVE_BRIDGE = True
except ImportError:
    HAVE_BRIDGE = False

try:
    from fla.modules import ShortConvolution

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

requires_bridge = pytest.mark.skipif(not HAVE_BRIDGE, reason="megatron.bridge not importable")
requires_gpu_fla = pytest.mark.skipif(
    not HAVE_FLA or not torch.cuda.is_available(), reason="needs FLA and a CUDA device"
)

KSIZE = 4


def _naive_reference(x: torch.Tensor, weight: torch.Tensor, cu_seqlens) -> torch.Tensor:
    """Per-document depthwise causal conv + silu, straight from F.conv1d."""
    w = weight.squeeze(1).to(torch.float32)  # [C, K]
    channels, ksize = w.shape
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    if cu_seqlens is None:
        bounds = [(0, x.shape[0])]
    else:
        cu = [int(v) for v in cu_seqlens]
        bounds = list(zip(cu[:-1], cu[1:]))
    for start, stop in bounds:
        if stop <= start:
            continue
        seg = x[start:stop].float().permute(1, 2, 0)  # [b, C, t]
        seg = F.pad(seg, (ksize - 1, 0))
        y = F.conv1d(seg, w.unsqueeze(1), groups=channels)
        out[start:stop] = y.permute(2, 0, 1)
    return F.silu(out).to(x.dtype)


def _case(seq_len, batch, channels, dtype, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(seq_len, batch, channels, generator=gen).to(device=device, dtype=dtype)
    weight = torch.randn(channels, 1, KSIZE, generator=gen).to(device=device, dtype=torch.float32)
    return x, weight


CU_CASES = [
    None,  # one implicit document
    (0, 192),  # one explicit document
    (0, 128, 192),  # the parity-test shape
    (0, 2, 5, 6, 38, 192),  # docs shorter than the kernel (2, 3, 1 tokens)
]


@requires_bridge
@pytest.mark.parametrize("cu", CU_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matches_naive_reference(cu, dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # batch=1: the FLA varlen path is batch==1 by contract (the production
    # THD path always is); batch>1 coverage lives in the fallback tests.
    x, weight = _case(192, 1, 8, dtype, device)
    cu_t = None if cu is None else torch.tensor(cu, dtype=torch.int32, device=device)
    got = _doc_aware_causal_conv(x, weight, cu_t)
    want = _naive_reference(x, weight, cu_t)
    assert got.dtype == x.dtype
    # bf16 tolerance is deliberately tight: both paths accumulate in fp32 and
    # round once on store, so anything looser would mask a weight downcast.
    tol = 1e-6 if dtype == torch.float32 else 4e-3
    torch.testing.assert_close(got.float(), want.float(), rtol=tol, atol=tol)


@requires_bridge
@pytest.mark.parametrize("cu", CU_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_torch_fallback_matches_naive_reference(cu, dtype):
    """The fallback also supports batch>1 with packed documents."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, weight = _case(192, 2, 8, dtype, device)
    cu_t = None if cu is None else torch.tensor(cu, dtype=torch.int32, device=device)
    got = _doc_aware_causal_conv_torch(x, weight, cu_t)
    want = _naive_reference(x, weight, cu_t)
    tol = 1e-6 if dtype == torch.float32 else 4e-3
    torch.testing.assert_close(got.float(), want.float(), rtol=tol, atol=tol)


@requires_gpu_fla
def test_bf16_large_realistic():
    S, C = 8192, 2048  # 32 local heads x head_dim 64
    x, weight = _case(S, 1, C, torch.bfloat16, "cuda")
    # doc lengths deliberately not multiples of the kernel's 64-token chunks
    cu = torch.tensor((0, 1337, 4096, 6000, 8192), dtype=torch.int32, device="cuda")
    got = _doc_aware_causal_conv(x, weight, cu)
    want = _naive_reference(x, weight, cu)
    # 1-ulp bf16 agreement: both sides accumulate in fp32 and round once on
    # store, so at 16.7M samples a handful of rounding-boundary straddles is
    # the ceiling (measured: 106 elements, all within 1 true ulp).
    torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=2e-3)


@requires_gpu_fla
def test_bf16_large_grads_no_worse_than_fallback():
    """fp64-anchored: FLA's backward must not be less accurate than the
    fallback's. Elementwise comparison is the wrong criterion here — the
    error of BOTH paths is dominated by the shared bf16 quantization of
    dL/dy, which explodes relative metrics at cancellation points."""
    S, C = 8192, 2048
    x, weight = _case(S, 1, C, torch.bfloat16, "cuda")
    cu = torch.tensor((0, 1337, 4096, 6000, 8192), dtype=torch.int32, device="cuda")

    def _grads(fn):
        xa = x.clone().requires_grad_(True)
        wa = weight.clone().requires_grad_(True)
        fn(xa, wa, cu).square().sum().backward()
        return xa.grad.double(), wa.grad.double()

    def _ref64():
        w64 = weight.squeeze(1).double()
        channels, ksize = w64.shape
        x64 = x.clone().double().requires_grad_(True)
        w64 = w64.clone().requires_grad_(True)
        out = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
        for start, stop in zip([int(v) for v in cu][:-1], [int(v) for v in cu][1:]):
            seg = F.pad(x64[start:stop].permute(1, 2, 0), (ksize - 1, 0))
            out[start:stop] = F.conv1d(seg, w64.unsqueeze(1), groups=channels).permute(2, 0, 1)
        F.silu(out).square().sum().backward()
        return x64.grad, w64.grad.unsqueeze(1)

    dx64, dw64 = _ref64()
    dx_fla, dw_fla = _grads(_doc_aware_causal_conv)
    dx_fb, dw_fb = _grads(_doc_aware_causal_conv_torch)

    for got, base, oracle in ((dx_fla, dx_fb, dx64), (dw_fla, dw_fb, dw64)):
        err_got = (got - oracle).abs()
        err_base = (base - oracle).abs()
        assert err_got.pow(2).mean() <= err_base.pow(2).mean() * 1.1**2
        assert err_got.max() <= err_base.max() * 1.2 + 1e-6


@requires_gpu_fla
def test_padding_tail_rejected_on_fla_path():
    """FLA leaves rows past cu_seqlens[-1] uninitialized; the helper refuses."""
    x, weight = _case(192, 1, 8, torch.bfloat16, "cuda")
    cu = torch.tensor((0, 64, 160), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="cu_seqlens"):
        _doc_aware_causal_conv(x, weight, cu)


@requires_gpu_fla
def test_varlen_batch_gt1_rejected_on_fla_path():
    """FLA varlen silently reads only batch 0; the helper refuses batch>1."""
    x, weight = _case(192, 2, 8, torch.bfloat16, "cuda")
    cu = torch.tensor((0, 128, 192), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="batch"):
        _doc_aware_causal_conv(x, weight, cu)


@requires_bridge
def test_document_isolation():
    """Perturbing document 0 must not change any output token of document 1."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, weight = _case(64, 1, 4, torch.float32, device)
    cu = torch.tensor((0, 32, 64), dtype=torch.int32, device=device)
    base = _doc_aware_causal_conv(x, weight, cu)
    x2 = x.clone()
    x2[:32] += 100.0
    perturbed = _doc_aware_causal_conv(x2, weight, cu)
    assert not torch.equal(perturbed[:32], base[:32])
    torch.testing.assert_close(perturbed[32:], base[32:], rtol=0, atol=0)


@requires_bridge
def test_gradients_match_naive_reference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, weight = _case(48, 1, 4, torch.float32, device)
    cu = torch.tensor((0, 12, 48), dtype=torch.int32, device=device)

    x_a = x.clone().requires_grad_(True)
    w_a = weight.clone().requires_grad_(True)
    _doc_aware_causal_conv(x_a, w_a, cu).square().sum().backward()

    x_b = x.clone().requires_grad_(True)
    w_b = weight.clone().requires_grad_(True)
    _naive_reference(x_b, w_b, cu).square().sum().backward()

    torch.testing.assert_close(x_a.grad, x_b.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(w_a.grad, w_b.grad, rtol=1e-5, atol=1e-5)


@requires_gpu_fla
def test_grad_flows_into_cp_slice_of_full_param():
    """Production pattern: weight is a CP channel-slice VIEW of the full parameter."""
    from megatron.core.ssm.gated_delta_net.common import get_parameter_local_cp

    class _FakeCPGroup:
        def __init__(self, rank, size):
            self._rank, self._size = rank, size

        def size(self):
            return self._size

        def rank(self):
            return self._rank

    c_full, cp, rank = 16, 2, 1
    x, _ = _case(192, 1, c_full // cp, torch.float32, "cuda")
    w_full = torch.randn(c_full, 1, KSIZE, device="cuda", dtype=torch.float32, requires_grad=True)
    cu = torch.tensor((0, 128, 192), dtype=torch.int32, device="cuda")

    w_local = get_parameter_local_cp(w_full, dim=0, cp_group=_FakeCPGroup(rank, cp))
    _doc_aware_causal_conv(x, w_local, cu).square().sum().backward()

    w_ref = w_full.detach()[8:16].clone().requires_grad_(True)
    _naive_reference(x.detach().clone(), w_ref, cu).square().sum().backward()

    assert torch.all(w_full.grad[:8] == 0), "grad leaked into the other rank's slice"
    torch.testing.assert_close(w_full.grad[8:16], w_ref.grad, rtol=1e-5, atol=1e-5)


@requires_gpu_fla
def test_fp32_weight_actually_used():
    """Canary: a weight delta below bf16 resolution must still move the output."""
    w = torch.full((8, 1, KSIZE), 0.25, device="cuda", dtype=torch.float32)
    delta = 2.0**-12  # below bf16 resolution at 0.25
    ones = torch.ones(64, 1, 8, device="cuda", dtype=torch.float32)
    y_hi = _doc_aware_causal_conv(ones, w + delta, None)
    y_lo = _doc_aware_causal_conv(ones, w, None)
    assert (y_hi - y_lo).abs().max() > delta


@requires_gpu_fla
@pytest.mark.parametrize("cu", [(0, 192), (0, 128, 192)])
def test_matches_fla_short_convolution_bitwise(cu):
    """cp=1 runs ShortConvolution; cp>1 runs the functional call on a channel
    slice. Both are the same Triton kernel now, so demand bitwise equality —
    a tolerance relaxation here means cp1 and cp>1 diverge inside KDA and
    needs explicit sign-off."""
    device = "cuda"
    c_full, cp = 16, 2
    x, weight = _case(192, 1, c_full, torch.bfloat16, device)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)

    conv = ShortConvolution(c_full, KSIZE, activation="silu", bias=False).to(device)
    conv.weight.data = conv.weight.data.float()  # KimiK3 fp32 policy
    with torch.no_grad():
        conv.weight.copy_(weight)

    want, _ = conv(x.transpose(0, 1), output_final_state=False, cu_seqlens=cu_t)  # [1, s, C]
    for rank in range(cp):
        sl = slice(rank * c_full // cp, (rank + 1) * c_full // cp)
        got = _doc_aware_causal_conv(x[:, :, sl], conv.weight[sl], cu_t)
        assert torch.equal(got.transpose(0, 1), want[:, :, sl]), f"rank {rank} slice diverges"


@requires_gpu_fla
@pytest.mark.parametrize("cu", CU_CASES)
def test_fla_path_matches_torch_fallback(cu):
    """The two dispatch targets must agree wherever both are defined."""
    x, weight = _case(192, 1, 8, torch.bfloat16, "cuda")
    cu_t = None if cu is None else torch.tensor(cu, dtype=torch.int32, device="cuda")
    got = _doc_aware_causal_conv(x, weight, cu_t)
    want = _doc_aware_causal_conv_torch(x, weight, cu_t)
    torch.testing.assert_close(got.float(), want.float(), rtol=4e-3, atol=4e-3)
