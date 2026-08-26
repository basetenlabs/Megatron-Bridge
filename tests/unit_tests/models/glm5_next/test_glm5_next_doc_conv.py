# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for ``_doc_aware_causal_conv`` (the CP-path conv).

Ground truth twice over: a naive per-document depthwise causal conv built
from ``F.conv1d`` (authoritative semantics, runs anywhere), and FLA's
``ShortConvolution`` — the module the cp=1 path actually uses and the one
the helper claims equivalence to (GPU + FLA only).
"""

import pytest
import torch
import torch.nn.functional as F

try:
    from megatron.bridge.models.glm5_next.modeling_glm5_next.kda import _doc_aware_causal_conv

    HAVE_BRIDGE = True
except ImportError:
    HAVE_BRIDGE = False

try:
    from fla.modules import ShortConvolution

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

requires_bridge = pytest.mark.skipif(not HAVE_BRIDGE, reason="megatron.bridge not importable")

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
    None,                       # one implicit document
    (0, 192),                   # one explicit document
    (0, 128, 192),              # the parity-test shape
    (0, 2, 5, 6, 38, 192),      # docs shorter than the kernel (2, 3, 1 tokens)
]


@requires_bridge
@pytest.mark.parametrize("cu", CU_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matches_naive_reference(cu, dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, weight = _case(192, 2, 8, dtype, device)
    cu_t = None if cu is None else torch.tensor(cu, dtype=torch.int32, device=device)
    got = _doc_aware_causal_conv(x, weight, cu_t)
    want = _naive_reference(x, weight, cu_t)
    assert got.dtype == x.dtype
    tol = 1e-6 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(got.float(), want.float(), rtol=tol, atol=tol)


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


@requires_bridge
@pytest.mark.skipif(
    not HAVE_FLA or not torch.cuda.is_available(), reason="needs FLA and a CUDA device"
)
@pytest.mark.parametrize("cu", [(0, 192), (0, 128, 192)])
def test_matches_fla_short_convolution(cu):
    """Equivalence with the module the cp=1 path actually runs."""
    device = "cuda"
    channels = 8
    x, weight = _case(192, 1, channels, torch.bfloat16, device)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)

    conv = ShortConvolution(channels, KSIZE, activation="silu", bias=False).to(device)
    with torch.no_grad():
        conv.weight.copy_(weight)

    # FLA takes [b, s, C] with varlen expressed through cu_seqlens at b=1.
    want, _ = conv(x.transpose(0, 1), output_final_state=False, cu_seqlens=cu_t)
    got = _doc_aware_causal_conv(x, weight, cu_t)
    torch.testing.assert_close(got.transpose(0, 1).float(), want.float(), rtol=2e-2, atol=2e-2)
