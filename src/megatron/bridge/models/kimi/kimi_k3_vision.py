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

"""Kimi K3 vision tower (MoonViT3d) and patch-merger projector in plain PyTorch.

Written natively rather than loaded from the checkpoint's remote code, which imports
transformers 4.x internals that 5.x removed and needs ``flash_attn`` for anything but
an O(seq^2) eager mask. Submodule names match the checkpoint one for one, so the whole
tower maps with a single wildcard.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn


if TYPE_CHECKING:
    from transformers import PretrainedConfig


# The released K3 tower. Other values are rejected rather than half-supported.
_SUPPORTED = {
    "pos_emb_type": "divided_fixed",
    "norm_type": "rmsnorm",
    "mlp_type": "mlp2",
    "merge_type": "sd2_tpool",
    "mm_projector_type": "patchmergerv2",
}
_ROPE_MAX_GRID = 512
_ROPE_THETA = 10000.0


def _check_supported(vision_config: "PretrainedConfig") -> None:
    for field, expected in _SUPPORTED.items():
        actual = getattr(vision_config, field)
        if actual != expected:
            raise ValueError(f"Kimi K3 vision: unsupported {field}={actual!r}; only {expected!r} is implemented")


def _sincos_1d(dim: int, length: int) -> Tensor:
    """Fixed temporal embedding, [length, dim] as [sin | cos]."""
    omega = 1.0 / _ROPE_THETA ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0))
    angles = torch.outer(torch.arange(length, dtype=torch.float32), omega)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class _DividedPosEmb(nn.Module):
    """Learnable 2D grid embedding, bilinearly resized per image, plus fixed time embedding."""

    def __init__(self, height: int, width: int, num_frames: int, dim: int, interpolation_mode: str) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        nn.init.normal_(self.weight)

    def forward(self, x: Tensor, grid: list[list[int]]) -> Tensor:
        pos_embs = []
        for t, h, w in grid:
            if t > self.num_frames:
                raise ValueError(f"Kimi K3 vision: {t} frames exceeds the {self.num_frames} the tower embeds")
            if (h, w) == tuple(self.weight.shape[:-1]):
                pos_2d = self.weight.flatten(end_dim=1)
            else:
                pos_2d = (
                    F.interpolate(
                        self.weight.permute(2, 0, 1).unsqueeze(0),
                        size=(h, w),
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .flatten(end_dim=1)
                )
            if t == 1:
                pos_embs.append(pos_2d)
            else:
                # Fixed, not a buffer: a meta-device build would leave a buffer unmaterialized.
                time_2d = _sincos_1d(pos_2d.size(-1), t).to(pos_2d).unsqueeze(1)
                pos_embs.append((pos_2d.unsqueeze(0) + time_2d).flatten(end_dim=1))
        return x + torch.cat(pos_embs)


class _PatchEmbed(nn.Module):
    def __init__(self, config: "PretrainedConfig") -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            config.vt_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=config.patch_embed_proj_bias,
        )
        self.pos_emb = _DividedPosEmb(
            config.init_pos_emb_height,
            config.init_pos_emb_width,
            config.init_pos_emb_time,
            config.vt_hidden_size,
            config.pos_emb_interpolation_mode,
        )

    def forward(self, pixel_values: Tensor, grid: list[list[int]]) -> Tensor:
        x = self.proj(pixel_values).view(pixel_values.size(0), -1)
        return self.pos_emb(x, grid)


def _rope_2d(grid: list[list[int]], head_dim: int, device: torch.device) -> Tensor:
    """cis(angle) per patch, [patches, head_dim // 2], pairs interleaved as (x_i, y_i)."""
    freqs = 1.0 / _ROPE_THETA ** (torch.arange(0, head_dim, 4, device=device)[: head_dim // 4].float() / head_dim)
    out = []
    for t, h, w in grid:
        if not (1 <= h <= _ROPE_MAX_GRID and 1 <= w <= _ROPE_MAX_GRID):
            raise ValueError(f"Kimi K3 vision: grid {h}x{w} exceeds the {_ROPE_MAX_GRID}x{_ROPE_MAX_GRID} rope table")
        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        x_cis = torch.polar(torch.ones(h * w, freqs.numel(), device=device), torch.outer(x.flatten(), freqs))
        y_cis = torch.polar(torch.ones(h * w, freqs.numel(), device=device), torch.outer(y.flatten(), freqs))
        out.append(torch.stack([x_cis, y_cis], dim=-1).flatten(start_dim=1).repeat(t, 1))
    return torch.cat(out)


def _apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Rotate [patches, heads, head_dim] in fp32, as the reference does."""
    rotated = torch.view_as_complex(x.float().unflatten(-1, (-1, 2))) * freqs_cis.unsqueeze(-2)
    return torch.view_as_real(rotated).flatten(-2).type_as(x)


class _MLP2(nn.Module):
    def __init__(self, hidden: int, intermediate: int, bias: bool) -> None:
        super().__init__()
        self.fc0 = nn.Linear(hidden, intermediate, bias=bias)
        self.fc1 = nn.Linear(intermediate, hidden, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc1(F.gelu(self.fc0(x), approximate="tanh"))


class _EncoderLayer(nn.Module):
    def __init__(self, config: "PretrainedConfig") -> None:
        super().__init__()
        hidden = config.vt_hidden_size
        qkv_hidden = config.qkv_hidden_size or hidden
        self.num_heads = config.vt_num_attention_heads
        self.head_dim = qkv_hidden // self.num_heads
        # Default eps (dtype epsilon), as in the reference tower and vLLM's.
        self.norm0 = nn.RMSNorm(hidden)
        self.norm1 = nn.RMSNorm(hidden)
        self.wqkv = nn.Linear(hidden, 3 * qkv_hidden, bias=config.attn_bias)
        self.wo = nn.Linear(qkv_hidden, hidden, bias=config.attn_bias)
        self.mlp = _MLP2(hidden, config.vt_intermediate_size, config.linear_bias)

    def _attention(self, x: Tensor, cu_seqlens: list[int], freqs_cis: Tensor) -> Tensor:
        q, k, v = self.wqkv(x).unflatten(-1, (3, self.num_heads, self.head_dim)).unbind(dim=-3)
        q, k = _apply_rope(q, freqs_cis), _apply_rope(k, freqs_cis)
        # Bidirectional within each image, nothing across images: one SDPA per image
        # keeps memory linear in the image count instead of quadratic in the batch.
        outs = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            q_i, k_i, v_i = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            outs.append(F.scaled_dot_product_attention(q_i, k_i, v_i).squeeze(0).transpose(0, 1))
        return self.wo(torch.cat(outs).flatten(start_dim=1))

    def forward(self, x: Tensor, cu_seqlens: list[int], freqs_cis: Tensor) -> Tensor:
        x = x + self._attention(self.norm0(x), cu_seqlens, freqs_cis)
        return x + self.mlp(self.norm1(x))


class _Encoder(nn.Module):
    def __init__(self, config: "PretrainedConfig") -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_EncoderLayer(config) for _ in range(config.vt_num_hidden_layers))
        self.final_layernorm = nn.RMSNorm(config.vt_hidden_size)

    def forward(self, x: Tensor, grid: list[list[int]]) -> Tensor:
        freqs_cis = _rope_2d(grid, self.blocks[0].head_dim, x.device)
        cu_seqlens = [0]
        for t, h, w in grid:
            cu_seqlens.append(cu_seqlens[-1] + t * h * w)
        for block in self.blocks:
            x = block(x, cu_seqlens, freqs_cis)
        return self.final_layernorm(x)


class KimiK3VisionTower(nn.Module):
    """MoonViT3d: patch embedding, bidirectional encoder, 2x2 spatial merge with temporal mean."""

    def __init__(self, vision_config: "PretrainedConfig") -> None:
        super().__init__()
        _check_supported(vision_config)
        self.merge_kernel_size = tuple(vision_config.merge_kernel_size)
        self.patch_embed = _PatchEmbed(vision_config)
        self.encoder = _Encoder(vision_config)

    def forward(self, pixel_values: Tensor, grid_thws: Tensor) -> list[Tensor]:
        """[patches, 3, p, p] -> one [merged_tokens, kh * kw, hidden] tensor per image."""
        grid = grid_thws.tolist()  # the tower's only host sync
        x = self.encoder(self.patch_embed(pixel_values, grid), grid)
        kh, kw = self.merge_kernel_size
        outputs = []
        offset = 0
        for t, h, w in grid:
            seq = x[offset : offset + t * h * w].view(t, h // kh, kh, w // kw, kw, -1)
            outputs.append(seq.permute(0, 1, 3, 2, 4, 5).mean(dim=0).reshape((h // kh) * (w // kw), kh * kw, -1))
            offset += t * h * w
        return outputs


class KimiK3VisionProjector(nn.Module):
    """Patch merger v2: concat each 2x2 group, two-layer MLP, then RMSNorm into the text width."""

    def __init__(self, vision_config: "PretrainedConfig") -> None:
        super().__init__()
        _check_supported(vision_config)
        kh, kw = vision_config.merge_kernel_size
        merged = vision_config.mm_hidden_size * kh * kw
        self.proj = nn.Sequential(
            nn.Linear(merged, merged, bias=False),
            nn.GELU(),
            nn.Linear(merged, vision_config.text_hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(vision_config.text_hidden_size, eps=vision_config.projector_ln_eps)

    def forward(self, image_features: list[Tensor]) -> Tensor:
        """Per-image merged patches -> [total_tokens, text_hidden], images in order."""
        x = torch.cat([feature.flatten(start_dim=1) for feature in image_features])
        return self.post_norm(self.proj(x))
