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

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.common.vision_splice import splice_features
from megatron.bridge.models.kimi.kimi_k3_vision import KimiK3VisionProjector, KimiK3VisionTower, _rope_2d


def _vision_config(**overrides) -> SimpleNamespace:
    """The released K3 tower layout at toy width."""
    config = dict(
        patch_size=14,
        init_pos_emb_height=8,
        init_pos_emb_width=8,
        init_pos_emb_time=4,
        pos_emb_type="divided_fixed",
        pos_emb_interpolation_mode="bilinear",
        vt_num_attention_heads=2,
        vt_num_hidden_layers=2,
        vt_hidden_size=32,
        vt_intermediate_size=64,
        qkv_hidden_size=48,
        merge_kernel_size=[2, 2],
        merge_type="sd2_tpool",
        norm_type="rmsnorm",
        mlp_type="mlp2",
        attn_bias=False,
        linear_bias=False,
        patch_embed_proj_bias=False,
        mm_projector_type="patchmergerv2",
        mm_hidden_size=32,
        text_hidden_size=40,
        projector_ln_eps=1e-5,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def _pixels(grid: list[list[int]]) -> torch.Tensor:
    return torch.randn(sum(t * h * w for t, h, w in grid), 3, 14, 14)


def test_parameter_names_match_checkpoint() -> None:
    """Names equal the released checkpoint's, so one wildcard mapping covers the tower."""
    tower = KimiK3VisionTower(_vision_config())
    projector = KimiK3VisionProjector(_vision_config())
    block = {"norm0", "norm1", "wqkv", "wo", "mlp.fc0", "mlp.fc1"}

    assert set(tower.state_dict()) == {
        "patch_embed.proj.weight",
        "patch_embed.pos_emb.weight",
        "encoder.final_layernorm.weight",
        *(f"encoder.blocks.{i}.{name}.weight" for i in range(2) for name in block),
    }
    assert set(projector.state_dict()) == {"proj.0.weight", "proj.2.weight", "post_norm.weight"}


def test_one_feature_row_per_merged_patch_in_image_order() -> None:
    config = _vision_config()
    tower, projector = KimiK3VisionTower(config), KimiK3VisionProjector(config)
    grid = [[1, 4, 6], [1, 2, 2]]

    merged = tower(_pixels(grid), torch.tensor(grid))
    features = projector(merged)

    assert [m.shape for m in merged] == [torch.Size([6, 4, 32]), torch.Size([1, 4, 32])]
    assert features.shape == (7, 40)


def test_images_do_not_attend_to_each_other() -> None:
    torch.manual_seed(0)
    config = _vision_config()
    tower = KimiK3VisionTower(config)
    grid = [[1, 4, 4], [1, 2, 4]]
    pixels = _pixels(grid)
    changed = pixels.clone()
    changed[16:] = torch.randn_like(changed[16:])

    with torch.no_grad():
        first, _ = tower(pixels, torch.tensor(grid))
        first_again, _ = tower(changed, torch.tensor(grid))

    torch.testing.assert_close(first, first_again, rtol=0, atol=0)


def test_rope_matches_reference_table_layout() -> None:
    """Per-grid angles equal slicing the reference's full-size row-major table."""
    head_dim, max_side = 16, 8
    freqs = 1.0 / 10000.0 ** (torch.arange(0, head_dim, 4)[: head_dim // 4].float() / head_dim)
    flat = torch.arange(max_side * max_side).float()
    x_cis = torch.polar(torch.ones(flat.numel(), freqs.numel()), torch.outer(flat % max_side, freqs))
    y_cis = torch.polar(torch.ones(flat.numel(), freqs.numel()), torch.outer(flat // max_side, freqs))
    table = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1).reshape(max_side, max_side, -1)
    grid = [[1, 3, 5], [2, 2, 2]]

    want = torch.cat([table[:h, :w].reshape(-1, head_dim // 2).repeat(t, 1) for t, h, w in grid])

    torch.testing.assert_close(_rope_2d(grid, head_dim, torch.device("cpu")), want, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,value",
    [("mm_projector_type", "patchmerger"), ("norm_type", "layernorm"), ("merge_type", "sd2")],
)
def test_unsupported_layouts_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        KimiK3VisionTower(_vision_config(**{field: value}))


def test_missing_layout_field_is_rejected() -> None:
    config = _vision_config()
    del config.merge_type
    with pytest.raises(ValueError, match="merge_type=None"):
        KimiK3VisionTower(config)


def test_splice_writes_features_in_placeholder_order() -> None:
    token = 7
    input_ids = torch.tensor([[1, token, token, 2], [token, 3, 4, 5]])
    embeds = torch.zeros(2, 4, 3)
    features = torch.arange(9, dtype=torch.float32).view(3, 3)

    out = splice_features(
        embeds, input_ids=input_ids, features=features, token_id=token, packed_seq_params=None, cp_group=None
    )

    torch.testing.assert_close(out[input_ids == token], features)
    assert torch.count_nonzero(out[input_ids != token]) == 0


def test_splice_rejects_feature_count_mismatch() -> None:
    input_ids = torch.tensor([[7, 7, 1]])
    with pytest.raises(ValueError, match="2 placeholder"):
        splice_features(
            torch.zeros(1, 3, 3),
            input_ids=input_ids,
            features=torch.zeros(3, 3),
            token_id=7,
            packed_seq_params=None,
            cp_group=None,
        )


def test_video_grids_are_rejected() -> None:
    """Frame pooling would emit fewer rows than the placeholders expanded upstream."""
    tower = KimiK3VisionTower(_vision_config())
    grid = [[2, 2, 2]]
    with pytest.raises(ValueError, match="images only"):
        tower(_pixels(grid), torch.tensor(grid))
