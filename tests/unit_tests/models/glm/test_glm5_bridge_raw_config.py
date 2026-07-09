# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Unit tests for the GLM-5 raw-config resolution used by the qk head-dim
workaround (transformers GlmMoeDsaConfig collapses qk_rope_head_dim, so
GLM5Bridge re-reads the on-disk config.json).
"""

import json

from megatron.bridge.models.glm_moe_dsa.glm5_bridge import _load_raw_hf_config


def test_local_snapshot_dir_reads_config_directly(tmp_path):
    dims = {"qk_nope_head_dim": 128, "qk_rope_head_dim": 64}
    (tmp_path / "config.json").write_text(json.dumps(dims))
    assert _load_raw_hf_config(str(tmp_path)) == dims


def test_hub_id_resolves_through_hub_cache(tmp_path, monkeypatch):
    cached = tmp_path / "config.json"
    cached.write_text(json.dumps({"qk_rope_head_dim": 64}))
    calls = {}

    def fake_download(repo_id, filename):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        return str(cached)

    monkeypatch.setattr(
        "megatron.bridge.models.glm_moe_dsa.glm5_bridge.hf_hub_download",
        fake_download,
    )
    assert _load_raw_hf_config("zai-org/GLM-5.2-FP8") == {"qk_rope_head_dim": 64}
    assert calls == {"repo_id": "zai-org/GLM-5.2-FP8", "filename": "config.json"}


def test_unresolvable_name_returns_none(monkeypatch):
    def fake_download(repo_id, filename):
        raise OSError("offline and not in the hub cache")

    monkeypatch.setattr(
        "megatron.bridge.models.glm_moe_dsa.glm5_bridge.hf_hub_download",
        fake_download,
    )
    assert _load_raw_hf_config("zai-org/GLM-5.2-FP8") is None
