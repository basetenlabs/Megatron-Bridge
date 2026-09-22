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

from pathlib import Path

import transformers.dynamic_module_utils as _hf_dyn

import megatron.bridge.models.conversion.transformers_compat  # noqa: F401  (installs the check_imports hook)


def _write_chain(tmp_path):
    (tmp_path / "modeling_entry.py").write_text("from .modeling_mid import Mid\n")
    (tmp_path / "modeling_mid.py").write_text("from .configuration_leaf import Leaf\n")
    (tmp_path / "configuration_leaf.py").write_text("LEAF = 1\n")


def test_full_local_dir_returns_transitive_closure(tmp_path):
    _write_chain(tmp_path)
    assert _hf_dyn.check_imports(tmp_path / "modeling_entry.py") == [
        "configuration_leaf",
        "modeling_mid",
    ]


def test_cold_cache_names_missing_sibling_without_crashing(tmp_path):
    # Hub path on a cold cache: only the entry file has been downloaded when
    # check_imports runs. The hook must name the missing sibling (so the
    # caller downloads it) instead of raising FileNotFoundError.
    _write_chain(tmp_path)
    (tmp_path / "modeling_mid.py").unlink()
    (tmp_path / "configuration_leaf.py").unlink()
    assert _hf_dyn.check_imports(tmp_path / "modeling_entry.py") == ["modeling_mid"]


def test_reentry_after_download_discovers_deeper_imports(tmp_path):
    # get_cached_module_file re-enters check_imports on each file it
    # downloads; the second hop must surface the leaf.
    _write_chain(tmp_path)
    (tmp_path / "configuration_leaf.py").unlink()
    assert _hf_dyn.check_imports(tmp_path / "modeling_mid.py") == ["configuration_leaf"]


def test_circular_imports_terminate(tmp_path):
    (tmp_path / "a.py").write_text("from .b import B\n")
    (tmp_path / "b.py").write_text("from .a import A\n")
    assert _hf_dyn.check_imports(tmp_path / "a.py") == ["a", "b"]


def test_hub_cold_cache_materializes_full_chain(tmp_path, monkeypatch):
    # End-to-end through the real get_cached_module_file with only the
    # network mocked: a cold cache must still end with the complete import
    # chain in the dynamic-module cache — an incomplete cache is exactly the
    # FileNotFoundError-at-import-time failure this hook was added to fix.
    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "modeling_entry.py").write_text("from .modeling_mid import Mid\n")
    (hub / "modeling_mid.py").write_text("from .configuration_leaf import Leaf\n")
    (hub / "configuration_leaf.py").write_text("LEAF = 1\n")

    snapshot = tmp_path / "snapshot"  # starts empty: cold HF cache
    snapshot.mkdir()

    def fake_cached_file(repo, filename, **kwargs):
        dest = snapshot / filename
        if not dest.exists():
            dest.write_bytes((hub / filename).read_bytes())
        return str(dest)

    monkeypatch.setattr(_hf_dyn, "cached_file", fake_cached_file)
    monkeypatch.setattr(_hf_dyn, "try_to_load_from_cache", lambda *a, **k: None)
    modcache = tmp_path / "modcache"
    monkeypatch.setattr(_hf_dyn, "HF_MODULES_CACHE", str(modcache))
    _hf_dyn.create_dynamic_module(_hf_dyn.TRANSFORMERS_DYNAMIC_MODULE_NAME)

    _hf_dyn.get_cached_module_file("fake/repo", "modeling_entry.py", _commit_hash="deadbeef")

    copied = sorted(p.name for p in modcache.rglob("*.py") if p.name != "__init__.py")
    assert copied == ["configuration_leaf.py", "modeling_entry.py", "modeling_mid.py"]
    # get_class_in_module's pre-import closure walk must succeed on the result.
    entry = modcache / _hf_dyn.TRANSFORMERS_DYNAMIC_MODULE_NAME / "fake" / "repo" / "deadbeef" / "modeling_entry.py"
    walked = sorted(Path(f).name for f in _hf_dyn.get_relative_import_files(entry))
    assert walked == ["configuration_leaf.py", "modeling_mid.py"]
