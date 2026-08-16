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

"""Fail-loudly guards for Kimi-K3's two unsupported paths.

Both guards execute before their function touches ``self``, so each is exercised
through the unbound function with a lightweight stand-in. That keeps these tests
free of model construction, CUDA, and a real checkpoint.
"""

from types import SimpleNamespace

import pytest

from megatron.bridge.models.kimi.kimi_k3_bridge import KimiK3Bridge
from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3Attention


class TestConfigOnlyExportIsRejected:
    """K3 export is source-backed only; config-only export used to AttributeError."""

    def test_config_only_pretrained_config_is_rejected_with_a_clear_error(self):
        # examples/conversion/convert_checkpoints.py export builds exactly this
        # shape: a config object with no `.state` at all.
        config_only = SimpleNamespace(architectures=["KimiK3ForCausalLM"])

        with pytest.raises(ValueError, match="requires a state-backed HF checkpoint"):
            KimiK3Bridge.build_conversion_tasks(SimpleNamespace(), config_only, megatron_model=None)

    def test_state_present_but_source_missing_is_also_rejected(self):
        half_built = SimpleNamespace(state=SimpleNamespace(source=None))

        with pytest.raises(ValueError, match="no `.state.source`"):
            KimiK3Bridge.build_conversion_tasks(SimpleNamespace(), half_built, megatron_model=None)

    def test_the_error_names_the_reason_not_just_the_symptom(self):
        with pytest.raises(ValueError) as excinfo:
            KimiK3Bridge.build_conversion_tasks(SimpleNamespace(), SimpleNamespace(), megatron_model=None)
        message = str(excinfo.value)
        # A reader must learn *why*, so the MXFP4 packed/scale reason is required.
        assert "weight_packed" in message
        assert "config-only export is not supported" in message


class TestCachedInferenceIsRejected:
    """MLA writes no KV cache and KDA carries no recurrent state across steps."""

    def test_an_inference_context_raises_instead_of_being_discarded(self):
        with pytest.raises(NotImplementedError, match="cached incremental inference"):
            KimiK3Attention.forward(
                SimpleNamespace(),
                hidden_states=None,
                inference_context=object(),
            )

    def test_the_error_explains_both_missing_caches(self):
        with pytest.raises(NotImplementedError) as excinfo:
            KimiK3Attention.forward(SimpleNamespace(), hidden_states=None, inference_context=object())
        message = str(excinfo.value)
        assert "KV cache" in message
        assert "recurrent" in message

    def test_no_context_does_not_trip_the_guard(self):
        """Prefix-recompute decoding passes no context and must not be blocked.

        The guard must not fire; the call then proceeds and fails later for an
        unrelated reason (the stand-in ``self`` has no real attributes), which is
        exactly what proves the guard let it through.
        """
        with pytest.raises(Exception) as excinfo:
            KimiK3Attention.forward(SimpleNamespace(), hidden_states=None, inference_context=None)
        assert not isinstance(excinfo.value, NotImplementedError)
