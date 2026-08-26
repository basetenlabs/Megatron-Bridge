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

"""Unit tests for the GLM-5.3-Flash heterogeneous block spec.

The interesting property is the per-layer assignment: KDA layers must get
``Glm5NextLinearAttention`` and MLA+DSA layers must get Megatron-Core's DSA spec, at the
right depths, with a valid input layernorm on both. Getting the depths wrong produces a
model that still builds and still trains, so it is worth testing directly rather than
relying on an end-to-end loss curve to reveal it.

The block-spec construction itself is stubbed. These tests are about the assignment
logic and the guards, not about Megatron-Core's spec builder.
"""

from types import SimpleNamespace

import pytest

from megatron.bridge.models.glm5_next import glm5_next_spec
from megatron.bridge.models.glm5_next.glm5_next_layers import Glm5NextLinearAttention
from megatron.bridge.models.glm5_next.glm5_next_spec import build_glm5_next_spec


pytestmark = pytest.mark.unit


NUM_LAYERS = 45
KDA_LAYERS = tuple(i + 1 for i in range(NUM_LAYERS) if i % 4 != 3)
DSA_LAYERS = tuple(n for n in range(1, NUM_LAYERS + 1) if n not in KDA_LAYERS)


class _FakeConfig(SimpleNamespace):
    """Stands in for Glm5NextModelProvider without pulling in the full dataclass."""

    def is_kda_layer(self, layer_number: int) -> bool:
        return layer_number in self.glm5_next_kda_layers


def make_config(**overrides) -> _FakeConfig:
    kwargs = dict(
        num_layers=NUM_LAYERS,
        glm5_next_kda_layers=KDA_LAYERS,
        experimental_attention_variant="dsa",
        multi_latent_attention=True,
        virtual_pipeline_model_parallel_size=None,
        normalization="RMSNorm",
    )
    kwargs.update(overrides)
    return _FakeConfig(**kwargs)


def _fake_dsa_spec():
    return SimpleNamespace(
        module="GlmAbsorbedMLASelfAttention",
        metainfo={"fuse_input_layernorm": False},
    )


def _fake_layer_spec():
    """A layer as the variant builder returns it: DSA attention, norm already chosen."""
    return SimpleNamespace(
        module="TransformerLayer",
        submodules=SimpleNamespace(
            self_attention=_fake_dsa_spec(),
            input_layernorm="LayerNorm",
        ),
    )


@pytest.fixture
def stub_megatron(monkeypatch):
    """Stub the Megatron-Core entry points the builder calls."""
    monkeypatch.setattr(
        glm5_next_spec,
        "get_transformer_block_with_experimental_attention_variant_spec",
        lambda config, vp_stage=None, pp_rank=None: SimpleNamespace(
            layer_specs=[_fake_layer_spec() for _ in range(config.num_layers)]
        ),
    )
    monkeypatch.setattr(glm5_next_spec, "get_transformer_layer_offset", lambda config, vp_stage: 0)


class TestLayerAssignment:
    def test_kda_and_dsa_land_at_the_right_depths(self, stub_megatron):
        block = build_glm5_next_spec(make_config())

        assert len(block.layer_specs) == NUM_LAYERS
        kda_at = [
            i + 1
            for i, s in enumerate(block.layer_specs)
            if s.submodules.self_attention.module is Glm5NextLinearAttention
        ]
        dsa_at = [
            i + 1
            for i, s in enumerate(block.layer_specs)
            if s.submodules.self_attention.module == "GlmAbsorbedMLASelfAttention"
        ]

        assert tuple(kda_at) == KDA_LAYERS
        assert tuple(dsa_at) == DSA_LAYERS
        assert len(kda_at) == 34
        assert len(dsa_at) == 11
        # The 3:1 schedule puts DSA on every fourth layer starting at 4.
        assert dsa_at == [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

    def test_the_builders_input_layernorm_is_preserved(self, stub_megatron):
        """The variant builder already chose a norm valid for both attention kinds.

        It picks input_layernorm from the attention spec's fuse_input_layernorm, and
        KDA declares the same value as DSA, so swapping the module keeps the norm
        correct. If that ever stopped holding, swapping would strand an IdentityOp and
        drop the pre-attention normalization silently -- hence the guard below.
        """
        block = build_glm5_next_spec(make_config())
        assert all(s.submodules.input_layernorm == "LayerNorm" for s in block.layer_specs)

    def test_a_fusing_dsa_spec_is_rejected(self, monkeypatch, stub_megatron):
        """If DSA started fusing its norm, KDA layers could not reuse the same one."""

        def fusing_layer_spec():
            spec = _fake_layer_spec()
            spec.submodules.self_attention.metainfo["fuse_input_layernorm"] = True
            spec.submodules.input_layernorm = "IdentityOp"
            return spec

        monkeypatch.setattr(
            glm5_next_spec,
            "get_transformer_block_with_experimental_attention_variant_spec",
            lambda config, vp_stage=None, pp_rank=None: SimpleNamespace(
                layer_specs=[fusing_layer_spec() for _ in range(config.num_layers)]
            ),
        )
        with pytest.raises(ValueError, match="fuse_input_layernorm"):
            build_glm5_next_spec(make_config())

    def test_pipeline_offset_shifts_the_schedule(self, monkeypatch, stub_megatron):
        """On a later PP rank, local index 0 is not global layer 1.

        With an offset of 4, this rank's first local layer is global layer 5 -- a KDA
        layer -- and its fourth is global layer 8, a DSA layer. Ignoring the offset
        would silently give every PP rank rank-0's schedule.
        """
        monkeypatch.setattr(glm5_next_spec, "get_transformer_layer_offset", lambda config, vp_stage: 4)
        config = make_config(num_layers=8)
        block = build_glm5_next_spec(config)

        modules = [s.submodules.self_attention.module for s in block.layer_specs]
        # Global layers 5..12 -> DSA at global 8 and 12, i.e. local indices 3 and 7.
        dsa_local = [i for i, m in enumerate(modules) if m == "GlmAbsorbedMLASelfAttention"]
        assert dsa_local == [3, 7]


class TestGuards:
    def test_virtual_pipeline_is_rejected(self, stub_megatron):
        with pytest.raises(ValueError, match="virtual pipeline"):
            build_glm5_next_spec(make_config(virtual_pipeline_model_parallel_size=2))

    def test_a_non_dsa_variant_is_rejected(self, stub_megatron):
        """Building on the wrong variant would silently change every layer's attention."""
        with pytest.raises(ValueError, match="experimental_attention_variant='dsa'"):
            build_glm5_next_spec(make_config(experimental_attention_variant=None))

    def test_mla_is_required(self, stub_megatron):
        with pytest.raises(ValueError, match="multi_latent_attention"):
            build_glm5_next_spec(make_config(multi_latent_attention=False))
