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

"""Unit tests for the GLM-5.3-Flash model provider.

These cover the architectural invariants the provider enforces at construction. They
matter more than usual here because the failure they guard against is silent: a
misdescribed layer schedule still builds, still trains, and still produces a plausible
loss curve while wiring KDA and MLA+DSA layers to the wrong depths.
"""

import pytest

from megatron.bridge.models.glm5_next import Glm5NextModelProvider


pytestmark = pytest.mark.unit


# GLM-5.3-Flash's real schedule: KDA everywhere except layer_idx % 4 == 3.
# 1-indexed for the provider, so DSA lands on layer numbers 4, 8, ... 44.
NUM_LAYERS = 45
KDA_LAYERS = tuple(i + 1 for i in range(NUM_LAYERS) if i % 4 != 3)
INDEX_TOPK = 2048
INDEX_KPOOL = 16


def make_provider(**overrides) -> Glm5NextModelProvider:
    """Build a provider with GLM-5.3-Flash's shape, overridable per test."""
    kwargs = dict(
        num_layers=NUM_LAYERS,
        hidden_size=4096,
        num_attention_heads=64,
        vocab_size=154880,
        glm5_next_kda_layers=KDA_LAYERS,
        dsa_indexer_topk=INDEX_TOPK,
        glm5_next_index_kpool=INDEX_KPOOL,
    )
    kwargs.update(overrides)
    return Glm5NextModelProvider(**kwargs)


class TestLayout:
    """The KDA/MLA+DSA schedule."""

    def test_real_schedule_constructs(self):
        provider = make_provider()
        assert len(provider.glm5_next_kda_layers) == 34
        # The complement is the 11 MLA+DSA layers.
        dsa = [n for n in range(1, NUM_LAYERS + 1) if n not in provider.glm5_next_kda_layers]
        assert dsa == [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

    def test_is_kda_layer_matches_schedule(self):
        provider = make_provider()
        assert provider.is_kda_layer(1)
        assert provider.is_kda_layer(3)
        assert not provider.is_kda_layer(4)
        assert not provider.is_kda_layer(44)
        assert provider.is_kda_layer(45)

    def test_empty_schedule_is_rejected(self):
        # An all-DSA stack is GLM-5.2's shape, never GLM-5.3's, so an empty schedule
        # means the bridge failed to populate it rather than a legitimate config.
        with pytest.raises(ValueError, match="glm5_next_kda_layers is empty"):
            make_provider(glm5_next_kda_layers=())

    def test_zero_indexed_schedule_is_rejected(self):
        """The invariant most likely to be violated in practice.

        HF's ``layer_types`` is 0-indexed and the provider is 1-indexed. Passing the
        HF list through unconverted puts a 0 in the tuple, which this must catch --
        otherwise the whole schedule silently shifts by one layer.
        """
        zero_indexed = tuple(i for i in range(NUM_LAYERS) if i % 4 != 3)
        assert 0 in zero_indexed
        with pytest.raises(ValueError, match="out-of-range"):
            make_provider(glm5_next_kda_layers=zero_indexed)

    def test_layer_number_past_the_end_is_rejected(self):
        with pytest.raises(ValueError, match="out-of-range"):
            make_provider(glm5_next_kda_layers=KDA_LAYERS + (NUM_LAYERS + 1,))

    def test_duplicate_layers_are_rejected(self):
        with pytest.raises(ValueError, match="duplicates"):
            make_provider(glm5_next_kda_layers=KDA_LAYERS + (KDA_LAYERS[0],))


class TestKPool:
    """k-pool indexing invariants."""

    def test_output_width_includes_the_tail(self):
        # 128 pools x 16 tokens = 2048, plus up to kpool-1 tail tokens.
        provider = make_provider()
        assert provider.kpool_output_width == INDEX_TOPK + INDEX_KPOOL - 1 == 2063

    def test_output_width_without_tail(self):
        provider = make_provider(glm5_next_index_kpool_always_select_tail=False)
        assert provider.kpool_output_width == INDEX_TOPK

    def test_topk_must_divide_by_kpool(self):
        # Selection takes topk // kpool pools and expands each by kpool; a non-multiple
        # would silently select fewer tokens than index_topk implies.
        with pytest.raises(ValueError, match="must be divisible"):
            make_provider(glm5_next_index_kpool=17)

    def test_non_positive_kpool_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            make_provider(glm5_next_index_kpool=0)


class TestAttentionInvariants:
    """Invariants the spec builder and the DSA path depend on."""

    def test_nope_is_the_default(self):
        # Overrides the MLA default of 64; GLM-5.3 has no RoPE at all.
        assert make_provider().qk_pos_emb_head_dim == 0

    def test_rope_dim_is_rejected(self):
        with pytest.raises(ValueError, match="NoPE"):
            make_provider(qk_pos_emb_head_dim=64)

    def test_experimental_attention_variant_must_be_unset(self):
        """GLM-5.2 sets this to "dsa"; GLM-5.3 cannot.

        ``get_gpt_decoder_layer_specs`` asserts the variant is None, and the block
        builder goes through it. Failing here names the reason; the bare Megatron-Core
        assert does not.
        """
        with pytest.raises(ValueError, match="experimental_attention_variant=None"):
            make_provider(experimental_attention_variant="dsa")

    def test_mla_is_required(self):
        with pytest.raises(ValueError, match="multi_latent_attention"):
            make_provider(multi_latent_attention=False)


class TestMarkers:
    def test_fp32_lm_head_is_requested_by_default(self):
        """The fp32 head cannot be inferred from experimental_attention_variant.

        That scalar is None for GLM-5.3 (see TestAttentionInvariants), so consumers
        that key off it would silently fall back to a bf16 head on both the trainer
        and the sampler -- consistently, so nothing raises.
        """
        assert make_provider().glm5_next_requires_fp32_lm_head is True
