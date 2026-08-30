# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Vision is additive: it appears when the config declares a tower, and only then.

The point of the split is that the text half stands on its own. A config with a
``vision_config`` gets the tower wired and the ``visual.**`` mapping; a text-only
config -- which no released GLM-5.3 checkpoint is, but every text-side test uses --
comes back configured exactly as the text bridge leaves it, and refuses to build a
model only when the tower it would need is missing.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.glm5_next.glm5_next_bridge import Glm5NextBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

from tests.unit_tests.models.glm5_next.test_glm5_next_bridge import text_config  # noqa: F401


pytestmark = pytest.mark.unit

IMAGE_TOKEN_ID = 154854


@pytest.fixture
def vision_config():
    return SimpleNamespace(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        image_size=448,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=4096,
    )


def _pretrained(text_config, vision_config=None):  # noqa: F811
    wrapper = Mock(spec=PreTrainedCausalLM)
    top = SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"],
        model_type="glm5_next",
        text_config=text_config,
        dtype="bfloat16",
    )
    if vision_config is not None:
        top.vision_config = vision_config
        top.image_token_id = IMAGE_TOKEN_ID
        top.video_token_id = 154855
        top.image_start_token_id = 154830
        top.image_end_token_id = 154831
        top.video_start_token_id = 154832
        top.video_end_token_id = 154833
    wrapper.config = top
    return wrapper


def test_vision_config_wires_the_tower_and_placeholder_ids(text_config, vision_config):  # noqa: F811
    provider = Glm5NextBridge().provider_bridge(_pretrained(text_config, vision_config))

    assert provider.vision_config is vision_config
    assert provider.image_token_id == IMAGE_TOKEN_ID
    # The splice needs embeddings the embedding layer has not reduce-scattered.
    assert provider.scatter_embedding_sequence_parallel is False
    # Text settings must survive untouched.
    assert provider.requires_packed_sequence is True
    assert provider.mtp_num_layers is None


def test_tower_weights_have_one_wildcard_mapping(text_config, vision_config):  # noqa: F811
    bridge = Glm5NextBridge()
    bridge.provider_bridge(_pretrained(text_config, vision_config))
    mapping = bridge.mapping_registry().hf_to_megatron_lookup("model.visual.blocks.0.attn.proj.bias")

    # Replicated, not Auto: the tower is plain torch modules held identically on
    # every rank, and AutoMapping cannot infer parallelism for a bare nn.Linear.
    assert isinstance(mapping, ReplicatedMapping)
    assert mapping.megatron_param == "visual.blocks.0.attn.proj.bias"


def test_backbone_weights_land_under_language_model(text_config, vision_config):  # noqa: F811
    """The wrapper holds the backbone one level down, so its names move with it.

    Without this the weight loader builds a conversion task for a module that is
    not there and dies on the first parameter, which is what a bare-name registry
    against the VL wrapper actually does.
    """
    bridge = Glm5NextBridge()
    bridge.provider_bridge(_pretrained(text_config, vision_config))
    registry = bridge.mapping_registry()

    embed = registry.hf_to_megatron_lookup("model.language_model.embed_tokens.weight")
    assert embed.megatron_param == "language_model.embedding.word_embeddings.weight"
    proj = registry.hf_to_megatron_lookup("model.language_model.layers.3.self_attn.o_proj.weight")
    assert proj.megatron_param == "language_model.decoder.layers.3.self_attention.linear_proj.weight"


def test_registry_prefixes_from_a_fresh_bridge(text_config, vision_config):  # noqa: F811
    """The names must not depend on provider_bridge having run on this instance.

    The weight-loading hook resolves its own bridge, so a registry that keyed off
    state left by provider construction came back un-prefixed and matched nothing
    on the wrapper -- 8 ranks dead at startup, with only a per-parameter warning
    to say why. Here mapping_registry is called on a bridge that has only ever
    seen hf_config, which is what the conversion machinery guarantees.
    """
    bridge = Glm5NextBridge()
    bridge.hf_config = _pretrained(text_config, vision_config).config
    embed = bridge.mapping_registry().hf_to_megatron_lookup(
        "model.language_model.embed_tokens.weight"
    )

    assert embed.megatron_param == "language_model.embedding.word_embeddings.weight"


def test_text_only_config_keeps_bare_backbone_names(text_config):  # noqa: F811
    """No tower, no wrapper, so the text bridge's own names stand unchanged."""
    bridge = Glm5NextBridge()
    bridge.provider_bridge(_pretrained(text_config))
    embed = bridge.mapping_registry().hf_to_megatron_lookup(
        "model.language_model.embed_tokens.weight"
    )

    assert embed.megatron_param == "embedding.word_embeddings.weight"


def test_text_only_config_leaves_vision_unset(text_config):  # noqa: F811
    provider = Glm5NextBridge().provider_bridge(_pretrained(text_config))

    assert provider.vision_config is None
    assert provider.requires_packed_sequence is True


def test_building_without_a_tower_is_refused(text_config):  # noqa: F811
    provider = Glm5NextBridge().provider_bridge(_pretrained(text_config))

    with pytest.raises(ValueError, match="requires vision_config"):
        provider.provide(pre_process=True, post_process=True)
