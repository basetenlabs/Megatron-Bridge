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

"""Model provider for the GLM-5.3-Flash (``glm5_next``) language backbone.

GLM-5.3 interleaves two attention mechanisms in a single stack:

* **KDA** (Kimi Delta Attention, linear) where ``layer_idx % 4 != 3`` -- 34 of 45 layers
* **MLA + DSA** (NoPE multi-latent attention with a sparse indexer) on the rest -- 11 layers

Megatron-Core selects an experimental attention variant with one scalar per model
(``experimental_attention_variant``), and its dispatch admits either *{linear attention
mixed with standard attention}* or *{every layer the variant}* -- never *{KDA mixed with
DSA}*. GLM-5.3 therefore leaves that scalar unset and assigns per-layer attention specs
in :func:`~megatron.bridge.models.glm5_next.glm5_next_spec.build_glm5_next_spec`.

Only the language backbone is provided. The vision tower is out of scope, following the
Kimi K3 precedent.
"""

from dataclasses import dataclass

from megatron.bridge.models.mla_provider import MLAModelProvider


@dataclass
class Glm5NextModelProvider(MLAModelProvider):
    """Megatron configuration and provider for the GLM-5.3-Flash text backbone.

    Fields inherited from ``MLATransformerConfig`` cover the MLA geometry
    (``q_lora_rank``, ``kv_lora_rank``, ``qk_head_dim``, ``qk_pos_emb_head_dim``,
    ``v_head_dim``) and the DSA indexer (``dsa_indexer_*``). Declared here are only the
    fields GLM-5.3 adds: the KDA layer schedule and geometry, k-pool indexing, and one
    explicit capability marker.
    """

    variable_seq_lengths: bool = True

    qk_pos_emb_head_dim: int = 0
    """GLM-5.3 attention is NoPE. Overrides the MLA default of 64.

    This is an architectural invariant rather than a tunable default: HF's
    ``Glm5NextTextConfig.validate_architecture`` rejects a nonzero RoPE dimension
    outright, and ``validate_attention`` below rejects it here. Left at the MLA
    default, every construction of this provider would fail that check.
    """

    # ----------------------------------------------------------------- layout
    glm5_next_kda_layers: tuple[int, ...] = ()
    """1-indexed *global* layer numbers that run KDA instead of MLA+DSA.

    1-indexed to match Megatron-Core's ``layer_number`` and the Kimi K3 convention
    (``layer_number in config.kimi_kda_layers``). HF's ``layer_types`` is 0-indexed, so
    the bridge converts once at the boundary. An off-by-one here silently shifts the
    whole attention schedule, producing a model that trains and looks plausible while
    being architecturally wrong -- see ``validate_layout``.
    """

    glm5_next_indexer_full_layers: tuple[int, ...] = ()
    """1-indexed layer numbers whose DSA indexer runs (rather than reusing a previous
    layer's top-k). Diagnostics and FLOP accounting only; cross-layer sharing itself is
    driven by ``dsa_indexer_topk_freq`` / ``dsa_indexer_skip_topk_offset``."""

    # -------------------------------------------------------------------- KDA
    glm5_next_linear_num_heads: int = 64
    """Number of KDA heads (HF ``linear_num_heads``)."""

    glm5_next_linear_head_dim: int = 128
    """Dimension per KDA head (HF ``linear_head_dim``)."""

    glm5_next_linear_conv_kernel_size: int = 4
    """Short-convolution kernel size applied to KDA q/k/v (HF ``linear_conv_kernel_dim``)."""

    glm5_next_kda_gate_lower_bound: float = -5.0
    """Lower clamp on the KDA forget gate (HF ``linear_lower_bound``)."""

    # --------------------------------------------------------------- k-pool
    glm5_next_index_kpool: int = 16
    """DSA indexer pool size.

    GLM-5.3 scores compressed *groups* of ``index_kpool`` keys rather than individual
    keys, selecting ``dsa_indexer_topk // index_kpool`` pools and expanding each winner
    back into raw token indices. This is the one genuinely new algorithm relative to
    GLM-5.2; see ``dsa_kpool``.
    """

    glm5_next_index_kpool_always_select_tail: bool = True
    """Whether the trailing incomplete pool is always appended to the selection.

    When True the selection stage emits ``dsa_indexer_topk + index_kpool - 1`` indices
    instead of ``dsa_indexer_topk``. Downstream consumers are width-agnostic, so this
    costs nothing beyond the wider tensor.
    """

    # -------------------------------------------------------------- markers
    glm5_next_requires_fp32_lm_head: bool = True
    """Whether the output projection must run in fp32.

    GLM-5.3 needs the fp32 head for the same reason GLM-5.2 does, but -- unlike
    GLM-5.2 -- it cannot advertise that through ``experimental_attention_variant``:
    the block builder requires that scalar to be ``None`` (see ``glm5_next_spec``).

    Downstream code that currently infers the fp32 head from
    ``experimental_attention_variant == "dsa"`` must key off this field instead. If it
    does not, the fp32 head silently turns off on the trainer *and* on the sampler,
    which mirrors the same condition -- consistent on both sides, so nothing raises and
    the only symptom is degraded numerics.
    """

    def is_kda_layer(self, layer_number: int) -> bool:
        """Whether the given 1-indexed global layer number runs KDA rather than MLA+DSA."""
        return layer_number in self.glm5_next_kda_layers

    @property
    def kpool_output_width(self) -> int:
        """Width of the index tensor the k-pool selection stage emits."""
        if self.dsa_indexer_topk is None:
            raise ValueError("dsa_indexer_topk must be set before querying kpool_output_width")
        tail = self.glm5_next_index_kpool - 1 if self.glm5_next_index_kpool_always_select_tail else 0
        return self.dsa_indexer_topk + tail

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_layout()
        self.validate_kpool()
        self.validate_attention()

    # ---------------------------------------------------------------- checks
    # Validated here rather than in the bridge so a hand-built provider (tests,
    # experiments) cannot silently produce a wrong architecture.

    def validate_layout(self) -> None:
        """Reject a KDA schedule that cannot correspond to this stack."""
        if not self.glm5_next_kda_layers:
            # A stack with no KDA layers is pure MLA+DSA -- valid in principle (it is
            # GLM-5.2's shape), but never what a GLM-5.3 checkpoint describes, so it
            # almost certainly means the bridge failed to populate the schedule.
            raise ValueError(
                "glm5_next_kda_layers is empty; GLM-5.3 interleaves KDA and MLA+DSA layers. "
                "Populate it from the checkpoint's layer_types."
            )

        out_of_range = [n for n in self.glm5_next_kda_layers if not 1 <= n <= self.num_layers]
        if out_of_range:
            raise ValueError(
                f"glm5_next_kda_layers holds 1-indexed global layer numbers in [1, {self.num_layers}], "
                f"but got out-of-range entries {out_of_range}. A 0-indexed list would show up here."
            )

        if len(set(self.glm5_next_kda_layers)) != len(self.glm5_next_kda_layers):
            raise ValueError(f"glm5_next_kda_layers contains duplicates: {self.glm5_next_kda_layers}")

    def validate_kpool(self) -> None:
        """Enforce the k-pool divisibility invariant."""
        if self.glm5_next_index_kpool < 1:
            raise ValueError(f"glm5_next_index_kpool must be positive, got {self.glm5_next_index_kpool}")

        if self.dsa_indexer_topk is not None and self.dsa_indexer_topk % self.glm5_next_index_kpool:
            # Mirrors Glm5NextTextConfig.validate_architecture. Selection takes
            # topk // kpool pools and expands each by kpool, so a non-multiple would
            # silently select fewer tokens than index_topk implies.
            raise ValueError(
                f"dsa_indexer_topk ({self.dsa_indexer_topk}) must be divisible by "
                f"glm5_next_index_kpool ({self.glm5_next_index_kpool})"
            )

    def validate_attention(self) -> None:
        """Enforce the invariants the spec builder and the DSA path rely on."""
        if not self.multi_latent_attention:
            raise ValueError("GLM-5.3 requires multi_latent_attention=True for its DSA layers")

        if self.experimental_attention_variant is not None:
            # get_gpt_decoder_layer_specs asserts this is None, and the DSA layers get
            # their spec directly instead. Fail here with the reason rather than at the
            # bare assert inside Megatron-Core.
            raise ValueError(
                "GLM-5.3 builds its block from get_gpt_decoder_block_spec, which requires "
                "experimental_attention_variant=None; DSA layers receive their module spec "
                f"directly. Got {self.experimental_attention_variant!r}."
            )

        if self.qk_pos_emb_head_dim:
            # GLM-5.3 is NoPE; HF's validate_architecture rejects a nonzero RoPE dim.
            # Checked here so a checkpoint that reintroduces RoPE fails at construction
            # rather than training without position information.
            raise ValueError(f"GLM-5.3 expects NoPE attention (qk_pos_emb_head_dim=0), got {self.qk_pos_emb_head_dim}")
