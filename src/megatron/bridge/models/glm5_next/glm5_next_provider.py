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
DSA}*. GLM-5.3 keeps that scalar at ``"dsa"`` and replaces attention on the KDA layers
afterwards, in
:func:`~megatron.bridge.models.glm5_next.glm5_next_spec.build_glm5_next_spec`. Keeping
the scalar is deliberate: consumers infer the fp32 output projection from it, on the
trainer and on the sampler alike.

The vision tower is included -- see ``Glm5NextVLModelProvider`` below and
``glm5_next_vl_model``. ``Glm5NextForConditionalGeneration`` is GLM-5.3's only
architecture, so the text path is simply the one that passes no images.
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

    experimental_attention_variant: str | None = "dsa"
    """Selects Megatron-Core's DSA attention for every layer.

    The block builder starts from the all-DSA block this produces and then replaces
    attention on the KDA layers (see ``glm5_next_spec``). Overrides the base default of
    ``None``, which would build a standard-attention block instead -- so, like
    ``qk_pos_emb_head_dim``, this is an architectural invariant rather than a tunable.
    """

    multi_latent_attention: bool = True
    """GLM-5.3's 11 sparse layers are MLA, so this is an architectural invariant rather
    than a tunable -- declared here, like ``experimental_attention_variant`` and
    ``qk_pos_emb_head_dim``, so ``validate_attention`` holds from construction. Assigning
    it after the fact would mean every provider is briefly alive in a state its own
    __post_init__ rejects."""

    position_embedding_type: str = "none"
    """GLM-5.3 has no position embeddings on the text side at all.

    NoPE means no rotary, and the checkpoint ships no learned table either. Megatron's
    default is ``"learned_absolute"``, which builds a position-embedding matrix that no
    checkpoint tensor maps onto -- so it would be randomly initialised, trained, and
    added to every token. Caught by the VL path, where the embedding is called with
    ``position_ids=None`` and raised ``embedding(): argument 'indices' must be Tensor,
    not NoneType``; the text path would instead have silently added a spurious learned
    positional signal."""

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

    # ------------------------------------------------------------------- KDA
    # GLM-5.3's KDA layers are Megatron-Core's ``KimiDeltaAttention`` (see
    # glm5_next_spec), so its own config fields are the single source of truth for KDA
    # geometry -- these override MCore defaults that do not match GLM-5.3. Verified
    # shape-by-shape against zai-org/GLM-5.3-Flash-BF16:
    #
    #   q/k/v_proj [8192, 4096]  -> 64 heads x 128 head_dim
    #   b_proj     [64, 4096]    -> beta_proj, width num_key_heads
    #   f_a/g_a    [128, 4096]   -> low-rank bottlenecks, width head_dim
    #   f_b/g_b    [8192, 128]   -> back out to qk_dim / v_dim
    #   A_log      [64]          -> per head
    #   dt_bias    [8192]        -> per channel
    #   q_conv1d   [8192, 1, 4]  -> depthwise, kernel 4 (fused across q/k/v in MCore)
    linear_num_key_heads: int | None = 64
    """KDA query/key heads (HF ``linear_num_heads``). MCore defaults to 16."""

    linear_num_value_heads: int | None = 64
    """KDA value heads. GLM-5.3 is not grouped, so this equals the key-head count.
    MCore defaults to 32."""

    linear_key_head_dim: int | None = 128
    """Dimension per KDA key head (HF ``linear_head_dim``)."""

    linear_value_head_dim: int | None = 128
    """Dimension per KDA value head."""

    linear_conv_kernel_dim: int | None = 4
    """KDA short-convolution kernel size (HF ``linear_conv_kernel_dim``)."""

    kda_f_lora_rank: int | None = 128
    """Rank of the forget-gate bottleneck. HF builds ``f_a_proj`` as
    ``Linear(hidden_size, linear_head_dim)``, so the rank *is* the head dim -- the
    bridge derives it rather than carrying a second constant.

    Setting this (with ``kda_gate_lora_rank``) is also what selects MCore's low-rank
    KDA projection layout over its legacy fused one. Left at MCore's ``None`` default,
    KDA would expect a single fused q/k/v/f/g projection that GLM-5.3 does not ship."""

    kda_gate_lora_rank: int | None = 128
    """Rank of the output-gate bottleneck (HF ``g_a_proj``), likewise the head dim."""

    kda_safe_gate: bool = True
    """Whether the KDA kernel bounds the forget gate. **Load-bearing, and MCore
    defaults it to False.**

    GLM-5.3 sets ``safe_gate`` (HF defaults it True) and ships
    ``gate_lower_bound = -5.0``, which selects a different gate function entirely::

        safe_gate:  g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias))   # in (-5, 0)
        otherwise:  g = -exp(A_log) * softplus(f + dt_bias)                 # unbounded

    ``fla.ops.kda.chunk_kda`` takes ``safe_gate``/``lower_bound`` and fuses whichever
    form is selected. With this left False the kernel raises nothing and trains the
    unbounded gate -- the same silent-divergence shape as the Kimi-K3 fla floor.

    No new dependency floor is needed: the support is present as far back as
    flash-linear-attention 0.4.2, verified on a B300 pod against the installed 0.4.2,
    where ``safe_gate``/``lower_bound`` are explicit parameters threaded through both
    forward and backward and validated to ``-5 <= lower_bound < 0`` -- the docstring
    there even says to set -5, which is GLM-5.3's value. (An earlier revision of this
    comment claimed 0.5.2; that came from checking only the 0.5.2 tag and was wrong.)"""

    kda_lower_bound: float | None = -5.0
    """Lower bound on the KDA forget gate (HF ``linear_lower_bound``). Only consumed
    when ``kda_safe_gate`` is set."""

    linear_cp_mode: str | None = "headwise"
    """How the KDA layers shard under context parallelism.

    ``"headwise"`` all-to-alls from sequence-sharded to head-sharded around the kernel,
    so each rank sees whole sequences for a subset of the 64 heads and the recurrent
    state is never split. This is the mode Megatron-Core's own KDA + gated-MLA
    functional test covers (``hybrid_..._ep8_cp2_kda_gated_mla_1N8G``).

    ``"chunkwise"`` (MCore's default) keeps the sequence sharded and passes a CP context
    into the kernel. Both are implemented; headwise is the validated one, and it is why
    GLM-5.3 is not stuck at cp=1 -- 34 of its 45 layers are KDA."""

    # ---------------------------------------------------------------- k-pool

    glm5_next_index_kpool: int = 4
    """DSA indexer pool size.

    GLM-5.3 scores compressed *groups* of ``index_kpool`` keys rather than individual
    keys, selecting ``dsa_indexer_topk // index_kpool`` pools and expanding each winner
    back into raw token indices. This is the one genuinely new algorithm relative to
    GLM-5.2.

    Defaults to 4, the value in the ``zai-org/GLM-5.3-Flash`` checkpoint qualified on
    B200 (revision ``84c6a6aa9497188e15a635ba793b0f95a79b1033``, which allocates
    ``index_kpool_compress_ape`` as ``[4, index_head_dim]`` and requires the sequence
    length to be divisible by 4). Note this differs from the ``Glm5NextTextConfig``
    class default of 16 -- the bridge reads the real value from the checkpoint, and
    this default exists only so a hand-built provider matches the shipped model rather
    than the library default.
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

    GLM-5.3 needs the fp32 head for the same reason GLM-5.2 does, and -- because the
    block is built on the experimental-variant path -- it *does* advertise
    ``experimental_attention_variant == "dsa"``, so consumers that infer the fp32 head
    from that scalar keep working.

    This field is the explicit signal for the same property. It exists because tying a
    numerics decision to a string scalar chosen for spec dispatch is fragile: if the
    block ever moves off the variant path, that inference flips to False on the trainer
    *and* on the sampler, which mirrors the same condition -- consistent on both sides,
    so nothing raises and the only symptom is degraded numerics. Prefer this field.
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

        if self.experimental_attention_variant != "dsa":
            # The block is built on the experimental-variant path as all-DSA, and the
            # KDA layers then replace their attention. Keeping the scalar at "dsa" also
            # keeps downstream inferences drawn from it correct -- see
            # glm5_next_requires_fp32_lm_head.
            raise ValueError(
                "GLM-5.3 builds on the experimental-variant block builder and requires "
                "experimental_attention_variant='dsa'; its KDA layers replace their "
                f"attention afterwards. Got {self.experimental_attention_variant!r}."
            )

        if self.qk_pos_emb_head_dim:
            # GLM-5.3 is NoPE; HF's validate_architecture rejects a nonzero RoPE dim.
            # Checked here so a checkpoint that reintroduces RoPE fails at construction
            # rather than training without position information.
            raise ValueError(f"GLM-5.3 expects NoPE attention (qk_pos_emb_head_dim=0), got {self.qk_pos_emb_head_dim}")


@dataclass
class Glm5NextVLModelProvider(Glm5NextModelProvider):
    """GLM-5.3-Flash with its vision tower.

    ``Glm5NextForConditionalGeneration`` is the model's only architecture, so this is
    the provider the bridge builds for every GLM-5.3 checkpoint. Passing no
    ``pixel_values`` gives the text path; the tower is then constructed but never
    called.

    Vision geometry is not declared here. The tower is HuggingFace's own
    ``Glm5NextVisionModel``, built from ``vision_config``, so the config object is the
    single source of truth and there is no second copy to drift. For reference, the
    shipped values are depth 24, hidden 1024, 16 heads, image 448, patch 14, spatial
    merge 2, temporal patch 2, out_hidden 4096 -- the GLM-4.5V tower rescaled.
    """

    # Set by the bridge from the HF config; typed loosely because it is a transformers
    # config object, not a Megatron dataclass.
    vision_config: object = None

    # Multimodal token ids. HF's get_placeholder_mask -- which this model reuses rather
    # than reimplementing -- reads image_token_id, video_start_token_id and
    # video_end_token_id off the *Megatron* config, so they have to live here. Defaults
    # are zai-org/GLM-5.3-Flash's; the bridge overwrites them from the checkpoint.
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833

    scatter_embedding_sequence_parallel: bool = False
    """The vision-language wrapper owns the sequence-parallel scatter, not the embedding.

    The splice has to happen on the *whole* sequence -- ``get_placeholder_mask`` and
    ``masked_scatter`` work over ``[batch, seq, hidden]`` with every image placeholder
    present -- so the wrapper needs unsharded embeddings and scatters afterwards. Left at
    Megatron's ``True`` default the sequence is scattered twice: measured at TP=8 with
    seq 8192, the DSA indexer saw ``x`` of 128 positions where 1024 was correct.

    It cannot be fixed by flipping the flag after construction:
    ``LanguageModelEmbedding`` derives ``reduce_scatter_embeddings`` from it in
    ``__init__`` and passes that into ``VocabParallelEmbedding``, so a late flip leaves
    the reduce-scatter path active and the sequence still sharded twice.

    ``GPTModel`` reads the same flag to decide whether to scatter a supplied
    ``decoder_input`` itself, so this keeps exactly one scatter in the graph."""

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build the VL model, applying any requested freezes."""
        from megatron.bridge.models.glm5_next.glm5_next_vl_model import Glm5NextVLModel

        if self.vision_config is None:
            raise ValueError(
                "Glm5NextVLModelProvider requires vision_config; GLM-5.3-Flash ships a "
                "vision tower and its weights would otherwise have nowhere to load."
            )

        # MTP + sequence parallelism + the vision splice cannot all hold at once, and the
        # symptom is an opaque shape error deep in MTP rather than anything nameable:
        #
        #   RuntimeError: The size of tensor a (65536) must match the size of tensor b
        #   (8192) at non-singleton dimension 0
        #   multi_token_prediction.py:2172 _concat_embeddings
        #
        # The splice needs unsharded embeddings, so this provider sets
        # scatter_embedding_sequence_parallel=False. MTP then re-embeds input_ids through
        # that same non-scattering embedding while its hidden states are SP-sharded, and
        # the two no longer line up. Measured at TP=8 / EP=1 / ETP=8, seq 8192.
        #
        # Not a blocker in practice: GLM-5.2's published B300 rows are tp=1 too, and
        # GLM-5.3 at tp=1 / ep=8 reaches 32K on one node with full recompute (215 GiB of
        # 268 per rank). Resolving it properly means teaching MTP to scatter its own
        # embedding output, which is a Megatron-Core change.
        if self.sequence_parallel and self.mtp_num_layers and self.tensor_model_parallel_size > 1:
            raise ValueError(
                "GLM-5.3 cannot combine MTP with sequence parallelism: the vision splice "
                "requires an embedding that does not scatter, and MTP's own embedding call "
                "then disagrees with its SP-sharded hidden states. Use "
                "tensor_model_parallel_size=1 (the shape validated on B300, which reaches "
                "32K with full recompute), or set mtp_num_layers=None."
            )

        model = Glm5NextVLModel(
            self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None):
        """Build only the Megatron language backbone, for the VL model to wrap."""
        return MLAModelProvider.provide(
            self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
