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

"""KDA, no-RoPE MLA, latent-MoE, and AttnRes layers for Kimi K3.

The initial implementation was adapted from the Apache-2.0 Miles Kimi K3 backend:
https://github.com/radixark/miles/tree/dc62a0bd4b7af1c59ee2084852eb18b5585ec082/miles_plugins/models/kimi_k3
"""

import copy

import torch
import torch.nn.functional as F
from einops import rearrange
from fla.modules import FusedRMSNormGated, ShortConvolution
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELinear,
    TERowParallelLinear,
)
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.common import (
    a2a_cp_to_hp,
    a2a_hp_to_cp,
    causal_conv1d,
    get_parameter_local_cp,
)
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer, get_transformer_layer_offset
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint
from torch import nn

from megatron.bridge.models.kimi.kimi_k3_ops import KimiRMSNorm, attn_res_aggregate, kda
from megatron.bridge.models.kimi.kimi_k3_pipeline import bank_num_rows, pack_stage_boundary, unpack_stage_boundary


def _mark_tp_replicated(module: nn.Module, *, reduction: str = "average") -> None:
    if reduction not in ("average", "sum"):
        raise ValueError(f"Unsupported TP gradient reduction: {reduction}")
    for parameter in module.parameters():
        setattr(parameter, f"{reduction}_gradients_across_tp_domain", True)


def _linear(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output, bias = module(inputs)
    if bias is not None:
        raise ValueError(f"Kimi K3 requires bias-free projections, got bias from {type(module).__name__}")
    return output


def _is_single_document(cu_seqlens: torch.Tensor | None) -> bool:
    return cu_seqlens is not None and cu_seqlens.numel() == 2 and int(cu_seqlens[0]) == 0


def _physical_cu_seqlens(packed_seq_params: PackedSeqParams | None) -> torch.Tensor | None:
    """The document boundaries as laid out in memory.

    The packer advances by the *padded* length per document but reports
    ``cu_seqlens_q`` as the unpadded cumulative lengths, emitting
    ``cu_seqlens_q_padded`` only when the two differ. The conv and KDA kernels
    segment the physical buffer, so they need the padded boundaries: fed the
    unpadded ones they reset state early and leave the real tail tokens
    uninitialized, with no error.
    """
    if packed_seq_params is None:
        return None
    cu_seqlens = packed_seq_params.cu_seqlens_q_padded
    if cu_seqlens is None:
        cu_seqlens = packed_seq_params.cu_seqlens_q
    if cu_seqlens is None:
        raise ValueError("Packed KDA input requires cu_seqlens_q")
    return cu_seqlens


def _prepare_kda_inputs(
    tensors: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor | None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None, int]:
    """Use dense KDA for one document and remove any trailing sequence padding.

    FLA's varlen KDA path mishandles a partial final chunk even when the pack
    holds a single document. Dense causal KDA is equivalent on the valid prefix
    ``[0, valid_length]``, so trim to it and drop ``cu_seqlens``.

    Args:
        tensors: ``[b, s, ...]`` KDA inputs that share the sequence dimension.
        cu_seqlens: cumulative document lengths, or ``None`` for a dense stream.

    Returns:
        The (possibly trimmed) tensors, the cu_seqlens to hand to ``kda``, and
        the valid sequence length the outputs cover.
    """
    sequence_length = tensors[0].shape[1]
    if not _is_single_document(cu_seqlens):
        # A [b, s, h, d] view of a transposed [s, b, C] tensor is non-contiguous
        # whenever b > 1. contiguous() is a no-op on the b == 1 shapes the packed
        # path produces, so this costs nothing in the common case.
        return tuple(tensor.contiguous() for tensor in tensors), cu_seqlens, sequence_length

    valid_length = int(cu_seqlens[-1])
    if not 0 < valid_length <= sequence_length:
        raise ValueError(f"single-document cu_seqlens ends at {valid_length}, expected 1..{sequence_length}")
    # A batch dimension of one can hide a padded stride while still reporting
    # contiguous, including when valid_length == sequence_length. TileLang's
    # backward kernels require canonical input strides.
    valid_tensors = tuple(tensor[:, :valid_length].clone(memory_format=torch.contiguous_format) for tensor in tensors)
    return valid_tensors, None, valid_length


def _short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Depthwise causal conv + silu that resets at packed-document boundaries.

    Equivalent to ``ShortConvolution(activation="silu", bias=False)`` on each
    document independently. Used on the CP head-parallel path, where the conv
    weight is a per-CP-rank channel slice: FLA's module is sized for the full
    local channel count, but its functional kernel takes explicit weights —
    the same Triton kernel the cp=1 ``ShortConvolution`` path runs, so cp>1
    conv numerics match cp=1 exactly.

    Args:
        x: ``[s, b, C]`` full-sequence activations for this rank's head shard.
        weight: ``[C, 1, K]`` depthwise conv weight slice (fp32 per K3 policy).
        cu_seqlens: global cumulative document lengths; ``None`` = one document.
    """
    if causal_conv1d is None:
        raise RuntimeError("KDA context parallelism requires flash-linear-attention's causal_conv1d")
    if cu_seqlens is not None:
        if x.shape[1] != 1:
            raise ValueError(
                f"FLA varlen conv requires batch == 1, got batch={x.shape[1]}; "
                "it silently reads only the first batch otherwise"
            )
        if int(cu_seqlens[-1]) != x.shape[0]:
            raise ValueError(
                f"cu_seqlens must span the full sequence (got {int(cu_seqlens[-1])} "
                f"!= {x.shape[0]}): FLA leaves rows past cu_seqlens[-1] uninitialized"
            )
    y, _ = causal_conv1d(
        x=x.transpose(0, 1),  # [b, s, C]
        weight=weight.squeeze(1),  # [C, K]
        bias=None,
        activation="silu",
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    return y.transpose(0, 1)


def kda_cp_split_sections(local_projection_size: int, local_num_heads: int) -> tuple[int, ...]:
    """Per-section widths of the fused KDA projection, local to this TP rank.

    The head-parallel all-to-all splits every section evenly across CP ranks,
    so the widths must be the ones this rank actually holds: the q/k/v/forget/
    gate projections are column-parallel (``local_projection_size`` wide) and
    ``b_proj`` emits one channel per local head.
    """
    return (local_projection_size,) * 5 + (local_num_heads,)


class KimiK3ShortConvolution(ShortConvolution):
    """TP-sharded KDA short convolution with FP32 state."""

    def __init__(self, *args, tp_group, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tp_group = tp_group
        self.weight.data = self.weight.data.float()
        self._keep_in_float32_parameter_names = ("weight",)
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Return the TP-sharded convolution state."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


class KimiK3Attention(MegatronModule):
    """Select KDA or Kimi's no-RoPE MLA according to the global layer number.

    Context parallel (cp>1) runs the KDA layers GDN-style head-parallel: an
    all-to-all converts the sequence-sharded activations into full-sequence
    shards over a head subset, the recurrence runs unchanged, and a second
    all-to-all restores sequence sharding. Per-head state (``A_log``/``dt_bias``)
    and the depthwise conv weights are sliced per CP rank; gradients flow into
    the full parameters through the slice views. The MLA layers get CP from
    ``TEDotProductAttention`` as usual.
    """

    def __init__(
        self,
        config,
        layer_number: int,
        cp_comm_type: str | None = None,
        pg_collection=None,
        pp_layer_offset: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        del pp_layer_offset, name
        self.cp_comm_type = cp_comm_type

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        elif not hasattr(pg_collection, "tp") or not hasattr(pg_collection, "cp"):
            raise ValueError("KimiK3Attention requires TP and CP process groups")
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size()
        self.cp_group = pg_collection.cp
        self.cp_size = self.cp_group.size()
        self.sequence_parallel = config.sequence_parallel
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False
        # K3 attention owns the sequence-parallel gather/scatter around the
        # complete attention block. Its internal linears must not also use TE
        # UserBuffers, which would duplicate collectives and expect the wrong
        # tensor shape. MLP linears retain overlap through the parent config.
        self.linear_config.tp_comm_overlap = False

        self.layer_idx = layer_number - 1
        self.is_kda = layer_number in config.kimi_kda_layers
        if self.is_kda:
            self._init_kda(config)
        else:
            self._init_mla(config)

    def _duplicated_linear(self, input_size: int, output_size: int) -> TELinear:
        return TELinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _column_linear(self, input_size: int, output_size: int) -> TEColumnParallelLinear:
        return TEColumnParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _row_linear(self, input_size: int, output_size: int) -> TERowParallelLinear:
        return TERowParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _init_kda(self, config) -> None:
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.kimi_linear_num_heads
        if self.num_heads % self.tp_size:
            raise ValueError(f"KDA heads {self.num_heads} must be divisible by TP size {self.tp_size}")
        self.local_num_heads = self.num_heads // self.tp_size
        self.head_dim = config.kimi_linear_head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.local_projection_size = self.local_num_heads * self.head_dim

        self.q_proj = self._column_linear(hidden_size, self.projection_size)
        self.k_proj = self._column_linear(hidden_size, self.projection_size)
        self.v_proj = self._column_linear(hidden_size, self.projection_size)
        conv_kwargs = {
            "hidden_size": self.local_projection_size,
            "kernel_size": config.kimi_linear_conv_kernel_size,
            "activation": "silu",
            "device": device,
            "dtype": dtype,
            "tp_group": self.tp_group,
        }
        self.q_conv1d = KimiK3ShortConvolution(**conv_kwargs)
        self.k_conv1d = KimiK3ShortConvolution(**conv_kwargs)
        self.v_conv1d = KimiK3ShortConvolution(**conv_kwargs)

        self.f_a_proj = self._duplicated_linear(hidden_size, self.head_dim)
        self.f_b_proj = self._column_linear(self.head_dim, self.projection_size)
        self.b_proj = self._column_linear(hidden_size, self.num_heads)
        self.g_proj = self._column_linear(hidden_size, self.projection_size)

        self.A_log = nn.Parameter(torch.empty(self.local_num_heads, dtype=torch.float32, device=device))
        self.dt_bias = nn.Parameter(torch.empty(self.local_projection_size, dtype=torch.float32, device=device))
        self._keep_in_float32_parameter_names = ("A_log", "dt_bias")
        set_tensor_model_parallel_attributes(self.A_log, True, 0, 1)
        set_tensor_model_parallel_attributes(self.dt_bias, True, 0, 1)

        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.layernorm_epsilon,
            activation="sigmoid",
            device=device,
            dtype=dtype,
        )
        _mark_tp_replicated(self.o_norm, reduction="sum")
        self.o_proj = self._row_linear(self.projection_size, hidden_size)
        self.gate_lower_bound = config.kimi_kda_gate_lower_bound
        if self.cp_size > 1 and self.local_num_heads % self.cp_size:
            raise ValueError(
                f"KDA heads per TP rank ({self.local_num_heads}) must be divisible by "
                f"context parallel size ({self.cp_size})"
            )

    def _init_mla(self, config) -> None:
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.num_attention_heads
        if self.num_heads % self.tp_size:
            raise ValueError(f"MLA heads {self.num_heads} must be divisible by TP size {self.tp_size}")
        self.local_num_heads = self.num_heads // self.tp_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_head_dim
        self.qk_extra_head_dim = config.qk_pos_emb_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_extra_head_dim

        self.q_a_proj = self._duplicated_linear(hidden_size, self.q_lora_rank)
        self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank, config.layernorm_epsilon, device=device, dtype=dtype)
        self.q_b_proj = self._column_linear(self.q_lora_rank, self.num_heads * self.q_head_dim)
        self.kv_a_proj_with_mqa = self._duplicated_linear(hidden_size, self.kv_lora_rank + self.qk_extra_head_dim)
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, config.layernorm_epsilon, device=device, dtype=dtype)
        self.kv_b_proj = self._column_linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        self.g_proj = self._column_linear(hidden_size, self.num_heads * self.v_head_dim)
        self.o_proj = self._row_linear(self.num_heads * self.v_head_dim, hidden_size)

        self.core_attention = TEDotProductAttention(
            config=self.config,
            layer_number=self.layer_idx + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=self.q_head_dim**-0.5,
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            cp_comm_type=self.cp_comm_type,
            pg_collection=self.pg_collection,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Return attention state with explicit KDA TP sharding."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if not self.is_kda:
            return sharded_state_dict

        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict.update(
            make_sharded_tensors_for_checkpoint(
                {"A_log": self.A_log, "dt_bias": self.dt_bias},
                prefix,
                {"A_log": 0, "dt_bias": 0},
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return sharded_state_dict

    def _gate_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Project the KDA output gate to ``[..., local_projection_size]``.

        Kimi K3 gates from a single column-parallel projection. Variants with a
        different gate (GLM-5 Next's two-stage low-rank gate, for one) override
        this and inherit both the dense and the context-parallel forward.
        """
        return _linear(self.g_proj, x)

    def _forward_kda(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        if self.cp_size > 1:
            return self._forward_kda_cp(hidden_states, packed_seq_params)
        x = hidden_states.transpose(0, 1)
        cu_seqlens = _physical_cu_seqlens(packed_seq_params)

        conv_kwargs = {"output_final_state": False, "cu_seqlens": cu_seqlens}
        q, _ = self.q_conv1d(x=_linear(self.q_proj, x), **conv_kwargs)
        k, _ = self.k_conv1d(x=_linear(self.k_proj, x), **conv_kwargs)
        v, _ = self.v_conv1d(x=_linear(self.v_proj, x), **conv_kwargs)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.local_num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.local_num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.local_num_heads)
        forget_gate = rearrange(
            _linear(self.f_b_proj, _linear(self.f_a_proj, x)),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        beta = _linear(self.b_proj, x).float().sigmoid()
        sequence_length = q.shape[1]
        (q, k, v, forget_gate, beta), kda_cu_seqlens, valid_length = _prepare_kda_inputs(
            (q, k, v, forget_gate, beta), cu_seqlens
        )
        output = kda(
            q,
            k,
            v,
            forget_gate,
            beta,
            self.A_log,
            self.dt_bias,
            self.gate_lower_bound,
            cu_seqlens=kda_cu_seqlens,
        )
        if valid_length < sequence_length:
            if output.requires_grad:
                output.register_hook(lambda grad: grad.clone(memory_format=torch.contiguous_format))
            output = F.pad(output, (0, 0, 0, 0, 0, sequence_length - valid_length))
        gate = rearrange(self._gate_projection(x), "b s (h d) -> b s h d", h=self.local_num_heads)
        output = self.o_norm(output.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim))
        output = output.view(*gate.shape).flatten(-2)
        return _linear(self.o_proj, output.to(hidden_states.dtype)).transpose(0, 1)

    def _resolve_global_cu_seqlens(
        self, packed_seq_params: PackedSeqParams | None, seq_len_global: int
    ) -> torch.Tensor | None:
        cu_seqlens = _physical_cu_seqlens(packed_seq_params)
        if cu_seqlens is None:
            return None
        if int(cu_seqlens[-1]) != seq_len_global:
            raise ValueError(
                f"cu_seqlens must be global under context parallelism: got total "
                f"{int(cu_seqlens[-1])} for global sequence length {seq_len_global}"
            )
        # _build_thd_cp_a2a_perm floor-divides each document length by 2*cp and
        # each start offset by cp. A remainder there does not raise; it produces
        # an out-of-range chunk index that either scrambles tokens across
        # documents or dies later in index_select with a device-side assert.
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if bool((lengths % (2 * self.cp_size) != 0).any()):
            raise ValueError(
                f"every packed document length must be a multiple of 2*cp "
                f"({2 * self.cp_size}) for the zigzag CP partition, got "
                f"{lengths.tolist()}"
            )
        return cu_seqlens

    def _forward_kda_cp(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        # hidden_states: [s_local, b, hidden], sequence-sharded over CP in the
        # attention load-balanced (zigzag) layout the THD packer emits.
        cp = self.cp_size
        seq_len_global = hidden_states.shape[0] * cp
        cu_seqlens = self._resolve_global_cu_seqlens(packed_seq_params, seq_len_global)

        sections = kda_cp_split_sections(self.local_projection_size, self.local_num_heads)
        packed = torch.cat(
            [
                _linear(self.q_proj, hidden_states),
                _linear(self.k_proj, hidden_states),
                _linear(self.v_proj, hidden_states),
                _linear(self.f_b_proj, _linear(self.f_a_proj, hidden_states)),
                self._gate_projection(hidden_states),
                _linear(self.b_proj, hidden_states),
            ],
            dim=-1,
        )
        # [s_local, b, sum(sections)] -> [s_global, b, sum(sections)/cp] in
        # natural token order (the THD perm folds the un-zigzag).
        packed, thd_cp_a2a_inv = a2a_cp_to_hp(
            packed,
            sections,
            cp,
            self.cp_group,
            cu_seqlens,
            seq_len_global,
            packed_seq_params,
        )
        q, k, v, forget_gate, gate, beta_logits = torch.split(packed, [section // cp for section in sections], dim=-1)

        conv_inputs = {"q": q, "k": k, "v": v}
        for name in conv_inputs:
            conv_weight = get_parameter_local_cp(getattr(self, f"{name}_conv1d").weight, dim=0, cp_group=self.cp_group)
            conv_inputs[name] = _short_conv(conv_inputs[name], conv_weight, cu_seqlens)

        heads_cp = self.local_num_heads // cp
        q = rearrange(conv_inputs["q"].transpose(0, 1), "b s (h d) -> b s h d", h=heads_cp)
        k = rearrange(conv_inputs["k"].transpose(0, 1), "b s (h d) -> b s h d", h=heads_cp)
        v = rearrange(conv_inputs["v"].transpose(0, 1), "b s (h d) -> b s h d", h=heads_cp)
        forget_gate = rearrange(forget_gate.transpose(0, 1), "b s (h d) -> b s h d", h=heads_cp)
        beta = beta_logits.transpose(0, 1).float().sigmoid()
        (q, k, v, forget_gate, beta), kda_cu_seqlens, _ = _prepare_kda_inputs((q, k, v, forget_gate, beta), cu_seqlens)

        output = kda(
            q,
            k,
            v,
            forget_gate,
            beta,
            get_parameter_local_cp(self.A_log, dim=0, cp_group=self.cp_group),
            get_parameter_local_cp(self.dt_bias, dim=0, cp_group=self.cp_group),
            self.gate_lower_bound,
            # The original metadata remains in use by the CP all-to-all and
            # document-aware convolution above.
            cu_seqlens=kda_cu_seqlens,
        )
        gate = rearrange(gate.transpose(0, 1), "b s (h d) -> b s h d", h=heads_cp)
        output = self.o_norm(output.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim))
        output = output.view(*gate.shape).flatten(-2).transpose(0, 1)
        # [s_global, b, proj/cp] -> [s_local, b, proj], back in zigzag layout.
        output = a2a_hp_to_cp(output, cp, self.cp_group, packed_seq_params, thd_cp_a2a_inv)
        return _linear(self.o_proj, output.to(hidden_states.dtype))

    def _forward_mla(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        x = hidden_states
        query = _linear(self.q_b_proj, self.q_a_layernorm(_linear(self.q_a_proj, x)))
        query = query.view(*query.shape[:-1], self.local_num_heads, self.q_head_dim)

        compressed_kv = _linear(self.kv_a_proj_with_mqa, x)
        kv_latent, key_extra = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_extra_head_dim],
            dim=-1,
        )
        key_value = _linear(self.kv_b_proj, self.kv_a_layernorm(kv_latent))
        key_value = key_value.view(
            *key_value.shape[:-1],
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_nope, value = torch.split(
            key_value,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        value = value.contiguous()
        key_extra = copy_to_tensor_model_parallel_region(key_extra, group=self.tp_group)
        key_extra = key_extra.unsqueeze(-2).expand(*key_nope.shape[:-1], -1)
        key = torch.cat((key_nope, key_extra), dim=-1)

        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if is_thd:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        output = self.core_attention(
            query,
            key,
            value,
            None,
            packed_seq_params=packed_seq_params,
            attn_mask_type=AttnMaskType.causal,
        )
        if is_thd:
            output = output.reshape(output.size(0), 1, -1)
        output = output * torch.sigmoid(_linear(self.g_proj, x))
        return _linear(self.o_proj, output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Run the attention implementation selected for this layer."""
        del (
            attention_mask,
            key_value_states,
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            sequence_len_offset,
            kwargs,
        )
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=self.tp_group,
            )
        output = (
            self._forward_kda(hidden_states, packed_seq_params)
            if self.is_kda
            else self._forward_mla(hidden_states, packed_seq_params)
        )
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output, None


class KimiK3MoELayer(MoELayer):
    """MCore latent MoE with Kimi's post-combine RMS normalization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.config.moe_latent_size is None:
            raise ValueError("Kimi K3 MoE layers require moe_latent_size")
        self.routed_expert_norm = KimiRMSNorm(
            self.config.moe_latent_size,
            self.config.layernorm_epsilon,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        _mark_tp_replicated(self.routed_expert_norm, reduction="sum")

    def postprocess(
        self,
        output: torch.Tensor,
        shared_expert_output: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize the combined routed-expert output before projecting up."""
        output = self.token_dispatcher.combine_postprocess(output)
        output = self.routed_expert_norm(output)
        output, _ = self.fc2_latent_proj(output)

        if shared_expert_output is not None:
            output = output + shared_expert_output
        elif self._latent_shared_expert_output is not None:
            raise ValueError("Kimi K3 does not support latent shared-expert overlap")
        return output


class KimiK3TransformerLayer(TransformerLayer):
    """Transformer layer implementing Kimi K3's AttnRes residual bank."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.config.hidden_dropout != 0.0:
            raise ValueError("Kimi K3 requires hidden_dropout=0")

        hidden_size = self.config.hidden_size
        eps = self.config.layernorm_epsilon
        device = torch.cuda.current_device()
        dtype = self.config.params_dtype
        self.attn_res_block_size = self.config.kimi_attn_res_block_size
        self.self_attention_res_norm = KimiRMSNorm(hidden_size, eps, device=device, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
        self.mlp_res_norm = KimiRMSNorm(hidden_size, eps, device=device, dtype=dtype)
        self.mlp_res_proj = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
        _mark_tp_replicated(self.self_attention_res_norm, reduction="sum")
        _mark_tp_replicated(self.self_attention_res_proj, reduction="sum")
        _mark_tp_replicated(self.mlp_res_norm, reduction="sum")
        _mark_tp_replicated(self.mlp_res_proj, reduction="sum")

        if self.layer_number == self.config.num_layers:
            self.output_attn_res_norm = KimiRMSNorm(hidden_size, eps, device=device, dtype=dtype)
            self.output_attn_res_proj = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)
            _mark_tp_replicated(self.output_attn_res_norm, reduction="sum")
            _mark_tp_replicated(self.output_attn_res_proj, reduction="sum")

        pp_size = self.config.pipeline_model_parallel_size
        stage_starts = {get_transformer_layer_offset(self.config, None, rank) for rank in range(pp_size)}
        layer_idx = self.layer_number - 1
        self.is_stage_entry = layer_idx in stage_starts and layer_idx > 0
        self.is_stage_exit = (layer_idx + 1) in stage_starts and layer_idx + 1 < self.config.num_layers

    @staticmethod
    def _add_bias(output_with_bias: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
        output, bias = output_with_bias
        return output if bias is None else output + bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply attention, MLP/MoE, and AttnRes state updates."""
        del context_mask, input_ids, kwargs
        layer_idx = self.layer_number - 1

        if context is not None:
            prefix_sum = hidden_states
            block_residual = context
        elif layer_idx == 0:
            prefix_sum = hidden_states
            block_residual = hidden_states.new_empty(*hidden_states.shape[:-1], 0, hidden_states.shape[-1])
        else:
            if not self.is_stage_entry:
                raise ValueError("Attention-residual snapshot bank is missing")
            prefix_sum, block_residual = unpack_stage_boundary(
                hidden_states,
                self.config.hidden_size,
                bank_num_rows(layer_idx, self.attn_res_block_size),
            )

        if block_residual.shape[-2] > 0:
            attention_input = attn_res_aggregate(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            attention_input = self.input_layernorm(prefix_sum)

        is_block_write_layer = layer_idx % self.attn_res_block_size == 0
        if is_block_write_layer:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)

        attention_output = self._add_bias(
            self.self_attention(
                attention_input,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        )
        prefix_sum = attention_output if is_block_write_layer else prefix_sum + attention_output

        mlp_input = attn_res_aggregate(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.pre_mlp_layernorm,
        )
        mlp_output = self._add_bias(self.mlp(mlp_input, padding_mask=padding_mask))
        prefix_sum = prefix_sum + mlp_output

        if self.layer_number == self.config.num_layers:
            prefix_sum = attn_res_aggregate(
                prefix_sum,
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
                nn.Identity(),
            )

        if self.is_stage_exit:
            return pack_stage_boundary(prefix_sum, block_residual), block_residual
        return prefix_sum, block_residual
