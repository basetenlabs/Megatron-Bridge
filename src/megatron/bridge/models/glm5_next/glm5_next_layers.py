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

"""KDA linear-attention layer for GLM-5.3-Flash.

Only the **KDA half** of GLM-5.3's attention lives here. The MLA+DSA layers reuse
Megatron-Core's ``get_dsa_module_spec_for_backend`` spec unmodified -- see
``glm5_next_spec``. That asymmetry is deliberate: it keeps the sparse-attention path
on a maintained upstream code path and confines new code to the linear-attention half.

GLM-5.3 and Kimi K3 use the same underlying linear-attention kernel
(``fla.ops.kda.chunk_kda``) and the same short-convolution + forget-gate structure, so
the layer is adapted from ``KimiK3Attention``'s KDA branch. Two structural differences:

* This module is instantiated **only** for KDA layers, so it has no ``is_kda`` branch.
  ``KimiK3Attention`` is built for every layer and dispatches internally, because K3
  hand-rolls its MLA half too.
* GLM-5.3's MLA is ungated, so there is no shared ``g_proj`` convention to preserve
  across the two halves; the ``g_proj`` here is KDA's own output gate.
"""

import copy

import torch
from einops import rearrange
from fla.modules import FusedRMSNormGated
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELinear,
    TERowParallelLinear,
)
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint
from torch import nn

# Same kernel wrapper and TP-sharded short convolution Kimi K3 uses. Imported rather
# than duplicated: these wrap fla ops and carry non-obvious fp32/TP handling that
# should not fork between two models using the identical kernel.
from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3ShortConvolution
from megatron.bridge.models.kimi.kimi_k3_ops import kda


def _mark_tp_replicated(module: nn.Module, *, reduction: str = "average") -> None:
    """Flag a replicated module's gradients for cross-TP reduction."""
    if reduction not in ("average", "sum"):
        raise ValueError(f"Unsupported TP gradient reduction: {reduction}")
    for parameter in module.parameters():
        setattr(parameter, f"{reduction}_gradients_across_tp_domain", True)


def _linear(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Apply a Megatron linear that is expected to be bias-free."""
    output, bias = module(inputs)
    if bias is not None:
        raise ValueError(f"GLM-5.3 requires bias-free projections, got bias from {type(module).__name__}")
    return output


class Glm5NextLinearAttention(MegatronModule):
    """KDA (Kimi Delta Attention) for GLM-5.3's 34 linear-attention layers.

    Instantiated only for layers listed in ``config.glm5_next_kda_layers``; the layer
    schedule is resolved by the spec builder, not here.
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
        self.layer_number = layer_number
        self.layer_idx = layer_number - 1

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        elif not hasattr(pg_collection, "tp") or not hasattr(pg_collection, "cp"):
            raise ValueError("Glm5NextLinearAttention requires TP and CP process groups")

        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size()
        self.cp_group = pg_collection.cp
        self.cp_size = self.cp_group.size()

        # KDA carries recurrent state across chunks, which context parallelism would
        # split across ranks. Kimi K3 refuses the same combination for the same reason.
        # Until it is solved, GLM-5.3 runs with cp=1 -- the main limit on how far its
        # sequence length can be pushed.
        if self.cp_size > 1:
            raise ValueError("GLM-5.3 KDA context parallelism is not supported yet")

        self.sequence_parallel = config.sequence_parallel
        # The KDA projections are built against a copy with sequence parallelism off:
        # forward() gathers the full sequence before the kernel and scatters after, so
        # the projections themselves see unsharded input.
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False

        self._init_kda(config)

    # ------------------------------------------------------------------ builders

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

    # ---------------------------------------------------------------------- init

    def _init_kda(self, config) -> None:
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype

        self.num_heads = config.glm5_next_linear_num_heads
        if self.num_heads % self.tp_size:
            raise ValueError(f"KDA heads {self.num_heads} must be divisible by TP size {self.tp_size}")
        self.local_num_heads = self.num_heads // self.tp_size
        self.head_dim = config.glm5_next_linear_head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.local_projection_size = self.local_num_heads * self.head_dim

        self.q_proj = self._column_linear(hidden_size, self.projection_size)
        self.k_proj = self._column_linear(hidden_size, self.projection_size)
        self.v_proj = self._column_linear(hidden_size, self.projection_size)

        conv_kwargs = {
            "hidden_size": self.local_projection_size,
            "kernel_size": config.glm5_next_linear_conv_kernel_size,
            "activation": "silu",
            "device": device,
            "dtype": dtype,
            "tp_group": self.tp_group,
        }
        self.q_conv1d = KimiK3ShortConvolution(**conv_kwargs)
        self.k_conv1d = KimiK3ShortConvolution(**conv_kwargs)
        self.v_conv1d = KimiK3ShortConvolution(**conv_kwargs)

        # Forget gate: a low-rank projection (f_a -> f_b) plus a per-head term (b_proj).
        self.f_a_proj = self._duplicated_linear(hidden_size, self.head_dim)
        self.f_b_proj = self._column_linear(self.head_dim, self.projection_size)
        self.b_proj = self._column_linear(hidden_size, self.num_heads)
        self.g_proj = self._column_linear(hidden_size, self.projection_size)

        # Kept in fp32: these feed the fused gate computation inside chunk_kda, where
        # bf16 rounding measurably changes the decay.
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

        self.gate_lower_bound = config.glm5_next_kda_gate_lower_bound

    # ----------------------------------------------------------------- forward

    def _forward_kda(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        # Megatron hands us [s, b, h]; the fla kernels are batch-first.
        x = hidden_states.transpose(0, 1)

        cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        if packed_seq_params is not None and cu_seqlens is None:
            # Without cu_seqlens the recurrence would run across packed-sequence
            # boundaries, silently mixing state between unrelated documents.
            raise ValueError("Packed KDA input requires cu_seqlens_q")

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

        # A_log and dt_bias are passed through so the kernel fuses them into the forget
        # gate. flash-linear-attention < 0.5.2 accepts and silently discards them,
        # training a different gate with nothing raised -- hence the pinned floor.
        output = kda(
            q,
            k,
            v,
            forget_gate,
            beta,
            self.A_log,
            self.dt_bias,
            self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
        )

        gate = rearrange(_linear(self.g_proj, x), "b s (h d) -> b s h d", h=self.local_num_heads)
        output = self.o_norm(output.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim))
        output = output.view(*gate.shape).flatten(-2)
        return _linear(self.o_proj, output.to(hidden_states.dtype)).transpose(0, 1)

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
        """Run KDA over the full sequence.

        Signature matches Megatron-Core's self-attention contract so this module can be
        dropped into ``TransformerLayerSubmodules.self_attention``. GLM-5.3 is NoPE and
        KDA is inherently causal, so the rotary and mask arguments are unused.

        Returns ``(output, bias)`` with ``bias=None``, as the layer's bias-dropout-add
        expects.
        """
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

        # The linear-attention recurrence needs the whole sequence on each rank, so
        # undo sequence-parallel sharding around the kernel.
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=self.tp_group,
            )

        output = self._forward_kda(hidden_states, packed_seq_params)

        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)

        return output, None

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Return attention state with explicit TP sharding for the fp32 KDA parameters."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
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
