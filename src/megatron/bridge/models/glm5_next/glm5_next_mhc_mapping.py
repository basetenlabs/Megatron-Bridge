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

"""Split GLM-5.3's packed mHC ``scale`` tensor onto Megatron-Core's three alphas.

GLM-5.3 stores one ``[3]`` tensor per hyper-connection; Megatron-Core keeps three
separate ``(1,)`` parameters. The order is fixed by HF's own unpack::

    pre_scale, post_scale, comb_scale = self.scale.unbind(0)

    scale[0] -> alpha_pre
    scale[1] -> alpha_post
    scale[2] -> alpha_res     (HF calls it "comb", the residual combine matrix)

which is the same order as ``mapping_proj``'s row blocks -- HF splits ``fn`` as
``[n, n, n*n]`` into (pre, post, comb) and Megatron-Core slices ``h`` as
``[:n]``, ``[n:2n]``, ``[2n:]``. Everything else about mHC is a 1:1 match, verified
term by term against ``transformers``:

* ``h = r * proj * alpha + bias``           <-> ``linear(input_norm(x), fn) * scale + base``
* ``h_pre  = sigmoid(.) + eps``             <-> ``sigmoid(.) + hc_eps``
* ``h_post = sigmoid(.) * 2``               <-> ``2 * sigmoid(.)``
* ``_sinkhorn_iterations``                  <-> HF's loop, line for line, including the
  asymmetric first ``sum(dim=-2)`` normalisation
* ``_MHC_COMPUTE_H_EPS = 1e-6``             <-> ``hc_eps: 1e-06``

The one place they differ is the block-exit contraction, which is handled by
``mhc_learned_output_contract=False`` on the provider rather than here: GLM-5.3
contracts by an unweighted mean and ships no ``hc_head_*`` weights, while DeepSeek-V4
introduced a learned gated sum.
"""

from typing import Dict, Optional

import torch
from torch import nn

from megatron.bridge.models.conversion.param_mapping import (
    MegatronParamMapping,
    get_module_and_param_from_name,
)

# HF's unbind order. Index into the packed [3] tensor -> Megatron parameter name.
MHC_SCALE_ORDER = ("alpha_pre", "alpha_post", "alpha_res")


class Glm5NextMhcScaleMapping(MegatronParamMapping[torch.Tensor]):
    """Map one element of GLM-5.3's packed mHC ``scale`` onto one Megatron alpha.

    Three of these share a single HF tensor, one per alpha. All parameters here are
    replicated (``mapping_proj`` is a plain ``nn.Linear`` and the alphas are scalars),
    so there is no tensor-parallel sharding to do in either direction.

    Args:
        megatron_param: full dotted path of the target alpha.
        hf_param: the packed ``hc_{attn,ffn}_scale`` tensor.
        index: position in HF's ``scale.unbind(0)`` -- 0 pre, 1 post, 2 res.
    """

    def __init__(self, megatron_param: str, hf_param: str, index: int):
        if not 0 <= index < len(MHC_SCALE_ORDER):
            raise ValueError(f"mHC scale index must be 0..{len(MHC_SCALE_ORDER) - 1}, got {index}")
        super().__init__(megatron_param, hf_param)
        self.index = index

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Take this alpha's element out of the packed tensor."""
        if hf_weights.numel() != len(MHC_SCALE_ORDER):
            raise ValueError(
                f"{self.hf_param} has {hf_weights.numel()} elements, expected "
                f"{len(MHC_SCALE_ORDER)} (pre, post, res). A different packing order "
                "would silently permute the mHC gating factors."
            )
        return hf_weights.reshape(-1)[self.index].reshape(1).clone()

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Repack the three alphas, from the ``alpha_pre`` mapping only.

        Three mappings target one HF tensor, so exactly one of them must own the export
        or they would emit conflicting values for the same key. Index 0 reads its two
        siblings off the parent module and writes the packed tensor; the others return
        nothing.
        """
        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )
        if megatron_weights is None or self.index != 0:
            return {}

        parent = self.megatron_param.rsplit(".", 1)[0]
        alphas = []
        for name in MHC_SCALE_ORDER:
            _, param = get_module_and_param_from_name(megatron_module, f"{parent}.{name}")
            alphas.append(self.maybe_dequantize(param.detach()).reshape(-1))
        return {self.hf_param: torch.cat(alphas, dim=0)}


def mhc_mappings(megatron_layer: str, hf_layer: str) -> list:
    """Build every mHC mapping for one decoder layer.

    GLM-5.3 carries two hyper-connections per layer -- one around attention and one
    around the MLP -- named ``hc_attn_*`` and ``hc_ffn_*`` on the HF side against
    ``self_attention_hyper_connection`` and ``mlp_hyper_connection`` in
    ``TransformerLayer``.
    """
    from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping

    mappings: list = []
    for hf_prefix, megatron_module in (
        ("hc_attn", "self_attention_hyper_connection"),
        ("hc_ffn", "mlp_hyper_connection"),
    ):
        base = f"{megatron_layer}.{megatron_module}"
        mappings += [
            # fn [(2+n)*n, n*hidden] -> mapping_proj.weight, same shape, replicated.
            ReplicatedMapping(
                megatron_param=f"{base}.mapping_proj.weight",
                hf_param=f"{hf_layer}.{hf_prefix}_fn",
            ),
            # base [(2+n)*n] -> the static bias term.
            ReplicatedMapping(
                megatron_param=f"{base}.bias",
                hf_param=f"{hf_layer}.{hf_prefix}_base",
            ),
        ]
        mappings += [
            Glm5NextMhcScaleMapping(
                megatron_param=f"{base}.{name}",
                hf_param=f"{hf_layer}.{hf_prefix}_scale",
                index=index,
            )
            for index, name in enumerate(MHC_SCALE_ORDER)
        ]
    return mappings
