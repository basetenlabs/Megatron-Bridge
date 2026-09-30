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

"""KimiK3VLModel.forward routing, with the backbone mocked out."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from megatron.bridge.models.kimi.kimi_k3_vl_model import KimiK3VLModel


TOKEN = 7
HIDDEN = 4


def _model(sequence_parallel: bool = False) -> SimpleNamespace:
    language_model = MagicMock()
    # [seq, batch, hidden], as Megatron's embedding returns it.
    language_model.embedding.side_effect = lambda input_ids, position_ids: torch.zeros(
        input_ids.shape[1], input_ids.shape[0], HIDDEN
    )
    model = SimpleNamespace(
        pre_process=True,
        language_model=language_model,
        config=SimpleNamespace(
            media_placeholder_token_id=TOKEN, sequence_parallel=sequence_parallel, _pg_collection=None
        ),
        vision_tower=lambda pixels, grid: pixels,
        mm_projector=MagicMock(side_effect=lambda features: features, post_norm=SimpleNamespace(weight=torch.ones(1))),
    )
    model._embed_with_images = partial(KimiK3VLModel._embed_with_images, model)
    return model


def test_text_only_batch_reaches_backbone_unembedded() -> None:
    """GPTModel embeds and, under SP, scatters a text batch itself when no decoder_input is given."""
    model = _model(sequence_parallel=True)
    input_ids = torch.tensor([[1, 2, 3]])

    KimiK3VLModel.forward(model, input_ids=input_ids)

    kwargs = model.language_model.forward.call_args.kwargs
    assert kwargs["decoder_input"] is None
    assert kwargs["input_ids"] is input_ids
    model.language_model.embedding.assert_not_called()


def test_image_batch_splices_features_into_decoder_input() -> None:
    model = _model()
    input_ids = torch.tensor([[1, TOKEN, TOKEN, 2]])
    features = torch.arange(2 * HIDDEN, dtype=torch.float32).view(2, HIDDEN) + 1

    KimiK3VLModel.forward(model, input_ids=input_ids, pixel_values=features, image_grid_thw=torch.tensor([[1, 2, 2]]))

    decoder_input = model.language_model.forward.call_args.kwargs["decoder_input"]
    assert decoder_input.shape == (4, 1, HIDDEN)
    torch.testing.assert_close(decoder_input[1:3, 0], features)
    assert torch.count_nonzero(decoder_input[[0, 3]]) == 0
