#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""The fixed-grid Qwen3-VL helpers must give exactly what the stock transformers model gives."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("transformers")

from transformers import Qwen3VLModel  # noqa: E402

from lerobot.policies.common.qwen3_vision import (  # noqa: E402
    Qwen3VisionGrid,
    qwen3_vision_forward,
    qwen3_vl_forward,
)
from tests.policies.common.fixed_grid_checks import data_dependent_nodes, save_and_load  # noqa: E402
from tests.policies.common.qwen3_vl_helpers import (  # noqa: E402
    IMAGE_TOKEN_ID,
    STOCK_DEEPSTACK_NODES,
    image_prompts,
    tiny_qwen3_vl_config,
)


# A fresh model per test: the stock forward keeps `rope_deltas` on the model, which would leak between tests.
@pytest.fixture
def model():
    torch.manual_seed(0)
    return Qwen3VLModel(tiny_qwen3_vl_config()).eval()


def _inputs(model):
    prompts = image_prompts(model.config)
    return tuple(prompts[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"))


class VisionProgram(torch.nn.Module):
    def __init__(self, visual, grid):
        super().__init__()
        self.visual, self.grid = visual, grid

    def forward(self, pixel_values, grid_thw):
        return qwen3_vision_forward(self.visual, pixel_values, grid_thw, self.grid)[0]


def _export_vision(visual, grid, pixel_values, grid_thw):
    return torch.export.export(VisionProgram(visual, grid), (pixel_values, grid_thw), strict=False).module()


# The grid check guards the grid the program holds, so saving or decomposing the program must keep it.
def test_qwen3_vl_forward_matches_stock_model(model):
    input_ids, attention_mask, pixel_values, grid_thw = _inputs(model)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    grid = Qwen3VisionGrid.build(model.visual, grid_thw)
    image_mask = input_ids == IMAGE_TOKEN_ID
    with torch.no_grad():
        expected = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
        ).last_hidden_state
        actual = qwen3_vl_forward(
            model, inputs_embeds, attention_mask, pixel_values, grid_thw, grid, image_mask
        )
        # Other pixels on the same grid: a saved program must read them, not replay its example output.
        other_pixels = pixel_values.flip(0).contiguous()
        expected_other = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=other_pixels,
            image_grid_thw=grid_thw,
        ).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(expected_other, expected)

    class Wrapper(torch.nn.Module):
        def forward(self, inputs_embeds, attention_mask, pixel_values, grid_thw, image_mask):
            return qwen3_vl_forward(
                model, inputs_embeds, attention_mask, pixel_values, grid_thw, grid, image_mask
            )

    args = (inputs_embeds, attention_mask, pixel_values, grid_thw, image_mask)
    # Same image and token counts, other shape: the exported program's grid check rejects it.
    changed = grid_thw.clone()
    changed[1, 1:] = changed[1, 1:].flip(0)
    with torch.no_grad():
        program = torch.export.export(Wrapper(), args, strict=False)
        torch.testing.assert_close(program.module()(*args), expected, rtol=0, atol=0)
        for lowered in (program, save_and_load(program), program.run_decompositions()):
            # The decompositions round differently.
            torch.testing.assert_close(lowered.module()(*args), expected)
            other = lowered.module()(inputs_embeds, attention_mask, other_pixels, grid_thw, image_mask)
            torch.testing.assert_close(other, expected_other)
            with pytest.raises(RuntimeError, match="Runtime assertion failed"):
                lowered.module()(inputs_embeds, attention_mask, pixel_values, changed, image_mask)
    assert data_dependent_nodes(program) == STOCK_DEEPSTACK_NODES


def test_qwen3_vl_forward_rejects_a_wrong_image_token_count(model):
    input_ids, attention_mask, pixel_values, grid_thw = _inputs(model)
    image_mask = input_ids == IMAGE_TOKEN_ID
    image_mask[0, image_mask[0].nonzero()[0, 0]] = False
    grid = Qwen3VisionGrid.build(model.visual, grid_thw)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    with torch.no_grad(), pytest.raises(ValueError, match="Image features and image tokens do not match"):
        qwen3_vl_forward(model, inputs_embeds, attention_mask, pixel_values, grid_thw, grid, image_mask)


def test_qwen3_vl_forward_needs_positions_once_rope_deltas_are_cached(model):
    input_ids, attention_mask, pixel_values, grid_thw = _inputs(model)
    image_mask = input_ids == IMAGE_TOKEN_ID
    grid = Qwen3VisionGrid.build(model.visual, grid_thw)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    with torch.no_grad():
        # A call with input_ids caches rope deltas, which the stock forward then adds to the positions it builds
        # for a call without input_ids.
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            mm_token_type_ids=image_mask.int(),
        )
        assert model.rope_deltas is not None
        with pytest.raises(ValueError, match="pass position_ids"):
            qwen3_vl_forward(model, inputs_embeds, attention_mask, pixel_values, grid_thw, grid, image_mask)


def test_vision_forward_matches_stock_tower_in_bfloat16(model):
    # Released checkpoints run in bfloat16, where the interpolated position embeddings come out in float32.
    visual = model.visual.to(torch.bfloat16)
    _, _, pixel_values, grid_thw = _inputs(model)
    grid = Qwen3VisionGrid.build(visual, grid_thw)
    with torch.no_grad():
        actual = qwen3_vision_forward(visual, pixel_values, grid_thw, grid)[0]
        expected = visual(pixel_values, grid_thw=grid_thw).pooler_output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_grid_follows_learned_position_updates(model):
    visual = deepcopy(model.visual)
    _, _, pixel_values, grid_thw = _inputs(model)
    with torch.inference_mode():  # as when it is built next to a prediction
        grid = Qwen3VisionGrid.build(visual, grid_thw)
    with torch.no_grad():
        # A training step changes pos_embed after the grid exists; an export must see the new weights.
        visual.pos_embed.weight.add_(1.0)
    # An ordinary grad-enabled export of the trainable tower, with a grid built in inference mode.
    exported = _export_vision(visual, grid, pixel_values, grid_thw)(pixel_values, grid_thw)
    exported.square().sum().backward()
    exported_grad = visual.pos_embed.weight.grad.clone()
    visual.zero_grad()
    stock = visual(pixel_values, grid_thw=grid_thw).pooler_output
    stock.square().sum().backward()
    torch.testing.assert_close(exported, stock, rtol=0, atol=0)
    torch.testing.assert_close(exported_grad, visual.pos_embed.weight.grad, rtol=0, atol=0)
