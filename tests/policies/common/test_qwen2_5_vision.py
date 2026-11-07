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

"""The fixed-grid Qwen2.5-VL helpers must give exactly what the stock transformers code gives."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig  # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)

from lerobot.policies.common.qwen2_5_vision import (  # noqa: E402
    Qwen2_5VisionGrid,
    qwen2_5_vision_forward,
)
from lerobot.policies.common.qwen_vision import (  # noqa: E402
    check_visible_images,
    is_exporting,
    qwen_image_rope_index,
)
from tests.policies.common.fixed_grid_checks import data_dependent_nodes  # noqa: E402

IMAGE_TOKEN_ID = 99
VISION_START_ID, VISION_END_ID = 97, 98
GRID = [[1, 8, 12]]


def _tiny_config() -> Qwen2_5_VLConfig:
    config = Qwen2_5_VLConfig(
        vision_config={
            "depth": 4,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 48,
            "fullatt_block_indexes": [1, 3],
            "window_size": 112,
        },
        text_config={"hidden_size": 48, "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 128},
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
    )
    config.vision_config._attn_implementation = "sdpa"
    return config


def _tiny_visual():
    torch.manual_seed(0)
    return Qwen2_5_VisionTransformerPretrainedModel._from_config(_tiny_config().vision_config).eval()


def _pixels(grid_thw: torch.Tensor) -> torch.Tensor:
    return torch.randn(int(grid_thw.prod(-1).sum()), 3 * 2 * 14 * 14)


class FixedGridVision(torch.nn.Module):
    def __init__(self, visual, grid):
        super().__init__()
        self.visual = visual
        self.grid = grid

    def forward(self, pixel_values, grid_thw):
        return qwen2_5_vision_forward(self.visual, pixel_values, grid_thw, self.grid)


# The two-frame grid checks the temporal axis.
def test_vision_forward_matches_stock_and_exports():
    visual = _tiny_visual()
    grid_thw = torch.tensor([[2, 8, 12], [1, 16, 22]])
    pixel_values = _pixels(grid_thw)
    with torch.no_grad():
        expected = visual(pixel_values, grid_thw=grid_thw).pooler_output
        model = FixedGridVision(visual, Qwen2_5VisionGrid.build(visual, grid_thw))
        exported = torch.export.export(model, (pixel_values, grid_thw), strict=False)
        torch.testing.assert_close(model(pixel_values, grid_thw), expected, rtol=0, atol=0)
        torch.testing.assert_close(exported.module()(pixel_values, grid_thw), expected, rtol=0, atol=0)
    assert data_dependent_nodes(exported) == []


def test_vision_forward_refuses_a_grid_built_for_another_call():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    one_image = Qwen2_5VisionGrid.build(visual, grid_thw)
    wider_windows = deepcopy(visual)
    wider_windows.window_size = 224
    smaller_patches = deepcopy(visual)
    smaller_patches.patch_size = 7
    cases = {
        "built for 2 images, but the call has 1": (
            visual,
            Qwen2_5VisionGrid.build(visual, torch.tensor(GRID * 2)),
        ),
        "built for cpu torch.float32, but the vision tower is on cpu torch.bfloat16": (
            deepcopy(visual).to(torch.bfloat16),
            one_image,
        ),
        # Other attention windows from the same grid.
        "but the tower has 224 and 14": (wider_windows, one_image),
        "but the tower has 112 and 7": (smaller_patches, one_image),
    }
    for problem, (tower, grid) in cases.items():
        pixel_values = _pixels(grid_thw).to(tower.dtype)
        with torch.no_grad(), pytest.raises(ValueError, match=problem):
            FixedGridVision(tower, grid)(pixel_values, grid_thw)


def test_compiled_code_does_not_take_the_export_path():
    @torch.compile(backend="eager", fullgraph=True)
    def exporting(x):
        return x + 1 if is_exporting() else x

    # torch.compile also reports torch.compiler.is_exporting() on some PyTorch releases, 2.11 among them.
    assert torch.equal(exporting(torch.zeros(2)), torch.zeros(2))


def test_masks_hiding_more_than_padding_are_refused():
    config = _tiny_config()
    image = [VISION_START_ID, *[IMAGE_TOKEN_ID] * 6, VISION_END_ID]
    # The second row has two images without boundary tokens, separated by one text token.
    input_ids = torch.tensor([[0, 1, *image, 2], [5, *[IMAGE_TOKEN_ID] * 4, 6, *[IMAGE_TOKEN_ID] * 4, 0]])
    attention_mask = input_ids.ne(0).long()
    check_visible_images(input_ids, attention_mask, config)
    # The vision start token with the text before it, the last image token and the vision end token with the
    # text after it (each leaves one visible run), and the text token between the second row's images (eager
    # would join them into one image).
    for row, columns in ((0, [1, 2]), (1, [9]), (0, [9, 10]), (1, [5])):
        hidden = attention_mask.clone()
        hidden[row, columns] = 0
        with pytest.raises(ValueError, match="must hide only padding"):
            check_visible_images(input_ids, hidden, config)


def test_image_rope_index_matches_stock():
    model = Qwen2_5_VLModel._from_config(_tiny_config())
    grids = [[1, 4, 6], [1, 4, 4], [1, 6, 2]]
    text = lambda n: list(range(1, n + 1))  # noqa: E731
    image = lambda t, h, w: [IMAGE_TOKEN_ID] * (t * h * w // 4)  # noqa: E731
    rows = [
        text(3) + image(*grids[0]) + text(2) + image(*grids[1]) + text(4),
        text(1) + image(*grids[2]) + text(2),
    ]
    length = max(map(len, rows)) + 1
    input_ids = torch.zeros(len(rows), length, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, row in enumerate(rows):  # right padding for the first prompt, left padding for the second
        span = slice(0, len(row)) if i == 0 else slice(length - len(row), length)
        input_ids[i, span] = torch.tensor(row)
        attention_mask[i, span] = 1

    expected, _ = model.get_rope_index(
        input_ids,
        (input_ids == IMAGE_TOKEN_ID).int(),
        image_grid_thw=torch.tensor(grids),
        attention_mask=attention_mask,
    )
    position_ids = qwen_image_rope_index(input_ids, attention_mask, grids, 2, IMAGE_TOKEN_ID)

    assert torch.equal(position_ids, expected)


def test_image_rope_index_rejects_video_grids():
    input_ids = torch.tensor([[1] + [IMAGE_TOKEN_ID] * 12 + [2]])
    with pytest.raises(ValueError, match="still images only"):
        qwen_image_rope_index(input_ids, torch.ones_like(input_ids), [[2, 4, 6]], 2, IMAGE_TOKEN_ID)


def test_image_tokens_that_do_not_match_the_grid_are_refused():
    input_ids = torch.tensor([[1, 2] + [IMAGE_TOKEN_ID] * 6 + [3]])
    attention_mask = torch.ones_like(input_ids)
    # One image token more than the grid has, which would leave masked_scatter short of values.
    wrong = input_ids.index_fill(1, torch.tensor([1]), IMAGE_TOKEN_ID)
    with pytest.raises(ValueError, match="image tokens do not match the image grid"):
        qwen_image_rope_index(wrong, attention_mask, [[1, 4, 6]], 2, IMAGE_TOKEN_ID)


def test_grid_built_in_inference_mode_can_train():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    pixel_values = _pixels(grid_thw)
    with torch.inference_mode():
        grid = Qwen2_5VisionGrid.build(visual, grid_thw)
    visual.train()
    # The grid feeds the exported computation, which is what would save its tensors for backward.
    program = torch.export.export(
        FixedGridVision(visual, grid), (pixel_values, grid_thw), strict=False
    ).module()
    program(pixel_values, grid_thw).square().sum().backward()
    cached_grad = visual.patch_embed.proj.weight.grad.clone()
    visual.zero_grad()
    visual(pixel_values, grid_thw=grid_thw).pooler_output.square().sum().backward()
    torch.testing.assert_close(cached_grad, visual.patch_embed.proj.weight.grad, rtol=0, atol=0)
