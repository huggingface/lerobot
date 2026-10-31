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
    qwen2_5_image_temporal_scale,
    qwen2_5_vision_forward,
)
from lerobot.policies.common.qwen_vision import (  # noqa: E402
    check_visible_images,
    is_exporting,
    prepare_image_grid,
    qwen_image_rope_index,
)
from tests.policies.common.fixed_grid_checks import data_dependent_nodes, save_and_load  # noqa: E402

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
        grid = Qwen2_5VisionGrid.track(self.grid, self.visual, grid_thw)
        return qwen2_5_vision_forward(self.visual, pixel_values, grid)


# Two images of 18x18 patches (as WALL-X gives for 224 px cameras) have uneven attention windows; the
# two-frame grid checks the temporal axis.
@pytest.mark.parametrize("grids", [[[1, 18, 18], [1, 18, 18]], [[2, 8, 12], [1, 16, 22]]])
def test_vision_forward_matches_stock_and_exports(grids):
    visual = _tiny_visual()
    grid_thw = torch.tensor(grids)
    pixel_values = _pixels(grid_thw)
    with torch.no_grad():
        expected = visual(pixel_values, grid_thw=grid_thw).pooler_output
        model = FixedGridVision(visual, Qwen2_5VisionGrid.build(visual, grid_thw))
        exported = torch.export.export(model, (pixel_values, grid_thw), strict=False)
        torch.testing.assert_close(model(pixel_values, grid_thw), expected, rtol=0, atol=0)
        torch.testing.assert_close(exported.module()(pixel_values, grid_thw), expected, rtol=0, atol=0)
    assert data_dependent_nodes(exported) == []


def test_export_without_a_fitting_prepared_grid_names_the_mismatch():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    one_image = Qwen2_5VisionGrid.build(visual, grid_thw)
    cases = {
        "no image grid was prepared": (visual, None),
        "the prepared grid has 2 images but the export inputs have 1": (
            visual,
            Qwen2_5VisionGrid.build(visual, torch.tensor(GRID * 2)),
        ),
        "was prepared for cpu torch.float32, but the tower is now on cpu torch.bfloat16": (
            deepcopy(visual).to(torch.bfloat16),
            one_image,
        ),
    }
    for problem, (tower, cache) in cases.items():
        pixel_values = _pixels(grid_thw).to(tower.dtype)
        with pytest.raises(RuntimeError, match=rf"{problem}.*inside `prepare_image_grid\(\)`"):
            torch.export.export(FixedGridVision(tower, cache), (pixel_values, grid_thw), strict=False)


@pytest.mark.parametrize(
    "changed", [[[1, 12, 8]], [[1, 8, 12], [1, 4, 4]]], ids=["same-count", "more-images"]
)
def test_preparation_rebuilds_a_changed_grid(changed):
    visual = _tiny_visual()
    with prepare_image_grid():
        cache = Qwen2_5VisionGrid.track(None, visual, torch.tensor(GRID))
        rebuilt = Qwen2_5VisionGrid.track(cache, visual, torch.tensor(changed))
        assert rebuilt.grid_thw == changed
        assert Qwen2_5VisionGrid.track(rebuilt, visual, torch.tensor(changed)) is rebuilt


def test_eager_outside_preparation_neither_reads_nor_caches_the_grid():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    cache = Qwen2_5VisionGrid.build(visual, grid_thw)

    class Unreadable(torch.Tensor):
        def tolist(self):
            raise AssertionError("the grid must not be read outside prepare_image_grid()")

    unreadable = grid_thw.as_subclass(Unreadable)
    assert Qwen2_5VisionGrid.track(None, visual, unreadable) is None
    assert Qwen2_5VisionGrid.track(cache, visual, torch.tensor([[1, 12, 8]]).as_subclass(Unreadable)) is cache


def test_a_call_without_images_has_no_grid():
    visual = _tiny_visual()
    cache = Qwen2_5VisionGrid.build(visual, torch.tensor(GRID))
    with prepare_image_grid():
        assert Qwen2_5VisionGrid.track(cache, visual, None) is None


def test_compiled_code_does_not_take_the_export_path():
    @torch.compile(backend="eager", fullgraph=True)
    def exporting(x):
        return x + 1 if is_exporting() else x, x + 1 if torch.compiler.is_exporting() else x

    ours, torchs = exporting(torch.zeros(2))
    # torch.compile on PyTorch 2.11 also reports torch.compiler.is_exporting(); later versions do not.
    if torch.equal(torchs, torch.zeros(2)):
        pytest.skip("torch.compile on this PyTorch does not report torch.compiler.is_exporting()")
    assert torch.equal(ours, torch.zeros(2))


# The grid check is the only guard of the cached grid, so transformers' TRANSFORMERS_DISABLE_TORCH_CHECK=1
# must not remove it, and saving or decomposing the program must keep it.
@pytest.mark.parametrize("transformers_checks_off", [False, True], ids=["switch-unset", "switch-set"])
def test_exported_program_rejects_a_changed_grid(monkeypatch, transformers_checks_off):
    if transformers_checks_off:
        monkeypatch.setenv("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    model = FixedGridVision(visual, Qwen2_5VisionGrid.build(visual, grid_thw))
    with torch.no_grad():
        program = torch.export.export(model, (_pixels(grid_thw), grid_thw), strict=False)
        for lowered in (program, save_and_load(program), program.run_decompositions()):
            lowered.module()(_pixels(grid_thw), grid_thw)
            # 12x8 has the image and patch counts of the prepared 8x12 grid.
            with pytest.raises(RuntimeError, match="Runtime assertion failed"):
                lowered.module()(_pixels(grid_thw), torch.tensor([[1, 12, 8]]))


def test_masks_hiding_more_than_padding_are_refused_also_after_export():
    config = _tiny_config()
    image = [VISION_START_ID, *[IMAGE_TOKEN_ID] * 6, VISION_END_ID]
    # The second row has two images without boundary tokens, separated by one text token.
    input_ids = torch.tensor([[0, 1, *image, 2], [5, *[IMAGE_TOKEN_ID] * 4, 6, *[IMAGE_TOKEN_ID] * 4, 0]])
    attention_mask = input_ids.ne(0).long()

    class CheckVisibleImages(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            check_visible_images(input_ids, attention_mask, config)
            return attention_mask + 0

    program = torch.export.export(CheckVisibleImages(), (input_ids, attention_mask), strict=False).module()
    more_padding = attention_mask.clone()
    more_padding[0, 1] = 0
    for mask in (attention_mask, more_padding):
        check_visible_images(input_ids, mask, config)
        program(input_ids, mask)
    # The vision start token with the text before it, the last image token and the vision end token with the
    # text after it (each leaves one visible run), the text token between the second row's images (eager would
    # join them into one image), and a whole row.
    for row, columns in ((0, [1, 2]), (1, [9]), (0, [9, 10]), (1, [5]), (1, list(range(11)))):
        hidden = attention_mask.clone()
        hidden[row, columns] = 0
        with pytest.raises(ValueError, match="must hide only padding"):
            check_visible_images(input_ids, hidden, config)
        with pytest.raises(RuntimeError):
            program(input_ids, hidden)


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
    # transformers 5.4.0 to 5.5.3 scale the temporal position of still images; the probe reads the scale.
    scale = qwen2_5_image_temporal_scale(model)
    position_ids = qwen_image_rope_index(
        input_ids, attention_mask, grids, 2, IMAGE_TOKEN_ID, image_temporal_scale=scale
    )

    assert torch.equal(position_ids, expected)


def test_image_temporal_scale_moves_only_the_temporal_start_of_images():
    input_ids = torch.tensor([[1, 2] + [IMAGE_TOKEN_ID] * 6 + [3]])
    attention_mask = torch.ones_like(input_ids)
    plain = qwen_image_rope_index(input_ids, attention_mask, [[1, 4, 6]], 2, IMAGE_TOKEN_ID)
    scaled = qwen_image_rope_index(
        input_ids, attention_mask, [[1, 4, 6]], 2, IMAGE_TOKEN_ID, image_temporal_scale=4
    )
    image = input_ids[0] == IMAGE_TOKEN_ID
    assert torch.equal(scaled[0, 0, image], plain[0, 0, image] * 4)  # every image token starts at 2
    assert torch.equal(scaled[0, 0, ~image], plain[0, 0, ~image])
    assert torch.equal(scaled[1:], plain[1:])


def test_image_rope_index_rejects_video_grids():
    input_ids = torch.tensor([[1] + [IMAGE_TOKEN_ID] * 12 + [2]])
    with pytest.raises(ValueError, match="still images only"):
        qwen_image_rope_index(input_ids, torch.ones_like(input_ids), [[2, 4, 6]], 2, IMAGE_TOKEN_ID)


def test_reused_grid_follows_dtype_changes():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    pixel_values = _pixels(grid_thw)
    with prepare_image_grid():
        grid = Qwen2_5VisionGrid.track(None, visual, grid_thw)
        visual.to(torch.bfloat16)
        grid = Qwen2_5VisionGrid.track(grid, visual, grid_thw)
    pixel_values = pixel_values.bfloat16()
    with torch.no_grad():
        program = torch.export.export(
            FixedGridVision(visual, grid), (pixel_values, grid_thw), strict=False
        ).module()
        torch.testing.assert_close(
            program(pixel_values, grid_thw),
            visual(pixel_values, grid_thw=grid_thw).pooler_output,
            rtol=0,
            atol=0,
        )


def test_grid_built_in_inference_mode_can_train():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    pixel_values = _pixels(grid_thw)
    with torch.inference_mode(), prepare_image_grid():
        grid = Qwen2_5VisionGrid.track(None, visual, grid_thw)
    visual.train()
    # The cache feeds the exported computation, which is what would save its tensors for backward.
    program = torch.export.export(
        FixedGridVision(visual, grid), (pixel_values, grid_thw), strict=False
    ).module()
    program(pixel_values, grid_thw).square().sum().backward()
    cached_grad = visual.patch_embed.proj.weight.grad.clone()
    visual.zero_grad()
    visual(pixel_values, grid_thw=grid_thw).pooler_output.square().sum().backward()
    torch.testing.assert_close(cached_grad, visual.patch_embed.proj.weight.grad, rtol=0, atol=0)
