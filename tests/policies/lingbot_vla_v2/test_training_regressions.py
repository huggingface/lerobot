"""Small CPU regressions for the fix-all behavior retained on master's layout."""

from types import MethodType

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.lingbot_vla_v2.model_core.qwen3vl_in_vla import (
    forward_without_grid_thw,
    preprcess_grid_thw,
)
from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import _prepare_camera_frame


def test_square_resize_scales_to_qwen_range():
    current = _prepare_camera_frame(torch.ones(3, 480, 640), (256, 256))
    assert current.shape == (3, 256, 256)
    # [0, 1] float inputs are scaled to the [0, 255] range Qwen3-VL expects.
    assert current.max().item() == pytest.approx(255.0, abs=1e-2)


class TinyVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = torch.nn.Embedding(4, 3)
        self.patch_embed = torch.nn.Identity()
        self.blocks = []
        self.deepstack_visual_indexes = []
        self.merger = torch.nn.Identity()
        self.spatial_merge_size = 1
        self.preprcess_grid_thw = MethodType(preprcess_grid_thw, self)
        self.forward = MethodType(forward_without_grid_thw, self)

    def rot_pos_emb(self, grid):
        return torch.zeros(4, 2)

    def fast_pos_embed_interpolate(self, grid):
        return self.pos_embed(torch.arange(4))


def test_cached_grid_allows_two_position_embedding_updates():
    visual = TinyVisual()
    grid = torch.tensor([[1, 2, 2]])
    pos, rope, cu, _, maxlen = visual.preprcess_grid_thw(grid)
    assert pos is None  # Never cache the trainable interpolation graph.
    optimizer = torch.optim.SGD(visual.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        before = visual.pos_embed.weight.detach().clone()
        out, _ = visual(torch.ones(4, 3), grid, pos, rope, cu, maxlen)
        out.square().sum().backward()
        assert visual.pos_embed.weight.grad is not None
        optimizer.step()
        assert not torch.equal(before, visual.pos_embed.weight)
