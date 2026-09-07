"""Small CPU regressions for the fix-all behavior retained on master's layout."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from lerobot.policies.lingbot_vla_v2.model_core.qwen3vl_in_vla import (
    forward_without_grid_thw,
    preprcess_grid_thw,
)
from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import (
    LingbotVLAV2FeatureTransformStep,
    _future_video_fps,
)
from lerobot.policies.lingbot_vla_v2.teachers.morgbd_teacher import MoRGBDTeacher
from lerobot.lerobot_types import TransitionKey


@pytest.mark.parametrize("future", [False, True])
def test_square_resize_preserves_current_future_layout_and_padding(future):
    pad = torch.tensor([[False, False, True, True]])
    step = SimpleNamespace(
        use_future_image=future,
        resize_imgs_with_padding=(256, 256),
        chunk_size=4,
        _feature_transform=SimpleNamespace(org_features={"actions": ["action"]}),
        _current_transition={TransitionKey.COMPLEMENTARY_DATA: {"action_is_pad": pad}},
    )
    image_shape = (1, 2, 3, 480, 640) if future else (1, 3, 480, 640)
    state_shape = (1, 2, 14) if future else (1, 14)
    observation = {
        "observation.state": torch.zeros(state_shape),
        "observation.images.cam_high": torch.ones(image_shape),
    }
    item, _ = next(
        LingbotVLAV2FeatureTransformStep._iter_items(step, observation, torch.zeros(1, 4, 14), ["task"])
    )
    assert item["observation.images.cam_high"].shape == ((2, 3, 256, 256) if future else (3, 256, 256))
    assert item["observation.state"].shape == (14,)
    torch.testing.assert_close(item["action_is_pad"], pad[0])


def test_future_fps_uses_each_samples_actual_tail():
    pad = torch.ones(3, 50, dtype=torch.bool)
    pad[0, :50] = False
    pad[1, :25] = False
    pad[2, :1] = False
    fps = _future_video_fps(50, 49, pad)
    torch.testing.assert_close(fps[:2], torch.tensor([50 / 49, 50 / 24]))
    assert torch.isinf(fps[2])  # A repeated final frame has zero temporal displacement.


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


def test_published_depth_embedding_loads_and_missing_weight_fails():
    model = torch.nn.Module()
    model.encoder = torch.nn.Module()
    model.encoder.backbone = torch.nn.Module()
    model.encoder.backbone.depth_mask_patch_embed = torch.nn.Module()
    model.encoder.backbone.depth_mask_patch_embed.proj = torch.nn.Conv2d(1, 2, 1)
    state = {key: torch.full_like(value, 0.75) for key, value in model.state_dict().items()}
    MoRGBDTeacher._load_checkpoint_state(model, state)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    missing = dict(state)
    del missing["encoder.backbone.depth_mask_patch_embed.proj.weight"]
    with pytest.raises(RuntimeError, match="complete runtime"):
        MoRGBDTeacher._load_checkpoint_state(model, missing, strict=False)
