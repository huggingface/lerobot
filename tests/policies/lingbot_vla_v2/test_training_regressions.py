"""Small CPU regressions for the fix-all behavior retained on master's layout."""

from types import MethodType

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.lingbot_vla_v2.model_core.qwen3vl_in_vla import (
    forward_without_grid_thw,
    preprcess_grid_thw,
)
from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import (
    LingbotVLAV2SlotMappingProcessorStep,
    _future_video_fps,
    _split_camera_frames,
)
from lerobot.policies.lingbot_vla_v2.teachers.morgbd_teacher import MoRGBDTeacher


@pytest.mark.parametrize("future", [False, True])
def test_square_resize_preserves_current_future_layout_and_padding(future):
    image_shape = (2, 3, 480, 640) if future else (3, 480, 640)
    current, future_frame = _split_camera_frames(torch.ones(image_shape), (256, 256), use_future_image=future)
    assert current.shape == (3, 256, 256)
    if future:
        assert future_frame.shape == (3, 256, 256)
    else:
        assert future_frame is None
    # [0, 1] float inputs are scaled to the [0, 255] range Qwen3-VL expects.
    assert current.max().item() == pytest.approx(255.0, abs=1e-2)


def test_slot_mapping_slices_future_state_to_current_frame():
    """With future-frame deltas stacked on the state, the policy state is frame 0."""
    step = LingbotVLAV2SlotMappingProcessorStep(
        robot_config={
            "states": [
                {
                    "observation.state.arm.position": {
                        "origin_keys": [{"observation.state": {"start": 0, "end": 14}}]
                    }
                },
            ],
            "actions": [],
        },
        canonical_joints={"arm.position": 14},
        max_state_dim=14,
        max_action_dim=14,
        use_future_image=True,
    )
    state = torch.stack([torch.zeros(1, 14), torch.ones(1, 14)], dim=1)  # (1, T=2, 14)
    transition = step({"observation": {"observation.state": state}})
    canonical = transition["observation"]["observation.state"]
    assert canonical.shape == (1, 14)
    torch.testing.assert_close(canonical, torch.zeros(1, 14))


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
