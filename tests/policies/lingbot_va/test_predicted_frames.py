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

"""LingBot-VA returns, with each action, the imagined camera frames that action belongs to."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
from lerobot.processor import (
    policy_output_to_transition,
    transition_to_policy_action,
    transition_to_prediction,
)

CAMS = ["observation.images.image", "observation.images.image2"]


def _policy(**state) -> SimpleNamespace:
    config = SimpleNamespace(obs_cam_keys=CAMS, height=2, width=2, camera_layout="width_concat")
    policy = SimpleNamespace(config=config, **state)
    policy._split_cameras = lambda frames: LingBotVAPolicy._split_cameras(policy, frames)
    return policy


def _clip(n_frames: int) -> torch.Tensor:
    """A decoded clip ``[B=2, T, H, W, 3]``: two 2x2 cameras side by side (width_concat).

    Pixel value = frame index + 10 * env + 100 on the second camera, so a test can tell them apart.
    """
    clip = torch.zeros(2, n_frames, 2, 4, 3, dtype=torch.uint8)
    for env in range(2):
        for t in range(n_frames):
            clip[env, t, :, :2] = t + 10 * env
            clip[env, t, :, 2:] = t + 10 * env + 100
    return clip


def _serve_chunk(chunk_len: int, n_frames: int) -> list[int | None]:
    """Frame index returned with each action of one chunk (``None`` for a bare action)."""
    policy = _policy(
        _chunk_frames=_clip(n_frames),
        _chunk_len=chunk_len,
        _shown_frame=-1,
        _action_queue=deque(range(chunk_len)),
    )
    shown: list[int | None] = []
    for _ in range(chunk_len):
        action = torch.zeros(2, 7) + policy._action_queue.popleft()
        out = LingBotVAPolicy._with_predicted_frame(policy, action)
        if isinstance(out, torch.Tensor):
            shown.append(None)
            continue
        step = policy_output_to_transition(out)
        assert transition_to_policy_action(step) is action
        predicted = transition_to_prediction(step)["observation"]
        assert list(predicted) == CAMS
        for camera, offset in zip(CAMS, (0, 100), strict=True):
            frames = predicted[camera]
            assert frames.shape == (2, 3, 2, 2)  # [B, C, H, W], like camera frames
            assert int(frames[1, 0, 0, 0]) == int(frames[0, 0, 0, 0]) + 10  # each env its own frame
            assert 0 <= int(frames[0, 0, 0, 0]) - offset < 100  # its own camera
        shown.append(int(predicted[CAMS[0]][0, 0, 0, 0]))
    return shown


def test_frames_are_spread_over_the_chunk_and_returned_once():
    # 4 imagined frames over 8 actions: a new frame every 2 actions, bare actions in between.
    assert _serve_chunk(chunk_len=8, n_frames=4) == [0, None, 1, None, 2, None, 3, None]


def test_more_frames_than_actions_skips_frames():
    assert _serve_chunk(chunk_len=2, n_frames=4) == [0, 2]


def test_no_decoded_clip_returns_the_bare_action():
    policy = SimpleNamespace(_chunk_frames=None)
    action = torch.zeros(1, 7)
    assert LingBotVAPolicy._with_predicted_frame(policy, action) is action


def test_robotwin_frames_split_back_into_head_and_wrists():
    # T-shape: half-res left|right wrists on top of the full-res head.
    policy = _policy()
    policy.config.obs_cam_keys = ["head", "left", "right"]
    policy.config.height, policy.config.width = 4, 4
    policy.config.camera_layout = "robotwin_tshape"
    frames = torch.zeros(2, 3, 6, 4)
    frames[:, :, :2, :2], frames[:, :, :2, 2:], frames[:, :, 2:] = 1, 2, 3
    views = LingBotVAPolicy._split_cameras(policy, frames)
    assert {k: (tuple(v.shape), int(v.unique())) for k, v in views.items()} == {
        "head": ((2, 3, 4, 4), 3),
        "left": ((2, 3, 2, 2), 1),
        "right": ((2, 3, 2, 2), 2),
    }


def test_decoded_clip_keeps_every_env_and_eval_keeps_the_first():
    # A fake VAE whose decode returns env-distinct videos in [-1, 1].
    video = torch.stack([torch.full((3, 2, 4, 4), -1.0), torch.full((3, 2, 4, 4), 1.0)])  # [B, C, F, H, W]
    vae = SimpleNamespace(
        config=SimpleNamespace(z_dim=1, latents_mean=[0.0], latents_std=[1.0]),
        dtype=torch.float32,
        parameters=lambda: iter([torch.zeros(1)]),
        decode=lambda latents, return_dict: (video,),
    )
    policy = SimpleNamespace(_vae=vae)
    policy._decode_video_batch = lambda latents: LingBotVAPolicy._decode_video_batch(policy, latents)
    latents = torch.zeros(2, 1, 1, 1, 1)

    clip = LingBotVAPolicy._decode_video_batch(policy, latents)
    assert clip.shape == (2, 2, 4, 4, 3) and clip.dtype == torch.uint8  # [B, T, H, W, 3]
    assert int(clip[0].max()) == 0 and int(clip[1].min()) == 255
    # lerobot-eval's saved predicted video still decodes the first env only.
    torch.testing.assert_close(LingBotVAPolicy._decode_predicted_video(policy, latents), clip[0])
