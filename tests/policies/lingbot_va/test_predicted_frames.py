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
from lerobot.processor import transition_to_prediction

CAMS = ["observation.images.image", "observation.images.image2"]


def _policy(**state) -> SimpleNamespace:
    config = SimpleNamespace(obs_cam_keys=CAMS, height=2, width=2, camera_layout="width_concat")
    policy = SimpleNamespace(config=config, **state)
    policy._split_cameras = lambda frame: LingBotVAPolicy._split_cameras(policy, frame)
    return policy


def _serve_chunk(chunk_len: int, n_frames: int) -> list[int | None]:
    """Frame index returned with each action of one chunk (``None`` for a bare action)."""
    # Two 2x2 cameras side by side, as the width_concat latent layout decodes.
    frames = torch.stack([torch.full((2, 4, 3), i, dtype=torch.uint8) for i in range(n_frames)])
    policy = _policy(
        _chunk_frames=frames, _chunk_len=chunk_len, _shown_frame=-1, _action_queue=deque(range(chunk_len))
    )
    shown: list[int | None] = []
    for _ in range(chunk_len):
        out = LingBotVAPolicy._with_predicted_frame(
            policy, torch.zeros(1, 7) + policy._action_queue.popleft()
        )
        if isinstance(out, torch.Tensor):
            shown.append(None)
        else:
            predicted = transition_to_prediction(out)["observation"]
            assert list(predicted) == CAMS
            frame = predicted[CAMS[0]]
            assert frame.shape == (3, 2, 2)  # [C, H, W], like a camera frame
            shown.append(int(frame[0, 0, 0]))
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


def test_robotwin_frame_splits_back_into_head_and_wrists():
    # T-shape: half-res left|right wrists on top of the full-res head.
    policy = _policy()
    policy.config.obs_cam_keys = ["head", "left", "right"]
    policy.config.height, policy.config.width = 4, 4
    policy.config.camera_layout = "robotwin_tshape"
    frame = torch.zeros(3, 6, 4)
    frame[:, :2, :2], frame[:, :2, 2:], frame[:, 2:] = 1, 2, 3
    views = LingBotVAPolicy._split_cameras(policy, frame)
    assert {k: (tuple(v.shape), int(v.unique())) for k, v in views.items()} == {
        "head": ((3, 4, 4), 3),
        "left": ((3, 2, 2), 1),
        "right": ((3, 2, 2), 2),
    }
