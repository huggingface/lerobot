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

"""LingBot-VA returns, with each chunk, the imagined camera frames its actions belong to."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy

CAMS = ["observation.images.image", "observation.images.image2"]


def _policy(**state) -> SimpleNamespace:
    config = SimpleNamespace(
        obs_cam_keys=CAMS, height=2, width=2, camera_layout="width_concat", return_predicted_video=True
    )
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


def _predict(chunk_len: int, n_frames: int, *, video: bool = True):
    """The first chunk's output, with stubbed sampling and decoding."""
    actions = torch.zeros(2, chunk_len, 7)
    policy = _policy(_first_chunk=True)
    policy.config.return_predicted_video = video
    policy._predict_chunk = lambda batch, keyframes, noise, video_noise: (actions, None)
    policy._decode_video_batch = lambda latents: _clip(n_frames)
    cameras = {key: torch.zeros(2, 1, 3, 2, 2) for key in CAMS}
    return LingBotVAPolicy.predict_action_chunk(policy, cameras), actions


def test_chunk_carries_its_imagined_clip_per_camera():
    out, actions = _predict(chunk_len=8, n_frames=4)
    assert out["action"] is actions
    keys = ["prediction.images.image", "prediction.images.image2"]
    assert set(out) == {"action", *keys, *(key + "_steps" for key in keys)}
    assert set(keys) == set(LingBotVAConfig.prediction_features.fget(_policy().config))
    for key, offset in zip(keys, (0, 100), strict=True):
        # 4 imagined frames over 8 actions: a new frame every 2 actions.
        assert out[key + "_steps"] == [0, 2, 4, 6]
        frames = out[key]
        assert frames.shape == (2, 4, 3, 2, 2)  # [B, F, C, H, W]
        assert frames[0, :, 0, 0, 0].tolist() == [offset + f for f in range(4)]  # its own camera
        assert torch.equal(frames[1].int(), frames[0].int() + 10)  # each env its own frames


def test_more_frames_than_actions_stay_within_the_chunk():
    out, _ = _predict(chunk_len=2, n_frames=4)
    assert out["prediction.images.image_steps"] == [0, 1, 1, 1]


def test_without_predicted_video_returns_only_the_chunk():
    out, actions = _predict(chunk_len=8, n_frames=4, video=False)
    assert out == {"action": actions}
