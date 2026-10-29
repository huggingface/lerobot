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
"""Depth and intrinsics through the eval path: env observation -> policy batch.

`LiberoEnv(use_depth=True)` emits `pixels["<camera>_depth"]` (float metres, top-left
origin) and `intrinsics["<camera>"]`. `preprocess_observation` used to push every
`pixels` entry through the RGB path, which failed on depth twice over (a negative
stride from the orientation flip, then the uint8 / channel-last asserts), dropped
the intrinsics, and `LiberoProcessorStep` would then have rotated depth away from
the orientation its intrinsics describe. None of this needs MuJoCo to test.
"""

import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep

B, H, W = 2, 8, 10


def env_observation():
    rgb = np.random.default_rng(0).integers(0, 255, (B, H, W, 3), dtype=np.uint8)
    depth = np.random.default_rng(1).uniform(0.6, 3.1, (B, H, W)).astype(np.float32)
    intrinsics = np.tile(np.array([[300.0, 0, 5], [0, 300.0, 4], [0, 0, 1]], np.float32), (B, 1, 1))
    return {
        "pixels": {"image": rgb, "image_depth": np.flip(depth, axis=-2)},  # a view, like the old orient_depth
        "intrinsics": {"image": intrinsics},
    }, depth


def test_depth_takes_its_own_path_through_preprocess_observation():
    obs, depth = env_observation()
    out = preprocess_observation(obs)

    assert out["observation.images.image"].shape == (B, 3, H, W)
    assert out["observation.images.image"].max() <= 1.0  # RGB is still scaled

    got = out["observation.images.image_depth"]
    assert got.shape == (B, 1, H, W)
    assert got.dtype == torch.float32
    # metres in, metres out: not divided by 255, not reoriented
    assert torch.equal(got[:, 0], torch.from_numpy(np.flip(depth, axis=-2).copy()))


def test_intrinsics_reach_the_batch():
    obs, _ = env_observation()
    out = preprocess_observation(obs)
    assert out["observation.intrinsics.image"].shape == (B, 3, 3)
    assert torch.equal(out["observation.intrinsics.image"], torch.from_numpy(obs["intrinsics"]["image"]))


def test_unbatched_depth_gets_a_batch_dimension():
    obs = {"pixels": {"image_depth": np.ones((H, W), np.float32)}, "intrinsics": {"image": np.eye(3)}}
    out = preprocess_observation(obs)
    assert out["observation.images.image_depth"].shape == (1, 1, H, W)
    assert out["observation.intrinsics.image"].shape == (1, 3, 3)


def test_libero_processor_rotates_rgb_but_not_depth():
    obs, _ = env_observation()
    batch = preprocess_observation(obs)
    rgb, depth = batch["observation.images.image"].clone(), batch["observation.images.image_depth"].clone()

    out = LiberoProcessorStep().observation(dict(batch))

    assert torch.equal(out["observation.images.image"], torch.flip(rgb, dims=[2, 3]))
    assert torch.equal(out["observation.images.image_depth"], depth)
