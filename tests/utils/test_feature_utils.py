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
"""Unit tests for ``lerobot.utils.feature_utils``."""

import numpy as np

from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features


def test_hw_to_dataset_features_routes_3d_shape_to_images():
    hw = {"front": (480, 640, 3)}
    out = hw_to_dataset_features(hw, OBS_STR, use_video=True)

    assert "observation.images.front" in out
    assert out["observation.images.front"]["dtype"] == "video"
    assert out["observation.images.front"]["shape"] == (480, 640, 3)
    assert out["observation.images.front"]["names"] == ["height", "width", "channels"]
    assert "info" not in out["observation.images.front"]


def test_hw_to_dataset_features_routes_2d_shape_to_depth():
    hw = {"front_depth": (480, 640)}
    out = hw_to_dataset_features(hw, OBS_STR, use_video=True)

    assert "observation.depth.front" in out, out
    feat = out["observation.depth.front"]
    assert feat["dtype"] == "video"
    assert feat["shape"] == (480, 640)
    assert feat["names"] == ["height", "width"]
    assert feat["info"] == {"video.is_depth_map": True}


def test_hw_to_dataset_features_handles_paired_color_and_depth():
    """A camera with use_depth=True is expected to emit both keys."""
    hw = {"front": (480, 640, 3), "front_depth": (480, 640)}
    out = hw_to_dataset_features(hw, OBS_STR, use_video=True)

    assert set(out) == {"observation.images.front", "observation.depth.front"}
    assert out["observation.images.front"]["shape"] == (480, 640, 3)
    assert out["observation.depth.front"]["shape"] == (480, 640)


def test_hw_to_dataset_features_keeps_bare_2d_key_when_no_suffix():
    """If the producer didn't use a "_depth" suffix, the bare name flows through."""
    hw = {"top": (240, 320)}
    out = hw_to_dataset_features(hw, OBS_STR, use_video=True)

    assert "observation.depth.top" in out


def test_build_dataset_frame_routes_depth_values():
    ds_features = hw_to_dataset_features(
        {"front": (4, 6, 3), "front_depth": (4, 6)},
        OBS_STR,
        use_video=True,
    )
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    depth = np.full((4, 6), 0.5, dtype=np.float32)
    values = {"front": rgb, "front_depth": depth}

    frame = build_dataset_frame(ds_features, values, OBS_STR)
    assert frame["observation.images.front"] is rgb
    assert frame["observation.depth.front"] is depth
