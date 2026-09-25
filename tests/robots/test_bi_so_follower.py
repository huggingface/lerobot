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

from unittest.mock import MagicMock, patch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig

_SO_FOLLOWER_MODULE = "lerobot.robots.so_follower.so_follower"


def _camera_config(index: int) -> OpenCVCameraConfig:
    return OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=30)


def _depth_camera_config(serial: str) -> RealSenseCameraConfig:
    return RealSenseCameraConfig(serial_number_or_name=serial, width=640, height=480, fps=30, use_depth=True)


def _fake_make_cameras(configs):
    """Build camera stubs mirroring the config, without touching camera SDKs."""
    cameras = {}
    for name, cfg in configs.items():
        cam = MagicMock()
        cam.height, cam.width = cfg.height, cfg.width
        cam.use_rgb = True
        cam.use_depth = getattr(cfg, "use_depth", False)
        cameras[name] = cam
    return cameras


def _make_robot(cameras: dict) -> BiSOFollower:
    bus_mock = MagicMock()
    bus_mock.motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    with (
        patch(f"{_SO_FOLLOWER_MODULE}.FeetechMotorsBus", return_value=bus_mock),
        patch(f"{_SO_FOLLOWER_MODULE}.make_cameras_from_configs", _fake_make_cameras),
    ):
        cfg = BiSOFollowerConfig(
            left_arm_config=SOFollowerConfig(port="/dev/null0"),
            right_arm_config=SOFollowerConfig(port="/dev/null1"),
            cameras=cameras,
        )
        return BiSOFollower(cfg)


def test_top_level_camera_key_stays_unprefixed():
    robot = _make_robot(cameras={"top": _camera_config(0)})
    assert set(robot._cameras_ft) == {"top"}


def test_top_level_depth_key_stays_unprefixed():
    """The depth sibling of a top-level camera must not get an arm prefix (regression:
    a top-level depth camera produced ``left_top_depth`` instead of ``top_depth``)."""
    robot = _make_robot(cameras={"top": _depth_camera_config("111")})
    assert set(robot._cameras_ft) == {"top", "top_depth"}
