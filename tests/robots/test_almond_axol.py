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

from unittest.mock import MagicMock

import pytest

pytest.importorskip("almond_axol")

import draccus  # noqa: E402

from lerobot.robots.almond_axol import (  # noqa: E402
    AlmondAxol,
    AlmondAxolCameraConfig,
    AlmondAxolConfig,
)
from lerobot.robots.config import RobotConfig  # noqa: E402
from lerobot.teleoperators.almond_axol_vr import (  # noqa: E402
    AlmondAxolVR,
    AlmondAxolVRConfig,
)
from lerobot.teleoperators.config import TeleoperatorConfig  # noqa: E402

JOINTS = (
    "shoulder_1",
    "shoulder_2",
    "shoulder_3",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
)
JOINT_KEYS = [f"{side}_{joint}.pos" for side in ("left", "right") for joint in JOINTS]


def test_robot_config_registration():
    cfg = draccus.decode(RobotConfig, {"type": "almond_axol"})
    assert isinstance(cfg, AlmondAxolConfig)


def test_teleop_config_registration():
    cfg = draccus.decode(TeleoperatorConfig, {"type": "almond_axol_vr", "port": 9000})
    assert isinstance(cfg, AlmondAxolVRConfig)
    assert cfg.port == 9000


def test_robot_features_without_cameras():
    robot = AlmondAxol(AlmondAxolConfig())
    assert sorted(robot.observation_features) == sorted(JOINT_KEYS)
    assert sorted(robot.action_features) == sorted(JOINT_KEYS)
    assert not robot.is_connected


def test_robot_features_with_torques():
    robot = AlmondAxol(AlmondAxolConfig(observe_torques=True))
    trq_keys = [k.replace(".pos", ".trq") for k in JOINT_KEYS]
    assert sorted(robot.observation_features) == sorted(JOINT_KEYS + trq_keys)
    assert sorted(robot.action_features) == sorted(JOINT_KEYS)


def test_sdk_config_mapping():
    config = AlmondAxolConfig(
        id="unit_test",
        left_channel="can_l",
        right_channel="can_r",
        telemetry_hz=42.0,
        observe_torques=True,
        cameras={
            "overhead": AlmondAxolCameraConfig(serial=41234567, stereo=True),
            "left_arm": AlmondAxolCameraConfig(serial=41234568, fps=30, width=640, height=360),
        },
    )
    sdk_config = AlmondAxol._make_sdk_config(config)

    assert sdk_config.id == "unit_test"
    assert sdk_config.left_channel == "can_l"
    assert sdk_config.right_channel == "can_r"
    assert sdk_config.telemetry_hz == 42.0
    assert sdk_config.observe_torques is True
    assert sdk_config.video_backend == "sdk"

    overhead = sdk_config.cameras["overhead"]
    assert overhead.serial == 41234567
    assert overhead.stereo is True
    assert overhead.eyes == "both"
    left_arm = sdk_config.cameras["left_arm"]
    assert (left_arm.serial, left_arm.fps, left_arm.width, left_arm.height) == (41234568, 30, 640, 360)
    assert left_arm.stereo is False


def test_robot_delegates_to_sdk(monkeypatch):
    sdk_robot = MagicMock(name="AxolRobotMock")
    sdk_robot.is_connected = False
    monkeypatch.setattr(AlmondAxol, "_make_sdk_robot", classmethod(lambda cls, config: sdk_robot))

    robot = AlmondAxol(AlmondAxolConfig())

    robot.connect()
    sdk_robot.connect.assert_called_once_with(True)

    robot.get_observation()
    sdk_robot.get_observation.assert_called_once()

    action = dict.fromkeys(JOINT_KEYS, 0.0)
    robot.send_action(action)
    sdk_robot.send_action.assert_called_once_with(action)

    robot.disconnect()
    sdk_robot.disconnect.assert_called_once()


def test_teleop_features_and_gripper():
    teleop = AlmondAxolVR(AlmondAxolVRConfig())
    assert sorted(teleop.action_features) == sorted(JOINT_KEYS)
    assert not teleop.is_connected

    gripperless = AlmondAxolVR(AlmondAxolVRConfig(has_gripper=False))
    keys = [k for k in JOINT_KEYS if not k.startswith(("left_gripper", "right_gripper"))]
    assert sorted(gripperless.action_features) == sorted(keys)


def test_teleop_sdk_config_mapping():
    sdk_config = AlmondAxolVR._make_sdk_config(AlmondAxolVRConfig(id="unit_test", port=9443))
    assert sdk_config.id == "unit_test"
    assert sdk_config.vr_server_config.port == 9443
    assert sdk_config.has_gripper is True


def test_teleop_delegates_to_sdk(monkeypatch):
    sdk_teleop = MagicMock(name="AxolVRTeleopMock")
    sdk_teleop.is_connected = False
    monkeypatch.setattr(AlmondAxolVR, "_make_sdk_teleop", classmethod(lambda cls, config: sdk_teleop))

    teleop = AlmondAxolVR(AlmondAxolVRConfig())

    teleop.connect()
    sdk_teleop.connect.assert_called_once_with(True)

    teleop.get_action()
    sdk_teleop.get_action.assert_called_once()

    teleop.get_teleop_events()
    sdk_teleop.get_teleop_events.assert_called_once()

    teleop.disconnect()
    sdk_teleop.disconnect.assert_called_once()
