#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import pytest

from lerobot.robots.reachy2 import (
    REACHY2_ANTENNAS_JOINTS,
    REACHY2_L_ARM_JOINTS,
    REACHY2_NECK_JOINTS,
    REACHY2_R_ARM_JOINTS,
    REACHY2_VEL,
    Reachy2Robot,
    Reachy2RobotConfig,
)

# {lerobot_keys: reachy2_sdk_keys}
REACHY2_JOINTS = {
    **REACHY2_NECK_JOINTS,
    **REACHY2_ANTENNAS_JOINTS,
    **REACHY2_R_ARM_JOINTS,
    **REACHY2_L_ARM_JOINTS,
}

PARAMS = [
    {},  # default config
    {"with_mobile_base": False},
    {"with_mobile_base": False, "with_l_arm": False, "with_antennas": False},
    {"with_r_arm": False, "with_neck": False, "with_antennas": False},
    {"use_external_commands": True, "disable_torque_on_disconnect": True},
    {"use_external_commands": True, "with_mobile_base": False, "with_neck": False},
    {"disable_torque_on_disconnect": False},
    {"max_relative_target": 5},
    {"with_right_teleop_camera": False},
    {"with_left_teleop_camera": False, "with_right_teleop_camera": False},
    {"with_left_teleop_camera": False, "with_torso_camera": True},
]


def _make_reachy2_sdk_mock():
    class JointSpy:
        __slots__ = (
            "present_position",
            "_goal_position",
            "_on_set",
        )

        def __init__(self, present_position=0.0, on_set=None):
            self.present_position = present_position
            self._goal_position = present_position
            self._on_set = on_set

        @property
        def goal_position(self):
            return self._goal_position

        @goal_position.setter
        def goal_position(self, v):
            self._goal_position = v
            if self._on_set:
                self._on_set()

    r = MagicMock(name="ReachySDKMock")
    r.is_connected.return_value = True

    def _connect():
        r.is_connected.return_value = True

    def _disconnect():
        r.is_connected.return_value = False

    # Global counter of goal_position sets
    r._goal_position_set_total = 0

    def _on_any_goal_set():
        r._goal_position_set_total += 1

    # Mock joints with some dummy positions
    joints = {
        k: JointSpy(
            present_position=float(i),
            on_set=_on_any_goal_set,
        )
        for i, k in enumerate(REACHY2_JOINTS.values())
    }
    r.joints = joints

    # Mock mobile base with some dummy odometry
    r.mobile_base = MagicMock()
    r.mobile_base.odometry = {
        "x": 0.1,
        "y": -0.2,
        "theta": 21.3,
        "vx": 0.001,
        "vy": 0.002,
        "vtheta": 0.0,
    }

    r.connect = MagicMock(side_effect=_connect)
    r.disconnect = MagicMock(side_effect=_disconnect)

    # Mock methods
    r.turn_on = MagicMock()
    r.reset_default_limits = MagicMock()
    r.send_goal_positions = MagicMock()
    r.turn_off_smoothly = MagicMock()
    r.mobile_base.set_goal_speed = MagicMock()
    r.mobile_base.send_speed_command = MagicMock()

    return r


def _make_reachy2_camera_mock(*args, **kwargs):
    cfg = args[0] if args else kwargs.get("config")
    name = getattr(cfg, "name", kwargs.get("name", "cam"))
    image_type = getattr(cfg, "image_type", kwargs.get("image_type", "cam"))
    width = getattr(cfg, "width", kwargs.get("width", 640))
    height = getattr(cfg, "height", kwargs.get("height", 480))

    cam = MagicMock(name=f"Reachy2CameraMock:{name}")
    cam.name = name
    cam.image_type = image_type
    cam.width = width
    cam.height = height
    cam.connect = MagicMock()
    cam.disconnect = MagicMock()
    cam.async_read = MagicMock(side_effect=lambda: np.zeros((height, width, 3), dtype=np.uint8))
    return cam


@pytest.fixture(params=PARAMS, ids=lambda p: "default" if not p else ",".join(p.keys()))
def reachy2(request):
    with (
        patch(
            "lerobot.robots.reachy2.robot_reachy2.ReachySDK",
            side_effect=lambda *a, **k: _make_reachy2_sdk_mock(),
        ),
        patch(
            "lerobot.cameras.reachy2_camera.reachy2_camera.Reachy2Camera",
            side_effect=_make_reachy2_camera_mock,
        ),
    ):
        overrides = request.param
        cfg = Reachy2RobotConfig(ip_address="192.168.0.200", **overrides)
        robot = Reachy2Robot(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(reachy2):
    assert not reachy2.is_connected

    reachy2.connect()
    assert reachy2.is_connected

    reachy2.reachy.turn_on.assert_called_once()
    reachy2.reachy.reset_default_limits.assert_called_once()

    reachy2.disconnect()
    assert not reachy2.is_connected

    if reachy2.config.disable_torque_on_disconnect:
        reachy2.reachy.turn_off_smoothly.assert_called_once()
    else:
        reachy2.reachy.turn_off_smoothly.assert_not_called()
    reachy2.reachy.disconnect.assert_called_once()


def test_get_joints_dict(reachy2):
    reachy2.connect()

    if reachy2.config.with_neck:
        assert "neck_yaw.pos" in reachy2.joints_dict
        assert "neck_pitch.pos" in reachy2.joints_dict
        assert "neck_roll.pos" in reachy2.joints_dict
    else:
        assert "neck_yaw.pos" not in reachy2.joints_dict
        assert "neck_pitch.pos" not in reachy2.joints_dict
        assert "neck_roll.pos" not in reachy2.joints_dict

    if reachy2.config.with_antennas:
        assert "l_antenna.pos" in reachy2.joints_dict
        assert "r_antenna.pos" in reachy2.joints_dict
    else:
        assert "l_antenna.pos" not in reachy2.joints_dict
        assert "r_antenna.pos" not in reachy2.joints_dict

    if reachy2.config.with_r_arm:
        assert "r_shoulder_pitch.pos" in reachy2.joints_dict
        assert "r_shoulder_roll.pos" in reachy2.joints_dict
        assert "r_elbow_yaw.pos" in reachy2.joints_dict
        assert "r_elbow_pitch.pos" in reachy2.joints_dict
        assert "r_wrist_roll.pos" in reachy2.joints_dict
        assert "r_wrist_pitch.pos" in reachy2.joints_dict
        assert "r_wrist_yaw.pos" in reachy2.joints_dict
        assert "r_gripper.pos" in reachy2.joints_dict
    else:
        assert "r_shoulder_pitch.pos" not in reachy2.joints_dict
        assert "r_shoulder_roll.pos" not in reachy2.joints_dict
        assert "r_elbow_yaw.pos" not in reachy2.joints_dict
        assert "r_elbow_pitch.pos" not in reachy2.joints_dict
        assert "r_wrist_roll.pos" not in reachy2.joints_dict
        assert "r_wrist_pitch.pos" not in reachy2.joints_dict
        assert "r_wrist_yaw.pos" not in reachy2.joints_dict
        assert "r_gripper.pos" not in reachy2.joints_dict

    if reachy2.config.with_l_arm:
        assert "l_shoulder_pitch.pos" in reachy2.joints_dict
        assert "l_shoulder_roll.pos" in reachy2.joints_dict
        assert "l_elbow_yaw.pos" in reachy2.joints_dict
        assert "l_elbow_pitch.pos" in reachy2.joints_dict
        assert "l_wrist_roll.pos" in reachy2.joints_dict
        assert "l_wrist_pitch.pos" in reachy2.joints_dict
        assert "l_wrist_yaw.pos" in reachy2.joints_dict
        assert "l_gripper.pos" in reachy2.joints_dict
    else:
        assert "l_shoulder_pitch.pos" not in reachy2.joints_dict
        assert "l_shoulder_roll.pos" not in reachy2.joints_dict
        assert "l_elbow_yaw.pos" not in reachy2.joints_dict
        assert "l_elbow_pitch.pos" not in reachy2.joints_dict
        assert "l_wrist_roll.pos" not in reachy2.joints_dict
        assert "l_wrist_pitch.pos" not in reachy2.joints_dict
        assert "l_wrist_yaw.pos" not in reachy2.joints_dict
        assert "l_gripper.pos" not in reachy2.joints_dict


def test_get_observation(reachy2):
    reachy2.connect()
    obs = reachy2.get_observation()

    expected_keys = set(reachy2.joints_dict)
    expected_keys.update(f"{v}" for v in REACHY2_VEL if reachy2.config.with_mobile_base)
    expected_keys.update(reachy2.cameras.keys())
    assert set(obs.keys()) == expected_keys

    for motor in reachy2.joints_dict:
        assert obs[motor] == reachy2.reachy.joints[REACHY2_JOINTS[motor]].present_position
    if reachy2.config.with_mobile_base:
        for vel in REACHY2_VEL:
            assert obs[vel] == reachy2.reachy.mobile_base.odometry[REACHY2_VEL[vel]]
    if reachy2.config.with_left_teleop_camera:
        assert obs["teleop_left"].shape == (
            reachy2.config.cameras["teleop_left"].height,
            reachy2.config.cameras["teleop_left"].width,
            3,
        )
    if reachy2.config.with_right_teleop_camera:
        assert obs["teleop_right"].shape == (
            reachy2.config.cameras["teleop_right"].height,
            reachy2.config.cameras["teleop_right"].width,
            3,
        )
    if reachy2.config.with_torso_camera:
        assert obs["torso_rgb"].shape == (
            reachy2.config.cameras["torso_rgb"].height,
            reachy2.config.cameras["torso_rgb"].width,
            3,
        )


def test_send_action(reachy2):
    reachy2.connect()

    action = {k: i * 10.0 for i, k in enumerate(reachy2.joints_dict.keys(), start=1)}
    if reachy2.config.with_mobile_base:
        action.update({k: i * 0.1 for i, k in enumerate(REACHY2_VEL.keys(), start=1)})

    previous_present_position = {
        k: reachy2.reachy.joints[REACHY2_JOINTS[k]].present_position for k in reachy2.joints_dict
    }
    returned = reachy2.send_action(action)

    if reachy2.config.max_relative_target is None:
        assert returned == action

    assert reachy2.reachy._goal_position_set_total == len(reachy2.joints_dict)
    for motor in reachy2.joints_dict:
        expected_pos = action[motor]
        real_pos = reachy2.reachy.joints[REACHY2_JOINTS[motor]].goal_position
        if reachy2.config.max_relative_target is None:
            assert real_pos == expected_pos
        else:
            assert real_pos == previous_present_position[motor] + np.sign(expected_pos) * min(
                abs(expected_pos - real_pos), reachy2.config.max_relative_target
            )

    if reachy2.config.with_mobile_base:
        goal_speed = [i * 0.1 for i, _ in enumerate(REACHY2_VEL.keys(), start=1)]
        reachy2.reachy.mobile_base.set_goal_speed.assert_called_once_with(*goal_speed)

    if reachy2.config.use_external_commands:
        reachy2.reachy.send_goal_positions.assert_not_called()
        if reachy2.config.with_mobile_base:
            reachy2.reachy.mobile_base.send_speed_command.assert_not_called()
    else:
        reachy2.reachy.send_goal_positions.assert_called_once()
        if reachy2.config.with_mobile_base:
            reachy2.reachy.mobile_base.send_speed_command.assert_called_once()


def test_no_part_declared():
    with pytest.raises(ValueError):
        _ = Reachy2RobotConfig(
            ip_address="192.168.0.200",
            with_mobile_base=False,
            with_l_arm=False,
            with_r_arm=False,
            with_neck=False,
            with_antennas=False,
        )
