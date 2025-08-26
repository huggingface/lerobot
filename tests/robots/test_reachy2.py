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
    Reachy2Robot,
    Reachy2RobotConfig,
)

# {lerobot_keys: reachy2_sdk_keys}
REACHY2_JOINTS = {
    "neck_yaw.pos": "head.neck.yaw",
    "neck_pitch.pos": "head.neck.pitch",
    "neck_roll.pos": "head.neck.roll",
    "l_antenna.pos": "head.l_antenna",
    "r_antenna.pos": "head.r_antenna",
    "r_shoulder_pitch.pos": "r_arm.shoulder.pitch",
    "r_shoulder_roll.pos": "r_arm.shoulder.roll",
    "r_elbow_yaw.pos": "r_arm.elbow.yaw",
    "r_elbow_pitch.pos": "r_arm.elbow.pitch",
    "r_wrist_roll.pos": "r_arm.wrist.roll",
    "r_wrist_pitch.pos": "r_arm.wrist.pitch",
    "r_wrist_yaw.pos": "r_arm.wrist.yaw",
    "r_gripper.pos": "r_arm.gripper",
    "l_shoulder_pitch.pos": "l_arm.shoulder.pitch",
    "l_shoulder_roll.pos": "l_arm.shoulder.roll",
    "l_elbow_yaw.pos": "l_arm.elbow.yaw",
    "l_elbow_pitch.pos": "l_arm.elbow.pitch",
    "l_wrist_roll.pos": "l_arm.wrist.roll",
    "l_wrist_pitch.pos": "l_arm.wrist.pitch",
    "l_wrist_yaw.pos": "l_arm.wrist.yaw",
    "l_gripper.pos": "l_arm.gripper",
}

REACHY2_VEL = {
    "mobile_base.vx": "vx",
    "mobile_base.vy": "vy",
    "mobile_base.vtheta": "vtheta",
}

PARAMS = [
    {},  # config par défaut
    {"with_mobile_base": False},
    {"with_mobile_base": False, "with_l_arm": False, "with_antennas": False},
    {"with_r_arm": False, "with_neck": False, "with_antennas": False},
    {"use_external_commands": True},
    {"use_external_commands": True, "with_mobile_base": False, "with_neck": False},
]


def _make_reachy2_sdk_mock():
    class JointSpy:
        __slots__ = (
            "present_position",
            "_goal_position",
            "set_calls",
            "set_values",
            "_on_set",
        )

        def __init__(self, present_position=0.0, initial_goal=None, on_set=None):
            self.present_position = present_position
            self._goal_position = initial_goal
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
            initial_goal=float(i) + 0.1,
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


@pytest.fixture(params=PARAMS, ids=lambda p: "default" if not p else ",".join(p.keys()))
def reachy2(request):
    # Mock cameras
    fake_cams = {
        "teleop_left": MagicMock(
            width=640,
            height=480,
            connect=MagicMock(),
            disconnect=MagicMock(),
            async_read=MagicMock(return_value=np.zeros((10, 10, 3), dtype=np.uint8)),
        ),
    }

    with (
        patch(
            "lerobot.robots.reachy2.robot_reachy2.ReachySDK",
            side_effect=lambda *a, **k: _make_reachy2_sdk_mock(),
        ),
        patch(
            "lerobot.robots.reachy2.robot_reachy2.make_cameras_from_configs",
            return_value=fake_cams,
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

    reachy2.reachy.turn_off_smoothly.assert_called_once()
    reachy2.reachy.disconnect.assert_called_once()


def test_get_observation(reachy2):
    reachy2.connect()
    obs = reachy2.get_observation()

    expected_keys = set(reachy2.joints_dict)
    expected_keys.update(f"{v}" for v in REACHY2_VEL.keys() if reachy2.config.with_mobile_base)
    expected_keys.update(reachy2.cameras.keys())
    assert set(obs.keys()) == expected_keys

    print(obs)

    for motor in reachy2.joints_dict.keys():
        assert obs[motor] == reachy2.reachy.joints[REACHY2_JOINTS[motor]].present_position
    if reachy2.config.with_mobile_base:
        for vel in REACHY2_VEL.keys():
            assert obs[vel] == reachy2.reachy.mobile_base.odometry[REACHY2_VEL[vel]]


def test_send_action(reachy2):
    reachy2.connect()

    action = {k: i * 10.0 for i, k in enumerate(reachy2.joints_dict.keys())}
    if reachy2.config.with_mobile_base:
        action.update({k: i * 0.1 for i, k in enumerate(REACHY2_VEL.keys(), start=1)})

    returned = reachy2.send_action(action)

    assert returned == action

    assert reachy2.reachy._goal_position_set_total == len(reachy2.joints_dict)
    for motor in reachy2.joints_dict.keys():
        expected_pos = action[motor]
        real_pos = reachy2.reachy.joints[REACHY2_JOINTS[motor]].goal_position
        assert real_pos == expected_pos

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
