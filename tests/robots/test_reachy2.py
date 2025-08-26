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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from lerobot.robots.reachy2 import (
    Reachy2Robot,
    Reachy2RobotConfig,
)


# {lerobot_keys: reachy2_sdk_keys}
REACHY2_JOINTS = {
    "head.neck.yaw": "neck_yaw.pos",
    "head.neck.pitch": "neck_pitch.pos",
    "head.neck.roll": "neck_roll.pos",
    "head.l_antenna": "l_antenna.pos",
    "head.r_antenna": "r_antenna.pos",
    "r_arm.shoulder.pitch": "r_shoulder_pitch.pos",
    "r_arm.shoulder.roll": "r_shoulder_roll.pos",
    "r_arm.elbow.yaw": "r_elbow_yaw.pos",
    "r_arm.elbow.pitch": "r_elbow_pitch.pos",
    "r_arm.wrist.roll": "r_wrist_roll.pos",
    "r_arm.wrist.pitch": "r_wrist_pitch.pos",
    "r_arm.wrist.yaw": "r_wrist_yaw.pos",
    "r_arm.gripper": "r_gripper.pos",
    "l_arm.shoulder.pitch": "l_shoulder_pitch.pos",
    "l_arm.shoulder.roll": "l_shoulder_roll.pos",
    "l_arm.elbow.yaw": "l_elbow_yaw.pos",
    "l_arm.elbow.pitch": "l_elbow_pitch.pos",
    "l_arm.wrist.roll": "l_wrist_roll.pos",
    "l_arm.wrist.pitch": "l_wrist_pitch.pos",
    "l_arm.wrist.yaw": "l_wrist_yaw.pos",
    "l_arm.gripper": "l_gripper.pos",
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
]


def _make_reachy2_sdk_mock():
    r = MagicMock(name="ReachySDKMock")
    r.is_connected.return_value = True

    def _connect():
        r.is_connected.return_value = True

    def _disconnect():
        r.is_connected.return_value = False

    # Mock joints with some dummy positions
    joints = {
        k: MagicMock(present_position=round(0.1 + 0.01 * i, 2))
        for i, k in enumerate(REACHY2_JOINTS.keys())
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

    expected_keys = {f"{REACHY2_JOINTS[m]}" for m in reachy2.joints_dict.values()}
    expected_keys.update(
        f"{v}" for v in REACHY2_VEL.keys() if reachy2.config.with_mobile_base
    )
    expected_keys.update(reachy2.cameras.keys())
    assert set(obs.keys()) == expected_keys

