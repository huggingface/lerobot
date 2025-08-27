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

from lerobot.teleoperators.reachy2_teleoperator import (
    Reachy2Teleoperator,
    Reachy2TeleoperatorConfig,
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
    {},  # default config
    {"with_mobile_base": False},
    {"with_mobile_base": False, "with_l_arm": False, "with_antennas": False},
    {"with_r_arm": False, "with_neck": False, "with_antennas": False},
    {"with_mobile_base": False, "with_neck": False},
]


def _make_reachy2_sdk_mock():
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
        k: MagicMock(
            goal_position=float(i),
        )
        for i, k in enumerate(REACHY2_JOINTS.values())
    }
    r.joints = joints

    # Mock mobile base with some dummy odometry
    r.mobile_base = MagicMock()
    r.mobile_base.last_cmd_vel = {
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
    with (
        patch(
            "lerobot.teleoperators.reachy2_teleoperator.reachy2_teleoperator.ReachySDK",
            side_effect=lambda *a, **k: _make_reachy2_sdk_mock(),
        ),
    ):
        overrides = request.param
        cfg = Reachy2TeleoperatorConfig(ip_address="192.168.0.200", **overrides)
        robot = Reachy2Teleoperator(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(reachy2):
    assert not reachy2.is_connected

    reachy2.connect()
    assert reachy2.is_connected

    reachy2.disconnect()
    assert not reachy2.is_connected

    reachy2.reachy.disconnect.assert_called_once()


def test_get_action(reachy2):
    reachy2.connect()
    action = reachy2.get_action()

    expected_keys = set(reachy2.joints_dict)
    expected_keys.update(
        f"{v}" for v in REACHY2_VEL.keys() if reachy2.config.with_mobile_base
    )
    assert set(action.keys()) == expected_keys

    for motor in reachy2.joints_dict.keys():
        assert (
            action[motor] == reachy2.reachy.joints[REACHY2_JOINTS[motor]].goal_position
        )
    if reachy2.config.with_mobile_base:
        for vel in REACHY2_VEL.keys():
            assert (
                action[vel] == reachy2.reachy.mobile_base.last_cmd_vel[REACHY2_VEL[vel]]
            )
