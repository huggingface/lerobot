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

import pytest

from lerobot.teleoperators.reachy2_teleoperator import (
    REACHY2_ANTENNAS_JOINTS,
    REACHY2_L_ARM_JOINTS,
    REACHY2_NECK_JOINTS,
    REACHY2_R_ARM_JOINTS,
    REACHY2_VEL,
    Reachy2Teleoperator,
    Reachy2TeleoperatorConfig,
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
    {"with_mobile_base": False, "with_neck": False},
    {"use_present_position": True},
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
        k: MagicMock(
            present_position=float(i),
            goal_position=float(i) + 0.5,
        )
        for i, k in enumerate(REACHY2_JOINTS.values())
    }
    r.joints = joints

    # Mock mobile base with some dummy odometry
    r.mobile_base = MagicMock()
    r.mobile_base.last_cmd_vel = {
        "vx": -0.2,
        "vy": 0.2,
        "vtheta": 11.0,
    }
    r.mobile_base.odometry = {
        "x": 1.0,
        "y": 2.0,
        "theta": 20.0,
        "vx": 0.1,
        "vy": -0.1,
        "vtheta": 8.0,
    }

    r.connect = MagicMock(side_effect=_connect)
    r.disconnect = MagicMock(side_effect=_disconnect)

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
    expected_keys.update(f"{v}" for v in REACHY2_VEL.keys() if reachy2.config.with_mobile_base)
    assert set(action.keys()) == expected_keys

    for motor in reachy2.joints_dict.keys():
        if reachy2.config.use_present_position:
            assert action[motor] == reachy2.reachy.joints[REACHY2_JOINTS[motor]].present_position
        else:
            assert action[motor] == reachy2.reachy.joints[REACHY2_JOINTS[motor]].goal_position
    if reachy2.config.with_mobile_base:
        if reachy2.config.use_present_position:
            for vel in REACHY2_VEL.keys():
                assert action[vel] == reachy2.reachy.mobile_base.odometry[REACHY2_VEL[vel]]
        else:
            for vel in REACHY2_VEL.keys():
                assert action[vel] == reachy2.reachy.mobile_base.last_cmd_vel[REACHY2_VEL[vel]]


def test_no_part_declared():
    with pytest.raises(ValueError):
        _ = Reachy2TeleoperatorConfig(
            ip_address="192.168.0.200",
            with_mobile_base=False,
            with_l_arm=False,
            with_r_arm=False,
            with_neck=False,
            with_antennas=False,
        )
