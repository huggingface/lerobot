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

import pytest

from lerobot.motors import MotorCalibration
from lerobot.teleoperators.openarm_v1_dynamixel import (
    OpenArmV1Dynamixel,
    OpenArmV1DynamixelConfig,
)
from lerobot.teleoperators.openarm_v1_dynamixel.openarm_v1_dynamixel import (
    GRIPPER_TELEOP_TO_DEGREES,
    SIDE_BASE_ID,
    SIDE_MOTORS_TO_FLIP,
)

_MODULE = "lerobot.teleoperators.openarm_v1_dynamixel.openarm_v1_dynamixel"

ALL_MOTORS = [
    f"{side}_{joint}"
    for side in ("right", "left")
    for joint in [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
]


def _make_leader(calibration: dict | None = None, **cfg_kwargs) -> OpenArmV1Dynamixel:
    """Build a leader with the Dynamixel bus mocked out -- no hardware, no serial port."""
    with patch(f"{_MODULE}.DynamixelMotorsBus") as bus_cls:
        bus = bus_cls.return_value
        bus.motors = {
            name: MagicMock(id=SIDE_BASE_ID[name.split("_")[0]] + (i % 8))
            for i, name in enumerate(ALL_MOTORS)
        }
        bus.is_connected = True
        leader = OpenArmV1Dynamixel(
            OpenArmV1DynamixelConfig(port="/dev/null", id="test_leader", **cfg_kwargs)
        )
    leader.calibration = calibration or {}
    return leader


def test_all_sixteen_servos_share_one_bus():
    """Both arms hang off a single OpenRB-150 bus: right ids 1-8, left ids 9-16."""
    leader = _make_leader()
    assert len(leader.bus.motors) == 16
    assert "right_joint_1" in leader.bus.motors
    assert "left_gripper" in leader.bus.motors


def test_action_features_cover_both_arms():
    features = _make_leader().action_features
    assert len(features) == 16
    assert "right_joint_1.pos" in features
    assert "left_gripper.pos" in features


def test_include_gripper_false_drops_only_grippers():
    features = _make_leader(include_gripper=False).action_features
    assert len(features) == 14
    assert not any("gripper" in key for key in features)


def test_get_action_flips_mirrored_joints():
    """Mirror-mounted servos read the opposite sign from what the follower expects."""
    leader = _make_leader()
    leader.bus.sync_read.return_value = dict.fromkeys(ALL_MOTORS, 10.0)

    action = leader.get_action()

    for side, joints in SIDE_MOTORS_TO_FLIP.items():
        for joint in joints:
            assert action[f"{side}_{joint}.pos"] == -10.0, f"{side}_{joint} should be flipped"
    assert action["right_joint_1.pos"] == 10.0  # not in the flip list


@pytest.mark.parametrize(
    ("drive_mode", "raw", "expected_degrees"),
    [
        # Normally wired gripper: teleop 0 = closed -> 0 deg, 100 = open -> -65 deg.
        (0, 0.0, 0.0),
        (0, 100.0, -65.0),
        # Reverse-wired gripper (encoder counts down while opening). `DynamixelMotorsBus`
        # sets apply_drive_mode = False, so normalisation does NOT undo this for us --
        # the teleoperator must, or a fully open leader gripper would clamp the follower.
        (1, 100.0, 0.0),
        (1, 0.0, -65.0),
    ],
)
def test_gripper_drive_mode_applied_by_teleoperator(drive_mode, raw, expected_degrees):
    calibration = {
        "right_gripper": MotorCalibration(
            id=8, drive_mode=drive_mode, homing_offset=0, range_min=0, range_max=4095
        )
    }
    leader = _make_leader(calibration=calibration)
    leader.bus.sync_read.return_value = {"right_gripper": raw}

    action = leader.get_action()

    assert action["right_gripper.pos"] == pytest.approx(expected_degrees)


def test_gripper_scale_matches_openarm_mini():
    """Both leaders drive the same follower, so the gripper scale must be identical."""
    from lerobot.teleoperators.openarm_mini.openarm_mini import (
        GRIPPER_TELEOP_TO_DEGREES as MINI_SCALE,
    )

    assert GRIPPER_TELEOP_TO_DEGREES == MINI_SCALE


def test_calibration_uses_extended_position_mode():
    """Single-turn POSITION silently ignores |Homing_Offset| > 1024 on Dynamixel."""
    from lerobot.motors.dynamixel import OperatingMode

    leader = _make_leader()
    leader.bus.read.side_effect = [0, 100] * 2  # gripper closed/open captures
    leader.bus.set_half_turn_homings.return_value = dict.fromkeys(ALL_MOTORS, 0)
    leader.bus.model_resolution_table = {"xl330-m288": 4096}
    leader.calibration = {}

    with patch("builtins.input", return_value=""), patch.object(leader, "_save_calibration"):
        leader.calibrate()

    modes = {call.args[2] for call in leader.bus.write.call_args_list if call.args[0] == "Operating_Mode"}
    assert modes == {OperatingMode.EXTENDED_POSITION.value}


def test_send_feedback_not_implemented():
    with pytest.raises(NotImplementedError):
        _make_leader().send_feedback({})


def test_resolvable_from_cli_type_string():
    """`--teleop.type=openarm_v1_dynamixel` must reach the factory."""
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    with patch(f"{_MODULE}.DynamixelMotorsBus"):
        teleop = make_teleoperator_from_config(OpenArmV1DynamixelConfig(port="/dev/null", id="test_leader"))
    assert isinstance(teleop, OpenArmV1Dynamixel)
