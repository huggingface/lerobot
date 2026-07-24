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

from lerobot.action_mapping import ActionMappingProfile, JointActionMapping
from lerobot.motors import MotorCalibration
from lerobot.robots.rebot_b601_follower import RebotB601FollowerRobotConfig
from lerobot.robots.so102_follower import SO102FollowerConfig
from lerobot.scripts.lerobot_teleoperate import configure_follower_action_conversion
from lerobot.so102 import (
    B601_JOINT_DIRECTIONS,
    DEFAULT_MOTOR_IDS,
    GRIPPER_ACTION_CLOSED,
    GRIPPER_ACTION_OPEN,
    JOINT_NAMES,
    build_b601_action_conversion,
)
from lerobot.teleoperators.so102_leader import SO102Leader, SO102LeaderConfig

_LEADER_MODULE = "lerobot.teleoperators.so102_leader.so102_leader"
_MODEL_RESOLUTION = 4096


def _calibration(range_min: int = 1000, range_max: int = 3000) -> dict[str, MotorCalibration]:
    return {
        joint: MotorCalibration(
            id=motor_id,
            drive_mode=0,
            homing_offset=0,
            range_min=range_min,
            range_max=range_max,
        )
        for joint, motor_id in DEFAULT_MOTOR_IDS.items()
    }


def _make_leader(tmp_path, positions: dict[str, float] | None = None) -> SO102Leader:
    bus = MagicMock(name="FeetechLeaderBusMock")
    bus.is_connected = True
    bus.is_calibrated = True
    bus.model_resolution_table = {"sts3215": _MODEL_RESOLUTION}
    bus.sync_read.return_value = positions or dict.fromkeys(JOINT_NAMES, 0.0)

    def _bus_factory(*_args, **kwargs):
        bus.motors = kwargs["motors"]
        bus.calibration = kwargs["calibration"]
        return bus

    with patch(f"{_LEADER_MODULE}.FeetechMotorsBus", side_effect=_bus_factory):
        leader = SO102Leader(
            SO102LeaderConfig(
                port="/dev/null",
                id="leader_so102_v1",
                calibration_dir=tmp_path,
            )
        )

    leader.calibration = _calibration()
    leader.bus.calibration = leader.calibration
    leader.b601_zero_positions = dict.fromkeys(JOINT_NAMES, 0.0)
    return leader


def test_joint_action_mapping_maps_endpoints_linearly_and_clamps():
    mapping = JointActionMapping(
        source_zero=0.0,
        source_for_target_min=100.0,
        source_for_target_max=-100.0,
        target_min=-150.0,
        target_zero=0.0,
        target_max=150.0,
    )

    assert mapping.map(100.0) == -150.0
    assert mapping.map(0.0) == 0.0
    assert mapping.map(-100.0) == 150.0
    assert mapping.map(200.0) == -150.0
    assert mapping.map(-200.0) == 150.0


def test_build_b601_mapping_uses_feetech_calibration_and_verified_directions(tmp_path):
    leader = _make_leader(tmp_path)
    robot_cfg = RebotB601FollowerRobotConfig(port="/dev/ttyACM2")
    leader.b601_zero_positions["wrist_flex"] = 25.0

    profile = build_b601_action_conversion(
        calibration=leader.calibration,
        motors=leader.bus.motors,
        model_resolution_table=leader.bus.model_resolution_table,
        joint_limits=robot_cfg.joint_limits,
        zero_positions=leader.b601_zero_positions,
    )

    source_half_span = (3000 - 1000) * 180 / (_MODEL_RESOLUTION - 1)
    pan = profile.joints["shoulder_pan"]
    elbow = profile.joints["elbow_flex"]

    assert B601_JOINT_DIRECTIONS["shoulder_pan"] == -1
    assert pan.source_for_target_min == pytest.approx(source_half_span)
    assert pan.source_for_target_max == pytest.approx(-source_half_span)
    assert pan.map(source_half_span) == -150.0
    assert pan.map(0.0) == 0.0
    assert pan.map(-source_half_span) == 150.0

    assert B601_JOINT_DIRECTIONS["elbow_flex"] == 1
    assert elbow.source_for_target_min == pytest.approx(-source_half_span)
    assert elbow.source_for_target_max is None
    assert elbow.map(-source_half_span) == -200.0
    assert elbow.map(0.0) == 0.0
    assert elbow.map(source_half_span) == 1.0

    wrist_flex = profile.joints["wrist_flex"]
    assert wrist_flex.map(25.0) == 0.0
    assert wrist_flex.map(-source_half_span) == -80.0
    assert wrist_flex.map(source_half_span) == 90.0

    gripper = profile.joints["gripper"]
    assert gripper.map(GRIPPER_ACTION_OPEN) == GRIPPER_ACTION_OPEN
    assert gripper.map(GRIPPER_ACTION_CLOSED) == GRIPPER_ACTION_CLOSED


def test_teleoperate_configures_mapping_only_for_so102_to_b601(tmp_path):
    leader = _make_leader(tmp_path)
    leader_cfg = leader.config
    b601_cfg = RebotB601FollowerRobotConfig(port="/dev/ttyACM2", id="follower")

    configure_follower_action_conversion(leader, leader_cfg, b601_cfg)

    assert isinstance(leader.b601_action_conversion, ActionMappingProfile)

    other_leader = _make_leader(tmp_path / "other")
    so102_follower_cfg = SO102FollowerConfig(port="/dev/ttyACM1", id="follower_so102_v1")

    configure_follower_action_conversion(other_leader, other_leader.config, so102_follower_cfg)

    assert other_leader.b601_action_conversion is None


def test_so102_get_action_outputs_b601_ranges_after_automatic_configuration(tmp_path):
    source_half_span = (3000 - 1000) * 180 / (_MODEL_RESOLUTION - 1)
    positions = {
        "shoulder_pan": source_half_span,
        "shoulder_lift": -source_half_span,
        "elbow_flex": source_half_span,
        "wrist_flex": -source_half_span,
        "wrist_yaw": source_half_span,
        "wrist_roll": -90.0,
        "gripper": 100.0,
    }
    leader = _make_leader(tmp_path, positions)
    robot_cfg = RebotB601FollowerRobotConfig(port="/dev/ttyACM2", id="follower")

    configure_follower_action_conversion(leader, leader.config, robot_cfg)
    action = leader.get_action()

    assert action["shoulder_pan.pos"] == -150.0
    assert action["shoulder_lift.pos"] == 1.0
    assert action["elbow_flex.pos"] == 1.0
    assert action["wrist_flex.pos"] == -80.0
    assert action["wrist_yaw.pos"] == 90.0
    assert action["wrist_roll.pos"] == 90.0
    assert action["gripper.pos"] == -270.0


def test_so102_get_action_stays_unmapped_for_so102_follower(tmp_path):
    positions = dict.fromkeys(JOINT_NAMES, 12.5)
    positions["gripper"] = 50.0
    leader = _make_leader(tmp_path, positions)
    leader.b601_zero_positions["shoulder_pan"] = 70.0
    leader.b601_zero_positions["wrist_flex"] = -40.0
    follower_cfg = SO102FollowerConfig(port="/dev/ttyACM1", id="follower_so102_v1")

    configure_follower_action_conversion(leader, leader.config, follower_cfg)
    action = leader.get_action()

    assert action["shoulder_pan.pos"] == 12.5
    assert action["gripper.pos"] == -135.0


def test_b601_mapping_rejects_missing_calibration_joint(tmp_path):
    leader = _make_leader(tmp_path)
    calibration = leader.calibration.copy()
    calibration.pop("wrist_yaw")

    with pytest.raises(ValueError, match="calibration joints"):
        build_b601_action_conversion(
            calibration=calibration,
            motors=leader.bus.motors,
            model_resolution_table=leader.bus.model_resolution_table,
            joint_limits=RebotB601FollowerRobotConfig(port="/dev/ttyACM2").joint_limits,
            zero_positions=leader.b601_zero_positions,
        )


def test_b601_mapping_requires_recorded_zero_pose(tmp_path):
    leader = _make_leader(tmp_path)
    leader.b601_zero_positions = {}

    with pytest.raises(ValueError, match="has no B601 zero pose"):
        configure_follower_action_conversion(
            leader,
            leader.config,
            RebotB601FollowerRobotConfig(port="/dev/ttyACM2", id="follower"),
        )


def test_zero_pose_may_equal_source_endpoint_when_b601_limit_is_one_degree(tmp_path):
    leader = _make_leader(tmp_path)
    source_half_span = (3000 - 1000) * 180 / (_MODEL_RESOLUTION - 1)
    leader.b601_zero_positions["elbow_flex"] = source_half_span

    profile = build_b601_action_conversion(
        calibration=leader.calibration,
        motors=leader.bus.motors,
        model_resolution_table=leader.bus.model_resolution_table,
        joint_limits=RebotB601FollowerRobotConfig(port="/dev/ttyACM2").joint_limits,
        zero_positions=leader.b601_zero_positions,
    )

    elbow = profile.joints["elbow_flex"]
    assert elbow.source_for_target_max is None
    assert elbow.map(source_half_span) == 0.0
    assert elbow.map(-source_half_span) == -200.0
