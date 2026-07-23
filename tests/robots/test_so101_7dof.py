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

from lerobot.motors import MotorNormMode
from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_7dof_follower import (
    SO1017DoFFollower,
    SO1017DoFFollowerConfig,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.so101_7dof import (
    ACTION_KEYS,
    DEFAULT_MOTOR_IDS,
    action_gripper_to_native,
    native_gripper_to_action,
)
from lerobot.teleoperators.so101_7dof_leader import SO1017DoFLeader, SO1017DoFLeaderConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config

_FOLLOWER_MODULE = "lerobot.robots.so101_7dof_follower.so101_7dof_follower"
_LEADER_MODULE = "lerobot.teleoperators.so101_7dof_leader.so101_7dof_leader"


def _bus_mock(native_positions: dict[str, float] | None = None) -> MagicMock:
    bus = MagicMock(name="FeetechMotorsBusMock")
    bus.is_connected = True
    bus.is_calibrated = True
    bus.sync_read.return_value = native_positions or dict.fromkeys(DEFAULT_MOTOR_IDS, 0.0)
    return bus


def _make_devices(tmp_path, native_positions: dict[str, float] | None = None):
    leader_bus = _bus_mock(native_positions)
    follower_bus = _bus_mock(native_positions)

    def _leader_bus_factory(*_args, **kwargs):
        leader_bus.motors = kwargs["motors"]
        return leader_bus

    def _follower_bus_factory(*_args, **kwargs):
        follower_bus.motors = kwargs["motors"]
        return follower_bus

    with (
        patch(f"{_LEADER_MODULE}.FeetechMotorsBus", side_effect=_leader_bus_factory),
        patch(f"{_FOLLOWER_MODULE}.FeetechMotorsBus", side_effect=_follower_bus_factory),
    ):
        leader = SO1017DoFLeader(SO1017DoFLeaderConfig(port="/dev/null", calibration_dir=tmp_path / "leader"))
        follower = SO1017DoFFollower(
            SO1017DoFFollowerConfig(port="/dev/null", calibration_dir=tmp_path / "follower")
        )
    return leader, follower


def test_leader_and_follower_expose_same_seven_action_keys(tmp_path):
    leader, follower = _make_devices(tmp_path)
    assert tuple(leader.action_features) == ACTION_KEYS
    assert tuple(follower.action_features) == ACTION_KEYS
    assert leader.action_features == follower.action_features
    assert len(ACTION_KEYS) == 7
    assert "wrist_yaw.pos" in ACTION_KEYS


def test_default_motor_mapping_is_ids_one_through_seven(tmp_path):
    leader, follower = _make_devices(tmp_path)
    assert {name: motor.id for name, motor in leader.bus.motors.items()} == DEFAULT_MOTOR_IDS
    assert {name: motor.id for name, motor in follower.bus.motors.items()} == DEFAULT_MOTOR_IDS
    for device in (leader, follower):
        assert device.bus.motors["gripper"].norm_mode is MotorNormMode.RANGE_0_100
        assert all(
            motor.norm_mode is MotorNormMode.DEGREES
            for name, motor in device.bus.motors.items()
            if name != "gripper"
        )


def test_factories_register_new_types(tmp_path):
    leader_bus = _bus_mock()
    follower_bus = _bus_mock()

    with (
        patch(f"{_LEADER_MODULE}.FeetechMotorsBus", return_value=leader_bus),
        patch(f"{_FOLLOWER_MODULE}.FeetechMotorsBus", return_value=follower_bus),
    ):
        leader = make_teleoperator_from_config(
            SO1017DoFLeaderConfig(port="/dev/null", calibration_dir=tmp_path / "leader")
        )
        follower = make_robot_from_config(
            SO1017DoFFollowerConfig(port="/dev/null", calibration_dir=tmp_path / "follower")
        )

    assert isinstance(leader, SO1017DoFLeader)
    assert isinstance(follower, SO1017DoFFollower)
    assert leader.config.type == "so101_7dof_leader"
    assert follower.config.type == "so101_7dof_follower"


@pytest.mark.parametrize(
    ("native", "action"),
    [(0.0, 0.0), (50.0, -135.0), (100.0, -270.0)],
)
def test_gripper_conversion(native, action):
    assert native_gripper_to_action(native) == pytest.approx(action)
    assert action_gripper_to_native(action) == pytest.approx(native)


def test_follower_writes_action_directly_without_limits_or_position_read(tmp_path):
    _, follower = _make_devices(tmp_path)
    action = dict.fromkeys(ACTION_KEYS, 0.0)
    action["shoulder_pan.pos"] = 999.0
    action["gripper.pos"] = -999.0

    returned = follower.send_action(action)

    assert returned["shoulder_pan.pos"] == 999.0
    assert returned["gripper.pos"] == -999.0
    native_goals = follower.bus.sync_write.call_args.args[1]
    assert native_goals["shoulder_pan"] == 999.0
    assert native_goals["gripper"] == pytest.approx(370.0)
    follower.bus.sync_read.assert_not_called()


def test_leader_returns_transformed_position_without_soft_limits(tmp_path):
    native_positions = dict.fromkeys(DEFAULT_MOTOR_IDS, 0.0)
    native_positions["shoulder_pan"] = 999.0
    leader, _ = _make_devices(tmp_path, native_positions)

    action = leader.get_action()

    assert action["shoulder_pan.pos"] == 999.0


def test_existing_so101_devices_remain_six_motor(tmp_path):
    leader_bus = _bus_mock()
    follower_bus = _bus_mock()

    def _leader_factory(*_args, **kwargs):
        leader_bus.motors = kwargs["motors"]
        return leader_bus

    def _follower_factory(*_args, **kwargs):
        follower_bus.motors = kwargs["motors"]
        return follower_bus

    with (
        patch("lerobot.teleoperators.so_leader.so_leader.FeetechMotorsBus", side_effect=_leader_factory),
        patch("lerobot.robots.so_follower.so_follower.FeetechMotorsBus", side_effect=_follower_factory),
    ):
        leader = SO101Leader(SO101LeaderConfig(port="/dev/null", calibration_dir=tmp_path / "old_leader"))
        follower = SO101Follower(
            SO101FollowerConfig(port="/dev/null", calibration_dir=tmp_path / "old_follower")
        )

    expected = {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "wrist_roll": 5,
        "gripper": 6,
    }
    assert {name: motor.id for name, motor in leader.bus.motors.items()} == expected
    assert {name: motor.id for name, motor in follower.bus.motors.items()} == expected
    assert "wrist_yaw" not in leader.bus.motors
    assert "wrist_yaw" not in follower.bus.motors
