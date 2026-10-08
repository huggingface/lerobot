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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots import make_robot_from_config
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot_robot_openarm_gripette import OpenArmGripetteFollower, OpenArmGripetteFollowerConfig

_ADAPTER = "lerobot.motors.damiao.motorbridge_bus"
_FOLLOWER = "lerobot.robots.openarm_follower.openarm_follower"
_JOINTS = [f"joint_{i}" for i in range(1, 8)]


@contextmanager
def _robot(**config_kwargs):
    controller = MagicMock()
    state = MagicMock(pos=0.0, vel=0.0, torq=0.0)
    controller.add_damiao_motor.side_effect = lambda *a: MagicMock(get_state=MagicMock(return_value=state))
    with (
        patch(f"{_FOLLOWER}.require_package", lambda *a, **kw: None),
        patch(f"{_ADAPTER}.require_package", lambda *a, **kw: None),
        patch(f"{_ADAPTER}.MotorBridgeController") as controller_cls,
        patch(f"{_ADAPTER}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_socketcanfd.return_value = controller
        config = OpenArmGripetteFollowerConfig(
            arm_config=OpenArmFollowerConfigBase(
                port="can0", joint_limits=dict.fromkeys(_JOINTS, (-180.0, 180.0))
            ),
            gripette_addr="gripette:50051",
            gripper_camera_height=4,
            gripper_camera_width=3,
            **config_kwargs,
        )
        yield make_robot_from_config(config)


def test_features_and_arm_bus_without_gripper():
    with _robot() as robot:
        assert isinstance(robot, OpenArmGripetteFollower)
        assert set(robot.arm.bus.motors) == set(_JOINTS)
        motors = {f"{j}.pos" for j in _JOINTS} | {"gripper.pos"}
        assert set(robot.action_features) == motors
        assert robot.observation_features == {**dict.fromkeys(motors, float), "gripette": (4, 3, 3)}


@pytest.mark.parametrize(
    ("goal", "expected_move", "expected_sent"),
    [(0.5, (0.5, 1.0), 0.5), (-1.0, (0.0, 0.0), 0.0), (3.0, (1.632, 2.025), 1.632)],
)
def test_connect_observe_act_disconnect(goal, expected_move, expected_sent):
    with _robot(gripper_motor_scales=(1.0, 2.0)) as robot:
        robot.connect(calibrate=False)
        client = robot._gripper._client
        client.torque_on.assert_called_once()

        obs = robot.get_observation()
        assert {f"{j}.pos" for j in _JOINTS} <= obs.keys()
        assert obs["gripper.pos"] == pytest.approx(0.4)
        assert obs["gripette"].shape == (4, 3, 3)
        assert obs["gripette"][0, 0, 2] > 100  # BGR blue decoded into the RGB blue channel

        sent = robot.send_action({"joint_1.pos": 1.0, "gripper.pos": goal})
        client.move.assert_called_once_with(*expected_move)
        assert sent["joint_1.pos"] == pytest.approx(1.0)
        assert sent["gripper.pos"] == pytest.approx(expected_sent)

        robot.disconnect()
        client.torque_off.assert_called_once()
        client.close.assert_called_once()
        assert not robot.is_connected
