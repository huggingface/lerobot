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

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots import make_robot_from_config
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot_robot_openarm_gripette import OpenArmGripetteFollower, OpenArmGripetteFollowerConfig
from lerobot_robot_openarm_gripette.manual_start import prepare_manual_start
from lerobot_robot_openarm_gripette.prediction_logger import PredictionLogger

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
        motors = {f"{j}.pos" for j in _JOINTS} | {"proximal.pos", "distal.pos"}
        assert set(robot.action_features) == motors
        assert robot.observation_features == {**dict.fromkeys(motors, float), "gripette": (4, 3, 3)}


def test_prepare_manual_start_disables_torque_captures_pose_and_holds():
    robot = MagicMock()
    events = []
    robot.arm.bus.disable_torque.side_effect = lambda: events.append("disable")
    robot.arm.bus.enable_torque.side_effect = lambda: events.append("enable")
    robot.arm.bus.configure_motors.side_effect = lambda: events.append("configure")
    robot.send_action.side_effect = lambda _: events.append("open_gripper")
    robot.arm.send_action.side_effect = lambda _: events.append("hold")
    robot.arm.get_observation.return_value = {f"joint_{i}.pos": float(i) for i in range(1, 8)}

    result = prepare_manual_start(
        robot,
        np.array([1, -1, 1, -1, 1, -1, 1]),
        gripper_action={"proximal.pos": 1.0, "distal.pos": 1.0},
        input_fn=lambda _: events.append("prompt"),
        sleep_fn=lambda _: events.append("sleep"),
    )

    assert events == [
        "disable",
        "sleep",
        "disable",
        "open_gripper",
        "prompt",
        "hold",
        "enable",
        "configure",
        "hold",
    ]
    assert result.tolist() == [1, -2, 3, -4, 5, -6, 7]
    robot.send_action.assert_called_once_with({"proximal.pos": 1.0, "distal.pos": 1.0})
    assert robot.arm.send_action.call_count == 2
    robot.arm.send_action.assert_called_with({f"joint_{i}.pos": float(i) for i in range(1, 8)})


def test_prediction_logger_writes_and_flushes_jsonl(tmp_path):
    logger = PredictionLogger(tmp_path)
    logger.write({"type": "metadata", "initial_joints_deg": np.array([1.0, 2.0])})
    logger.write(
        {
            "type": "prediction",
            "chunk_index": 0,
            "action_index": 1,
            "raw": {"x": np.float32(0.25), "closure": np.float64(0.8)},
            "joint_targets_deg": np.arange(7, dtype=np.float32),
            "executed": False,
        }
    )

    with logger.path.open() as f:
        records = [json.loads(line) for line in f]
    logger.close()

    assert records == [
        {"type": "metadata", "initial_joints_deg": [1.0, 2.0]},
        {
            "type": "prediction",
            "chunk_index": 0,
            "action_index": 1,
            "raw": {"x": 0.25, "closure": 0.8},
            "joint_targets_deg": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "executed": False,
        },
    ]


@pytest.mark.parametrize(
    ("goals", "expected_move"),
    [
        ({"proximal.pos": 0.5, "distal.pos": 1.0}, (0.5, 1.0)),
        ({"proximal.pos": -1.0, "distal.pos": 3.0}, (0.0, 2.025)),
        ({"proximal.pos": 3.0}, (1.632, 0.5)),  # distal holds its observed position
    ],
)
def test_connect_observe_act_disconnect(goals, expected_move):
    with _robot() as robot:
        robot.connect(calibrate=False)
        client = robot._gripper._client
        client.torque_on.assert_called_once()

        obs = robot.get_observation()
        assert {f"{j}.pos" for j in _JOINTS} <= obs.keys()
        assert (obs["proximal.pos"], obs["distal.pos"]) == pytest.approx((0.4, 0.5))
        assert obs["gripette"].shape == (4, 3, 3)
        assert obs["gripette"][0, 0, 2] > 100  # BGR blue decoded into the RGB blue channel

        sent = robot.send_action({"joint_1.pos": 1.0, **goals})
        client.move.assert_called_once_with(*expected_move)
        assert sent["joint_1.pos"] == pytest.approx(1.0)
        assert (sent["proximal.pos"], sent["distal.pos"]) == pytest.approx(expected_move)

        robot.disconnect()
        client.torque_off.assert_called_once()
        client.close.assert_called_once()
        assert not robot.is_connected
