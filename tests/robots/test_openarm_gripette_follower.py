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

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from lerobot.cameras.gripette import GripetteCamera, GripetteCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.openarm_follower import OpenArmFollower
from lerobot.robots.openarm_gripette_follower import OpenArmGripetteFollower, OpenArmGripetteFollowerConfig

_ADAPTER = "lerobot.motors.damiao.motorbridge_bus"
_FOLLOWER = "lerobot.robots.openarm_follower.openarm_follower"
_CAMERA = "lerobot.cameras.gripette.camera_gripette"
_ROBOT = "lerobot.robots.openarm_gripette_follower.openarm_gripette_follower"
_JOINTS = [f"joint_{i}" for i in range(1, 8)]


class FakeGripperClient:
    """Stands in for ``gripette.client.GripperClient``."""

    def __init__(self, addr: str):
        self.addr = addr
        self._closed = threading.Event()
        self.connect = MagicMock()
        self.torque_on = MagicMock()
        self.torque_off = MagicMock()
        self.move = MagicMock()
        self.read_motors = MagicMock(return_value=(0.4, 0.5))
        self.close = MagicMock(side_effect=self._closed.set)
        ok, buf = cv2.imencode(".jpg", np.full((8, 6, 3), (200, 0, 0), dtype=np.uint8))
        assert ok
        self._jpeg = buf.tobytes()

    def stream(self):
        while not self._closed.is_set():
            yield SimpleNamespace(jpeg_data=self._jpeg, motor1=0.4, motor2=0.5)
            self._closed.wait(0.01)


@contextmanager
def _robot(**config_kwargs):
    controller = MagicMock()
    state = MagicMock(pos=0.0, vel=0.0, torq=0.0, t_mos=0.0, t_rotor=0.0)
    controller.add_damiao_motor.side_effect = lambda *a: MagicMock(get_state=MagicMock(return_value=state))
    with (
        patch(f"{_FOLLOWER}.require_package", lambda *a, **kw: None),
        patch(f"{_FOLLOWER}.platform.system", return_value="Darwin"),
        patch(f"{_ADAPTER}.require_package", lambda *a, **kw: None),
        patch(f"{_ADAPTER}._is_macos", return_value=False),
        patch(f"{_ADAPTER}.MotorBridgeController") as controller_cls,
        patch(f"{_ADAPTER}.MotorBridgeMode", MagicMock()),
        patch(f"{_CAMERA}.require_package", lambda *a, **kw: None),
        patch(f"{_CAMERA}.GripperClient", FakeGripperClient),
        patch(f"{_ROBOT}.require_package", lambda *a, **kw: None),
        patch(f"{_ROBOT}.GripperClient", FakeGripperClient),
    ):
        controller_cls.from_socketcanfd.return_value = controller
        config = OpenArmGripetteFollowerConfig(
            port="can0",
            joint_limits=dict.fromkeys(_JOINTS, (-180.0, 180.0)),
            gripette_addr="gripette:50051",
            **config_kwargs,
        )
        yield make_robot_from_config(config)


def test_features_and_arm_bus_without_gripper():
    with _robot() as robot:
        assert isinstance(robot, OpenArmGripetteFollower)
        assert isinstance(robot, OpenArmFollower)
        assert robot.config.type == "openarm_gripette_follower"
        assert robot.cameras == {}
        assert set(robot.bus.motors) == set(_JOINTS)
        motors = {f"{j}.pos" for j in _JOINTS} | {"proximal.pos", "distal.pos"}
        assert set(robot.action_features) == motors
        assert robot.observation_features == dict.fromkeys(motors, float)

    cameras = {"gripette": GripetteCameraConfig(addr="gripette:50051", width=3, height=4, fps=10)}
    with _robot(cameras=cameras) as robot:
        assert isinstance(robot.cameras["gripette"], GripetteCamera)
        assert robot.observation_features["gripette"] == (4, 3, 3)


@pytest.mark.parametrize(
    ("goals", "expected_move"),
    [
        ({"proximal.pos": 0.5, "distal.pos": 1.0}, (0.5, 1.0)),
        ({"proximal.pos": -1.0, "distal.pos": 3.0}, (0.0, 2.025)),
        ({"proximal.pos": 3.0}, (1.632, 0.5)),
    ],
)
def test_connect_observe_act_disconnect(goals, expected_move):
    cameras = {"gripette": GripetteCameraConfig(addr="gripette:50051", width=3, height=4, fps=10)}
    with _robot(cameras=cameras) as robot:
        robot.connect(calibrate=False)
        client = robot._gripper_client
        client.torque_on.assert_called_once()
        client.read_motors.assert_called()
        camera_client = robot.cameras["gripette"]._client
        assert camera_client is not client

        obs = robot.get_observation()
        assert {f"{j}.pos" for j in _JOINTS} <= obs.keys()
        assert (obs["proximal.pos"], obs["distal.pos"]) == pytest.approx((0.4, 0.5))
        assert obs["gripette"].shape == (4, 3, 3)
        assert obs["gripette"][0, 0, 2] > 100

        sent = robot.send_action({"joint_1.pos": 1.0, **goals})
        client.move.assert_called_once_with(*expected_move)
        assert sent["joint_1.pos"] == pytest.approx(1.0)
        assert (sent["proximal.pos"], sent["distal.pos"]) == pytest.approx(expected_move)

        robot.disconnect()
        client.torque_off.assert_called_once()
        client.close.assert_called_once()
        camera_client.close.assert_called_once()
        assert not robot.is_connected
