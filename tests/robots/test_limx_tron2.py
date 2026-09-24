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

"""Tests for the LimX TRON2 robot class.

These drive the *real* ``tron2_env`` motion controller against an in-memory transport, so they need
no robot, no network and no WebSocket server -- the same approach ``tron2_env``'s own
``mock_quickstart`` example uses. Without the ``tron2-env`` extra the module skips; config and
joint-layout coverage lives in ``test_limx_tron2_utils.py`` and runs unconditionally.
"""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("tron2_env", reason="tron2-env is required (install lerobot[limx_tron2])")

from lerobot.robots.limx_tron2 import LimxTron2, LimxTron2Config  # noqa: E402
from lerobot.robots.limx_tron2.joints import SERVOJ_DIM, STATE_DIM  # noqa: E402


class FakeTransport:
    """In-memory transport: records setpoints, replays a fixed state."""

    def __init__(self, state: np.ndarray | None = None) -> None:
        self._connected = True
        self._lock = threading.Lock()
        self._sent: list[np.ndarray] = []
        self._state = np.zeros(STATE_DIM) if state is None else np.asarray(state, dtype=float)
        self.gripper_commands: list[tuple[float, float]] = []
        self.disconnect_calls = 0

    def send_joint_cmd(self, q: np.ndarray) -> None:
        with self._lock:
            self._sent.append(np.asarray(q, dtype=np.float64).copy())

    def get_joint_state(self, timeout: float = 1.0, max_age: float | None = None) -> dict:
        return {"timestamp": int(time.time() * 1000), "states": self._state.tolist()}

    def get_head_position(self) -> np.ndarray:
        return self._state[16:18].copy()

    def set_gripper(self, left_opening: float, right_opening: float) -> None:
        self.gripper_commands.append((left_opening, right_opening))

    def wait_until_reached(self, target_joints, tolerance: float = 0.05, timeout: float = 10.0) -> bool:
        return True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def sent_frames(self) -> list[np.ndarray]:
        with self._lock:
            return list(self._sent)


class HarnessRobot(LimxTron2):
    """``LimxTron2`` wired to a fake transport instead of a real WebSocket."""

    def __init__(self, config: LimxTron2Config, transport: FakeTransport) -> None:
        super().__init__(config)
        self._transport = transport

    def _make_controller(self):
        from tron2_env.motion import MotionController

        return MotionController(
            transport=self._transport,
            publish_rate=self.config.publish_rate,
            eta_default=self.config.eta_default,
        )


def make_config(**overrides) -> LimxTron2Config:
    base = {"robot_ip": "127.0.0.1", "publish_rate": 100.0}
    base.update(overrides)
    return LimxTron2Config(**base)


# ------------------------------------------------------------------ features


def test_observation_features_match_get_observation():
    transport = FakeTransport(state=np.arange(STATE_DIM, dtype=float))
    robot = HarnessRobot(make_config(), transport)
    robot.connect()
    try:
        assert set(robot.get_observation()) == set(robot.observation_features)
        assert set(robot.action_features) == set(robot.observation_features)
    finally:
        robot.disconnect()


def test_observation_reports_measured_state():
    state = np.arange(STATE_DIM, dtype=float)
    robot = HarnessRobot(make_config(), FakeTransport(state=state))
    robot.connect()
    try:
        observation = robot.get_observation()
        # Spelled out independently of the robot's own slicing helpers:
        # state = [L_arm(0:7), L_grip(7), R_arm(8:15), R_grip(15), head(16:18)]
        expected = np.concatenate((state[0:7], state[8:15], state[16:18]))
        for name, value in zip(robot.config.joint_names, expected, strict=True):
            assert observation[f"{name}.pos"] == pytest.approx(value)
    finally:
        robot.disconnect()


# ------------------------------------------------------------------- actions


def test_action_reaches_the_transport_as_a_16_value_setpoint():
    transport = FakeTransport()
    robot = HarnessRobot(make_config(), transport)
    robot.connect()
    try:
        action = {f"{name}.pos": 0.25 for name in robot.config.joint_names}
        robot.send_action(action)
        time.sleep(0.12)
    finally:
        robot.disconnect()

    frames = transport.sent_frames()
    assert frames, "controller published no setpoints"
    assert frames[-1].shape == (SERVOJ_DIM,)
    np.testing.assert_allclose(frames[-1], 0.25, atol=1e-6)


def test_action_order_follows_joint_names():
    transport = FakeTransport()
    robot = HarnessRobot(make_config(), transport)
    robot.connect()
    try:
        action = {f"{name}.pos": float(index) for index, name in enumerate(robot.config.joint_names)}
        robot.send_action(action)
        time.sleep(0.12)
    finally:
        robot.disconnect()

    np.testing.assert_allclose(transport.sent_frames()[-1], np.arange(SERVOJ_DIM), atol=1e-6)


def test_missing_joint_key_is_reported():
    robot = HarnessRobot(make_config(), FakeTransport())
    robot.connect()
    try:
        with pytest.raises(KeyError, match="missing joint keys"):
            robot.send_action({"left_arm_shoulder_pitch.pos": 0.0})
    finally:
        robot.disconnect()


def test_half_specified_gripper_pair_is_rejected():
    robot = HarnessRobot(make_config(include_grippers=True), FakeTransport())
    robot.connect()
    try:
        action = {f"{name}.pos": 0.0 for name in robot.config.joint_names}
        action["left_gripper.pos"] = 0.5
        with pytest.raises(KeyError, match="only one side"):
            robot.send_action(action)
    finally:
        robot.disconnect()


def test_grippers_are_forwarded_when_enabled():
    transport = FakeTransport()
    robot = HarnessRobot(make_config(include_grippers=True), transport)
    robot.connect()
    try:
        action = {f"{name}.pos": 0.0 for name in robot.config.joint_names}
        action["left_gripper.pos"] = 0.25
        action["right_gripper.pos"] = 0.75
        robot.send_action(action)
        time.sleep(0.12)
    finally:
        robot.disconnect()

    assert transport.gripper_commands, "no gripper command reached the transport"
    # The runtime's end-effector path scales normalised openings to 0..100.
    left, right = transport.gripper_commands[-1]
    assert left == pytest.approx(25.0)
    assert right == pytest.approx(75.0)


# ----------------------------------------------------------------- lifecycle


def test_operations_require_a_connection():
    robot = HarnessRobot(make_config(), FakeTransport())
    assert not robot.is_connected
    with pytest.raises(RuntimeError, match="not connected"):
        robot.get_observation()
    with pytest.raises(RuntimeError, match="not connected"):
        robot.send_action({})


def test_context_manager_connects_and_disconnects():
    transport = FakeTransport()
    robot = HarnessRobot(make_config(), transport)
    with robot as connected:
        assert connected.is_connected
    assert not robot.is_connected
    assert transport.disconnect_calls == 1, "transport was not closed exactly once"
    assert not transport.is_connected()


def test_disconnect_is_idempotent():
    robot = HarnessRobot(make_config(), FakeTransport())
    robot.connect()
    robot.disconnect()
    robot.disconnect()
    assert not robot.is_connected


def test_configure_surfaces_controller_faults():
    robot = HarnessRobot(make_config(), FakeTransport())
    robot.connect()
    try:
        robot.configure()  # healthy controller: no fault, no error
    finally:
        robot.disconnect()
