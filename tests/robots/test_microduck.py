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

"""The Microduck robot speaks robotd's NDJSON JSON-RPC protocol.

A fake robotd on a Unix socket answers requests, streams ``robot.state`` to subscribers and records the
notifications it receives, so these tests pin the wire messages LeRobot sends and how it reads the state.
"""

import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from lerobot.robots.microduck import MicroDuck, MicroDuckConfig
from lerobot.robots.microduck.microduck import JOINT_NAMES
from lerobot.utils.errors import DeviceNotConnectedError

STATE = {
    "t": 1.0,
    "move": {"requested": [0.1, 0.0, 0.2], "applied": [0.1, 0.0, 0.2]},
    "head": [0.0, 0.0, 0.0, 0.0],
    "policy": "walk",
    "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
    "loop": {"hz": 50.0, "missed": 0},
    "joints": [0.01 * i for i in range(len(JOINT_NAMES))],
    "targets": [0.0] * len(JOINT_NAMES),
    "odom": {"position": [1.0, 2.0, 0.0], "yaw": 0.5},
    "imu": {"gyro": [0.1, 0.2, 0.3], "quat": [1.0, 0.0, 0.0, 0.0]},
}


class FakeRobotd:
    def __init__(self, path: str):
        self.path = path
        self.requests: list[dict] = []
        self.notifications: list[dict] = []
        self.state_connection: socket.socket | None = None
        self.close_before_first_state = False
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(path)
        self._server.listen()
        self._stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket):
        reader = conn.makefile("r")
        for line in reader:
            message = json.loads(line)
            if "id" not in message:
                self.notifications.append(message)
                continue
            self.requests.append(message)
            method = message["method"]
            result = {"api_version": 41} if method == "hello" else {"accepted": True}
            conn.sendall(
                (json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}) + "\n").encode()
            )
            if method == "robot.subscribe":
                self.state_connection = conn
                if self.close_before_first_state:
                    self.close_state()
                    return
                while not self._stop.is_set():
                    state = {"jsonrpc": "2.0", "method": "robot.state", "params": STATE}
                    try:
                        conn.sendall((json.dumps(state) + "\n").encode())
                    except OSError:
                        return
                    time.sleep(0.01)
                return

    def close(self):
        self._stop.set()
        self._server.close()

    def methods(self) -> list[str]:
        return [r["method"] for r in self.requests]

    def close_state(self):
        self.state_connection.shutdown(socket.SHUT_RDWR)
        self.state_connection.close()


@pytest.fixture
def robotd():
    # Unix socket paths are limited to ~104 bytes on macOS, so stay out of pytest's long tmp_path.
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        server = FakeRobotd(str(Path(tmp) / "robotd.sock"))
        yield server
        server.close()


@pytest.fixture
def robot(robotd):
    robot = MicroDuck(MicroDuckConfig(robotd_address=robotd.path, use_camera=False))
    robot.connect()
    yield robot
    if robot.is_connected:
        robot.disconnect()


def test_connect_says_hello_subscribes_and_enables(robot, robotd):
    methods = robotd.methods()
    assert methods[0] == "hello"
    assert "robot.subscribe" in methods
    assert methods[-1] == "robot.enable"
    enable = robotd.requests[-1]
    assert enable["params"] == {"on": True}


def test_observation_reads_the_state_stream(robot):
    obs = robot.get_observation()
    assert set(obs) == set(robot.observation_features)
    assert obs["left_hip_yaw.pos"] == 0.0
    assert obs["right_ankle.pos"] == pytest.approx(0.14)
    assert (obs["gyro.x"], obs["gyro.y"], obs["gyro.z"]) == (0.1, 0.2, 0.3)
    assert obs["gravity.z"] == -1.0
    assert (obs["odom.x"], obs["odom.y"], obs["odom.yaw"]) == (1.0, 2.0, 0.5)
    assert obs["vyaw.applied"] == 0.2


def test_send_action_clips_and_notifies_move_and_head(robot, robotd):
    sent = robot.send_action({"vx": 1.0, "vy": -0.1, "vyaw": 0.5, "head_yaw.pos": 2.0})
    assert sent["vx"] == pytest.approx(0.3)
    assert sent["head_yaw.pos"] == pytest.approx(0.8)
    assert set(sent) == set(robot.action_features)

    deadline = time.monotonic() + 1.0
    while len(robotd.notifications) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    move, head = robotd.notifications[-2:]
    assert move["method"] == "robot.move"
    assert move["params"] == {"vx": pytest.approx(0.3), "vy": pytest.approx(-0.1), "vyaw": 0.5}
    assert head["method"] == "robot.head"
    assert head["params"] == {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.8, "head_roll": 0.0}


def test_disconnect_stops_without_relaxing(robot, robotd):
    robot.disconnect()
    assert not robot.is_connected
    assert "robot.stop" in robotd.methods()
    assert "robot.relax" not in robotd.methods()


def test_state_eof_rejects_observations_and_actions_but_allows_cleanup(robot, robotd):
    robotd.close_state()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            robot.get_observation()
        except DeviceNotConnectedError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("The closed state stream still supplies observations.")

    assert robot.is_connected
    with pytest.raises(DeviceNotConnectedError, match="robot.state stream stopped"):
        robot.get_observation()
    with pytest.raises(DeviceNotConnectedError, match="robot.state stream stopped"):
        robot.send_action({"vx": 0.1})
    assert robotd.notifications == []
    robot.disconnect()
    assert not robot.is_connected
    assert not robot._cmd.is_open
    assert not robot._state_rpc.is_open
    assert robot._threads == []
    assert "robot.stop" in robotd.methods()

    robot.connect()
    assert robot.get_observation()["right_ankle.pos"] == pytest.approx(0.14)
    robot.send_action({})


def test_state_eof_before_first_state_closes_both_connections(robotd, tmp_path):
    robotd.close_before_first_state = True
    robot = MicroDuck(MicroDuckConfig(robotd_address=robotd.path, use_camera=False, calibration_dir=tmp_path))
    with pytest.raises(DeviceNotConnectedError, match="robot.state stream stopped"):
        robot.connect()
    assert not robot.is_connected
    assert not robot._cmd.is_open
    assert not robot._state_rpc.is_open
    assert robot._threads == []


def test_observation_keeps_validated_state_if_eof_arrives_after_check(robot, robotd, monkeypatch):
    check_state = robot._check_state_stream

    def fail_after_check():
        snapshot = check_state()
        # Force the former check/snapshot race without scheduling a thread.
        with robot._state_lock:
            robot._state_error = ConnectionError("EOF after state validation")
            robot._latest_state = None
        robotd.close_state()
        return snapshot

    monkeypatch.setattr(robot, "_check_state_stream", fail_after_check)
    observation = robot.get_observation()
    assert observation["right_ankle.pos"] == pytest.approx(0.14)
    assert observation["gravity.z"] == -1
    with pytest.raises(DeviceNotConnectedError, match="robot.state stream stopped"):
        robot.get_observation()


@pytest.mark.parametrize("method", ["hello", "robot.subscribe", "robot.enable"])
def test_connect_error_closes_both_connections(robotd, tmp_path, monkeypatch, method):
    robot = MicroDuck(MicroDuckConfig(robotd_address=robotd.path, use_camera=False, calibration_dir=tmp_path))
    rpc = robot._state_rpc if method == "robot.subscribe" else robot._cmd
    request = rpc.request

    def fail_request(name, *args, **kwargs):
        if name == method:
            raise ConnectionError("Injected handshake failure")
        return request(name, *args, **kwargs)

    monkeypatch.setattr(rpc, "request", fail_request)
    start = time.monotonic()
    with pytest.raises(ConnectionError, match="Injected handshake failure"):
        robot.connect()
    assert time.monotonic() - start < robot.config.rpc_timeout_s + robot.config.camera_timeout_s + 1
    assert not robot.is_connected
    assert not robot._cmd.is_open
    assert not robot._state_rpc.is_open
    assert robot._threads == []
