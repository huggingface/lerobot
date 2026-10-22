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

"""Real SDK lifecycle/hold behavior over an in-memory transport, without network I/O."""

import threading
from math import radians
from unittest.mock import Mock

import pytest

from lerobot.robots.asimov import Asimov, AsimovConfig
from lerobot.robots.asimov.asimov import ASIMOV_JOINTS

menlo = pytest.importorskip("menlo.asimov")


@pytest.fixture
def sdk_robot(monkeypatch, tmp_path):
    transport = Mock(
        kind="udp",
        endpoint="in-memory",
        capabilities=frozenset({"drive", "state"}),
        default_outcome_timeout=0.1,
    )
    zero_sent = threading.Event()
    commands = []

    def send(command):
        commands.append(command)
        if isinstance(command, menlo.Velocity) and command.is_zero:
            zero_sent.set()
        return len(commands)

    transport.send.side_effect = send

    def open_transport():
        state = menlo.State(
            mode=menlo.Mode.MOVE,
            joints=tuple(menlo.Joint(name, 0.0, None, None, None) for name in ASIMOV_JOINTS),
            gravity=(0.0, 0.0, -1.0),
            gyro=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            error_flags=0,
            alerts=(),
            sequence=1,
            fw_timestamp_us=1,
            protocol_version=1,
        )
        transport.subscribe_state.call_args.args[0](state)

    transport.open.side_effect = open_transport
    # Leave ConnectionConfig/UdpConfig, Robot, Limits, State and the SDK hold thread real.
    monkeypatch.setattr(menlo.ConnectionConfig, "transport_for", lambda *args, **kwargs: transport)
    robot = Asimov(AsimovConfig(host="192.0.2.10", calibration_dir=tmp_path))
    try:
        yield robot, transport, commands, zero_sent
    finally:
        robot.disconnect()


def test_real_sdk_clamping_and_close_zero(sdk_robot):
    robot, transport, commands, _ = sdk_robot
    robot.connect()
    assert not commands  # Connection is observational.
    assert set(robot.get_observation()) == set(robot.observation_features)
    sent = robot.send_action({"x.vel": 1.0, "y.vel": -1.0, "theta.vel": 90.0})
    assert sent == pytest.approx({"x.vel": 0.3, "y.vel": -0.2, "theta.vel": 30.0})
    assert commands[0] == menlo.Velocity(0.3, -0.2, radians(30.0))
    robot.disconnect()
    assert commands[-1] == menlo.Velocity()
    assert all(isinstance(command, menlo.Velocity) for command in commands)
    transport.close.assert_called_once()


def test_real_sdk_expires_command_when_caller_stalls(sdk_robot):
    robot, _, commands, zero_sent = sdk_robot
    robot.connect()
    robot.send_action({"x.vel": 0.1})
    # No further actions or state samples: the 0.25 s hold must end before the 2 s link timeout.
    assert zero_sent.wait(timeout=1.0), "The SDK did not stop an expired command"
    assert commands[-1] == menlo.Velocity()
    assert all(isinstance(command, menlo.Velocity) for command in commands)
