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

"""Mocked CAN hardware for the Metal follower and leader tests."""

import math
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.motors.damiao import damiao as damiao_module
from lerobot.motors.damiao.tables import MOTOR_LIMIT_PARAMS, MotorType
from lerobot.robots.metal_follower import MetalFollower, metal_follower as metal_module, mit_bus
from lerobot.robots.yam_follower import mit_control


def without_hardware_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build Metal arms without the metal extra; their hardware is mocked, and CAN frames are namespaces."""
    monkeypatch.setattr(metal_module, "require_package", lambda *a, **kw: None)
    monkeypatch.setattr(damiao_module, "require_package", lambda *a, **kw: None)
    frames = SimpleNamespace(Message=SimpleNamespace, interface=SimpleNamespace(Bus=lambda **kw: FakeCan()))
    monkeypatch.setattr(damiao_module, "can", frames)
    monkeypatch.setattr(mit_bus, "can", frames, raising=False)


def raw_states(gripper=0.0):
    position = np.array([0, 0, 0, 0, 0, 0, gripper], dtype=float)
    return mit_control.MotorStates(position=position, velocity=np.zeros(7), torque=np.zeros(7))


def mock_bus(states=None):
    bus = MagicMock(spec=mit_bus.MetalMotorsBus)
    bus.motor_names = mit_bus.MOTOR_NAMES
    bus.enabled = False
    bus.read_states.return_value = raw_states() if states is None else states
    bus.read_fresh.return_value = (raw_states(), np.ones(7, dtype=bool), np.ones(7, dtype=bool))
    bus.send_confirmed.return_value = []
    counts = iter(range(1, 1_000_000))
    bus.sample.side_effect = lambda: (*bus.read_fresh()[:2], dict.fromkeys(mit_bus.MOTOR_NAMES, next(counts)))
    return bus


def attach_bus(arm: MetalFollower, bus):
    arm.bus = arm.controller._bus = bus
    return bus


def mock_hardware(arm: MetalFollower, monkeypatch: pytest.MonkeyPatch):
    """Mock the arm's bus, and replace its control loop with one that returns at once."""
    monkeypatch.setattr(arm.controller, "_run", lambda: None)
    return attach_bus(arm, mock_bus())


def mark_running(controller: mit_control.MitController):
    """Make the controller report active, with a finished thread that stop() can join."""
    controller._thread = threading.Thread(target=lambda: None)
    controller._thread.start()


def make_ready(arm: MetalFollower, position=None):
    """Mark the arm connected, with its controller active at `position` (zeros by default)."""
    arm._connected = True
    position = np.zeros(7) if position is None else np.asarray(position, dtype=float)
    zeros = np.zeros(7)
    # The base method skips the start check, so tests can start anywhere.
    mit_control.MitController._set_initial_state(
        arm.controller, mit_control.JointState(position=position, velocity=zeros, torque=zeros)
    )
    arm._start_position = position.copy()
    mark_running(arm.controller)


class FakeCan:
    """A CAN interface with seven Metal motors, which answer refresh, enable, disable and MIT frames.

    Motor states are by name, positions in degrees. Motors in `silent` don't answer; motors in `stuck` answer
    but ignore enable and disable. `queue` holds the replies not received yet.
    """

    def __init__(self):
        self.motors = {
            name: {"position": 0.0, "status": 0, "temperature": 30} for name in mit_bus.MOTOR_NAMES
        }
        self.silent: set[str] = set()
        self.stuck: set[str] = set()
        self.queue: list = []
        self.sent: list = []

    def send(self, msg):
        self.sent.append(msg)
        data = bytes(msg.data)
        motor_id = data[0] if msg.arbitration_id == 0x7FF else msg.arbitration_id
        name = mit_bus.MOTOR_NAMES[motor_id - 1]
        if data[:7] == b"\xff" * 7 and data[7] in (0xFC, 0xFD) and name not in self.stuck:
            self.motors[name]["status"] = int(data[7] == 0xFC)
        if name not in self.silent:
            self.queue.append(SimpleNamespace(arbitration_id=0x10 + motor_id, data=self.state_frame(name)))

    def recv(self, timeout=None):
        return self.queue.pop(0) if self.queue else None

    def shutdown(self):
        pass

    def state_frame(self, name):
        motor = self.motors[name]
        motor_id = mit_bus.MOTOR_NAMES.index(name) + 1
        pmax, _, _ = MOTOR_LIMIT_PARAMS[MotorType[mit_bus.MOTOR_TYPES[motor_id - 1].upper()]]
        q = round((math.radians(motor["position"]) + pmax) / (2 * pmax) * 0xFFFF)
        dq, tau = 0x7FF, 0x7FF  # zero velocity and torque
        return bytes(
            [
                (motor["status"] << 4) | motor_id,
                q >> 8,
                q & 0xFF,
                dq >> 4,
                ((dq & 0xF) << 4) | (tau >> 8),
                tau & 0xFF,
                motor["temperature"],
                motor["temperature"],
            ]
        )

    def mit_frames(self):
        return [
            msg for msg in self.sent if msg.arbitration_id != 0x7FF and bytes(msg.data)[:7] != b"\xff" * 7
        ]


def fake_can(arm_or_bus, feedback_timeout_s=None) -> FakeCan:
    """Connect an arm's (or a bare) `MetalMotorsBus` to a `FakeCan`, and record its MIT batches in `batches`."""
    bus = arm_or_bus.bus if isinstance(arm_or_bus, MetalFollower) else arm_or_bus
    fake = FakeCan()
    bus.bus.canbus, bus.bus._is_connected = fake, True
    if feedback_timeout_s is not None:
        bus.feedback_timeout_s = feedback_timeout_s
    fake.batches = []
    send_batch = bus._send_batch

    def record(commands):
        fake.batches.append(dict(commands))
        return send_batch(commands)

    bus._send_batch = record
    return fake
