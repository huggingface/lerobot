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

"""Mocked CAN hardware for the single-arm and bimanual YAM follower tests."""

import math
import threading
from unittest.mock import MagicMock

import numpy as np

from lerobot.robots.yam_follower import YamFollower, mit_control, yam_follower as yam_module
from lerobot.robots.yam_follower.config_yam_follower import MOTOR_NAMES


def gripper_calibration(closed=0.1, opened=6.1):
    """Gripper calibration with limits at exactly `closed` and `opened` radians; `calibrate()` would round them."""
    low, high = sorted(math.degrees(value) for value in (closed, opened))
    return yam_module.MotorCalibration(
        id=7, drive_mode=int(opened < closed), homing_offset=0, range_min=low, range_max=high
    )


def calibrate_gripper(arm: YamFollower, closed=0.1, opened=6.1):
    arm.calibration = {"gripper": gripper_calibration(closed, opened)}
    arm.controller.set_gripper_params(yam_module._yam_gripper_params(arm.config, arm.calibration["gripper"]))


def raw_states(gripper=0.1):
    position = np.array([0, 0, 0, 0, 0, 0, gripper], dtype=float)
    return mit_control.MotorStates(position=position, velocity=np.zeros(7), torque=np.zeros(7))


def mock_bus(states=None):
    bus = MagicMock(spec=yam_module._YamMotorsBus)
    bus.enabled = False
    bus.read_states.return_value = raw_states() if states is None else states
    bus.can_timeouts.return_value = dict.fromkeys(MOTOR_NAMES, 8000)
    return bus


def attach_bus(arm: YamFollower, bus):
    arm.bus = arm.controller._bus = bus
    return bus


def mock_hardware(arm: YamFollower, monkeypatch):
    """Mock the arm's bus, and replace its control loop with one that returns at once."""
    attach_bus(arm, mock_bus())
    monkeypatch.setattr(arm.controller, "_run", lambda: None)


def mark_running(controller: mit_control.MitController):
    """Make the controller report active, with a finished thread that stop() can join."""
    controller._thread = threading.Thread(target=lambda: None)
    controller._thread.start()


def make_ready(arm: YamFollower):
    """Mark the arm connected, with its controller active at the zero position."""
    arm._connected = True
    zeros = np.zeros(7)
    arm.controller._set_initial_state(mit_control.JointState(position=zeros, velocity=zeros, torque=zeros))
    mark_running(arm.controller)
