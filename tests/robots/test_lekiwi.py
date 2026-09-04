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

from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig

_MODULE = "lerobot.robots.lekiwi.lekiwi"


@pytest.fixture
def lekiwi(tmp_path):
    bus_mock = MagicMock(name="FeetechBusMock")
    bus_mock.is_connected = True

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        bus_mock.sync_read.return_value = dict.fromkeys(bus_mock.motors, 0.0)
        return bus_mock

    with patch(f"{_MODULE}.FeetechMotorsBus", side_effect=_bus_side_effect):
        yield LeKiwi(LeKiwiConfig(port="/dev/null", calibration_dir=tmp_path, cameras={}))


def _action(lekiwi, arm_target: float = 100.0) -> dict[str, float]:
    return {
        **{f"{motor}.pos": arm_target for motor in lekiwi.arm_motors},
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }


def test_send_action_clamps_scalar_relative_target(lekiwi):
    lekiwi.config.max_relative_target = 10.0
    lekiwi.config.num_read_retries = 7

    returned = lekiwi.send_action(_action(lekiwi))

    assert {key: returned[key] for key in returned if key.endswith(".pos")} == {
        f"{motor}.pos": 10.0 for motor in lekiwi.arm_motors
    }
    lekiwi.bus.sync_read.assert_called_once_with("Present_Position", lekiwi.arm_motors, num_retry=7)
    lekiwi.bus.sync_write.assert_any_call("Goal_Position", dict.fromkeys(lekiwi.arm_motors, 10.0))


def test_send_action_clamps_per_motor_relative_target(lekiwi):
    relative_limits = {motor: float(index) for index, motor in enumerate(lekiwi.arm_motors, 1)}
    lekiwi.config.max_relative_target = relative_limits

    returned = lekiwi.send_action(_action(lekiwi))

    assert {key: returned[key] for key in returned if key.endswith(".pos")} == {
        f"{motor}.pos": limit for motor, limit in relative_limits.items()
    }
    lekiwi.bus.sync_write.assert_any_call("Goal_Position", relative_limits)
