# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
Tests for physical motors and their mocked versions.
If the physical motors are not connected to the computer, or not working,
the test will be skipped.

Example of running a specific test:
```bash
pytest -sx tests/test_motors.py::test_find_port
pytest -sx tests/test_motors.py::test_motors_bus
```

Example of running test on real dynamixel motors connected to the computer:
```bash
pytest -sx 'tests/test_motors.py::test_motors_bus[dynamixel-False]'
```

Example of running test on a mocked version of dynamixel motors:
```bash
pytest -sx 'tests/test_motors.py::test_motors_bus[dynamixel-True]'
```
"""

# TODO(rcadene): measure fps in nightly?
# TODO(rcadene): test logs
# TODO(rcadene): test calibration
# TODO(rcadene): add compatibility with other motors bus

import time

import numpy as np
import pytest

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.scripts.find_motors_bus_port import find_port
from tests.utils import TEST_MOTOR_TYPES, make_motors_bus, require_motor


@pytest.mark.parametrize("motor_type, mock", TEST_MOTOR_TYPES)
@require_motor
def test_find_port(request, motor_type, mock):
    if mock:
        request.getfixturevalue("patch_builtins_input")
        with pytest.raises(OSError):
            find_port()
    else:
        find_port()


@pytest.mark.parametrize("motor_type, mock", TEST_MOTOR_TYPES)
@require_motor
def test_motors_bus(request, motor_type, mock):
    if mock:
        request.getfixturevalue("patch_builtins_input")

    motors_bus = make_motors_bus(motor_type, mock=mock)

    # Test reading and writing before connecting raises an error
    with pytest.raises(DeviceNotConnectedError):
        motors_bus.read("Torque_Enable")
    with pytest.raises(DeviceNotConnectedError):
        motors_bus.write("Torque_Enable", 1)
    with pytest.raises(DeviceNotConnectedError):
        motors_bus.disconnect()

    # Test deleting the object without connecting first
    del motors_bus

    # Test connecting
    motors_bus = make_motors_bus(motor_type, mock=mock)
    motors_bus.connect()

    # Test connecting twice raises an error
    with pytest.raises(DeviceAlreadyConnectedError):
        motors_bus.connect()

    # Test disabling torque and reading torque on all motors
    motors_bus.write("Torque_Enable", 0)
    values = motors_bus.read("Torque_Enable")
    assert isinstance(values, np.ndarray)
    assert len(values) == len(motors_bus.motors)
    assert (values == 0).all()

    # Test writing torque on a specific motor
    motors_bus.write("Torque_Enable", 1, "gripper")

    # Test reading torque from this specific motor. It is now 1
    values = motors_bus.read("Torque_Enable", "gripper")
    assert len(values) == 1
    assert values[0] == 1

    # Test reading torque from all motors. It is 1 for the specific motor,
    # and 0 on the others.
    values = motors_bus.read("Torque_Enable")
    gripper_index = motors_bus.motor_names.index("gripper")
    assert values[gripper_index] == 1
    assert values.sum() == 1  # gripper is the only motor to have torque 1

    # Test writing torque on all motors and it is 1 for all.
    motors_bus.write("Torque_Enable", 1)
    values = motors_bus.read("Torque_Enable")
    assert (values == 1).all()

    # Test ordering the motors to move slightly (+1 value among 4096) and this move
    # can be executed and seen by the motor position sensor
    values = motors_bus.read("Present_Position")
    motors_bus.write("Goal_Position", values + 1)
    # Give time for the motors to move to the goal position
    time.sleep(1)
    new_values = motors_bus.read("Present_Position")
    assert (new_values == values).all()
