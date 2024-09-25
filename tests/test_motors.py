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

from lerobot import available_motors
from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from tests.utils import TEST_MOTOR_TYPES, require_motor

DYNAMIXEL_PORT = "/dev/tty.usbmodem575E0032081"
DYNAMIXEL_MOTORS = {
    "shoulder_pan": [1, "xl430-w250"],
    "shoulder_lift": [2, "xl430-w250"],
    "elbow_flex": [3, "xl330-m288"],
    "wrist_flex": [4, "xl330-m288"],
    "wrist_roll": [5, "xl330-m288"],
    "gripper": [6, "xl330-m288"],
}


def make_motors_bus(motor_type: str, **kwargs) -> MotorsBus:
    if motor_type == "dynamixel":
        from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

        port = kwargs.pop("port", DYNAMIXEL_PORT)
        motors = kwargs.pop("motors", DYNAMIXEL_MOTORS)
        return DynamixelMotorsBus(port, motors, **kwargs)

    else:
        raise ValueError(f"The motor type '{motor_type}' is not valid.")


# TODO(rcadene): implement mocked version of this test
@pytest.mark.parametrize("motor_type, mock", [(m, False) for m in available_motors])
@require_motor
def test_find_port(request, motor_type, mock):
    from lerobot.common.robot_devices.motors.dynamixel import find_port

    find_port()


# TODO(rcadene): implement mocked version of this test
@pytest.mark.parametrize("motor_type, mock", [(m, False) for m in available_motors])
@require_motor
def test_configure_motors_all_ids_1(request, motor_type, mock):
    input("Are you sure you want to re-configure the motors? Press enter to continue...")
    # This test expect the configuration was already correct.
    motors_bus = make_motors_bus(motor_type)
    motors_bus.connect()
    motors_bus.write("Baud_Rate", [0] * len(motors_bus.motors))
    motors_bus.set_bus_baudrate(9_600)
    motors_bus.write("ID", [1] * len(motors_bus.motors))
    del motors_bus

    # Test configure
    motors_bus = make_motors_bus(motor_type)
    motors_bus.connect()
    assert motors_bus.are_motors_configured()
    del motors_bus


@pytest.mark.parametrize("motor_type, mock", TEST_MOTOR_TYPES)
@require_motor
def test_motors_bus(request, motor_type, mock):
    motors_bus = make_motors_bus(motor_type)

    # Test reading and writting before connecting raises an error
    with pytest.raises(RobotDeviceNotConnectedError):
        motors_bus.read("Torque_Enable")
    with pytest.raises(RobotDeviceNotConnectedError):
        motors_bus.write("Torque_Enable", 1)
    with pytest.raises(RobotDeviceNotConnectedError):
        motors_bus.disconnect()

    # Test deleting the object without connecting first
    del motors_bus

    # Test connecting
    motors_bus = make_motors_bus(motor_type)
    motors_bus.connect()

    # Test connecting twice raises an error
    with pytest.raises(RobotDeviceAlreadyConnectedError):
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
