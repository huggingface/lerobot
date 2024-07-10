import time

import numpy as np
import pytest

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from tests.utils import require_koch


@require_koch
def test_motors_bus(request):
    # TODO(rcadene): measure fps in nightly?
    # TODO(rcadene): test logs
    # TODO(rcadene): test calibration
    # TODO(rcadene): add compatibility with other motors bus
    from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

    # Test instantiating a common motors structure.
    # Here the one from Alexander Koch follower arm.
    port = "/dev/tty.usbmodem575E0032081"
    motors = {
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    }
    motors_bus = DynamixelMotorsBus(port, motors)

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
    motors_bus = DynamixelMotorsBus(port, motors)
    motors_bus.connect()

    # Test connecting twice raises an error
    with pytest.raises(RobotDeviceAlreadyConnectedError):
        motors_bus.connect()

    # Test disabling torque and reading torque on all motors
    motors_bus.write("Torque_Enable", 0)
    values = motors_bus.read("Torque_Enable")
    assert isinstance(values, np.ndarray)
    assert len(values) == len(motors)
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


@require_koch
def test_find_port(request):
    from lerobot.common.robot_devices.motors.dynamixel import find_port

    find_port()
