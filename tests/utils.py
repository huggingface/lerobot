#!/usr/bin/env python

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
import json
import os
import platform
from functools import wraps
from pathlib import Path

import pytest
import torch

from lerobot import available_cameras, available_motors, available_robots
from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.cameras.utils import make_camera as make_camera_device
from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.motors.utils import make_motors_bus as make_motors_bus_device
from lerobot.common.utils.import_utils import is_package_available

DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"

TEST_ROBOT_TYPES = []
for robot_type in available_robots:
    TEST_ROBOT_TYPES += [(robot_type, True), (robot_type, False)]

TEST_CAMERA_TYPES = []
for camera_type in available_cameras:
    TEST_CAMERA_TYPES += [(camera_type, True), (camera_type, False)]

TEST_MOTOR_TYPES = []
for motor_type in available_motors:
    TEST_MOTOR_TYPES += [(motor_type, True), (motor_type, False)]

# Camera indices used for connecting physical cameras
OPENCV_CAMERA_INDEX = int(os.environ.get("LEROBOT_TEST_OPENCV_CAMERA_INDEX", 0))
INTELREALSENSE_SERIAL_NUMBER = int(os.environ.get("LEROBOT_TEST_INTELREALSENSE_SERIAL_NUMBER", 128422271614))

DYNAMIXEL_PORT = os.environ.get("LEROBOT_TEST_DYNAMIXEL_PORT", "/dev/tty.usbmodem575E0032081")
DYNAMIXEL_MOTORS = {
    "shoulder_pan": [1, "xl430-w250"],
    "shoulder_lift": [2, "xl430-w250"],
    "elbow_flex": [3, "xl330-m288"],
    "wrist_flex": [4, "xl330-m288"],
    "wrist_roll": [5, "xl330-m288"],
    "gripper": [6, "xl330-m288"],
}

FEETECH_PORT = os.environ.get("LEROBOT_TEST_FEETECH_PORT", "/dev/tty.usbmodem585A0080971")
FEETECH_MOTORS = {
    "shoulder_pan": [1, "sts3215"],
    "shoulder_lift": [2, "sts3215"],
    "elbow_flex": [3, "sts3215"],
    "wrist_flex": [4, "sts3215"],
    "wrist_roll": [5, "sts3215"],
    "gripper": [6, "sts3215"],
}


def require_x86_64_kernel(func):
    """
    Decorator that skips the test if plateform device is not an x86_64 cpu.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if platform.machine() != "x86_64":
            pytest.skip("requires x86_64 plateform")
        return func(*args, **kwargs)

    return wrapper


def require_cpu(func):
    """
    Decorator that skips the test if device is not cpu.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if DEVICE != "cpu":
            pytest.skip("requires cpu")
        return func(*args, **kwargs)

    return wrapper


def require_cuda(func):
    """
    Decorator that skips the test if cuda is not available.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("requires cuda")
        return func(*args, **kwargs)

    return wrapper


def require_env(func):
    """
    Decorator that skips the test if the required environment package is not installed.
    As it need 'env_name' in args, it also checks whether it is provided as an argument.
    If 'env_name' is None, this check is skipped.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Determine if 'env_name' is provided and extract its value
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        if "env_name" in arg_names:
            # Get the index of 'env_name' and retrieve the value from args
            index = arg_names.index("env_name")
            env_name = args[index] if len(args) > index else kwargs.get("env_name")
        else:
            raise ValueError("Function does not have 'env_name' as an argument.")

        # Perform the package check
        package_name = f"gym_{env_name}"
        if env_name is not None and not is_package_available(package_name):
            pytest.skip(f"gym-{env_name} not installed")

        return func(*args, **kwargs)

    return wrapper


def require_package_arg(func):
    """
    Decorator that skips the test if the required package is not installed.
    This is similar to `require_env` but more general in that it can check any package (not just environments).
    As it need 'required_packages' in args, it also checks whether it is provided as an argument.
    If 'required_packages' is None, this check is skipped.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Determine if 'required_packages' is provided and extract its value
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        if "required_packages" in arg_names:
            # Get the index of 'required_packages' and retrieve the value from args
            index = arg_names.index("required_packages")
            required_packages = args[index] if len(args) > index else kwargs.get("required_packages")
        else:
            raise ValueError("Function does not have 'required_packages' as an argument.")

        if required_packages is None:
            return func(*args, **kwargs)

        # Perform the package check
        for package in required_packages:
            if not is_package_available(package):
                pytest.skip(f"{package} not installed")

        return func(*args, **kwargs)

    return wrapper


def require_package(package_name):
    """
    Decorator that skips the test if the specified package is not installed.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_package_available(package_name):
                pytest.skip(f"{package_name} not installed")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_robot(func):
    """
    Decorator that skips the test if a robot is not available

    The decorated function must have two arguments `request` and `robot_type`.

    Example of usage:
    ```python
    @pytest.mark.parametrize(
        "robot_type", ["koch", "aloha"]
    )
    @require_robot
    def test_require_robot(request, robot_type):
        pass
    ```
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Access the pytest request context to get the is_robot_available fixture
        request = kwargs.get("request")
        robot_type = kwargs.get("robot_type")
        mock = kwargs.get("mock")

        if robot_type is None:
            raise ValueError("The 'robot_type' must be an argument of the test function.")
        if request is None:
            raise ValueError("The 'request' fixture must be an argument of the test function.")
        if mock is None:
            raise ValueError("The 'mock' variable must be an argument of the test function.")

        # Run test with a real robot. Skip test if robot connection fails.
        if not mock and not request.getfixturevalue("is_robot_available"):
            pytest.skip(f"A {robot_type} robot is not available.")

        return func(*args, **kwargs)

    return wrapper


def require_camera(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Access the pytest request context to get the is_camera_available fixture
        request = kwargs.get("request")
        camera_type = kwargs.get("camera_type")
        mock = kwargs.get("mock")

        if request is None:
            raise ValueError("The 'request' fixture must be an argument of the test function.")
        if camera_type is None:
            raise ValueError("The 'camera_type' must be an argument of the test function.")
        if mock is None:
            raise ValueError("The 'mock' variable must be an argument of the test function.")

        if not mock and not request.getfixturevalue("is_camera_available"):
            pytest.skip(f"A {camera_type} camera is not available.")

        return func(*args, **kwargs)

    return wrapper


def require_motor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Access the pytest request context to get the is_motor_available fixture
        request = kwargs.get("request")
        motor_type = kwargs.get("motor_type")
        mock = kwargs.get("mock")

        if request is None:
            raise ValueError("The 'request' fixture must be an argument of the test function.")
        if motor_type is None:
            raise ValueError("The 'motor_type' must be an argument of the test function.")
        if mock is None:
            raise ValueError("The 'mock' variable must be an argument of the test function.")

        if not mock and not request.getfixturevalue("is_motor_available"):
            pytest.skip(f"A {motor_type} motor is not available.")

        return func(*args, **kwargs)

    return wrapper


def mock_calibration_dir(calibration_dir):
    # TODO(rcadene): remove this hack
    # calibration file produced with Moss v1, but works with Koch, Koch bimanual and SO-100
    example_calib = {
        "homing_offset": [-1416, -845, 2130, 2872, 1950, -2211],
        "drive_mode": [0, 0, 1, 1, 1, 0],
        "start_pos": [1442, 843, 2166, 2849, 1988, 1835],
        "end_pos": [2440, 1869, -1106, -1848, -926, 3235],
        "calib_mode": ["DEGREE", "DEGREE", "DEGREE", "DEGREE", "DEGREE", "LINEAR"],
        "motor_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    }
    Path(str(calibration_dir)).mkdir(parents=True, exist_ok=True)
    with open(calibration_dir / "main_follower.json", "w") as f:
        json.dump(example_calib, f)
    with open(calibration_dir / "main_leader.json", "w") as f:
        json.dump(example_calib, f)
    with open(calibration_dir / "left_follower.json", "w") as f:
        json.dump(example_calib, f)
    with open(calibration_dir / "left_leader.json", "w") as f:
        json.dump(example_calib, f)
    with open(calibration_dir / "right_follower.json", "w") as f:
        json.dump(example_calib, f)
    with open(calibration_dir / "right_leader.json", "w") as f:
        json.dump(example_calib, f)


# TODO(rcadene, aliberts): remove this dark pattern that overrides
def make_camera(camera_type: str, **kwargs) -> Camera:
    if camera_type == "opencv":
        camera_index = kwargs.pop("camera_index", OPENCV_CAMERA_INDEX)
        return make_camera_device(camera_type, camera_index=camera_index, **kwargs)

    elif camera_type == "intelrealsense":
        serial_number = kwargs.pop("serial_number", INTELREALSENSE_SERIAL_NUMBER)
        return make_camera_device(camera_type, serial_number=serial_number, **kwargs)
    else:
        raise ValueError(f"The camera type '{camera_type}' is not valid.")


# TODO(rcadene, aliberts): remove this dark pattern that overrides
def make_motors_bus(motor_type: str, **kwargs) -> MotorsBus:
    if motor_type == "dynamixel":
        port = kwargs.pop("port", DYNAMIXEL_PORT)
        motors = kwargs.pop("motors", DYNAMIXEL_MOTORS)
        return make_motors_bus_device(motor_type, port=port, motors=motors, **kwargs)

    elif motor_type == "feetech":
        port = kwargs.pop("port", FEETECH_PORT)
        motors = kwargs.pop("motors", FEETECH_MOTORS)
        return make_motors_bus_device(motor_type, port=port, motors=motors, **kwargs)

    else:
        raise ValueError(f"The motor type '{motor_type}' is not valid.")
