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
import os
import platform
import traceback
from functools import wraps

import pytest
import torch

from lerobot import available_cameras, available_motors, available_robots
from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.robots.factory import make_robot as make_robot_from_cfg
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.import_utils import is_package_available
from lerobot.common.utils.utils import init_hydra_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pass this as the first argument to init_hydra_config.
DEFAULT_CONFIG_PATH = "lerobot/configs/default.yaml"

ROBOT_CONFIG_PATH_TEMPLATE = "lerobot/configs/robot/{robot}.yaml"

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
INTELREALSENSE_CAMERA_INDEX = int(os.environ.get("LEROBOT_TEST_INTELREALSENSE_CAMERA_INDEX", 128422271614))

DYNAMIXEL_PORT = "/dev/tty.usbmodem575E0032081"
DYNAMIXEL_MOTORS = {
    "shoulder_pan": [1, "xl430-w250"],
    "shoulder_lift": [2, "xl430-w250"],
    "elbow_flex": [3, "xl330-m288"],
    "wrist_flex": [4, "xl330-m288"],
    "wrist_roll": [5, "xl330-m288"],
    "gripper": [6, "xl330-m288"],
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
        # Access the pytest request context to get the mockeypatch fixture
        request = kwargs.get("request")
        robot_type = kwargs.get("robot_type")

        if robot_type is None:
            raise ValueError("The 'robot_type' must be an argument of the test function.")
        if request is None:
            raise ValueError("The 'request' fixture must be an argument of the test function.")

        # Run test with a real robot. Skip test if robot connection fails.
        if not request.getfixturevalue("is_robot_available"):
            pytest.skip(f"A {robot_type} robot is not available.")

        return func(*args, **kwargs)

    return wrapper


def require_camera(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        request = kwargs.get("request")
        camera_type = kwargs.get("camera_type")

        if request is None:
            raise ValueError("The 'request' fixture must be an argument of the test function.")
        if camera_type is None:
            raise ValueError("The 'camera_type' must be an argument of the test function.")

        if not request.getfixturevalue("is_camera_available"):
            pytest.skip(f"A {camera_type} camera is not available.")

        return func(*args, **kwargs)

    return wrapper


def require_motor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Access the pytest request context to get the mockeypatch fixture
        request = kwargs.get("request")
        motor_type = kwargs.get("motor_type")

        if request is None:
            raise ValueError("The 'request' fixture must be an argument of the test function.")
        if motor_type is None:
            raise ValueError("The 'motor_type' must be an argument of the test function.")

        if not request.getfixturevalue("is_motor_available"):
            pytest.skip(f"A {motor_type} motor is not available.")

        return func(*args, **kwargs)

    return wrapper


def require_mock_robot(func):
    """
    Decorator over test function to mock the robot

    The decorated function must have two arguments `monkeypatch` and `robot_type`.

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
        # Access the pytest request context to get the mockeypatch fixture
        monkeypatch = kwargs.get("monkeypatch")
        robot_type = kwargs.get("robot_type")

        if monkeypatch is None:
            raise ValueError("The 'monkeypatch' fixture must be an argument of the test function.")

        if robot_type is None:
            raise ValueError("The 'robot_type' must be an argument of the test function.")

        mock_robot(monkeypatch, robot_type)
        return func(*args, **kwargs)

    return wrapper


def require_mock_camera(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Access the pytest request context to get the mockeypatch fixture
        monkeypatch = kwargs.get("monkeypatch")
        camera_type = kwargs.get("camera_type")

        if monkeypatch is None:
            raise ValueError("The 'monkeypatch' fixture must be an argument of the test function.")
        if camera_type is None:
            raise ValueError("The 'camera_type' must be an argument of the test function.")

        mock_camera(monkeypatch, camera_type)
        return func(*args, **kwargs)

    return wrapper


def require_mock_motor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Access the pytest request context to get the mockeypatch fixture
        monkeypatch = kwargs.get("monkeypatch")
        motor_type = kwargs.get("motor_type")

        if monkeypatch is None:
            raise ValueError("The 'monkeypatch' fixture must be an argument of the test function.")
        if motor_type is None:
            raise ValueError("The 'motor_type' must be an argument of the test function.")

        mock_motor(monkeypatch, motor_type)
        return func(*args, **kwargs)

    return wrapper


def mock_robot(monkeypatch, robot_type):
    if robot_type not in available_robots:
        raise ValueError(
            f"The camera type '{robot_type}' is not valid. Expected one of these '{available_robots}"
        )

    if robot_type in ["koch", "koch_bimanual"]:
        mock_camera(monkeypatch, "opencv")
        mock_motor(monkeypatch, "dynamixel")
    elif robot_type == "aloha":
        mock_camera(monkeypatch, "intelrealsense")
        mock_motor(monkeypatch, "dynamixel")
    else:
        raise NotImplementedError("Implement mocking logic for new robot.")

    # To run calibration without user input
    mock_builtins_input(monkeypatch)


def mock_camera(monkeypatch, camera_type):
    if camera_type not in available_cameras:
        raise ValueError(
            f"The motor type '{camera_type}' is not valid. Expected one of these '{available_cameras}"
        )

    if camera_type == "opencv":
        try:
            import cv2

            from tests.mock_opencv import MockVideoCapture

            monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
        except ImportError:
            traceback.print_exc()
            pytest.skip("To avoid skipping tests mocking opencv cameras, run `pip install opencv-python`.")

    elif camera_type == "intelrealsense":
        try:
            import pyrealsense2 as rs

            from tests.mock_intelrealsense import (
                MockConfig,
                MockContext,
                MockFormat,
                MockPipeline,
                MockStream,
            )

            monkeypatch.setattr(rs, "config", MockConfig)
            monkeypatch.setattr(rs, "pipeline", MockPipeline)
            monkeypatch.setattr(rs, "stream", MockStream)
            monkeypatch.setattr(rs, "format", MockFormat)
            monkeypatch.setattr(rs, "context", MockContext)
        except ImportError:
            traceback.print_exc()
            pytest.skip(
                "To avoid skipping tests mocking intelrealsense cameras, run `pip install pyrealsense2`."
            )
    else:
        raise NotImplementedError("Implement mocking logic for new camera.")


def mock_motor(monkeypatch, motor_type):
    if motor_type not in available_motors:
        raise ValueError(
            f"The motor type '{motor_type}' is not valid. Expected one of these '{available_motors}"
        )

    if motor_type == "dynamixel":
        try:
            import dynamixel_sdk

            from tests.mock_dynamixel import (
                MockGroupSyncRead,
                MockGroupSyncWrite,
                MockPacketHandler,
                MockPortHandler,
                mock_convert_to_bytes,
            )

            monkeypatch.setattr(dynamixel_sdk, "GroupSyncRead", MockGroupSyncRead)
            monkeypatch.setattr(dynamixel_sdk, "GroupSyncWrite", MockGroupSyncWrite)
            monkeypatch.setattr(dynamixel_sdk, "PacketHandler", MockPacketHandler)
            monkeypatch.setattr(dynamixel_sdk, "PortHandler", MockPortHandler)

            # Import dynamixel AFTER mocking dynamixel_sdk to use mocked classes
            from lerobot.common.robot_devices.motors import dynamixel

            # TODO(rcadene): remove need to mock `convert_to_bytes` by implemented the inverse transform
            # `convert_bytes_to_value`
            monkeypatch.setattr(dynamixel, "convert_to_bytes", mock_convert_to_bytes)
        except ImportError:
            traceback.print_exc()
            pytest.skip("To avoid skipping tests mocking dynamixel motors, run `pip install dynamixel-sdk`.")
    else:
        raise NotImplementedError("Implement mocking logic for new motor.")


def mock_builtins_input(monkeypatch):
    def print_text(text=None):
        if text is not None:
            print(text)

    monkeypatch.setattr("builtins.input", print_text)


def make_robot(robot_type: str, overrides: list[str] | None = None) -> Robot:
    config_path = ROBOT_CONFIG_PATH_TEMPLATE.format(robot=robot_type)
    robot_cfg = init_hydra_config(config_path, overrides)
    robot = make_robot_from_cfg(robot_cfg)
    return robot


def make_camera(camera_type, **kwargs) -> Camera:
    if camera_type == "opencv":
        from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

        camera_index = kwargs.pop("camera_index", OPENCV_CAMERA_INDEX)
        return OpenCVCamera(camera_index, **kwargs)

    elif camera_type == "intelrealsense":
        from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera

        camera_index = kwargs.pop("camera_index", INTELREALSENSE_CAMERA_INDEX)
        return IntelRealSenseCamera(camera_index, **kwargs)

    else:
        raise ValueError(f"The camera type '{camera_type}' is not valid.")


def make_motors_bus(motor_type: str, **kwargs) -> MotorsBus:
    if motor_type == "dynamixel":
        from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

        port = kwargs.pop("port", DYNAMIXEL_PORT)
        motors = kwargs.pop("motors", DYNAMIXEL_MOTORS)
        return DynamixelMotorsBus(port, motors, **kwargs)

    else:
        raise ValueError(f"The motor type '{motor_type}' is not valid.")
