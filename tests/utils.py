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
import platform
import traceback
from functools import wraps

import pytest
import torch

from lerobot import available_cameras, available_motors, available_robots
from lerobot.common.utils.import_utils import is_package_available

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pass this as the first argument to init_hydra_config.
DEFAULT_CONFIG_PATH = "lerobot/configs/default.yaml"

ROBOT_CONFIG_PATH_TEMPLATE = "lerobot/configs/robot/{robot}.yaml"

TEST_ROBOT_TYPES = available_robots + [f"mocked_{robot_type}" for robot_type in available_robots]
TEST_CAMERA_TYPES = available_cameras + [f"mocked_{camera_type}" for camera_type in available_cameras]
TEST_MOTOR_TYPES = available_motors + [f"mocked_{motor_type}" for motor_type in available_motors]


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

        if robot_type is None:
            raise ValueError("The 'robot_type' must be an argument of the test function.")

        if robot_type not in TEST_ROBOT_TYPES:
            raise ValueError(
                f"The camera type '{robot_type}' is not valid. Expected one of these '{TEST_ROBOT_TYPES}"
            )

        if request is None:
            raise ValueError("The 'request' fixture must be an argument of the test function.")

        # Run test with a monkeypatched version of the robot devices.
        if robot_type.startswith("mocked_"):
            kwargs["robot_type"] = robot_type.replace("mocked_", "")
            mock_cameras(request)
            mock_motors(request)

        # Run test with a real robot. Skip test if robot connection fails.
        else:
            # `is_robot_available` is defined in `tests/conftest.py`
            if not request.getfixturevalue("is_robot_available"):
                pytest.skip(f"A {robot_type} robot is not available.")

        return func(*args, **kwargs)

    return wrapper


def require_camera(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Access the pytest request context to get the is_camera_available fixture
        request = kwargs.get("request")
        camera_type = kwargs.get("camera_type")

        if camera_type is None:
            raise ValueError("The 'camera_type' must be an argument of the test function.")

        if camera_type not in TEST_CAMERA_TYPES:
            raise ValueError(
                f"The camera type '{camera_type}' is not valid. Expected one of these '{TEST_CAMERA_TYPES}"
            )

        if request is None:
            raise ValueError("The 'request' fixture must be an argument of the test function.")

        # Run test with a monkeypatched version of the robot devices.
        if camera_type.startswith("mocked_"):
            kwargs["camera_type"] = camera_type.replace("mocked_", "")
            mock_cameras(request)

        # Run test with a real robot. Skip test if robot connection fails.
        else:
            # `is_camera_available` is defined in `tests/conftest.py`
            if not request.getfixturevalue("is_camera_available"):
                pytest.skip(f"A {camera_type} camera is not available.")

        return func(*args, **kwargs)

    return wrapper


def require_motor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Access the pytest request context to get the is_motor_available fixture
        request = kwargs.get("request")
        motor_type = kwargs.get("motor_type")

        if motor_type is None:
            raise ValueError("The 'motor_type' must be an argument of the test function.")

        if motor_type not in TEST_MOTOR_TYPES:
            raise ValueError(
                f"The motor type '{motor_type}' is not valid. Expected one of these '{TEST_MOTOR_TYPES}"
            )

        if request is None:
            raise ValueError("The 'request' fixture must be an argument of the test function.")

        # Run test with a monkeypatched version of the robot devices.
        if motor_type.startswith("mocked_"):
            kwargs["motor_type"] = motor_type.replace("mocked_", "")
            mock_motors(request)

        # Run test with a real robot. Skip test if robot connection fails.
        else:
            # `is_motor_available` is defined in `tests/conftest.py`
            if not request.getfixturevalue("is_motor_available"):
                pytest.skip(f"A {motor_type} motor is not available.")

        return func(*args, **kwargs)

    return wrapper


def mock_cameras(request):
    monkeypatch = request.getfixturevalue("monkeypatch")

    try:
        import cv2

        from tests.mock_opencv import MockVideoCapture

        monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    except ImportError:
        traceback.print_exc()

    try:
        import pyrealsense2 as rs

        from tests.mock_intelrealsense import MockConfig, MockFormat, MockPipeline, MockStream

        monkeypatch.setattr(rs, "config", MockConfig)
        monkeypatch.setattr(rs, "pipeline", MockPipeline)
        monkeypatch.setattr(rs, "stream", MockStream)
        monkeypatch.setattr(rs, "format", MockFormat)
    except ImportError:
        traceback.print_exc()


def mock_motors(request):
    monkeypatch = request.getfixturevalue("monkeypatch")

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
