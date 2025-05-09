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

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_opencv.py::test_connect
# ```

import os
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.common.cameras.configs import Cv2Rotation
from lerobot.common.cameras.intel import RealSenseCamera, RealSenseCameraConfig
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

TEST_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "cameras")
BAG_FILE_PATH = os.path.join(TEST_ARTIFACTS_DIR, "test.bag")


if not os.path.exists(BAG_FILE_PATH):
    print(f"Warning: Bag file not found at {BAG_FILE_PATH}. Some tests might fail or be skipped.")


def mock_rs_config_enable_device_from_file(rs_config_instance, sn):
    if not os.path.exists(BAG_FILE_PATH):
        raise FileNotFoundError(f"Test bag file not found: {BAG_FILE_PATH}")
    return rs_config_instance.enable_device_from_file(BAG_FILE_PATH, repeat_playback=True)


def mock_rs_config_enable_device_bad_file(rs_config_instance, sn):
    return rs_config_instance.enable_device_from_file("non_existent_file.bag", repeat_playback=True)


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_connect(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42)
    camera = RealSenseCamera(config)

    camera.connect()
    assert camera.is_connected


def test_base_class_implementation():
    config = RealSenseCameraConfig(serial_number=42)
    _ = RealSenseCamera(config)


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_connect_already_connected(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42)
    camera = RealSenseCamera(config)
    camera.connect()

    with pytest.raises(DeviceAlreadyConnectedError):
        camera.connect()


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_bad_file)
def test_connect_invalid_camera_path(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42)
    camera = RealSenseCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect()


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_invalid_width_connect(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42, width=99999, height=480, fps=30)
    camera = RealSenseCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect()


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_read(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42, width=640, height=480, fps=30)
    camera = RealSenseCamera(config)
    camera.connect()

    img = camera.read()
    assert isinstance(img, np.ndarray)


def test_read_before_connect():
    config = RealSenseCameraConfig(serial_number=42)
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_disconnect(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42)
    camera = RealSenseCamera(config)
    camera.connect()

    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    config = RealSenseCameraConfig(serial_number=42)
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_async_read(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42, width=640, height=480, fps=30)
    camera = RealSenseCamera(config)
    camera.connect()

    img = camera.async_read()

    assert camera.thread is not None
    assert camera.thread.is_alive()
    assert isinstance(img, np.ndarray)
    camera.disconnect()  # To stop/join the thread. Otherwise get warnings when the test ends


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_async_read_timeout(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42, width=640, height=480, fps=30)
    camera = RealSenseCamera(config)
    camera.connect()

    with pytest.raises(TimeoutError):
        camera.async_read(timeout_ms=0)

    camera.disconnect()


def test_async_read_before_connect():
    config = RealSenseCameraConfig(serial_number=42)
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


@pytest.mark.parametrize(
    "rotation",
    [
        Cv2Rotation.NO_ROTATION,
        Cv2Rotation.ROTATE_90,
        Cv2Rotation.ROTATE_180,
        Cv2Rotation.ROTATE_270,
    ],
)
@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_all_rotations(mock_enable_device, rotation):
    config = RealSenseCameraConfig(serial_number=42, rotation=rotation)
    camera = RealSenseCamera(config)
    camera.connect()

    img = camera.read()
    assert isinstance(img, np.ndarray)

    if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
        assert camera.width == 480
        assert camera.height == 640
        assert img.shape[:2] == (640, 480)
    else:
        assert camera.width == 640
        assert camera.height == 480
        assert img.shape[:2] == (480, 640)
