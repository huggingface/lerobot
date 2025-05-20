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

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.common.cameras.configs import Cv2Rotation
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

try:
    from lerobot.common.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("pyrealsense2 not available", allow_module_level=True)

TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"
BAG_FILE_PATH = TEST_ARTIFACTS_DIR / "test_rs.bag"

# NOTE(Steven): Missing tests for depth
# NOTE(Steven): Takes 20sec, the patch being the biggest bottleneck
# NOTE(Steven): more tests + assertions?


def mock_rs_config_enable_device_from_file(rs_config_instance, sn):
    return rs_config_instance.enable_device_from_file(str(BAG_FILE_PATH), repeat_playback=True)


def mock_rs_config_enable_device_bad_file(rs_config_instance, sn):
    return rs_config_instance.enable_device_from_file("non_existent_file.bag", repeat_playback=True)


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    config = RealSenseCameraConfig(serial_number=42)
    _ = RealSenseCamera(config)


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_connect(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42)
    camera = RealSenseCamera(config)

    camera.connect(warmup=False)
    assert camera.is_connected


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_connect_already_connected(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42)
    camera = RealSenseCamera(config)
    camera.connect(warmup=False)

    with pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_bad_file)
def test_connect_invalid_camera_path(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42)
    camera = RealSenseCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_invalid_width_connect(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42, width=99999, height=480, fps=30)
    camera = RealSenseCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_read(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42, width=640, height=480, fps=30)
    camera = RealSenseCamera(config)
    camera.connect(warmup=False)

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
    camera.connect(warmup=False)

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
    camera.connect(warmup=False)

    try:
        img = camera.async_read()

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(img, np.ndarray)
    finally:
        if camera.is_connected:
            camera.disconnect()  # To stop/join the thread. Otherwise get warnings when the test ends


@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_async_read_timeout(mock_enable_device):
    config = RealSenseCameraConfig(serial_number=42, width=640, height=480, fps=30)
    camera = RealSenseCamera(config)
    camera.connect(warmup=False)

    try:
        with pytest.raises(TimeoutError):
            camera.async_read(
                timeout_ms=0
            )  # NOTE(Steven): This is flaky as sdometimes we actually get a frame
    finally:
        if camera.is_connected:
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
    ids=["no_rot", "rot90", "rot180", "rot270"],
)
@patch("pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file)
def test_rotation(mock_enable_device, rotation):
    config = RealSenseCameraConfig(serial_number=42, rotation=rotation)
    camera = RealSenseCamera(config)
    camera.connect(warmup=False)

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
