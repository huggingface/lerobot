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

import numpy as np
import pytest

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# NOTE(Steven): more tests + assertions?
TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"
DEFAULT_PNG_FILE_PATH = TEST_ARTIFACTS_DIR / "image_160x120.png"
TEST_IMAGE_SIZES = ["128x128", "160x120", "320x180", "480x270"]
TEST_IMAGE_PATHS = [TEST_ARTIFACTS_DIR / f"image_{size}.png" for size in TEST_IMAGE_SIZES]


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    config = OpenCVCameraConfig(index_or_path=0)

    _ = OpenCVCamera(config)


def test_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        assert camera.is_connected


def test_connect_already_connected():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera, pytest.raises(DeviceAlreadyConnectedError):
        camera.connect()


def test_connect_invalid_camera_path():
    config = OpenCVCameraConfig(index_or_path="nonexistent/camera.png")

    camera = OpenCVCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


def test_invalid_width_connect():
    config = OpenCVCameraConfig(
        index_or_path=DEFAULT_PNG_FILE_PATH,
        width=99999,  # Invalid width to trigger error
        height=480,
    )

    camera = OpenCVCamera(config)
    with pytest.raises(RuntimeError):
        camera.connect(warmup=False)


@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
def test_read(index_or_path):
    config = OpenCVCameraConfig(index_or_path=index_or_path, warmup_s=0)

    with OpenCVCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)


def test_read_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)

    camera = OpenCVCamera(config)
    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


def test_disconnect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.disconnect()


@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
def test_async_read(index_or_path):
    config = OpenCVCameraConfig(index_or_path=index_or_path, warmup_s=0)

    with OpenCVCamera(config) as camera:
        img = camera.async_read()

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(img, np.ndarray)


def test_async_read_timeout():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera, pytest.raises(TimeoutError):
        camera.async_read(timeout_ms=0)


def test_async_read_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


def test_read_latest():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        # ensure at least one fresh frame is captured
        frame = camera.read()
        latest, ts = camera.read_latest()

        assert isinstance(latest, np.ndarray)
        assert isinstance(ts, float)
        assert latest.shape == frame.shape


def test_read_latest_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)

    camera = OpenCVCamera(config)
    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read_latest()


def test_read_latest_high_frequency():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        # prime to ensure frames are available
        ref = camera.read()

        timestamps = []
        for _ in range(20):
            latest, ts = camera.read_latest()
            assert isinstance(latest, np.ndarray)
            assert isinstance(ts, float)
            assert latest.shape == ref.shape
            timestamps.append(ts)

        assert max(timestamps) == timestamps[0]


def test_fourcc_configuration():
    """Test FourCC configuration validation and application."""

    # Test MJPG specifically (main use case)
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc="MJPG")
    camera = OpenCVCamera(config)
    assert camera.config.fourcc == "MJPG"

    # Test a few other common formats
    valid_fourcc_codes = ["YUYV", "YUY2", "RGB3"]

    for fourcc in valid_fourcc_codes:
        config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc=fourcc)
        camera = OpenCVCamera(config)
        assert camera.config.fourcc == fourcc

    # Test invalid FOURCC codes
    invalid_fourcc_codes = ["ABC", "ABCDE", ""]

    for fourcc in invalid_fourcc_codes:
        with pytest.raises(ValueError):
            OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc=fourcc)


def test_fourcc_with_camera():
    """Test FourCC functionality with actual camera connection."""
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc="MJPG", warmup_s=0)

    # Connect should work with MJPG specified
    with OpenCVCamera(config) as camera:
        assert camera.is_connected

        # Read should work normally
        img = camera.read()
        assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
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
def test_rotation(rotation, index_or_path):
    filename = Path(index_or_path).name
    dimensions = filename.split("_")[-1].split(".")[0]  # Assumes filenames format (_wxh.png)
    original_width, original_height = map(int, dimensions.split("x"))

    config = OpenCVCameraConfig(index_or_path=index_or_path, rotation=rotation, warmup_s=0)
    with OpenCVCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)

        if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
            assert camera.width == original_height
            assert camera.height == original_width
            assert img.shape[:2] == (original_width, original_height)
        else:
            assert camera.width == original_width
            assert camera.height == original_height
            assert img.shape[:2] == (original_height, original_width)
