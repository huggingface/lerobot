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

import sys
from pathlib import Path

import cv2
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

def _check_opencv_backends_available():
    """Check if OpenCV has working backends for image files."""
    try:
        if not DEFAULT_PNG_FILE_PATH.exists():
            return False

        # Check if FFmpeg backend works
        cap = cv2.VideoCapture(str(DEFAULT_PNG_FILE_PATH), cv2.CAP_FFMPEG)
        ffmpeg_works = cap.isOpened()
        cap.release()

        if ffmpeg_works:
            return True

        # Try DirectShow backend (Windows)
        cap = cv2.VideoCapture(str(DEFAULT_PNG_FILE_PATH), cv2.CAP_DSHOW)
        dshow_works = cap.isOpened()
        cap.release()

        return dshow_works
    except Exception:
        return False


def _check_opencv_image_support():
    """Check if OpenCV can handle image files on this platform."""
    if sys.platform == "win32":
        # On Windows, VideoCapture with DirectShow backend doesn't support image files
        # This is a known limitation - DirectShow can't open static image files as video sources
        return False
    else:
        # On Linux/macOS, assume it works (usually does)
        return True


# Reusable skip conditions
SKIP_NO_OPENCV_IMAGE_SUPPORT = pytest.mark.skipif(
    not _check_opencv_image_support(),
    reason="OpenCV cannot open image files as video sources. "
           "This is common on Windows without proper codecs. "
           "To fix: Install FFmpeg or use a different OpenCV build with image support. "
           "Run: conda install ffmpeg or pip install opencv-python-headless"
)

SKIP_NO_OPENCV_BACKENDS = pytest.mark.skipif(
    not _check_opencv_backends_available(),
    reason="OpenCV backends (FFmpeg/DirectShow) not available for image files. "
           "Install FFmpeg or use OpenCV with proper codec support."
)


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    config = OpenCVCameraConfig(index_or_path=0)

    _ = OpenCVCamera(config)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)

    camera.connect(warmup=False)

    assert camera.is_connected


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_connect_already_connected():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)
    camera.connect(warmup=False)

    with pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)


def test_connect_invalid_camera_path():
    config = OpenCVCameraConfig(index_or_path="nonexistent/camera.png")
    camera = OpenCVCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_invalid_width_connect():
    config = OpenCVCameraConfig(
        index_or_path=DEFAULT_PNG_FILE_PATH,
        width=99999,  # Invalid width to trigger error
        height=480,
    )
    camera = OpenCVCamera(config)

    with pytest.raises(RuntimeError):
        camera.connect(warmup=False)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
def test_read(index_or_path):
    config = OpenCVCameraConfig(index_or_path=index_or_path)
    camera = OpenCVCamera(config)
    camera.connect(warmup=False)

    img = camera.read()

    assert isinstance(img, np.ndarray)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_read_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_disconnect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_disconnect_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.disconnect()


@SKIP_NO_OPENCV_IMAGE_SUPPORT
@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
def test_async_read(index_or_path):
    config = OpenCVCameraConfig(index_or_path=index_or_path)
    camera = OpenCVCamera(config)
    camera.connect(warmup=False)

    try:
        img = camera.async_read()

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(img, np.ndarray)
    finally:
        if camera.is_connected:
            camera.disconnect()  # To stop/join the thread. Otherwise get warnings when the test ends


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_async_read_timeout():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)
    camera.connect(warmup=False)

    try:
        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=0)
    finally:
        if camera.is_connected:
            camera.disconnect()


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_async_read_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


@SKIP_NO_OPENCV_IMAGE_SUPPORT
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


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_fourcc_with_camera():
    """Test FourCC functionality with actual camera connection."""
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc="MJPG")
    camera = OpenCVCamera(config)

    # Connect should work with MJPG specified
    camera.connect(warmup=False)
    assert camera.is_connected

    # Read should work normally
    img = camera.read()
    assert isinstance(img, np.ndarray)

    camera.disconnect()


@SKIP_NO_OPENCV_IMAGE_SUPPORT
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

    config = OpenCVCameraConfig(index_or_path=index_or_path, rotation=rotation)
    camera = OpenCVCamera(config)
    camera.connect(warmup=False)

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
