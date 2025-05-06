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

import numpy as np
import pytest

from lerobot.common.cameras.configs import Cv2Rotation
from lerobot.common.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# NOTE(Steven): Patch get/set calls


def test_base_class_implementation():
    config = OpenCVCameraConfig(index_or_path=0)

    _ = OpenCVCamera(config)


def test_connect():
    config = OpenCVCameraConfig(index_or_path="tests/artifacts/cameras/fake_cam.png")
    camera = OpenCVCamera(config)

    camera.connect()

    assert camera.is_connected


def test_connect_already_connected():
    config = OpenCVCameraConfig(index_or_path="tests/artifacts/cameras/fake_cam.png")
    camera = OpenCVCamera(config)
    camera.connect()

    with pytest.raises(DeviceAlreadyConnectedError):
        camera.connect()


def test_connect_invalid_camera_path():
    config = OpenCVCameraConfig(index_or_path="nonexistent/camera.png")
    camera = OpenCVCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect()


def test_invalid_width_connect():
    config = OpenCVCameraConfig(
        index_or_path="tests/artifacts/cameras/fake_cam.png",
        width=99999,  # Invalid width to trigger error
        height=480,
    )
    camera = OpenCVCamera(config)

    with pytest.raises(RuntimeError):
        camera.connect()


def test_read():
    config = OpenCVCameraConfig(index_or_path="tests/artifacts/cameras/fake_cam.png")
    camera = OpenCVCamera(config)
    camera.connect()

    img = camera.read()

    assert isinstance(img, np.ndarray)


def test_read_before_connect():
    config = OpenCVCameraConfig(index_or_path="tests/artifacts/cameras/fake_cam.png")
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


def test_disconnect():
    config = OpenCVCameraConfig(index_or_path="tests/artifacts/cameras/fake_cam.png")
    camera = OpenCVCamera(config)
    camera.connect()

    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    config = OpenCVCameraConfig(index_or_path="tests/artifacts/cameras/fake_cam.png")
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.disconnect()


def test_async_read():
    config = OpenCVCameraConfig(index_or_path="tests/artifacts/cameras/fake_cam.png")
    camera = OpenCVCamera(config)
    camera.connect()

    img = camera.async_read()

    assert camera.thread is not None
    assert camera.thread.is_alive()
    assert isinstance(img, np.ndarray)
    camera.disconnect()  # To stop/join the thread. Otherwise get warnings when the test ends


def test_async_read_timeout():
    config = OpenCVCameraConfig(index_or_path="tests/artifacts/cameras/fake_cam.png")
    camera = OpenCVCamera(config)
    camera.connect()

    with pytest.raises(TimeoutError):
        camera.async_read(timeout_ms=0)

    camera.disconnect()


def test_async_read_before_connect():
    config = OpenCVCameraConfig(index_or_path="tests/artifacts/cameras/fake_cam.png")
    camera = OpenCVCamera(config)

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
def test_all_rotations(rotation):
    config = OpenCVCameraConfig(index_or_path="tests/artifacts/cameras/fake_cam.png", rotation=rotation)
    camera = OpenCVCamera(config)
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

    camera.disconnect()
