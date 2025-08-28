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

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.cameras.reachy2_camera import Reachy2Camera, Reachy2CameraConfig
from lerobot.errors import DeviceNotConnectedError

PARAMS = [
    ("teleop", "left"),
    ("teleop", "right"),
    ("depth", "rgb"),
    # ("depth", "depth"),  # Depth camera is not available yet
]


def _make_cam_manager_mock():
    c = MagicMock(name="CameraManagerMock")

    teleop = MagicMock(name="TeleopCam")
    teleop.width = 640
    teleop.height = 480
    teleop.get_frame = MagicMock(
        side_effect=lambda *_, **__: (
            np.zeros((480, 640, 3), dtype=np.uint8),
            time.time(),
        )
    )

    depth = MagicMock(name="DepthCam")
    depth.width = 640
    depth.height = 480
    depth.get_frame = MagicMock(
        side_effect=lambda *_, **__: (
            np.zeros((480, 640, 3), dtype=np.uint8),
            time.time(),
        )
    )

    c.is_connected.return_value = True
    c.teleop = teleop
    c.depth = depth

    def _connect():
        c.teleop = teleop
        c.depth = depth
        c.is_connected.return_value = True

    def _disconnect():
        c.teleop = None
        c.depth = None
        c.is_connected.return_value = False

    c.connect = MagicMock(side_effect=_connect)
    c.disconnect = MagicMock(side_effect=_disconnect)

    # Mock methods
    c.initialize_cameras = MagicMock()

    return c


@pytest.fixture(
    params=PARAMS,
    # ids=["teleop-left", "teleop-right", "torso-rgb", "torso-depth"],
    ids=["teleop-left", "teleop-right", "torso-rgb"],
)
def camera(request):
    name, image_type = request.param
    with (
        patch(
            "lerobot.cameras.reachy2_camera.reachy2_camera.CameraManager",
            side_effect=lambda *a, **k: _make_cam_manager_mock(),
        ),
    ):
        config = Reachy2CameraConfig(name=name, image_type=image_type)
        cam = Reachy2Camera(config)
        yield cam
        if cam.is_connected:
            cam.disconnect()


def test_connect(camera):
    camera.connect()
    assert camera.is_connected
    camera.cam_manager.initialize_cameras.assert_called_once()


def test_read(camera):
    camera.connect()

    img = camera.read()
    if camera.config.name == "teleop":
        camera.cam_manager.teleop.get_frame.assert_called_once()
    elif camera.config.name == "depth":
        camera.cam_manager.depth.get_frame.assert_called_once()
    assert isinstance(img, np.ndarray)
    assert img.shape == (480, 640, 3)


def test_disconnect(camera):
    camera.connect()

    camera.disconnect()
    assert not camera.is_connected


def test_async_read(camera):
    camera.connect()
    try:
        img = camera.async_read()

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(img, np.ndarray)
    finally:
        if camera.is_connected:
            camera.disconnect()


def test_async_read_timeout(camera):
    camera.connect()
    try:
        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=0)
    finally:
        if camera.is_connected:
            camera.disconnect()


def test_read_before_connect(camera):
    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


def test_disconnect_before_connect(camera):
    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


def test_async_read_before_connect(camera):
    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


def test_wrong_camera_name():
    with pytest.raises(ValueError):
        _ = Reachy2CameraConfig(name="wrong-name", image_type="left")


def test_wrong_image_type():
    with pytest.raises(ValueError):
        _ = Reachy2CameraConfig(name="teleop", image_type="rgb")
    with pytest.raises(ValueError):
        _ = Reachy2CameraConfig(name="depth", image_type="left")


def test_wrong_color_mode():
    with pytest.raises(ValueError):
        _ = Reachy2CameraConfig(name="teleop", image_type="left", color_mode="wrong-color")
