#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.gripette import GripetteCamera, GripetteCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs

_CAMERA = "lerobot.cameras.gripette.camera_gripette"


class FakeGripperClient:
    def __init__(self, addr: str):
        self.addr = addr
        self._closed = threading.Event()
        self.connect = MagicMock()
        self.close = MagicMock(side_effect=self._closed.set)
        ok, buf = cv2.imencode(".jpg", np.full((8, 6, 3), (200, 0, 0), dtype=np.uint8))
        assert ok
        self._jpeg = buf.tobytes()

    def stream(self):
        while not self._closed.is_set():
            yield SimpleNamespace(jpeg_data=self._jpeg, motor1=0.4, motor2=0.5)
            self._closed.wait(0.01)


def test_config_defaults_and_registration():
    cfg = GripetteCameraConfig(addr="172.18.7.107:50051")
    assert cfg.type == "gripette"
    assert (cfg.width, cfg.height, cfg.fps) == (648, 486, 10)
    assert cfg.color_mode == ColorMode.RGB

    with pytest.raises(ValueError, match="addr"):
        GripetteCameraConfig(addr="")


def test_make_cameras_from_configs_builds_gripette_camera():
    cfg = GripetteCameraConfig(addr="gripette:50051", width=3, height=4, fps=10)
    with patch(f"{_CAMERA}.require_package", lambda *a, **kw: None):
        cameras = make_cameras_from_configs({"wrist": cfg})
    assert isinstance(cameras["wrist"], GripetteCamera)
    assert cameras["wrist"].width == 3
    assert cameras["wrist"].height == 4


def test_read_decodes_rgb():
    cfg = GripetteCameraConfig(addr="gripette:50051", width=3, height=4, fps=10)
    with (
        patch(f"{_CAMERA}.require_package", lambda *a, **kw: None),
        patch(f"{_CAMERA}.GripperClient", FakeGripperClient),
    ):
        camera = GripetteCamera(cfg)
        camera.connect()
        try:
            latest = camera.read_latest()
            fresh = camera.async_read(timeout_ms=1000)
            assert latest.shape == (4, 3, 3)
            assert fresh.shape == (4, 3, 3)
            assert latest[0, 0, 2] > 100
            with pytest.raises(NotImplementedError):
                GripetteCamera.find_cameras()
        finally:
            camera.disconnect()
        assert not camera.is_connected


def test_bgr_color_mode_skips_channel_swap():
    cfg = GripetteCameraConfig(addr="gripette:50051", width=6, height=8, fps=10, color_mode=ColorMode.BGR)
    with (
        patch(f"{_CAMERA}.require_package", lambda *a, **kw: None),
        patch(f"{_CAMERA}.GripperClient", FakeGripperClient),
    ):
        camera = GripetteCamera(cfg)
        camera.connect()
        try:
            frame = camera.read_latest()
            assert frame.shape == (8, 6, 3)
            assert frame[0, 0, 0] > 100
        finally:
            camera.disconnect()
