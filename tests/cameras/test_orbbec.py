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
# pytest tests/cameras/test_orbbec.py::test_connect
# ```

# NOTE: These tests do not require the `pyorbbecsdk` package nor real hardware.
# The Orbbec SDK (imported as `pyorbbecsdk`) is mocked so the camera logic can be
# exercised in CI where the SDK and the device are unavailable.

from enum import Enum
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
SERIAL_NUMBER = "CP1234567890"
DEVICE_NAME = "Orbbec Gemini 335"


class FakeOBFormat(Enum):
    RGB = 0
    BGR = 1
    MJPG = 2
    YUYV = 3
    UYVY = 4
    UNKNOWN = 99


class FakeOBSensorType:
    COLOR_SENSOR = "COLOR_SENSOR"


def _make_profile(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS, fmt=FakeOBFormat.RGB):
    profile = MagicMock(name="StreamProfile")
    profile.get_width.return_value = width
    profile.get_height.return_value = height
    profile.get_fps.return_value = fps
    profile.get_format.return_value = fmt
    profile.get_type.return_value = FakeOBSensorType.COLOR_SENSOR
    profile.is_video_stream_profile.return_value = True
    profile.as_video_stream_profile.return_value = profile
    return profile


def _make_color_frame(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fmt=FakeOBFormat.RGB):
    frame = MagicMock(name="ColorFrame")
    frame.get_width.return_value = width
    frame.get_height.return_value = height
    frame.get_format.return_value = fmt
    frame.get_data.return_value = np.zeros(height * width * 3, dtype=np.uint8)
    return frame


def _make_device(serial=SERIAL_NUMBER, name=DEVICE_NAME):
    info = MagicMock(name="DeviceInfo")
    info.get_serial_number.return_value = serial
    info.get_name.return_value = name
    info.get_uid.return_value = "usb-1"
    info.get_vid.return_value = 0x2BC5
    info.get_pid.return_value = 0x0001
    info.get_firmware_version.return_value = "1.2.3"
    info.get_hardware_version.return_value = "1.0"
    info.get_connection_type.return_value = "USB3.0"
    device = MagicMock(name="Device")
    device.get_device_info.return_value = info
    return device


def _make_fake_ob(devices=None, profile=None, color_frame=None):
    """Build a fake `pyorbbecsdk` module (`ob`) emulating the subset of the API used."""
    if devices is None:
        devices = [_make_device()]
    if profile is None:
        profile = _make_profile()
    if color_frame is None:
        color_frame = _make_color_frame()

    profile_list = MagicMock(name="ProfileList")
    profile_list.get_count.return_value = 1
    profile_list.get_stream_profile_by_index.side_effect = lambda _i: profile
    profile_list.get_default_video_stream_profile.return_value = profile

    frames = MagicMock(name="FrameSet")
    frames.get_color_frame.return_value = color_frame

    pipeline = MagicMock(name="Pipeline")
    pipeline.get_stream_profile_list.return_value = profile_list
    pipeline.wait_for_frames.return_value = frames

    device_list = MagicMock(name="DeviceList")
    device_list.get_count.return_value = len(devices)
    device_list.get_device_by_index.side_effect = lambda i: devices[i]

    context = MagicMock(name="Context")
    context.query_devices.return_value = device_list

    ob = MagicMock(name="pyorbbecsdk")
    ob.OBFormat = FakeOBFormat
    ob.OBSensorType = FakeOBSensorType
    ob.Context.return_value = context
    ob.Pipeline.return_value = pipeline
    ob.Config.return_value = MagicMock(name="Config")
    return ob


@pytest.fixture(name="fake_ob")
def fixture_fake_ob():
    """Patch the SDK module and `require_package` so tests run without pyorbbecsdk."""
    ob = _make_fake_ob()
    with (
        patch("lerobot.cameras.orbbec.camera_orbbec.ob", ob),
        patch("lerobot.cameras.orbbec.camera_orbbec.require_package", return_value=None),
    ):
        yield ob


# --------------------------------------------------------------------------- #
# Configuration tests (no SDK required)
# --------------------------------------------------------------------------- #


def test_config_defaults():
    config = OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER)
    assert config.serial_number_or_name == SERIAL_NUMBER
    assert config.color_mode == ColorMode.RGB
    assert config.rotation == Cv2Rotation.NO_ROTATION
    assert config.warmup_s == 1
    # The config registers itself under the "orbbec" choice.
    assert config.type == "orbbec"


def test_config_invalid_color_mode():
    with pytest.raises(ValueError):
        OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER, color_mode="not_a_color_mode")


def test_config_invalid_rotation():
    with pytest.raises(ValueError):
        OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER, rotation=42)


def test_config_partial_resolution_raises():
    with pytest.raises(ValueError):
        OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER, fps=30, width=640)


def test_config_full_resolution_ok():
    config = OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER, fps=30, width=640, height=480)
    assert (config.fps, config.width, config.height) == (30, 640, 480)


# --------------------------------------------------------------------------- #
# Camera tests (SDK mocked)
# --------------------------------------------------------------------------- #


def test_find_cameras(fake_ob):
    cameras = OrbbecCamera.find_cameras()
    assert len(cameras) == 1
    assert cameras[0]["id"] == SERIAL_NUMBER
    assert cameras[0]["name"] == DEVICE_NAME
    assert cameras[0]["type"] == "Orbbec"
    assert cameras[0]["default_stream_profile"]["width"] == DEFAULT_WIDTH


def test_connect_and_is_connected(fake_ob):
    config = OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER)
    camera = OrbbecCamera(config)
    assert not camera.is_connected
    camera.connect(warmup=False)
    assert camera.is_connected
    # Resolution/fps are inferred from the default stream profile.
    assert camera.width == DEFAULT_WIDTH
    assert camera.height == DEFAULT_HEIGHT
    assert camera.fps == DEFAULT_FPS
    camera.disconnect()
    assert not camera.is_connected


def test_connect_already_connected(fake_ob):
    config = OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER)
    camera = OrbbecCamera(config)
    camera.connect(warmup=False)
    with pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)
    camera.disconnect()


def test_connect_unknown_serial_raises(fake_ob):
    config = OrbbecCameraConfig(serial_number_or_name="does-not-exist")
    camera = OrbbecCamera(config)
    with pytest.raises(ValueError):
        camera.connect(warmup=False)


def test_read_rgb(fake_ob):
    config = OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER)
    camera = OrbbecCamera(config)
    camera.connect(warmup=False)
    frame = camera.read()
    assert frame.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
    assert frame.dtype == np.uint8
    camera.disconnect()


def test_read_bgr(fake_ob):
    config = OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER, color_mode=ColorMode.BGR)
    camera = OrbbecCamera(config)
    camera.connect(warmup=False)
    frame = camera.read()
    assert frame.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
    camera.disconnect()


def test_read_not_connected_raises(fake_ob):
    config = OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER)
    camera = OrbbecCamera(config)
    with pytest.raises(DeviceNotConnectedError):
        camera.read()


def test_async_read(fake_ob):
    config = OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER)
    camera = OrbbecCamera(config)
    camera.connect(warmup=False)
    frame = camera.async_read(timeout_ms=2000)
    assert frame.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
    camera.disconnect()


def test_disconnect_not_connected_raises(fake_ob):
    config = OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER)
    camera = OrbbecCamera(config)
    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()
