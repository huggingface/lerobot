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

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_orbbec.py::test_connect
# ```

# NOTE: These tests do not require the `pyorbbecsdk` package nor real hardware.
# The Orbbec SDK (imported as `pyorbbecsdk`) is mocked so the camera logic can be
# exercised in CI where the SDK and the device are unavailable.

import time
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
DEPTH_VALUE = 500


def _make_profile(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS):
    profile = MagicMock(name="VideoStreamProfile")
    profile.get_width.return_value = width
    profile.get_height.return_value = height
    profile.get_fps.return_value = fps
    return profile


def _make_profile_list():
    """Profile list returning a profile with the requested settings (0 means "any" -> default)."""
    profile_list = MagicMock(name="StreamProfileList")
    profile_list.get_default_video_stream_profile.return_value = _make_profile()
    profile_list.get_video_stream_profile.side_effect = lambda w, h, fmt, fps: _make_profile(
        w or DEFAULT_WIDTH, h or DEFAULT_HEIGHT, fps or DEFAULT_FPS
    )
    return profile_list


def _make_frame(channels, dtype, value):
    frame = MagicMock(name="VideoFrame")
    frame.get_width.return_value = DEFAULT_WIDTH
    frame.get_height.return_value = DEFAULT_HEIGHT
    frame.get_depth_scale.return_value = 1.0
    data = np.full(DEFAULT_HEIGHT * DEFAULT_WIDTH * channels, value, dtype=dtype)
    frame.get_data.return_value = data.view(np.uint8)
    return frame


def _make_fake_ob(devices=((SERIAL_NUMBER, DEVICE_NAME),)):
    """Build a fake `pyorbbecsdk` module (`ob`) emulating the subset of the API used."""
    frames = MagicMock(name="FrameSet")
    frames.get_color_frame.return_value = _make_frame(3, np.uint8, 128)
    frames.get_depth_frame.return_value = _make_frame(1, np.uint16, DEPTH_VALUE)

    pipeline = MagicMock(name="Pipeline")
    profile_lists = {"COLOR": _make_profile_list(), "DEPTH": _make_profile_list()}
    pipeline.get_stream_profile_list.side_effect = profile_lists.__getitem__
    pipeline.wait_for_frames.return_value = frames

    def get_device_by_serial_number(serial_number):
        if serial_number not in [sn for sn, _ in devices]:
            raise RuntimeError(f"Device not found by serial number: {serial_number}")
        return MagicMock(name="Device")

    device_list = MagicMock(name="DeviceList")
    device_list.get_count.return_value = len(devices)
    device_list.get_device_serial_number_by_index.side_effect = lambda i: devices[i][0]
    device_list.get_device_name_by_index.side_effect = lambda i: devices[i][1]
    device_list.get_device_by_serial_number.side_effect = get_device_by_serial_number

    def get_device_by_index(i):
        device = MagicMock(name="Device")
        device.get_device_info.return_value.get_serial_number.return_value = devices[i][0]
        device.get_device_info.return_value.get_name.return_value = devices[i][1]
        return device

    device_list.get_device_by_index.side_effect = get_device_by_index

    ob = MagicMock(name="pyorbbecsdk")
    ob.Context.return_value.query_devices.return_value = device_list
    ob.Pipeline.return_value = pipeline
    ob.OBFormat.RGB, ob.OBFormat.BGR, ob.OBFormat.Y16 = "RGB", "BGR", "Y16"
    ob.OBSensorType.COLOR_SENSOR, ob.OBSensorType.DEPTH_SENSOR = "COLOR", "DEPTH"
    ob.AlignFilter.return_value.process.side_effect = lambda f: MagicMock(as_frame_set=lambda: f)
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
    assert config.use_rgb and not config.use_depth
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


def test_config_no_stream_raises():
    with pytest.raises(ValueError, match="use_rgb"):
        OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER, use_rgb=False, use_depth=False)


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


def test_init_resolves_serial_number_from_name(fake_ob):
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=DEVICE_NAME))
    assert camera.serial_number == SERIAL_NUMBER


def test_init_unknown_camera_raises(fake_ob):
    with pytest.raises(ValueError, match="No Orbbec camera found"):
        OrbbecCamera(OrbbecCameraConfig(serial_number_or_name="does-not-exist"))


def test_init_duplicate_name_raises():
    ob = _make_fake_ob(devices=(("SN1", DEVICE_NAME), ("SN2", DEVICE_NAME)))
    with (
        patch("lerobot.cameras.orbbec.camera_orbbec.ob", ob),
        patch("lerobot.cameras.orbbec.camera_orbbec.require_package", return_value=None),
        pytest.raises(ValueError, match="Multiple Orbbec cameras"),
    ):
        OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=DEVICE_NAME))


def test_connect_and_disconnect(fake_ob):
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER))
    assert not camera.is_connected
    camera.connect()
    assert camera.is_connected
    assert camera.thread is not None and camera.thread.is_alive()
    # Resolution/fps are inferred from the default stream profile.
    assert (camera.width, camera.height, camera.fps) == (DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS)
    camera.disconnect()
    assert not camera.is_connected
    assert camera.thread is None
    fake_ob.Pipeline.return_value.stop.assert_called_once()


def test_connect_already_connected(fake_ob):
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER))
    camera.connect(warmup=False)
    with pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)
    camera.disconnect()


def test_connect_start_failure_raises(fake_ob):
    fake_ob.Pipeline.return_value.start.side_effect = RuntimeError("boom")
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER))
    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)
    assert not camera.is_connected
    assert camera.fps is None and camera.width is None


def test_warmup_failure_cleans_up(fake_ob):
    # Like the real SDK, block for a while before timing out with no frames.
    fake_ob.Pipeline.return_value.wait_for_frames.side_effect = lambda _timeout_ms: time.sleep(0.5)
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER))
    with pytest.raises((ConnectionError, TimeoutError)):
        camera.connect()
    assert not camera.is_connected
    assert camera.thread is None
    fake_ob.Pipeline.return_value.stop.assert_called_once()


def test_reconnect(fake_ob):
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER))
    camera.connect()
    camera.disconnect()
    camera.connect()
    assert camera.read().shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
    camera.disconnect()


@pytest.mark.parametrize("color_mode", [ColorMode.RGB, ColorMode.BGR])
def test_read(fake_ob, color_mode):
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER, color_mode=color_mode))
    camera.connect()
    frame = camera.read()
    assert frame.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
    assert frame.dtype == np.uint8
    # The SDK is asked for the output color mode directly, so no conversion is needed.
    color_profiles = fake_ob.Pipeline.return_value.get_stream_profile_list("COLOR")
    assert color_profiles.get_video_stream_profile.call_args.args[2] == color_mode.name
    camera.disconnect()


def test_async_read_and_read_latest(fake_ob):
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER))
    camera.connect()
    assert camera.async_read(timeout_ms=2000).shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
    assert camera.read_latest().shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
    camera.disconnect()


def test_rotation(fake_ob):
    config = OrbbecCameraConfig(
        serial_number_or_name=SERIAL_NUMBER,
        fps=DEFAULT_FPS,
        width=DEFAULT_HEIGHT,
        height=DEFAULT_WIDTH,
        rotation=Cv2Rotation.ROTATE_90,
    )
    camera = OrbbecCamera(config)
    camera.connect()
    assert camera.read().shape == (DEFAULT_WIDTH, DEFAULT_HEIGHT, 3)
    camera.disconnect()


@pytest.mark.parametrize("depth_scale", [1.0, 0.1])
def test_read_depth(fake_ob, depth_scale):
    frames = fake_ob.Pipeline.return_value.wait_for_frames.return_value
    frames.get_depth_frame.return_value.get_depth_scale.return_value = depth_scale
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER, use_depth=True))
    camera.connect()
    depth = camera.read_depth()
    assert depth.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 1)
    assert depth.dtype == np.uint16
    assert (depth == int(DEPTH_VALUE * depth_scale)).all()
    assert camera.async_read_depth(timeout_ms=2000).shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 1)
    assert camera.read_latest_depth().shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 1)
    assert camera.read().shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
    # Depth is aligned to color when both streams are enabled.
    fake_ob.AlignFilter.assert_called_once()
    camera.disconnect()


def test_read_depth_disabled_raises(fake_ob):
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER))
    camera.connect()
    with pytest.raises(RuntimeError, match="use_depth=False"):
        camera.read_depth()
    camera.disconnect()


def test_read_not_connected_raises(fake_ob):
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER))
    with pytest.raises(DeviceNotConnectedError):
        camera.read()


def test_disconnect_not_connected_raises(fake_ob):
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name=SERIAL_NUMBER))
    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()
