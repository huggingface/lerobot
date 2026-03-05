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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.cameras.configs import Cv2Rotation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

pytest.importorskip("pyorbbecsdk")

from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

MOCK_SERIAL = "CL4K2310053"
MOCK_WIDTH = 640
MOCK_HEIGHT = 480
MOCK_FPS = 30
MOCK_DEPTH_SCALE = 1.0

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_color_frame(width: int = MOCK_WIDTH, height: int = MOCK_HEIGHT) -> MagicMock:
    """Create a mock color VideoFrame that returns RGB data."""
    frame = MagicMock()
    frame.get_width.return_value = width
    frame.get_height.return_value = height
    # OBFormat.RGB = 1 in pyorbbecsdk — we import the real enum below
    from pyorbbecsdk import OBFormat

    frame.get_format.return_value = OBFormat.RGB
    frame.get_data.return_value = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()
    return frame


def _make_depth_frame(width: int = MOCK_WIDTH, height: int = MOCK_HEIGHT) -> MagicMock:
    """Create a mock depth VideoFrame that returns uint16 data."""
    frame = MagicMock()
    frame.get_width.return_value = width
    frame.get_height.return_value = height
    frame.get_depth_scale.return_value = MOCK_DEPTH_SCALE
    frame.get_data.return_value = np.random.randint(0, 5000, (height, width), dtype=np.uint16).tobytes()
    return frame


def _make_frameset(
    color: bool = True, depth: bool = False, width: int = MOCK_WIDTH, height: int = MOCK_HEIGHT
) -> MagicMock:
    """Create a mock FrameSet containing colour and/or depth frames."""
    fs = MagicMock()
    fs.get_color_frame.return_value = _make_color_frame(width, height) if color else None
    fs.get_depth_frame.return_value = _make_depth_frame(width, height) if depth else None
    return fs


def _make_video_stream_profile(
    width: int = MOCK_WIDTH, height: int = MOCK_HEIGHT, fps: int = MOCK_FPS
) -> MagicMock:
    """Create a mock VideoStreamProfile."""
    vsp = MagicMock()
    vsp.get_width.return_value = width
    vsp.get_height.return_value = height
    vsp.get_fps.return_value = fps
    return vsp


def _make_profile_list(
    default_width: int = MOCK_WIDTH,
    default_height: int = MOCK_HEIGHT,
    default_fps: int = MOCK_FPS,
) -> MagicMock:
    """Create a mock StreamProfileList with a default profile."""
    pl = MagicMock()
    default_profile = _make_video_stream_profile(default_width, default_height, default_fps)
    pl.get_default_video_stream_profile.return_value = default_profile
    pl.get_video_stream_profile.return_value = default_profile
    return pl


def _make_device(serial_number: str = MOCK_SERIAL, has_color: bool = True, has_depth: bool = True):
    """Create a mock Orbbec Device with DeviceInfo and sensors."""
    from pyorbbecsdk import OBSensorType

    device = MagicMock()
    info = MagicMock()
    info.get_serial_number.return_value = serial_number
    info.get_name.return_value = "Orbbec MockCam"
    info.get_pid.return_value = 0x0001
    info.get_vid.return_value = 0x2BC5
    info.get_connection_type.return_value = "USB3.2"
    device.get_device_info.return_value = info

    sensors = []
    if has_color:
        color_sensor = MagicMock()
        color_sensor.get_type.return_value = OBSensorType.COLOR_SENSOR
        sensors.append(color_sensor)
    if has_depth:
        depth_sensor = MagicMock()
        depth_sensor.get_type.return_value = OBSensorType.DEPTH_SENSOR
        sensors.append(depth_sensor)

    sensor_list = MagicMock()
    sensor_list.get_count.return_value = len(sensors)
    sensor_list.get_sensor_by_index.side_effect = lambda i: sensors[i]
    device.get_sensor_list.return_value = sensor_list

    return device


def _make_device_list(devices: list | None = None):
    """Create a mock DeviceList."""
    if devices is None:
        devices = [_make_device()]
    dl = MagicMock()
    dl.get_count.return_value = len(devices)
    dl.get_device_by_index.side_effect = lambda i: devices[i]
    dl.__getitem__ = lambda self, i: devices[i]
    return dl


# ---------------------------------------------------------------------------
# autouse fixture: patch pyorbbecsdk Pipeline/Context/Config/AlignFilter
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_orbbec_sdk():
    """Patch SDK classes so no real hardware is needed."""

    device = _make_device()
    device_list = _make_device_list([device])

    mock_ctx_cls = MagicMock()
    mock_ctx_cls.return_value.query_devices.return_value = device_list

    mock_pipeline_cls = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    # Pipeline.get_stream_profile_list returns appropriate profile lists
    def _profile_list_for_sensor(sensor_type):
        return _make_profile_list()

    mock_pipeline_instance.get_stream_profile_list.side_effect = _profile_list_for_sensor
    mock_pipeline_instance.wait_for_frames.return_value = _make_frameset(color=True, depth=True)
    mock_pipeline_instance.start.return_value = None
    mock_pipeline_instance.stop.return_value = None

    mock_config_cls = MagicMock()
    mock_align_cls = MagicMock()

    # Determine module path for patching
    module_path = "lerobot.cameras.orbbec.camera_orbbec"

    with (
        patch(f"{module_path}.Context", mock_ctx_cls),
        patch(f"{module_path}.Pipeline", mock_pipeline_cls),
        patch(f"{module_path}.Config", mock_config_cls),
        patch(f"{module_path}.AlignFilter", mock_align_cls),
    ):
        yield {
            "context": mock_ctx_cls,
            "pipeline_cls": mock_pipeline_cls,
            "pipeline": mock_pipeline_instance,
            "config": mock_config_cls,
            "align": mock_align_cls,
            "device": device,
            "device_list": device_list,
        }


# ---------------------------------------------------------------------------
# Tests — Config validation
# ---------------------------------------------------------------------------


def test_abc_implementation():
    """Instantiation should succeed when abstract methods are implemented."""
    config = OrbbecCameraConfig(index_or_serial_number=0)
    _ = OrbbecCamera(config)


def test_config_invalid_index_type():
    with pytest.raises(ValueError, match="index_or_serial_number"):
        OrbbecCameraConfig(index_or_serial_number=3.14)


def test_config_empty_serial():
    with pytest.raises(ValueError, match="must not be an empty string"):
        OrbbecCameraConfig(index_or_serial_number="  ")


def test_config_align_depth_without_use_depth():
    with pytest.raises(ValueError, match="align_depth=True requires use_depth=True"):
        OrbbecCameraConfig(index_or_serial_number=0, use_depth=False, align_depth=True)


# ---------------------------------------------------------------------------
# Tests — Connection
# ---------------------------------------------------------------------------


def test_connect():
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480, warmup_s=1)

    camera = OrbbecCamera(config)
    camera.connect()

    assert camera.is_connected
    camera.disconnect()


def test_connect_context_manager():
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480, warmup_s=1)

    with OrbbecCamera(config) as camera:
        assert camera.is_connected

    assert not camera.is_connected


def test_connect_already_connected():
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480, warmup_s=1)

    with OrbbecCamera(config) as camera, pytest.raises(DeviceAlreadyConnectedError):
        camera.connect()


def test_connect_by_serial(patch_orbbec_sdk):
    config = OrbbecCameraConfig(index_or_serial_number=MOCK_SERIAL, fps=30, width=640, height=480, warmup_s=1)

    with OrbbecCamera(config) as camera:
        assert camera.is_connected
        assert camera.serial_number == MOCK_SERIAL


def test_connect_invalid_index(patch_orbbec_sdk):
    config = OrbbecCameraConfig(index_or_serial_number=99, fps=30, width=640, height=480)
    camera = OrbbecCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


def test_connect_invalid_serial(patch_orbbec_sdk):
    config = OrbbecCameraConfig(index_or_serial_number="NONEXISTENT_SN", fps=30, width=640, height=480)
    camera = OrbbecCamera(config)

    with pytest.raises(ConnectionError, match="No Orbbec device with serial"):
        camera.connect(warmup=False)


def test_connect_pipeline_start_failure(patch_orbbec_sdk):
    patch_orbbec_sdk["pipeline"].start.side_effect = RuntimeError("USB error")
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480)
    camera = OrbbecCamera(config)

    with pytest.raises(ConnectionError, match="Failed to start pipeline"):
        camera.connect(warmup=False)


# ---------------------------------------------------------------------------
# Tests — Synchronous read
# ---------------------------------------------------------------------------


def test_read():
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480, warmup_s=1)

    with OrbbecCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)
        assert img.shape == (MOCK_HEIGHT, MOCK_WIDTH, 3)
        assert img.dtype == np.uint8


def test_read_depth(patch_orbbec_sdk):
    patch_orbbec_sdk["pipeline"].wait_for_frames.return_value = _make_frameset(color=True, depth=True)
    config = OrbbecCameraConfig(
        index_or_serial_number=0, fps=30, width=640, height=480, use_depth=True, warmup_s=1
    )

    with OrbbecCamera(config) as camera:
        depth = camera.read_depth()
        assert isinstance(depth, np.ndarray)
        assert depth.shape == (MOCK_HEIGHT, MOCK_WIDTH, 1)
        assert depth.dtype == np.uint16


def test_read_depth_not_enabled():
    config = OrbbecCameraConfig(
        index_or_serial_number=0, fps=30, width=640, height=480, use_depth=False, warmup_s=1
    )

    with OrbbecCamera(config) as camera, pytest.raises(RuntimeError, match="depth stream not enabled"):
        camera.read_depth()


def test_read_before_connect():
    config = OrbbecCameraConfig(index_or_serial_number=0)
    camera = OrbbecCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        camera.read()


# ---------------------------------------------------------------------------
# Tests — Async read
# ---------------------------------------------------------------------------


def test_async_read():
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480, warmup_s=1)

    with OrbbecCamera(config) as camera:
        img = camera.async_read()

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(img, np.ndarray)
        assert img.shape == (MOCK_HEIGHT, MOCK_WIDTH, 3)


def test_async_read_timeout():
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480, warmup_s=1)

    with OrbbecCamera(config) as camera, pytest.raises(TimeoutError):
        camera.async_read(timeout_ms=0)  # consume any available frame
        camera.async_read(timeout_ms=0)  # request another immediately


def test_async_read_before_connect():
    config = OrbbecCameraConfig(index_or_serial_number=0)
    camera = OrbbecCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        camera.async_read()


def test_async_read_depth(patch_orbbec_sdk):
    patch_orbbec_sdk["pipeline"].wait_for_frames.return_value = _make_frameset(color=True, depth=True)
    config = OrbbecCameraConfig(
        index_or_serial_number=0, fps=30, width=640, height=480, use_depth=True, warmup_s=1
    )

    with OrbbecCamera(config) as camera:
        depth = camera.async_read_depth()
        assert isinstance(depth, np.ndarray)
        assert depth.shape == (MOCK_HEIGHT, MOCK_WIDTH, 1)
        assert depth.dtype == np.uint16


def test_async_read_depth_not_enabled():
    config = OrbbecCameraConfig(
        index_or_serial_number=0, fps=30, width=640, height=480, use_depth=False, warmup_s=1
    )

    with OrbbecCamera(config) as camera:
        result = camera.async_read_depth()
        assert result is None


# ---------------------------------------------------------------------------
# Tests — read_latest
# ---------------------------------------------------------------------------


def test_read_latest():
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480, warmup_s=1)

    with OrbbecCamera(config) as camera:
        img = camera.read()
        latest = camera.read_latest()

        assert isinstance(latest, np.ndarray)
        assert latest.shape == img.shape


def test_read_latest_high_frequency():
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480, warmup_s=1)

    with OrbbecCamera(config) as camera:
        ref = camera.read()

        for _ in range(20):
            latest = camera.read_latest()
            assert isinstance(latest, np.ndarray)
            assert latest.shape == ref.shape


def test_read_latest_before_connect():
    config = OrbbecCameraConfig(index_or_serial_number=0)
    camera = OrbbecCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        camera.read_latest()


def test_read_latest_too_old():
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480, warmup_s=1)

    with OrbbecCamera(config) as camera:
        _ = camera.read()

        with pytest.raises(TimeoutError):
            camera.read_latest(max_age_ms=0)  # immediately too old


# ---------------------------------------------------------------------------
# Tests — Disconnect
# ---------------------------------------------------------------------------


def test_disconnect():
    config = OrbbecCameraConfig(index_or_serial_number=0, fps=30, width=640, height=480, warmup_s=1)
    camera = OrbbecCamera(config)
    camera.connect()
    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    config = OrbbecCameraConfig(index_or_serial_number=0)
    camera = OrbbecCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


# ---------------------------------------------------------------------------
# Tests — Rotation
# ---------------------------------------------------------------------------


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
def test_rotation(rotation):
    config = OrbbecCameraConfig(
        index_or_serial_number=0, fps=30, width=640, height=480, rotation=rotation, warmup_s=1
    )

    with OrbbecCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)

        if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
            assert camera.width == MOCK_HEIGHT
            assert camera.height == MOCK_WIDTH
            assert img.shape[:2] == (MOCK_WIDTH, MOCK_HEIGHT)
        else:
            assert camera.width == MOCK_WIDTH
            assert camera.height == MOCK_HEIGHT
            assert img.shape[:2] == (MOCK_HEIGHT, MOCK_WIDTH)


# ---------------------------------------------------------------------------
# Tests — find_cameras static method
# ---------------------------------------------------------------------------


def test_find_cameras():
    cameras = OrbbecCamera.find_cameras()
    assert isinstance(cameras, list)
    assert len(cameras) == 1
    assert cameras[0]["type"] == "Orbbec"
    assert cameras[0]["id"] == MOCK_SERIAL
    assert cameras[0]["name"] == "Orbbec MockCam"


def test_find_cameras_no_devices(patch_orbbec_sdk):
    empty_list = MagicMock()
    empty_list.get_count.return_value = 0
    patch_orbbec_sdk["context"].return_value.query_devices.return_value = empty_list

    cameras = OrbbecCamera.find_cameras()
    assert cameras == []
