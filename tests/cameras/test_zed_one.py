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
# pytest tests/cameras/test_zed_one.py::test_connect
# ```

# The shared read/teardown machinery is covered by tests/cameras/test_zed.py; this file checks
# what is specific to the monocular `sl.CameraOne` API (no depth, its own parameters and device
# list, flat calibration) plus a smoke pass over the common surface. Same fake SDK, no hardware.

import logging
import time
from dataclasses import fields

import draccus
import numpy as np
import pytest

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.zed import ZedOneCamera, ZedOneCameraConfig, base_zed, camera_zed_one
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from tests.mocks.mock_pyzed import (
    BGRA_PIXEL,
    CAPTURE_FPS,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DEFAULT_SERIAL,
    RECTIFIED_FX,
    STEREO_SERIAL,
    UNRECTIFIED_FX,
    FakeCameraOne,
    FakeErrorCode,
    FakeInitParameters,
    FakeInitParametersOne,
    FakeResolution,
    FakeUnit,
    FakeView,
    ZedTestHarness,
    fake_device,
)


@pytest.fixture(name="zed_one")
def fixture_zed_one(monkeypatch):
    harness = ZedTestHarness(monkeypatch, (base_zed, camera_zed_one), ZedOneCamera, ZedOneCameraConfig)
    yield harness
    harness.cleanup()


# --- configuration (no ZED SDK involved at all) ---------------------------------------------------


def test_config_defaults():
    config = ZedOneCameraConfig()

    assert config.serial_number is None
    assert config.resolution == "AUTO"
    assert config.rectified is True
    assert config.color_mode is ColorMode.RGB
    assert config.rotation is Cv2Rotation.NO_ROTATION
    assert config.warmup_s == 1
    assert config.fps is None
    assert config.width is None
    assert config.height is None


def test_config_has_no_depth_settings():
    """A single sensor cannot triangulate, so the depth knobs of the stereo config must not exist."""
    config = ZedOneCameraConfig()
    names = {f.name for f in fields(config)}

    assert "use_depth" not in names
    assert "depth_mode" not in names
    assert not hasattr(config, "use_depth")
    assert not hasattr(config, "depth_mode")


@pytest.mark.parametrize(
    "kwargs",
    [{"fps": 30}, {"width": 1920}, {"height": 1200}, {"fps": 30, "width": 1920}],
    ids=["fps", "width", "height", "fps_width"],
)
def test_config_partial_capture_settings_raise(kwargs):
    with pytest.raises(ValueError, match="either all of them"):
        ZedOneCameraConfig(**kwargs)


def test_config_full_capture_settings_are_allowed():
    config = ZedOneCameraConfig(fps=30, width=1920, height=1200)

    assert (config.fps, config.width, config.height) == (30, 1920, 1200)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("rgb", ColorMode.RGB), ("bgr", ColorMode.BGR), (ColorMode.BGR, ColorMode.BGR)],
    ids=["str_rgb", "str_bgr", "enum_bgr"],
)
def test_config_color_mode_coercion(value, expected):
    assert ZedOneCameraConfig(color_mode=value).color_mode is expected


def test_config_invalid_color_mode_raises():
    with pytest.raises(ValueError, match="color_mode"):
        ZedOneCameraConfig(color_mode="rgba")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, Cv2Rotation.NO_ROTATION),
        (90, Cv2Rotation.ROTATE_90),
        (180, Cv2Rotation.ROTATE_180),
        (-90, Cv2Rotation.ROTATE_270),
    ],
    ids=["int_0", "int_90", "int_180", "int_-90"],
)
def test_config_rotation_coercion(value, expected):
    assert ZedOneCameraConfig(rotation=value).rotation is expected


@pytest.mark.parametrize("value", [45, 270], ids=["45", "270"])
def test_config_invalid_rotation_raises(value):
    with pytest.raises(ValueError, match="rotation"):
        ZedOneCameraConfig(rotation=value)


def test_config_type_is_registered():
    assert CameraConfig.get_known_choices()["zed_one"] is ZedOneCameraConfig
    assert ZedOneCameraConfig().type == "zed_one"


def test_config_decodes_from_dict():
    config = draccus.decode(
        CameraConfig,
        {
            "type": "zed_one",
            "serial_number": DEFAULT_SERIAL,
            "resolution": "HD1080",
            "rectified": False,
            "color_mode": "bgr",
            "rotation": 90,
        },
    )

    assert isinstance(config, ZedOneCameraConfig)
    assert config.serial_number == DEFAULT_SERIAL
    assert config.resolution == "HD1080"
    assert config.rectified is False
    assert config.color_mode is ColorMode.BGR
    assert config.rotation is Cv2Rotation.ROTATE_90


# --- missing ZED SDK ------------------------------------------------------------------------------


def test_missing_pyzed_raises_import_error(monkeypatch):
    for module in (base_zed, camera_zed_one):
        monkeypatch.setattr(module, "sl", None)
    monkeypatch.setattr(base_zed, "_pyzed_import_error", None)

    with pytest.raises(ImportError, match="get_python_api.py"):
        ZedOneCamera(ZedOneCameraConfig(serial_number=DEFAULT_SERIAL))


def test_broken_pyzed_import_reports_the_cause(monkeypatch):
    for module in (base_zed, camera_zed_one):
        monkeypatch.setattr(module, "sl", None)
    monkeypatch.setattr(
        base_zed, "_pyzed_import_error", OSError("libsl_zed.so: cannot open shared object file")
    )

    with pytest.raises(ImportError, match="libsl_zed.so") as excinfo:
        ZedOneCamera(ZedOneCameraConfig(serial_number=DEFAULT_SERIAL))
    assert isinstance(excinfo.value.__cause__, OSError)
    with pytest.raises(ImportError, match="failed to import"):
        ZedOneCamera.find_cameras()


def test_missing_pyzed_raises_import_error_in_find_cameras(monkeypatch):
    for module in (base_zed, camera_zed_one):
        monkeypatch.setattr(module, "sl", None)
    monkeypatch.setattr(base_zed, "_pyzed_import_error", None)

    with pytest.raises(ImportError, match="ZED SDK"):
        ZedOneCamera.find_cameras()


# --- connect / disconnect ------------------------------------------------------------------------


def test_abc_implementation(zed_one):
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    _ = zed_one.camera()


def test_str(zed_one):
    assert str(zed_one.camera()) == f"ZedOneCamera({DEFAULT_SERIAL})"
    assert str(ZedOneCamera(ZedOneCameraConfig())) == "ZedOneCamera(auto)"


def test_find_cameras(zed_one):
    assert ZedOneCamera.find_cameras() == [
        {
            "name": "ZED X One GS",
            "type": "zed_one",
            "id": DEFAULT_SERIAL,
            "serial_number": DEFAULT_SERIAL,
        }
    ]


def test_find_cameras_uses_the_monocular_device_list(zed_one):
    """Stereo devices are `ZedCamera.find_cameras()`'s business, even if they share a bus.

    On a ZED Box Mini a GMSL stereo camera and a ZED X One both report `/dev/i2c-2`; discovery
    that treated a shared path as evidence of a duplicate would discard the real X One.
    """
    zed_one.patch_sl(
        stereo_device_list=[fake_device("ZED X", STEREO_SERIAL, "/dev/i2c-2")],
        mono_device_list=[fake_device("ZED X One GS", DEFAULT_SERIAL, "/dev/i2c-2")],
    )

    assert [cam["serial_number"] for cam in ZedOneCamera.find_cameras()] == [DEFAULT_SERIAL]


def test_connect(zed_one):
    with zed_one.camera() as camera:
        assert camera.is_connected
        assert isinstance(zed_one.last_camera, FakeCameraOne)  # the single-sensor API, not sl.Camera
        assert (camera.width, camera.height) == (CAPTURE_WIDTH, CAPTURE_HEIGHT)
        assert camera.fps == CAPTURE_FPS
        assert camera.intrinsics == {
            "fx": RECTIFIED_FX,
            "fy": 701.0,
            "cx": CAPTURE_WIDTH / 2,
            "cy": CAPTURE_HEIGHT / 2,
        }


def test_connect_init_parameters(zed_one):
    """The single-sensor API takes `InitParametersOne`, which has no depth settings at all."""
    camera = zed_one.camera(resolution="HD1080")
    camera.connect(warmup=False)

    init = zed_one.last_camera.init_parameters
    assert isinstance(init, FakeInitParametersOne)
    assert not isinstance(init, FakeInitParameters)
    assert init.camera_resolution is FakeResolution.HD1080
    assert init.coordinate_units is FakeUnit.MILLIMETER  # SDK default; DEPTH_U16_MM does not depend on it
    assert init.serial_number == DEFAULT_SERIAL
    assert not hasattr(init, "depth_mode")  # the backend must not set one


def test_connect_forwards_configured_fps(zed_one):
    camera = zed_one.camera(fps=CAPTURE_FPS, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
    camera.connect(warmup=False)

    assert zed_one.last_camera.init_parameters.camera_fps == CAPTURE_FPS
    assert (camera.width, camera.height, camera.fps) == (CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS)


def test_connect_rejects_stream_settings_the_camera_ignores(zed_one):
    camera = zed_one.camera(fps=CAPTURE_FPS // 2, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)

    with pytest.raises(ValueError, match="did not honour"):
        camera.connect(warmup=False)

    assert not camera.is_connected
    assert zed_one.last_camera.close_count == 1


def test_connect_without_serial_number_opens_first_device(zed_one):
    camera = ZedOneCamera(ZedOneCameraConfig(warmup_s=0))
    camera.connect(warmup=False)
    try:
        assert zed_one.last_camera.init_parameters.serial_number is None
    finally:
        camera.disconnect()


def test_connect_unknown_resolution_raises(zed_one):
    camera = zed_one.camera(resolution="HD4K")

    with pytest.raises(ValueError, match="unknown resolution") as exc_info:
        camera.connect(warmup=False)

    assert "HD1080" in str(exc_info.value)
    assert "LAST" not in str(exc_info.value)
    assert not camera.is_connected


def test_connect_failure_raises_connection_error(zed_one):
    zed_one.patch_sl(open_status=FakeErrorCode.CAMERA_NOT_DETECTED)
    camera = zed_one.camera()

    with pytest.raises(ConnectionError, match="Failed to open"):
        camera.connect(warmup=False)

    assert not camera.is_connected
    assert camera.cam is None
    assert zed_one.last_camera.close_count == 1


def test_connect_with_an_sdk_warning_succeeds_and_logs_it(zed_one, caplog):
    zed_one.patch_sl(open_status=FakeErrorCode.CONFIGURATION_FALLBACK)
    camera = zed_one.camera()

    with caplog.at_level(logging.WARNING, logger="lerobot.cameras.zed.base_zed"):
        camera.connect(warmup=False)

    assert camera.is_connected
    assert any(str(FakeErrorCode.CONFIGURATION_FALLBACK) in r.getMessage() for r in caplog.records)


def test_connect_warmup_waits_and_primes_a_frame(zed_one):
    camera = ZedOneCamera(zed_one.config(warmup_s=0.2))
    start = time.perf_counter()
    camera.connect()
    try:
        assert time.perf_counter() - start >= 0.2
        assert camera.latest_color_frame is not None
    finally:
        camera.disconnect()


def test_connect_cleans_up_after_warmup_failure(zed_one):
    zed_one.patch_sl(grab_status=FakeErrorCode.FAILURE, grab_delay_s=0.05)
    camera = ZedOneCamera(zed_one.config(warmup_s=0.2))

    with pytest.raises(TimeoutError):
        camera.connect(warmup=True)

    assert not camera.is_connected
    assert camera.cam is None
    assert camera.thread is None
    assert zed_one.last_camera.close_count == 1


def test_connect_already_connected(zed_one):
    with zed_one.camera() as camera, pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)


def test_disconnect(zed_one):
    camera = zed_one.camera()
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected
    assert camera.cam is None
    assert camera.thread is None
    assert zed_one.last_camera.close_count == 1


def test_disconnect_before_connect(zed_one):
    camera = zed_one.camera()

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


def test_grab_takes_no_runtime_parameters(zed_one):
    """`sl.CameraOne.grab()` has no `RuntimeParameters` overload, unlike `sl.Camera.grab()`."""
    with zed_one.camera() as camera:
        camera.read()

        assert zed_one.last_camera.grab_arg_counts  # the read loop did grab
        assert set(zed_one.last_camera.grab_arg_counts) == {0}
        assert camera.runtime is None


# --- reads ---------------------------------------------------------------------------------------


def test_read(zed_one):
    with zed_one.camera() as camera:
        frame = camera.read()

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)
        assert frame.dtype == np.uint8


def test_read_returns_rgb_by_default(zed_one):
    with zed_one.camera() as camera:
        frame = camera.read()

        blue, green, red, _ = BGRA_PIXEL
        assert tuple(frame[0, 0]) == (red, green, blue)


def test_read_bgr_color_mode_keeps_channel_order(zed_one):
    with zed_one.camera(color_mode=ColorMode.BGR) as camera:
        frame = camera.read()

        assert tuple(frame[0, 0]) == BGRA_PIXEL[:3]


def test_read_color_mode_argument_is_deprecated_and_ignored(zed_one, caplog):
    with (
        caplog.at_level(logging.WARNING, logger="lerobot.cameras.zed.base_zed"),
        zed_one.camera() as camera,
    ):
        frame = camera.read(color_mode=ColorMode.BGR)

        blue, green, red, _ = BGRA_PIXEL
        assert tuple(frame[0, 0]) == (red, green, blue)
    assert any("color_mode parameter is deprecated" in r.getMessage() for r in caplog.records)


def test_async_read(zed_one):
    with zed_one.camera() as camera:
        frame = camera.async_read(timeout_ms=2000)

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert frame.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)


def test_async_read_consumes_the_frame_it_returns(zed_one):
    zed_one.patch_sl(grab_delay_s=0.3)
    with zed_one.camera() as camera:
        camera.async_read(timeout_ms=2000)

        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=1)


def test_read_latest(zed_one):
    with zed_one.camera() as camera:
        frame = camera.read()
        latest = camera.read_latest()

        assert isinstance(latest, np.ndarray)
        assert latest.shape == frame.shape


def test_read_latest_too_old(zed_one):
    with zed_one.camera() as camera:
        _ = camera.read()

        with pytest.raises(TimeoutError):
            _ = camera.read_latest(max_age_ms=0)


def test_read_latest_without_any_frame_raises(zed_one):
    zed_one.patch_sl(grab_status=FakeErrorCode.FAILURE, grab_delay_s=0.05)
    camera = zed_one.camera()
    camera.connect(warmup=False)

    with pytest.raises(RuntimeError, match="has not captured any frames"):
        camera.read_latest()


@pytest.mark.parametrize("method_name", ["read", "async_read", "read_latest"])
def test_read_before_connect(zed_one, method_name):
    camera = zed_one.camera()

    with pytest.raises(DeviceNotConnectedError):
        _ = getattr(camera, method_name)()


def test_latest_hw_timestamp_ns(zed_one):
    with zed_one.camera() as camera:
        assert camera.latest_hw_timestamp_ns is None

        _ = camera.read()

        assert isinstance(camera.latest_hw_timestamp_ns, int)
        assert camera.latest_hw_timestamp_ns > 0


# --- no depth on a single sensor ------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_name",
    ["read_depth", "async_read_depth", "read_latest_depth"],
    ids=["read_depth", "async_read_depth", "read_latest_depth"],
)
def test_depth_reads_are_refused(zed_one, method_name):
    with (
        zed_one.camera() as camera,
        pytest.raises(NotImplementedError, match="monocular"),
    ):
        _ = getattr(camera, method_name)()


@pytest.mark.parametrize(
    "method_name",
    ["read_depth", "async_read_depth", "read_latest_depth"],
    ids=["read_depth", "async_read_depth", "read_latest_depth"],
)
def test_depth_reads_are_refused_before_connect(zed_one, method_name):
    """Refusing depth does not depend on the connection state: the camera simply has none."""
    camera = zed_one.camera()

    with pytest.raises(NotImplementedError, match="monocular"):
        _ = getattr(camera, method_name)()


def test_camera_reports_no_depth_stream(zed_one):
    camera = zed_one.camera()

    assert camera.use_rgb is True
    assert camera.use_depth is False


# --- rectified vs unrectified --------------------------------------------------------------------


def test_rectified_by_default_retrieves_the_rectified_view(zed_one):
    camera = zed_one.camera()
    camera.connect(warmup=False)
    camera.read()

    assert zed_one.last_camera.requested_views
    assert set(zed_one.last_camera.requested_views) == {FakeView.LEFT}


def test_unrectified_retrieves_the_raw_view_and_its_own_intrinsics(zed_one):
    """`rectified=False` must switch both the retrieved view and the reported (flat) calibration."""
    camera = zed_one.camera(rectified=False)
    camera.connect(warmup=False)
    camera.read()

    assert set(zed_one.last_camera.requested_views) == {FakeView.LEFT_UNRECTIFIED}
    assert camera.intrinsics["fx"] == UNRECTIFIED_FX
    assert camera.intrinsics["cx"] == CAPTURE_WIDTH / 2 + 9


# --- rotation --------------------------------------------------------------------------------------


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
def test_rotation(zed_one, rotation):
    with zed_one.camera(rotation=rotation) as camera:
        frame = camera.read()

        if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
            assert (camera.width, camera.height) == (CAPTURE_HEIGHT, CAPTURE_WIDTH)
            assert frame.shape == (CAPTURE_WIDTH, CAPTURE_HEIGHT, 3)
        else:
            assert (camera.width, camera.height) == (CAPTURE_WIDTH, CAPTURE_HEIGHT)
            assert frame.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)

        # The capture resolution reported by the SDK is kept unrotated.
        assert (camera.capture_width, camera.capture_height) == (CAPTURE_WIDTH, CAPTURE_HEIGHT)


# --- SDK grab() contract ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "warning", [FakeErrorCode.CONFIGURATION_FALLBACK, FakeErrorCode.SENSOR_CONFIGURATION_CHANGED]
)
def test_grab_warning_still_delivers_a_frame(zed_one, warning):
    """A WARNING (negative code) carries a usable frame; only codes above SUCCESS are errors."""
    zed_one.patch_sl(grab_status=warning)
    with zed_one.camera() as camera:
        frame = camera.read()

        assert frame.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)


def test_corrupted_frames_are_dropped(zed_one):
    zed_one.patch_sl(grab_status=FakeErrorCode.CORRUPTED_FRAME, grab_delay_s=0.01)
    camera = zed_one.camera()
    camera.connect(warmup=False)
    try:
        time.sleep(0.1)

        assert zed_one.last_camera.grab_count > 1
        with pytest.raises(RuntimeError, match="has not captured any frames"):
            camera.read_latest()
    finally:
        camera.disconnect()


def test_sdk_verbosity_follows_the_logger(zed_one, caplog):
    with zed_one.camera():
        assert zed_one.last_camera.init_parameters.sdk_verbose == 0

    with caplog.at_level(logging.DEBUG, logger="lerobot.cameras.zed.base_zed"), zed_one.camera():
        assert zed_one.last_camera.init_parameters.sdk_verbose == 1
