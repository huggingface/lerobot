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
# pytest tests/cameras/test_zed.py::test_connect
# ```

# NOTE: `pyzed` ships with the ZED SDK and is not installable from PyPI, so it cannot be a CI
# dependency. `lerobot.cameras.zed` imports it behind a try/except, which lets these tests run
# everywhere: `sl` is replaced by the stand-in module from `tests/mocks/mock_pyzed.py`. No ZED
# hardware and no ZED SDK are required.

import inspect
import logging
import time

import draccus
import numpy as np
import pytest

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.zed import ZedCamera, ZedCameraConfig, base_zed, camera_zed
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from tests.mocks.mock_pyzed import (
    BGRA_PIXEL,
    CAPTURE_FPS,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DEFAULT_SERIAL,
    DEPTH_MM,
    INVALID_DEPTH,
    RECTIFIED_FX,
    UNRECTIFIED_FX,
    FakeCamera,
    FakeDepthMode,
    FakeErrorCode,
    FakeMeasure,
    FakeResolution,
    FakeTimeReference,
    FakeUnit,
    FakeView,
    ZedTestHarness,
)


@pytest.fixture(name="zed")
def fixture_zed(monkeypatch):
    harness = ZedTestHarness(monkeypatch, (base_zed, camera_zed), ZedCamera, ZedCameraConfig)
    yield harness
    harness.cleanup()


def wait_for_thread_exit(camera, timeout_s=3.0):
    deadline = time.perf_counter() + timeout_s
    while camera.thread is not None and camera.thread.is_alive() and time.perf_counter() < deadline:
        time.sleep(0.01)


# --- configuration (no ZED SDK involved at all) ---------------------------------------------------


def test_config_defaults():
    config = ZedCameraConfig()

    assert config.serial_number is None
    assert config.resolution == "AUTO"
    assert config.rectified is True
    assert config.depth_mode == "NEURAL"
    assert config.color_mode is ColorMode.RGB
    assert config.use_rgb is True
    assert config.use_depth is False
    assert config.rotation is Cv2Rotation.NO_ROTATION
    assert config.warmup_s == 1
    assert config.fps is None
    assert config.width is None
    assert config.height is None


def test_config_requires_rgb_or_depth():
    with pytest.raises(ValueError, match="use_rgb"):
        ZedCameraConfig(use_rgb=False, use_depth=False)


def test_config_depth_only_is_allowed():
    config = ZedCameraConfig(use_rgb=False, use_depth=True)

    assert config.use_rgb is False
    assert config.use_depth is True


def test_config_depth_with_unrectified_is_rejected():
    """Depth is computed from the rectified pair, so it cannot align with a raw image."""
    with pytest.raises(ValueError, match="cannot be combined with `use_depth=True`"):
        ZedCameraConfig(rectified=False, use_depth=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 30},
        {"width": 1280},
        {"height": 720},
        {"fps": 30, "width": 1280},
        {"width": 1280, "height": 720},
    ],
    ids=["fps", "width", "height", "fps_width", "width_height"],
)
def test_config_partial_capture_settings_raise(kwargs):
    with pytest.raises(ValueError, match="either all of them"):
        ZedCameraConfig(**kwargs)


def test_config_full_capture_settings_are_allowed():
    config = ZedCameraConfig(fps=30, width=1280, height=720)

    assert (config.fps, config.width, config.height) == (30, 1280, 720)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("rgb", ColorMode.RGB),
        ("bgr", ColorMode.BGR),
        (ColorMode.BGR, ColorMode.BGR),
    ],
    ids=["str_rgb", "str_bgr", "enum_bgr"],
)
def test_config_color_mode_coercion(value, expected):
    assert ZedCameraConfig(color_mode=value).color_mode is expected


def test_config_invalid_color_mode_raises():
    with pytest.raises(ValueError, match="color_mode"):
        ZedCameraConfig(color_mode="rgba")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, Cv2Rotation.NO_ROTATION),
        (90, Cv2Rotation.ROTATE_90),
        (180, Cv2Rotation.ROTATE_180),
        (-90, Cv2Rotation.ROTATE_270),
        (Cv2Rotation.ROTATE_270, Cv2Rotation.ROTATE_270),
    ],
    ids=["int_0", "int_90", "int_180", "int_-90", "enum_270"],
)
def test_config_rotation_coercion(value, expected):
    assert ZedCameraConfig(rotation=value).rotation is expected


@pytest.mark.parametrize("value", [45, 270], ids=["45", "270"])
def test_config_invalid_rotation_raises(value):
    # `Cv2Rotation` spells a counter-clockwise quarter turn -90, not 270
    with pytest.raises(ValueError, match="rotation"):
        ZedCameraConfig(rotation=value)


def test_config_type_is_registered():
    assert CameraConfig.get_known_choices()["zed"] is ZedCameraConfig
    assert ZedCameraConfig().type == "zed"


def test_make_cameras_from_configs_builds_zed_backends(zed):
    # `make_cameras_from_configs` has no explicit branch for zed: it resolves through the generic
    # `make_device_from_device_class` fallback, so this is what `lerobot-record` actually relies on.
    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.cameras.zed import ZedOneCamera, ZedOneCameraConfig

    cameras = make_cameras_from_configs(
        {"stereo": ZedCameraConfig(serial_number=DEFAULT_SERIAL), "mono": ZedOneCameraConfig()}
    )

    assert isinstance(cameras["stereo"], ZedCamera)
    assert isinstance(cameras["mono"], ZedOneCamera)


def test_config_decodes_from_dict():
    config = draccus.decode(
        CameraConfig,
        {
            "type": "zed",
            "serial_number": DEFAULT_SERIAL,
            "resolution": "HD720",
            "depth_mode": "NEURAL_PLUS",
            "color_mode": "bgr",
            "use_depth": True,
            "rotation": 90,
        },
    )

    assert isinstance(config, ZedCameraConfig)
    assert config.serial_number == DEFAULT_SERIAL
    assert config.resolution == "HD720"
    assert config.depth_mode == "NEURAL_PLUS"
    assert config.color_mode is ColorMode.BGR
    assert config.use_depth is True
    assert config.rotation is Cv2Rotation.ROTATE_90


# --- missing ZED SDK ------------------------------------------------------------------------------


def test_missing_pyzed_raises_import_error(monkeypatch):
    # the constructor's availability check lives in the shared base module
    for module in (base_zed, camera_zed):
        monkeypatch.setattr(module, "sl", None)
    monkeypatch.setattr(base_zed, "_pyzed_import_error", None)

    with pytest.raises(ImportError, match="get_python_api.py"):
        ZedCamera(ZedCameraConfig(serial_number=DEFAULT_SERIAL))


def test_broken_pyzed_import_reports_the_cause(monkeypatch):
    # pyzed present on disk but its shared libraries fail to load: neither the constructor nor
    # find_cameras() may fall through to an AttributeError on `sl is None`
    for module in (base_zed, camera_zed):
        monkeypatch.setattr(module, "sl", None)
    monkeypatch.setattr(
        base_zed, "_pyzed_import_error", OSError("libsl_zed.so: cannot open shared object file")
    )

    with pytest.raises(ImportError, match="libsl_zed.so") as excinfo:
        ZedCamera(ZedCameraConfig(serial_number=DEFAULT_SERIAL))
    assert isinstance(excinfo.value.__cause__, OSError)
    with pytest.raises(ImportError, match="failed to import"):
        ZedCamera.find_cameras()


def test_missing_pyzed_raises_import_error_in_find_cameras(monkeypatch):
    for module in (base_zed, camera_zed):
        monkeypatch.setattr(module, "sl", None)
    monkeypatch.setattr(base_zed, "_pyzed_import_error", None)

    with pytest.raises(ImportError, match="ZED SDK"):
        ZedCamera.find_cameras()


# --- connect / disconnect ------------------------------------------------------------------------


def test_abc_implementation(zed):
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    _ = zed.camera()


def test_str(zed):
    assert str(zed.camera()) == f"ZedCamera({DEFAULT_SERIAL})"
    assert str(ZedCamera(ZedCameraConfig())) == "ZedCamera(auto)"


def test_find_cameras(zed):
    assert ZedCamera.find_cameras() == [
        {
            "name": "ZED X",
            "type": "zed",
            "id": DEFAULT_SERIAL,
            "serial_number": DEFAULT_SERIAL,
        }
    ]


def test_connect(zed):
    with zed.camera() as camera:
        assert camera.is_connected
        assert isinstance(zed.last_camera, FakeCamera)  # the stereo API, not sl.CameraOne
        assert (camera.width, camera.height) == (CAPTURE_WIDTH, CAPTURE_HEIGHT)
        assert camera.fps == CAPTURE_FPS
        assert camera.intrinsics == {
            "fx": RECTIFIED_FX,
            "fy": 701.0,
            "cx": CAPTURE_WIDTH / 2,
            "cy": CAPTURE_HEIGHT / 2,
        }


def test_connect_init_parameters(zed):
    camera = zed.camera(resolution="HD720", use_depth=True, depth_mode="NEURAL_PLUS")
    camera.connect(warmup=False)

    init = zed.last_camera.init_parameters
    assert init.camera_resolution is FakeResolution.HD720
    assert init.depth_mode is FakeDepthMode.NEURAL_PLUS
    # left at the SDK default: DEPTH_U16_MM is millimetres whatever coordinate_units says
    assert init.coordinate_units is FakeUnit.MILLIMETER  # SDK default; DEPTH_U16_MM does not depend on it
    assert init.serial_number == DEFAULT_SERIAL


def test_connect_forwards_configured_fps(zed):
    camera = zed.camera(fps=CAPTURE_FPS, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
    camera.connect(warmup=False)

    assert zed.last_camera.init_parameters.camera_fps == CAPTURE_FPS
    assert (camera.width, camera.height, camera.fps) == (CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS)


def test_connect_rejects_stream_settings_the_camera_ignores(zed):
    """A requested fps/width/height the camera does not honour must fail loudly at connect.

    Robots publish observation shapes from the config before connecting, so silently adopting
    the camera's own mode would surface later as a confusing frame-shape mismatch.
    """
    camera = zed.camera(fps=CAPTURE_FPS // 2, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)

    with pytest.raises(ValueError, match="did not honour"):
        camera.connect(warmup=False)

    assert not camera.is_connected
    assert zed.last_camera.close_count == 1  # no camera handle left open behind the failure
    # the requested settings survive for a retry; nothing auto-detected leaks out of the failure
    assert (camera.width, camera.height, camera.fps) == (CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS // 2)
    assert camera.intrinsics is None


def test_connect_without_depth_disables_depth_mode(zed):
    camera = zed.camera(use_depth=False, depth_mode="NEURAL_PLUS")
    camera.connect(warmup=False)

    assert zed.last_camera.init_parameters.depth_mode is FakeDepthMode.NONE


def test_connect_without_serial_number_opens_first_device(zed):
    camera = ZedCamera(ZedCameraConfig(warmup_s=0))
    camera.connect(warmup=False)
    try:
        assert zed.last_camera.init_parameters.serial_number is None
    finally:
        camera.disconnect()


def test_connect_unknown_resolution_raises(zed):
    camera = zed.camera(resolution="HD4K")

    with pytest.raises(ValueError, match="unknown resolution") as exc_info:
        camera.connect(warmup=False)

    assert "HD720" in str(exc_info.value)  # the error lists the valid values
    assert "LAST" not in str(exc_info.value)  # ... but not the enum sentinel
    assert not camera.is_connected


@pytest.mark.parametrize("mode", ["NEURAL", "NEURAL_PLUS", "NEURAL_LIGHT"])
def test_documented_depth_modes_resolve(zed, mode):
    """The modes the config docstring advertises are the non-deprecated ones the SDK exposes."""
    camera = zed.camera(use_depth=True, depth_mode=mode)
    camera.connect(warmup=False)

    assert zed.last_camera.init_parameters.depth_mode is getattr(FakeDepthMode, mode)


def test_connect_unknown_depth_mode_raises(zed):
    camera = zed.camera(use_depth=True, depth_mode="MAGIC")

    with pytest.raises(ValueError, match="unknown depth_mode") as exc_info:
        camera.connect(warmup=False)

    assert "NEURAL_PLUS" in str(exc_info.value)
    assert "LAST" not in str(exc_info.value)
    assert not camera.is_connected


def test_connect_failure_raises_connection_error(zed):
    zed.patch_sl(open_status=FakeErrorCode.CAMERA_NOT_DETECTED)
    camera = zed.camera()

    with pytest.raises(ConnectionError, match="Failed to open"):
        camera.connect(warmup=False)

    assert not camera.is_connected
    assert camera.cam is None
    assert zed.last_camera.close_count == 1


def test_connect_with_an_sdk_warning_succeeds_and_logs_it(zed, caplog):
    """Negative `open()` codes (e.g. CONFIGURATION_FALLBACK) are warnings: the camera is usable."""
    zed.patch_sl(open_status=FakeErrorCode.CONFIGURATION_FALLBACK)
    camera = zed.camera()

    with caplog.at_level(logging.WARNING, logger="lerobot.cameras.zed.base_zed"):
        camera.connect(warmup=False)

    assert camera.is_connected
    assert any(str(FakeErrorCode.CONFIGURATION_FALLBACK) in r.getMessage() for r in caplog.records)


def test_failed_configure_closes_the_camera(zed, monkeypatch):
    """Any failure after open() must release the SDK handle, whatever raised."""
    camera = zed.camera()

    def boom(self, conf):
        raise AttributeError("SDK build without calibration_parameters_raw")

    monkeypatch.setattr(type(camera), "_read_calibration", boom)

    with pytest.raises(AttributeError):
        camera.connect(warmup=False)

    assert not camera.is_connected
    assert camera.thread is None
    assert zed.last_camera.close_count == 1


def test_connect_warmup_waits_and_primes_a_frame(zed):
    camera = ZedCamera(zed.config(warmup_s=0.2))
    start = time.perf_counter()
    camera.connect()
    try:
        assert time.perf_counter() - start >= 0.2
        assert camera.latest_color_frame is not None  # a frame is ready before connect() returns
    finally:
        camera.disconnect()


def test_connect_warmup_uses_depth_when_rgb_is_off(zed):
    camera = ZedCamera(zed.config(warmup_s=0.1, use_rgb=False, use_depth=True))
    camera.connect()
    try:
        assert camera.latest_depth_frame is not None
        assert camera.latest_color_frame is None
    finally:
        camera.disconnect()


def test_connect_skips_warmup_when_disabled(zed):
    camera = ZedCamera(zed.config(warmup_s=5))
    start = time.perf_counter()
    camera.connect(warmup=False)
    try:
        assert time.perf_counter() - start < 1.0
    finally:
        camera.disconnect()


def test_connect_cleans_up_after_warmup_failure_and_allows_retry(zed):
    """A camera that grabs nothing during warmup must be fully released, then reconnectable."""
    zed.patch_sl(grab_status=FakeErrorCode.FAILURE, grab_delay_s=0.05)
    camera = ZedCamera(zed.config(warmup_s=0.2))

    with pytest.raises(TimeoutError):
        camera.connect(warmup=True)

    assert not camera.is_connected
    assert camera.cam is None
    assert camera.thread is None
    assert zed.last_camera.close_count == 1

    zed.patch_sl()  # the camera comes back
    camera.connect(warmup=True)
    try:
        assert camera.is_connected
        assert camera.latest_color_frame is not None
    finally:
        camera.disconnect()


def test_connect_already_connected(zed):
    with zed.camera() as camera, pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)


def test_is_connected_follows_the_sdk_camera(zed):
    with zed.camera() as camera:
        assert camera.is_connected
        zed.last_camera.close()  # camera drops off the bus behind our back
        assert not camera.is_connected


def test_disconnect(zed):
    camera = zed.camera()
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected
    assert camera.cam is None
    assert camera.thread is None
    assert zed.last_camera.close_count == 1


def test_disconnect_clears_published_state(zed):
    camera = zed.camera(use_depth=True)
    camera.connect(warmup=False)
    camera.read()

    camera.disconnect()

    assert camera.latest_color_frame is None
    assert camera.latest_depth_frame is None
    assert camera.latest_timestamp is None
    assert camera.latest_hw_timestamp_ns is None
    assert not camera.new_frame_event.is_set()


def test_reconnect_starts_from_a_clean_frame_event(zed):
    zed.patch_sl(grab_delay_s=0.3)
    camera = zed.camera()
    camera.connect(warmup=False)
    camera.read()
    camera.disconnect()

    camera.connect(warmup=False)
    try:
        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=1)  # nothing from the previous session may leak through
    finally:
        camera.disconnect()


def test_disconnect_before_connect(zed):
    camera = zed.camera()

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


def test_disconnect_twice_raises(zed):
    camera = zed.camera()
    camera.connect(warmup=False)
    camera.disconnect()

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


# --- reads ---------------------------------------------------------------------------------------


def test_read(zed):
    with zed.camera() as camera:
        frame = camera.read()

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)
        assert frame.dtype == np.uint8


def test_read_returns_rgb_by_default(zed):
    with zed.camera() as camera:
        frame = camera.read()

        blue, green, red, _ = BGRA_PIXEL
        assert tuple(frame[0, 0]) == (red, green, blue)


def test_read_bgr_color_mode_keeps_channel_order(zed):
    with zed.camera(color_mode=ColorMode.BGR) as camera:
        frame = camera.read()

        assert tuple(frame[0, 0]) == BGRA_PIXEL[:3]


def test_read_color_mode_argument_is_deprecated_and_ignored(zed, caplog):
    """Matches the other backends: `read(color_mode=...)` warns and returns the configured order."""
    with caplog.at_level(logging.WARNING, logger="lerobot.cameras.zed.base_zed"), zed.camera() as camera:
        frame = camera.read(color_mode=ColorMode.BGR)

        blue, green, red, _ = BGRA_PIXEL
        assert tuple(frame[0, 0]) == (red, green, blue)
    assert any("color_mode parameter is deprecated" in r.getMessage() for r in caplog.records)


def test_read_timeout_ms_argument_is_deprecated(zed, caplog):
    with caplog.at_level(logging.WARNING, logger="lerobot.cameras.zed.base_zed"), zed.camera() as camera:
        camera.read(timeout_ms=500)
    assert any("timeout_ms parameter is deprecated" in r.getMessage() for r in caplog.records)


def test_read_does_not_serve_a_cached_frame(zed):
    """`read()` owes the caller the *next* grab, not the one already buffered."""
    zed.patch_sl(grab_delay_s=0.3)
    with zed.camera() as camera:
        first = camera.read()
        start = time.perf_counter()
        second = camera.read()

        assert time.perf_counter() - start >= 0.1  # had to wait for the read thread
        assert second is not first


def test_read_depth(zed):
    with zed.camera(use_depth=True) as camera:
        depth = camera.read_depth()

        assert isinstance(depth, np.ndarray)
        assert depth.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 1)
        assert depth.dtype == np.uint16
        assert depth[10, 10, 0] == DEPTH_MM
        # invalid measurements reach the caller as 0, the sentinel LeRobot's depth path expects
        assert depth[0, 0, 0] == INVALID_DEPTH
        assert set(zed.last_camera.requested_measures) == {FakeMeasure.DEPTH_U16_MM}


def test_depth_is_not_retrieved_when_disabled(zed):
    with zed.camera(use_depth=False) as camera:
        camera.read()

        assert zed.last_camera.requested_measures == []


def test_color_is_not_retrieved_when_disabled(zed):
    with zed.camera(use_rgb=False, use_depth=True) as camera:
        camera.read_depth()

        assert zed.last_camera.requested_views == []
        assert camera.latest_color_frame is None


def test_async_read(zed):
    with zed.camera() as camera:
        frame = camera.async_read(timeout_ms=2000)

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)


def test_async_read_depth(zed):
    with zed.camera(use_depth=True) as camera:
        depth = camera.async_read_depth(timeout_ms=2000)

        assert depth.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 1)
        assert depth.dtype == np.uint16


def test_async_read_consumes_the_frame_it_returns(zed):
    zed.patch_sl(grab_delay_s=0.3)
    with zed.camera() as camera:
        camera.async_read(timeout_ms=2000)

        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=1)  # that frame is spent; nothing new has arrived


def test_read_latest(zed):
    with zed.camera(use_depth=True) as camera:
        frame = camera.read()
        latest = camera.read_latest()
        latest_depth = camera.read_latest_depth()

        assert isinstance(latest, np.ndarray)
        assert latest.shape == frame.shape
        assert latest_depth.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 1)
        assert latest_depth.dtype == np.uint16


def test_read_latest_does_not_consume_the_frame(zed):
    """`read_latest()` + `read_latest_depth()` back to back must come from the same grab."""
    zed.patch_sl(grab_delay_s=0.3)
    with zed.camera(use_depth=True) as camera:
        camera.read()
        color = camera.read_latest()
        depth = camera.read_latest_depth()

        assert color is camera.latest_color_frame
        assert depth is camera.latest_depth_frame


def test_read_latest_too_old(zed):
    with zed.camera() as camera:
        _ = camera.read()  # prime to ensure frames are available

        with pytest.raises(TimeoutError):
            _ = camera.read_latest(max_age_ms=0)  # immediately too old


def test_read_latest_without_any_frame_raises(zed):
    # A camera that never grabs successfully: the read thread stays alive but publishes nothing.
    zed.patch_sl(grab_status=FakeErrorCode.FAILURE, grab_delay_s=0.05)
    camera = zed.camera()
    camera.connect(warmup=False)

    with pytest.raises(RuntimeError, match="has not captured any frames"):
        camera.read_latest()


@pytest.mark.parametrize(
    "method_name",
    ["read", "read_depth", "async_read", "async_read_depth", "read_latest", "read_latest_depth"],
)
def test_read_before_connect(zed, method_name):
    camera = zed.camera(use_depth=True)

    with pytest.raises(DeviceNotConnectedError):
        _ = getattr(camera, method_name)()


@pytest.mark.parametrize(
    "method_name", ["read", "async_read", "read_latest"], ids=["read", "async_read", "read_latest"]
)
def test_color_reads_raise_without_rgb(zed, method_name):
    with (
        zed.camera(use_rgb=False, use_depth=True) as camera,
        pytest.raises(RuntimeError, match="use_rgb=False"),
    ):
        _ = getattr(camera, method_name)()


@pytest.mark.parametrize(
    "method_name",
    ["read_depth", "async_read_depth", "read_latest_depth"],
    ids=["read_depth", "async_read_depth", "read_latest_depth"],
)
def test_depth_reads_raise_without_depth(zed, method_name):
    with zed.camera(use_depth=False) as camera, pytest.raises(RuntimeError, match="use_depth=False"):
        _ = getattr(camera, method_name)()


def test_documented_read_defaults(zed):
    camera = zed.camera()

    assert inspect.signature(camera.async_read).parameters["timeout_ms"].default == 200
    assert inspect.signature(camera.async_read_depth).parameters["timeout_ms"].default == 200
    assert inspect.signature(camera.read_latest).parameters["max_age_ms"].default == 500
    assert inspect.signature(camera.read_latest_depth).parameters["max_age_ms"].default == 500


def test_latest_hw_timestamp_ns(zed):
    with zed.camera() as camera:
        assert camera.latest_hw_timestamp_ns is None

        _ = camera.read()

        assert isinstance(camera.latest_hw_timestamp_ns, int)
        assert camera.latest_hw_timestamp_ns > 0
        # the sensor's own image clock, not the host's
        assert set(zed.last_camera.time_references) == {FakeTimeReference.IMAGE}


# --- frame post-processing -----------------------------------------------------------------------


def test_postprocess_depth(zed):
    camera = zed.camera(use_depth=True)
    camera.capture_width, camera.capture_height = 3, 2
    # DEPTH_U16_MM is already uint16 millimetres, so post-processing only adds the channel axis.
    raw_mm = np.array([[0, 0, 0], [0, 1234, 65000]], dtype=np.uint16)

    depth = camera._postprocess_depth(raw_mm)

    assert depth.dtype == np.uint16
    assert depth.shape == (2, 3, 1)
    assert depth[..., 0].tolist() == [[0, 0, 0], [0, 1234, 65000]]


def test_postprocess_depth_copies_off_the_reused_mat(zed):
    """The SDK hands back a view on a buffer it overwrites on the next grab()."""
    camera = zed.camera(use_depth=True)
    camera.capture_width, camera.capture_height = 3, 2
    raw_mm = np.full((2, 3), 1500, dtype=np.uint16)

    depth = camera._postprocess_depth(raw_mm)
    raw_mm[:] = 42  # simulate the SDK reusing the Mat for the next frame

    assert depth[..., 0].tolist() == [[1500, 1500, 1500], [1500, 1500, 1500]]


def test_postprocess_color_rgb(zed):
    camera = zed.camera()
    camera.capture_width, camera.capture_height = 2, 2
    bgra = np.zeros((2, 2, 4), dtype=np.uint8)
    bgra[:, :] = BGRA_PIXEL

    frame = camera._postprocess_color(bgra)

    blue, green, red, _ = BGRA_PIXEL
    assert frame.shape == (2, 2, 3)
    assert frame.dtype == np.uint8
    assert tuple(frame[0, 0]) == (red, green, blue)


def test_postprocess_color_bgr(zed):
    camera = zed.camera(color_mode=ColorMode.BGR)
    camera.capture_width, camera.capture_height = 2, 2
    bgra = np.zeros((2, 2, 4), dtype=np.uint8)
    bgra[:, :] = BGRA_PIXEL

    frame = camera._postprocess_color(bgra)

    assert frame.shape == (2, 2, 3)
    assert tuple(frame[0, 0]) == BGRA_PIXEL[:3]


def test_postprocess_rejects_frames_of_the_wrong_size(zed):
    camera = zed.camera(use_depth=True)
    camera.capture_width, camera.capture_height = 4, 2

    with pytest.raises(RuntimeError, match="do not match"):
        camera._postprocess_color(np.zeros((2, 3, 4), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="do not match"):
        camera._postprocess_depth(np.zeros((3, 4), dtype=np.uint16))


def test_postprocess_color_rejects_non_bgra_frames(zed):
    camera = zed.camera()
    camera.capture_width, camera.capture_height = 2, 2

    with pytest.raises(RuntimeError, match="BGRA"):
        camera._postprocess_color(np.zeros((2, 2, 3), dtype=np.uint8))


def test_read_loop_drops_frames_that_do_not_match_the_camera_mode(zed, caplog):
    """A frame of the wrong size never reaches the caller; the read thread logs and publishes nothing."""
    zed.patch_sl(served_size=(CAPTURE_WIDTH // 2, CAPTURE_HEIGHT // 2), grab_delay_s=0.05)
    camera = zed.camera()
    with caplog.at_level(logging.WARNING, logger="lerobot.cameras.zed.base_zed"):
        camera.connect(warmup=False)
        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=200)

    assert camera.thread.is_alive()
    assert any("do not match" in r.getMessage() for r in caplog.records)


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
def test_rotation(zed, rotation):
    with zed.camera(rotation=rotation, use_depth=True) as camera:
        frame = camera.read()
        depth = camera.read_depth()

        if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
            assert (camera.width, camera.height) == (CAPTURE_HEIGHT, CAPTURE_WIDTH)
            assert frame.shape == (CAPTURE_WIDTH, CAPTURE_HEIGHT, 3)
            assert depth.shape == (CAPTURE_WIDTH, CAPTURE_HEIGHT, 1)
        else:
            assert (camera.width, camera.height) == (CAPTURE_WIDTH, CAPTURE_HEIGHT)
            assert frame.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)
            assert depth.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 1)

        # The capture resolution reported by the SDK is kept unrotated.
        assert (camera.capture_width, camera.capture_height) == (CAPTURE_WIDTH, CAPTURE_HEIGHT)

        # intrinsics describe the frames handed out: fake calibration fx=700, fy=701, principal
        # point at the sensor centre. Check by mapping a known sensor pixel through the rotation.
        fx, fy, cx, cy = (camera.intrinsics[k] for k in ("fx", "fy", "cx", "cy"))
        w0, h0 = CAPTURE_WIDTH, CAPTURE_HEIGHT
        expected = {
            Cv2Rotation.NO_ROTATION: (700.0, 701.0, w0 / 2, h0 / 2),
            Cv2Rotation.ROTATE_90: (701.0, 700.0, h0 - 1 - h0 / 2, w0 / 2),
            Cv2Rotation.ROTATE_180: (700.0, 701.0, w0 - 1 - w0 / 2, h0 - 1 - h0 / 2),
            Cv2Rotation.ROTATE_270: (701.0, 700.0, h0 / 2, w0 - 1 - w0 / 2),
        }[rotation]
        assert (fx, fy, cx, cy) == expected
        assert 0 <= cx < camera.width and 0 <= cy < camera.height


def test_rotated_frames_are_actually_rotated(zed):
    """A marked corner pixel must land where a clockwise quarter turn puts it."""
    zed.patch_sl(width=4, height=2)
    camera = zed.camera(rotation=Cv2Rotation.ROTATE_90)
    camera.connect(warmup=False)
    try:
        fake = zed.last_camera
        fake._color[:] = 0
        fake._color[0, 0] = (0, 0, 255, 255)  # top-left, red in BGRA
        frame = camera.read()

        assert frame.shape == (4, 2, 3)
        assert tuple(frame[0, -1]) == (255, 0, 0)  # top-left → top-right after 90° clockwise
    finally:
        camera.disconnect()


# --- rectified vs unrectified --------------------------------------------------------------------


def test_rectified_by_default_retrieves_the_rectified_view(zed):
    camera = zed.camera()
    camera.connect(warmup=False)
    camera.read()

    assert zed.last_camera.requested_views
    assert set(zed.last_camera.requested_views) == {FakeView.LEFT}


def test_unrectified_retrieves_the_raw_view_and_its_own_intrinsics(zed):
    """`rectified=False` must switch both the retrieved view and the reported calibration."""
    camera = zed.camera(rectified=False)
    camera.connect(warmup=False)
    camera.read()

    assert set(zed.last_camera.requested_views) == {FakeView.LEFT_UNRECTIFIED}
    # raw calibration, not the rectified one (the fake gives them different values)
    assert camera.intrinsics["fx"] == UNRECTIFIED_FX
    assert camera.intrinsics["cx"] == CAPTURE_WIDTH / 2 + 9


# --- SDK grab() contract ---------------------------------------------------------------------------


def test_grab_receives_runtime_parameters(zed):
    with zed.camera() as camera:
        camera.read()

        assert zed.last_camera.grab_runtime_parameters
        assert all(p is camera.runtime for p in zed.last_camera.grab_runtime_parameters)


@pytest.mark.parametrize(
    "warning",
    [
        FakeErrorCode.POTENTIAL_CALIBRATION_ISSUE,
        FakeErrorCode.CONFIGURATION_FALLBACK,
        FakeErrorCode.SENSOR_CONFIGURATION_CHANGED,
    ],
)
def test_grab_warning_still_delivers_a_frame(zed, warning, caplog):
    """`grab()` below SUCCESS is a WARNING that still carries a frame (see the ZED SDK samples).

    Treating these as failures used to trip the consecutive-failure guard and kill the read
    thread, e.g. on POTENTIAL_CALIBRATION_ISSUE from a low-texture scene.
    """
    zed.patch_sl(grab_status=warning)
    with caplog.at_level(logging.WARNING, logger="lerobot.cameras.zed.base_zed"), zed.camera() as camera:
        frame = camera.read()
        camera.read()

        assert frame.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)
    warnings = [r for r in caplog.records if "grab() warning" in r.getMessage()]
    assert len(warnings) == 1, "one warning per streak, not one per frame"
    assert str(warning) in warnings[0].getMessage()


@pytest.mark.parametrize("code", [FakeErrorCode.CORRUPTED_FRAME, FakeErrorCode.CAMERA_REBOOTING])
def test_grab_warnings_without_an_image_drop_the_frame(zed, code, caplog):
    """CORRUPTED_FRAME (green/purple image) and CAMERA_REBOOTING (no new image) must not be published."""
    zed.patch_sl(grab_status=code, grab_delay_s=0.01)
    camera = zed.camera()
    with caplog.at_level(logging.WARNING, logger="lerobot.cameras.zed.base_zed"):
        camera.connect(warmup=False)
        try:
            time.sleep(0.1)
            fake = zed.last_camera

            assert fake.grab_count > 1
            assert not fake.requested_views  # nothing retrieved behind a frameless warning
            with pytest.raises(RuntimeError, match="has not captured any frames"):
                camera.read_latest()
            assert camera.thread.is_alive()  # a bad streak is not a read failure

            fake.grab_status = FakeErrorCode.SUCCESS
            frame = camera.read()

            assert frame.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)
        finally:
            camera.disconnect()
    assert sum("grab() warning" in r.getMessage() for r in caplog.records) == 1


def test_grab_error_is_still_an_error(zed, caplog):
    zed.patch_sl(grab_status=FakeErrorCode.CAMERA_NOT_DETECTED, grab_delay_s=0.05)
    camera = zed.camera()
    with caplog.at_level(logging.WARNING, logger="lerobot.cameras.zed.base_zed"):
        camera.connect(warmup=False)

        with pytest.raises(RuntimeError, match="has not captured any frames"):
            camera.read_latest()
        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=100)

    assert any("grab() failed" in r.getMessage() for r in caplog.records)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_read_thread_gives_up_after_repeated_failures(zed):
    """A hopeless camera must end the read thread, not log forever."""
    zed.patch_sl(grab_status=FakeErrorCode.CAMERA_NOT_DETECTED, grab_delay_s=0.001)
    camera = zed.camera()
    camera.connect(warmup=False)

    wait_for_thread_exit(camera)

    assert not camera.thread.is_alive()
    with pytest.raises(RuntimeError, match="read thread is not running"):
        camera.read()


def test_read_thread_recovers_from_transient_failures(zed):
    """Isolated failures are logged and the counter resets on the next good frame."""
    zed.patch_sl(grab_delay_s=0.01)
    camera = zed.camera()
    camera.connect(warmup=False)
    try:
        fake = zed.last_camera
        fake.grab_status = FakeErrorCode.CAMERA_NOT_DETECTED
        time.sleep(0.05)  # a few failures, fewer than the limit
        fake.grab_status = FakeErrorCode.SUCCESS

        frame = camera.read()

        assert frame.shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)
        assert camera.thread.is_alive()
    finally:
        camera.disconnect()


def test_image_validity_check_is_enabled(zed):
    """pyzed leaves it off; without it the SDK never reports CORRUPTED_FRAME."""
    with zed.camera():
        assert zed.last_camera.init_parameters.enable_image_validity_check == 1


def test_sdk_verbosity_follows_the_logger(zed, caplog):
    """The SDK prints its own INFO lines to stdout; keep them for DEBUG runs only."""
    with zed.camera():
        assert zed.last_camera.init_parameters.sdk_verbose == 0

    with caplog.at_level(logging.DEBUG, logger="lerobot.cameras.zed.base_zed"), zed.camera():
        assert zed.last_camera.init_parameters.sdk_verbose == 1
