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
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

pytest.importorskip("pyrealsense2")

import pyrealsense2 as rs

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"
BAG_FILE_PATH = TEST_ARTIFACTS_DIR / "test_rs.bag"

# NOTE(Steven): For some reason these tests take ~20sec in macOS but only ~2sec in Linux.


def mock_rs_config_enable_device_from_file(rs_config_instance, _sn):
    return rs_config_instance.enable_device_from_file(str(BAG_FILE_PATH), repeat_playback=True)


def mock_rs_config_enable_device_bad_file(rs_config_instance, _sn):
    return rs_config_instance.enable_device_from_file("non_existent_file.bag", repeat_playback=True)


@pytest.fixture(name="patch_realsense", autouse=True)
def fixture_patch_realsense():
    """Automatically mock pyrealsense2.config.enable_device for all tests."""
    with patch(
        "pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file
    ) as mock:
        yield mock


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    config = RealSenseCameraConfig(serial_number_or_name="042")
    _ = RealSenseCamera(config)


@pytest.mark.parametrize("option", ["exposure", "gain", "white_balance"])
def test_manual_color_option_requires_rgb(option):
    with pytest.raises(ValueError, match="use_rgb=True"):
        RealSenseCameraConfig(
            serial_number_or_name="042",
            use_rgb=False,
            use_depth=True,
            **{option: 100},
        )


def test_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)

    with RealSenseCamera(config) as camera:
        assert camera.is_connected


def test_connect_already_connected():
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)
    with RealSenseCamera(config) as camera, pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)


def test_connect_invalid_camera_path(patch_realsense):
    patch_realsense.side_effect = mock_rs_config_enable_device_bad_file
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


def test_connect_cleans_up_when_sensor_configuration_fails():
    config = RealSenseCameraConfig(serial_number_or_name="042", exposure=120)
    camera = RealSenseCamera(config)
    pipeline = MagicMock()
    pipeline.start.return_value = MagicMock()

    with (
        patch("lerobot.cameras.realsense.camera_realsense.rs.pipeline", return_value=pipeline),
        patch.object(camera, "_configure_rs_pipeline_config"),
        patch.object(camera, "_configure_capture_settings"),
        patch.object(camera, "_configure_sensor_options", side_effect=ValueError("invalid exposure")),
        pytest.raises(ValueError, match="invalid exposure"),
    ):
        camera.connect(warmup=False)

    pipeline.stop.assert_called_once_with()
    assert camera.rs_pipeline is None
    assert camera.rs_profile is None
    assert not camera.is_connected


def test_invalid_width_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=99999, height=480, fps=30)
    camera = RealSenseCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


def test_connect_cleans_up_after_warmup_failure_and_allows_retry():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30)
    camera = RealSenseCamera(config)
    read_threads = []

    def fail_warmup(*_args, **_kwargs):
        read_threads.append(camera.thread)
        raise TimeoutError("no frame")

    # Every attempt times out, so connect() exhausts its retries and reports the failure.
    with (
        patch.object(camera, "async_read", side_effect=fail_warmup),
        patch.object(camera, "_hardware_reset"),
        pytest.raises(ConnectionError, match="failed to capture frames"),
    ):
        camera.connect(warmup=True)

    assert read_threads[0] is not None
    assert not read_threads[0].is_alive()

    with patch.object(camera, "_run_warmup") as mock_warmup:
        camera.connect(warmup=False)
    mock_warmup.assert_not_called()
    assert camera.is_connected
    camera.disconnect()


def test_read():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, warmup_s=0)
    with RealSenseCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)


# TODO(Steven): Fix this test for the latest version of pyrealsense2.
@pytest.mark.skip("Skipping test: pyrealsense2 version > 2.55.1.6486")
def test_read_depth():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, use_depth=True)
    camera = RealSenseCamera(config)
    camera.connect(warmup=False)

    img = camera.read_depth(timeout_ms=2000)  # NOTE(Steven): Reading depth takes longer in CI environments.
    assert isinstance(img, np.ndarray)


# These exercise _postprocess_image directly rather than read(): the bag playback returns
# non-deterministic frames we can't compare against, and the depth read() path is skipped
# (see test_read_depth) with the current pyrealsense2 version.
def test_color_mode_conversion(img_array_factory):
    """RGB (native for RealSense) is passed through; BGR reverses the channel axis."""
    color = img_array_factory(height=3, width=4)

    outputs = {}
    for color_mode in (ColorMode.RGB, ColorMode.BGR):
        camera = RealSenseCamera(RealSenseCameraConfig(serial_number_or_name="042", color_mode=color_mode))
        camera.capture_height, camera.capture_width = color.shape[:2]
        outputs[color_mode] = camera._postprocess_image(color)

    np.testing.assert_array_equal(outputs[ColorMode.RGB], color)
    np.testing.assert_array_equal(outputs[ColorMode.BGR], color[..., ::-1])


def test_depth_frame_not_color_converted(img_array_factory):
    """Depth frames must bypass color conversion, even when a BGR color_mode is set."""
    camera = RealSenseCamera(RealSenseCameraConfig(serial_number_or_name="042", color_mode=ColorMode.BGR))
    depth = img_array_factory(height=3, width=4, channels=1, dtype=np.uint16)[..., 0]
    camera.capture_height, camera.capture_width = depth.shape

    np.testing.assert_array_equal(camera._postprocess_image(depth, depth_frame=True), depth)


def test_read_before_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


def test_disconnect():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


def test_async_read():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, warmup_s=0)

    with RealSenseCamera(config) as camera:
        img = camera.async_read()

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(img, np.ndarray)


def test_async_read_timeout():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, warmup_s=0)
    with RealSenseCamera(config) as camera, pytest.raises(TimeoutError):
        camera.async_read(timeout_ms=0)  # consumes any available frame by then
        camera.async_read(timeout_ms=0)  # request immediately another one


def test_async_read_before_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


def test_read_latest():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, warmup_s=0)
    with RealSenseCamera(config) as camera:
        img = camera.read()
        latest = camera.read_latest()

        assert isinstance(latest, np.ndarray)
        assert latest.shape == img.shape


def test_read_latest_high_frequency():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, warmup_s=0)
    with RealSenseCamera(config) as camera:
        # prime with one read to ensure frames are available
        ref = camera.read()

        for _ in range(20):
            latest = camera.read_latest()
            assert isinstance(latest, np.ndarray)
            assert latest.shape == ref.shape


def test_read_latest_before_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read_latest()


def test_read_latest_too_old():
    config = RealSenseCameraConfig(serial_number_or_name="042")

    with RealSenseCamera(config) as camera:
        # prime to ensure frames are available
        _ = camera.read()

        with pytest.raises(TimeoutError):
            _ = camera.read_latest(max_age_ms=0)  # immediately too old


def _make_mock_sensor(name: str, supported_options: set | None = None) -> MagicMock:
    """Build a fake rs.sensor that reports a name and a configurable supported-options set."""
    supported = supported_options if supported_options is not None else set()
    sensor = MagicMock()
    sensor.get_info.return_value = name
    sensor.supports.side_effect = lambda opt: opt in supported
    return sensor


def _attach_mock_color_sensor(camera: RealSenseCamera, sensor: MagicMock) -> None:
    """Wire camera.rs_profile so _get_color_sensor finds the given sensor."""
    profile = MagicMock()
    device = MagicMock()
    device.query_sensors.return_value = [sensor]
    profile.get_device.return_value = device
    camera.rs_profile = profile


def test_get_color_sensor_prefers_rgb_camera():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    rgb = _make_mock_sensor("RGB Camera")
    stereo = _make_mock_sensor("Stereo Module")
    profile = MagicMock()
    device = MagicMock()
    device.query_sensors.return_value = [stereo, rgb]
    profile.get_device.return_value = device
    camera.rs_profile = profile

    assert camera._get_color_sensor() is rgb


def test_get_color_sensor_raises_without_dedicated_rgb_module():
    """D405 has no separate RGB module; we refuse to touch the shared Stereo Module."""
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    stereo = _make_mock_sensor("Stereo Module")
    _attach_mock_color_sensor(camera, stereo)

    with pytest.raises(RuntimeError, match="dedicated 'RGB Camera' module"):
        camera._get_color_sensor()


def test_get_color_sensor_raises_with_available_sensors():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    other = _make_mock_sensor("Motion Module")
    _attach_mock_color_sensor(camera, other)

    with pytest.raises(RuntimeError, match="Motion Module"):
        camera._get_color_sensor()


def test_configure_sensor_options_skipped_when_none():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with patch.object(RealSenseCamera, "_get_color_sensor") as mock_get:
        camera._configure_sensor_options()
        mock_get.assert_not_called()


def test_configure_sensor_options_applies_all_values():
    config = RealSenseCameraConfig(serial_number_or_name="042", exposure=120, gain=64, white_balance=4600)
    camera = RealSenseCamera(config)

    sensor = _make_mock_sensor(
        "RGB Camera",
        supported_options={
            rs.option.enable_auto_exposure,
            rs.option.exposure,
            rs.option.gain,
            rs.option.enable_auto_white_balance,
            rs.option.white_balance,
        },
    )
    _attach_mock_color_sensor(camera, sensor)

    camera._configure_sensor_options()

    sensor.set_option.assert_any_call(rs.option.enable_auto_exposure, 0)
    sensor.set_option.assert_any_call(rs.option.exposure, 120)
    sensor.set_option.assert_any_call(rs.option.gain, 64)
    sensor.set_option.assert_any_call(rs.option.enable_auto_white_balance, 0)
    sensor.set_option.assert_any_call(rs.option.white_balance, 4600)


@pytest.mark.parametrize(
    ("config_field", "option", "label"),
    [
        ("exposure", rs.option.exposure, "exposure"),
        ("gain", rs.option.gain, "gain"),
        ("white_balance", rs.option.white_balance, "white balance"),
    ],
)
def test_configure_sensor_options_raises_when_requested_option_is_unsupported(config_field, option, label):
    config = RealSenseCameraConfig(serial_number_or_name="042", **{config_field: 100})
    camera = RealSenseCamera(config)

    sensor = _make_mock_sensor("RGB Camera", supported_options=set())
    _attach_mock_color_sensor(camera, sensor)

    with pytest.raises(ValueError, match=label):
        camera._configure_sensor_options()

    sensor.supports.assert_any_call(option)
    sensor.set_option.assert_not_called()


@pytest.mark.parametrize(
    ("config_field", "option", "value"),
    [
        ("exposure", rs.option.exposure, 120),
        ("gain", rs.option.gain, 64),
    ],
)
def test_configure_sensor_options_exposure_or_gain_disables_auto_exposure(config_field, option, value):
    """white_balance=None should not touch auto white balance."""
    config = RealSenseCameraConfig(serial_number_or_name="042", **{config_field: value})
    camera = RealSenseCamera(config)

    sensor = _make_mock_sensor(
        "RGB Camera",
        supported_options={rs.option.enable_auto_exposure, option},
    )
    _attach_mock_color_sensor(camera, sensor)

    camera._configure_sensor_options()

    calls = [call.args for call in sensor.set_option.call_args_list]
    assert (rs.option.enable_auto_exposure, 0) in calls
    assert (option, value) in calls
    for opt, _ in calls:
        assert opt != rs.option.enable_auto_white_balance
        assert opt != rs.option.white_balance


def test_configure_sensor_options_warns_when_auto_exposure_control_is_unsupported(caplog):
    config = RealSenseCameraConfig(serial_number_or_name="042", exposure=120)
    camera = RealSenseCamera(config)

    sensor = _make_mock_sensor("RGB Camera", supported_options={rs.option.exposure})
    _attach_mock_color_sensor(camera, sensor)

    with caplog.at_level("WARNING"):
        camera._configure_sensor_options()

    sensor.set_option.assert_called_once_with(rs.option.exposure, 120)
    assert "does not support disabling auto-exposure" in caplog.text


def test_configure_sensor_options_warns_when_auto_white_balance_control_is_unsupported(caplog):
    config = RealSenseCameraConfig(serial_number_or_name="042", white_balance=4600)
    camera = RealSenseCamera(config)

    sensor = _make_mock_sensor("RGB Camera", supported_options={rs.option.white_balance})
    _attach_mock_color_sensor(camera, sensor)

    with caplog.at_level("WARNING"):
        camera._configure_sensor_options()

    sensor.set_option.assert_called_once_with(rs.option.white_balance, 4600)
    assert "does not support disabling auto white balance" in caplog.text


def test_configure_sensor_options_out_of_range_raises_value_error():
    """set_option errors should be re-raised as ValueError with range diagnostics."""
    config = RealSenseCameraConfig(serial_number_or_name="042", exposure=999999)
    camera = RealSenseCamera(config)

    sensor = _make_mock_sensor(
        "RGB Camera",
        supported_options={rs.option.enable_auto_exposure, rs.option.exposure},
    )

    def fake_set_option(option, value):
        if option == rs.option.exposure:
            raise RuntimeError("value out of range")

    sensor.set_option.side_effect = fake_set_option

    option_range = MagicMock(min=1, max=10000, step=1, default=156)
    sensor.get_option_range.return_value = option_range

    _attach_mock_color_sensor(camera, sensor)

    with pytest.raises(ValueError, match="exposure") as exc_info:
        camera._configure_sensor_options()

    msg = str(exc_info.value)
    assert "999999" in msg
    assert "min=1" in msg
    assert "max=10000" in msg


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
    config = RealSenseCameraConfig(serial_number_or_name="042", rotation=rotation, warmup_s=0)
    with RealSenseCamera(config) as camera:
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


# --- connect() retry/state-machine tests ---


def test_connect_open_failure_propagates(patch_realsense):
    """A pipeline that cannot be opened at all fails immediately, with no hardware reset."""
    patch_realsense.side_effect = mock_rs_config_enable_device_bad_file
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with (
        patch.object(camera, "_hardware_reset") as mock_reset,
        pytest.raises(ConnectionError),
    ):
        camera.connect(warmup=False)

    mock_reset.assert_not_called()


@pytest.mark.parametrize(
    "warmup_error",
    [ConnectionError("no frames"), TimeoutError("timed out")],
    ids=["connection_error", "timeout_error"],
)
def test_connect_retries_without_reset_first(patch_realsense, warmup_error):
    """A failed warmup tears down and is first retried with a plain stop/start cycle.

    Both failure types are covered: warmup itself raises ConnectionError, while a
    stalled read surfaces as TimeoutError from async_read.
    """
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)
    camera = RealSenseCamera(config)

    with (
        patch.object(camera, "_run_warmup", side_effect=[warmup_error, None]) as mock_warmup,
        patch.object(camera, "_hardware_reset") as mock_reset,
    ):
        camera.connect(warmup=True)

    assert mock_warmup.call_count == 2
    mock_reset.assert_not_called()
    assert camera.is_connected
    camera.disconnect()


def test_connect_resets_before_final_attempt(patch_realsense):
    """When plain retries keep failing, the device is reset before the final attempt."""
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)
    camera = RealSenseCamera(config)
    real_release = camera._release_after_failed_setup
    calls = []

    def tracked_release():
        calls.append("teardown")
        real_release()

    failures = [ConnectionError("no frames")] * (RealSenseCamera._MAX_CONNECT_ATTEMPTS - 1)

    with (
        patch.object(camera, "_run_warmup", side_effect=[*failures, None]) as mock_warmup,
        patch.object(camera, "_release_after_failed_setup", side_effect=tracked_release),
        patch.object(camera, "_hardware_reset", side_effect=lambda: calls.append("reset")),
    ):
        camera.connect(warmup=True)

    assert mock_warmup.call_count == RealSenseCamera._MAX_CONNECT_ATTEMPTS
    # every failed attempt is torn down, and the reset happens after the last teardown
    assert calls == ["teardown"] * len(failures) + ["reset"]
    assert camera.is_connected
    camera.disconnect()


def test_connect_exhausts_attempts_and_cleans_up(patch_realsense):
    """When every attempt fails, connect() raises and leaves the camera fully torn down."""
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)
    camera = RealSenseCamera(config)
    max_attempts = RealSenseCamera._MAX_CONNECT_ATTEMPTS

    with (
        patch.object(camera, "_run_warmup", side_effect=ConnectionError("no frames")) as mock_warmup,
        patch.object(camera, "_hardware_reset") as mock_reset,
        pytest.raises(ConnectionError, match=f"after {max_attempts} attempts"),
    ):
        camera.connect(warmup=True)

    assert mock_warmup.call_count == max_attempts
    # the hardware reset is a last resort, used only before the final attempt
    assert mock_reset.call_count == 1
    assert not camera.is_connected
    assert camera.thread is None
    assert camera.rs_pipeline is None


def test_connect_setup_failure_after_start_tears_down_and_is_not_retried(patch_realsense):
    """A failure in _configure_capture_settings/_start_read_thread after a successful pipeline
    start must tear down and propagate unchanged, not be retried."""
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)
    camera = RealSenseCamera(config)

    with (
        patch.object(camera, "_configure_capture_settings", side_effect=RuntimeError("boom")),
        patch.object(camera, "_hardware_reset") as mock_reset,
        pytest.raises(RuntimeError, match="boom"),
    ):
        camera.connect(warmup=False)

    mock_reset.assert_not_called()
    assert not camera.is_connected
    assert camera.rs_pipeline is None


def test_connect_unexpected_warmup_exception_tears_down_and_propagates(patch_realsense):
    """An unexpected (non-Timeout/Connection) exception from warmup tears down and propagates
    unchanged, not treated as retry-worthy."""
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)
    camera = RealSenseCamera(config)

    with (
        patch.object(camera, "_run_warmup", side_effect=ValueError("unexpected")),
        patch.object(camera, "_hardware_reset") as mock_reset,
        pytest.raises(ValueError, match="unexpected"),
    ):
        camera.connect(warmup=True)

    mock_reset.assert_not_called()
    assert not camera.is_connected
    assert camera.thread is None
    assert camera.rs_pipeline is None


def test_read_loop_does_not_publish_after_stop_requested():
    """A read landing after a stop was requested must not repopulate the frame buffer.

    `_stop_read_thread` gives up joining after 2s while a hardware read can block for up
    to 10s, so a late frame would otherwise resurrect the buffer that was just cleared and
    be seen as a fresh frame by the next connect attempt.
    """
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)
    camera = RealSenseCamera(config)
    camera.stop_event = Event()

    def read_then_request_stop():
        # the stop lands while this read is in flight
        camera.stop_event.set()
        return MagicMock()

    with (
        patch.object(camera, "_read_from_hardware", side_effect=read_then_request_stop),
        patch.object(camera, "_postprocess_image", return_value=np.zeros((480, 640, 3), np.uint8)),
    ):
        camera._read_loop()

    assert camera.latest_color_frame is None
    assert camera.latest_timestamp is None
    assert not camera.new_frame_event.is_set()
