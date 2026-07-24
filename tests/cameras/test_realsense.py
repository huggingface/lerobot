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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.cameras.configs import Cv2Rotation
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


def test_get_color_sensor_falls_back_to_stereo_module():
    """D405 has no separate RGB module; color comes from Stereo Module."""
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    stereo = _make_mock_sensor("Stereo Module")
    _attach_mock_color_sensor(camera, stereo)

    assert camera._get_color_sensor() is stereo


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
