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

"""Native JSON preset integration, without a physical camera or a recorded bag."""

from unittest.mock import MagicMock, call

import draccus
import pytest

pytest.importorskip("pyrealsense2")

import pyrealsense2 as rs

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig, camera_realsense


@pytest.fixture
def connection(monkeypatch):
    def make(**kwargs):
        events = []
        camera = RealSenseCamera(RealSenseCameraConfig(serial_number_or_name="042", **kwargs))
        pipeline = MagicMock(spec=rs.pipeline)
        profile = MagicMock(spec=rs.pipeline_profile)
        pipeline.start.side_effect = lambda _config: events.append("start") or profile
        pipeline.stop.side_effect = lambda: events.append("stop")
        factory = MagicMock(return_value=pipeline)
        serializable = MagicMock(spec=rs.serializable_device)
        serializable.load_json.side_effect = lambda _text: events.append("preset")
        serializer = MagicMock(return_value=serializable)
        monkeypatch.setattr(camera_realsense.rs, "pipeline", factory)
        monkeypatch.setattr(camera_realsense.rs, "serializable_device", serializer)
        monkeypatch.setattr(camera, "_configure_rs_pipeline_config", MagicMock())
        monkeypatch.setattr(camera, "_configure_capture_settings", lambda: events.append("capture"))
        monkeypatch.setattr(camera, "_start_read_thread", lambda: events.append("thread"))
        return camera, factory, pipeline, profile, serializer, serializable, events

    return make


def test_json_config_path_round_trip():
    config = RealSenseCameraConfig(serial_number_or_name="042", json_config_path="presets/camera.json")
    restored = draccus.decode(RealSenseCameraConfig, draccus.encode(config))
    assert restored == config
    assert draccus.decode(CameraConfig, {"type": "intelrealsense", **draccus.encode(config)}) == config


def test_existing_positional_color_options_are_preserved():
    config = RealSenseCameraConfig(
        "042", ColorMode.RGB, True, False, Cv2Rotation.NO_ROTATION, 1, 120, 64, 4600
    )
    assert (config.exposure, config.gain, config.white_balance) == (120, 64, 4600)
    assert config.json_config_path is None


def test_no_json_preserves_existing_connection(connection):
    camera, _, pipeline, _, serializer, _, events = connection()
    camera.connect(warmup=False)
    serializer.assert_not_called()
    assert events == ["start", "capture", "thread"]
    camera.disconnect()
    pipeline.stop.assert_called_once_with()


def test_json_is_loaded_on_selected_device_before_capture(connection, tmp_path):
    path = tmp_path / "camera preset 日本語.json"
    text = '{"parameters": {}, "description": "depth preset"}\n'
    path.write_text(text, encoding="utf-8")
    camera, _, _, profile, serializer, serializable, events = connection(json_config_path=str(path))
    camera.connect(warmup=False)
    serializer.assert_called_once_with(profile.get_device.return_value)
    serializable.load_json.assert_called_once_with(text)
    assert events == ["start", "preset", "capture", "thread"]
    camera.disconnect()


def test_explicit_color_settings_override_json(connection, tmp_path):
    path = tmp_path / "preset.json"
    path.write_text("{}")
    camera, _, _, profile, _, _, events = connection(json_config_path=str(path), exposure=120)
    sensor = MagicMock(spec=rs.sensor)
    sensor.get_info.return_value = "RGB Camera"
    sensor.supports.return_value = True
    sensor.set_option.side_effect = lambda _option, _value: events.append("color")
    profile.get_device.return_value.query_sensors.return_value = [sensor]
    camera.connect(warmup=False)
    assert sensor.set_option.call_args_list == [
        call(rs.option.enable_auto_exposure, 0),
        call(rs.option.exposure, 120),
    ]
    assert events == ["start", "preset", "capture", "color", "color", "thread"]
    camera.disconnect()


@pytest.mark.parametrize("directory", [False, True], ids=["missing", "directory"])
def test_unreadable_json_fails_before_pipeline_creation(connection, tmp_path, directory):
    path = tmp_path / "preset.json"
    if directory:
        path.mkdir()
    camera, factory, _, _, serializer, _, _ = connection(json_config_path=str(path))
    with pytest.raises(OSError):
        camera.connect(warmup=False)
    factory.assert_not_called()
    serializer.assert_not_called()
    assert not camera.is_connected


@pytest.mark.parametrize("stage", ["device", "preset"])
def test_sdk_rejection_cleans_up_and_preserves_error(connection, tmp_path, stage):
    path = tmp_path / "bad-preset.json"
    path.write_text("not a valid preset")
    camera, factory, pipeline, _, serializer, serializable, events = connection(json_config_path=str(path))
    original = RuntimeError("unsupported device" if stage == "device" else "invalid JSON")
    if stage == "device":
        serializer.side_effect = original
    else:
        serializable.load_json.side_effect = original
    with pytest.raises(RuntimeError, match="bad-preset.json") as caught:
        camera.connect()
    assert caught.value.__cause__ is original
    assert events == ["start", "stop"]
    factory.assert_called_once_with()
    pipeline.stop.assert_called_once_with()
    assert camera.rs_pipeline is None and camera.rs_profile is None
    assert camera.thread is None and not camera.is_connected


def test_warmup_retry_reapplies_json(connection, monkeypatch, tmp_path):
    path = tmp_path / "preset.json"
    path.write_text("{}")
    camera, _, pipeline, _, _, serializable, _ = connection(json_config_path=str(path))
    monkeypatch.setattr(camera, "_run_warmup", MagicMock(side_effect=[TimeoutError("no frames"), None]))
    reset = MagicMock()
    monkeypatch.setattr(camera, "_hardware_reset", reset)
    camera.connect()
    assert camera.is_connected
    assert serializable.load_json.call_args_list == [call("{}"), call("{}")]
    assert pipeline.start.call_count == 2
    reset.assert_not_called()
    camera.disconnect()


def test_reconnect_reads_updated_preset(connection, tmp_path):
    path = tmp_path / "preset.json"
    path.write_text('{"preset": "first"}')
    camera, _, _, _, _, serializable, _ = connection(json_config_path=str(path))
    camera.connect(warmup=False)
    camera.disconnect()
    path.write_text('{"preset": "second"}')
    camera.connect(warmup=False)
    assert serializable.load_json.call_args_list == [
        call('{"preset": "first"}'),
        call('{"preset": "second"}'),
    ]
    camera.disconnect()
