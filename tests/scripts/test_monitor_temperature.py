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

from unittest.mock import MagicMock

import pytest

from lerobot.scripts import lerobot_monitor_temperature as monitor


@pytest.fixture
def bus(monkeypatch):
    robot = MagicMock()
    robot.bus.motors = {
        name: object()
        for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    }
    robot.bus.is_connected = True
    monkeypatch.setattr(monitor, "SO101Follower", lambda _: robot)
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    return robot.bus


@pytest.mark.parametrize("mode", [None, "rerun", "foxglove"])
def test_per_motor_crossings_and_read_only_cleanup(bus, monkeypatch, mode):
    cool = dict.fromkeys(bus.motors, 30)
    first = {**cool, "shoulder_pan": 52}
    second = {**first, "gripper": 53}
    bus.sync_read.side_effect = [cool, first, first, second, cool, first, KeyboardInterrupt()]
    notify, init, log, shutdown = (MagicMock() for _ in range(4))
    for name, mock in zip(
        ("send_notification", "init_visualization", "log_visualization_data", "shutdown_visualization"),
        (notify, init, log, shutdown),
        strict=True,
    ):
        monkeypatch.setattr(monitor, name, mock)
    monitor.monitor_temperature("unused", "test", 52, 1, mode)
    assert [call.args[0].split(":")[0] for call in notify.call_args_list] == [
        "shoulder_pan",
        "gripper",
        "shoulder_pan",
    ]
    bus.connect.assert_called_once_with()
    bus.disconnect.assert_called_once_with(disable_torque=False)
    assert all(
        call.args == ("Present_Temperature",) and call.kwargs == {"normalize": False}
        for call in bus.sync_read.call_args_list
    )
    bus.write.assert_not_called()
    if mode:
        init.assert_called_once_with(mode, session_name="lerobot_temperature")
        assert log.call_count == 6
        assert log.call_args.kwargs["observation"]["shoulder_pan.temperature"] == 52.0
        shutdown.assert_called_once_with(mode)
    else:
        init.assert_not_called()


@pytest.mark.parametrize("value", [None, float("nan"), 151])
def test_invalid_telemetry_warns_and_closes(bus, monkeypatch, value):
    notify = MagicMock()
    monkeypatch.setattr(monitor, "send_notification", notify)
    bus.sync_read.return_value = (
        {} if value is None else {**dict.fromkeys(bus.motors, 30), "shoulder_pan": value}
    )
    with pytest.raises(ValueError, match="telemetry"):
        monitor.monitor_temperature("unused", "test", 52, 1)
    notify.assert_called_once()
    bus.disconnect.assert_called_once_with(disable_torque=False)


def test_read_failure_warns_and_closes(bus, monkeypatch):
    monkeypatch.setattr(monitor, "send_notification", MagicMock())
    bus.sync_read.side_effect = ConnectionError("No reply")
    with pytest.raises(ConnectionError):
        monitor.monitor_temperature("unused", "test", 52, 1)
    bus.disconnect.assert_called_once_with(disable_torque=False)


def test_notification_failure_does_not_stop_monitoring(monkeypatch, caplog):
    monkeypatch.setattr(monitor.platform, "system", lambda: "Darwin")
    command = MagicMock(side_effect=OSError("notification denied"))
    monkeypatch.setattr(monitor.subprocess, "run", command)
    monitor.send_notification('Motor "hot": 52 C')
    assert command.call_args.args[0][-2] == 'Motor "hot": 52 C'
    assert "notification denied" in caplog.text


@pytest.mark.parametrize("option", [["--interval", "nan"], ["--interval", "0"], ["--threshold", "0"]])
def test_cli_rejects_invalid_options(monkeypatch, option):
    monkeypatch.setattr("sys.argv", ["lerobot-monitor-temperature", "--port", "unused", *option])
    with pytest.raises(SystemExit) as exc:
        monitor.main()
    assert exc.value.code == 2


def test_cli_defaults_to_52_degrees(monkeypatch):
    monkeypatch.setattr("sys.argv", ["lerobot-monitor-temperature", "--port", "unused"])
    run = MagicMock()
    monkeypatch.setattr(monitor, "monitor_temperature", run)
    monitor.main()
    run.assert_called_once_with("unused", "temperature_monitor", 52, 1, None)
