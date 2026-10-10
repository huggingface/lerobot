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

import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.scripts import lerobot_monitor_temperature as cli
from lerobot.utils import motor_temperature as thermal

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@pytest.fixture
def setup(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(thermal.time, "monotonic", lambda: clock[0])
    bus = MagicMock(motors=dict.fromkeys(NAMES), is_connected=True)
    bus.sync_read.return_value = dict.fromkeys(NAMES, 30)
    monitor = thermal.MotorTemperatureMonitor(bus)
    monkeypatch.setattr(monitor, "_warn", MagicMock())
    yield monitor, bus, clock
    monitor.notifications.shutdown(wait=True)


def test_stages_once_per_motor_and_rearm_after_cooling(setup):
    monitor, bus, clock = setup
    for second, value in enumerate((49, 50, 50, 54.9, 55, 55, 64.9, 65, 65, 70, 55, 55, 65, 65, 49, 50, 50)):
        clock[0] = second
        bus.sync_read.return_value = {**dict.fromkeys(NAMES, 30), "shoulder_pan": value}
        assert len(monitor.poll()) == 6
    messages = [call.args[0] for call in monitor._warn.call_args_list]
    assert len(messages) == 4
    assert "Heating" in messages[0] and "50 C" in messages[0]
    assert "Overheating" in messages[1] and "55 C" in messages[1]
    assert "cool down before continuing" in messages[2] and "65 C" in messages[2]
    assert "50 C" in messages[3]
    bus.connect.assert_not_called()
    bus.disconnect.assert_not_called()
    bus.write.assert_not_called()


def test_new_hot_motor_and_direct_jump_to_critical(setup):
    monitor, bus, clock = setup
    bus.sync_read.return_value = dict.fromkeys(NAMES, 30)
    monitor.poll()
    clock[0] = 1
    bus.sync_read.return_value = {**bus.sync_read.return_value, "gripper": 65}
    monitor.poll()
    assert monitor._warn.call_count == 0
    clock[0] = 2
    monitor.poll()
    assert monitor._warn.call_count == 1
    assert "gripper=65" in monitor._warn.call_args.args[0]
    assert "cool down" in monitor._warn.call_args.args[0]


def test_impossible_temperature_is_rejected_as_unavailable(setup):
    monitor, bus, clock = setup
    bus.sync_read.return_value = {**dict.fromkeys(NAMES, 30), "shoulder_pan": 128}

    assert monitor.poll() == {}
    assert monitor.unavailable
    assert monitor._warn.call_count == 1
    assert "UNAVAILABLE" in monitor._warn.call_args.args[0]
    assert monitor.warned == {}

    clock[0] = 1
    bus.sync_read.return_value = dict.fromkeys(NAMES, 30)
    assert len(monitor.poll()) == 6
    assert not monitor.unavailable


@pytest.mark.parametrize("failure", [ConnectionError("No reply"), {}, dict.fromkeys(NAMES, float("nan"))])
def test_retry_at_one_hz_without_treating_missing_readings_as_cool(setup, failure):
    monitor, bus, clock = setup
    if isinstance(failure, Exception):
        bus.sync_read.side_effect = failure
    else:
        bus.sync_read.return_value = failure
    monitor.warned["shoulder_pan"] = 3
    for value in (0, 0.1, 0.9, 1, 1.5, 2):
        clock[0] = value
        assert monitor.poll() == {}
    assert bus.sync_read.call_count == 3
    assert monitor._warn.call_count == 1
    assert monitor.warned["shoulder_pan"] == 3
    bus.sync_read.side_effect = None
    bus.sync_read.return_value = dict.fromkeys(NAMES, 30)
    clock[0] = 3
    assert len(monitor.poll()) == 6
    assert not monitor.unavailable and not monitor.warned


def test_slow_desktop_notification_does_not_block_sensor_reads(setup, monkeypatch):
    monitor, bus, clock = setup
    started, release = threading.Event(), threading.Event()

    def blocked_notification(_):
        started.set()
        release.wait(3)

    monkeypatch.setattr(thermal, "_desktop_notification", blocked_notification)
    monkeypatch.setattr(
        monitor, "_warn", lambda message: monitor.notifications.submit(thermal._desktop_notification, message)
    )
    try:
        bus.sync_read.return_value = dict.fromkeys(NAMES, 50)
        monitor.poll()
        clock[0] = 1
        monitor.poll()
        assert started.wait(1)
        clock[0] = 2
        assert len(monitor.poll()) == 6
        assert bus.sync_read.call_count == 3
    finally:
        release.set()


@pytest.mark.parametrize("mode", ["rerun", "foxglove"])
def test_standalone_closes_without_changing_torque(setup, monkeypatch, mode):
    monitor, bus, _ = setup
    robot = MagicMock(bus=bus)
    monkeypatch.setattr(cli, "SO101Follower", lambda _: robot)
    monkeypatch.setattr(cli, "MotorTemperatureMonitor", lambda _: monitor)
    init, log, shutdown = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(cli, "init_visualization", init)
    monkeypatch.setattr(cli, "log_visualization_data", log)
    monkeypatch.setattr(cli, "shutdown_visualization", shutdown)
    monkeypatch.setattr(cli.time, "sleep", MagicMock(side_effect=KeyboardInterrupt))
    cli.monitor_temperature("unused", "test", mode)
    init.assert_called_once_with(mode, session_name="lerobot_temperature")
    assert len(log.call_args.kwargs["observation"]) == 6
    bus.disconnect.assert_called_once_with(disable_torque=False)
    robot.connect.assert_not_called()
    shutdown.assert_called_once_with(mode)


@pytest.mark.parametrize("loop_name", ["record", "teleop"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("display", [False, True])
def test_control_loops_share_bus_and_keep_temperature_out_of_dataset(
    setup, monkeypatch, loop_name, enabled, display
):
    from lerobot.scripts import lerobot_record as record, lerobot_teleoperate as teleop

    monitor, bus, clock = setup
    bus.sync_read.side_effect = [dict.fromkeys(NAMES, value) for value in (50, 55, 65)]
    monkeypatch.setattr(thermal.time, "perf_counter", lambda: clock[0])

    class Timer:
        def tick(self):
            pass

        def section(self, _):
            return contextlib.nullcontext()

        def wait(self):
            clock[0] = round(clock[0] + 0.1, 1)

        def log_run_summary(self):
            pass

    robot = MagicMock(action_features={"joint.pos": float})
    robot.get_observation.return_value = {"joint.pos": 1.0}
    leader = MagicMock(spec=record.Teleoperator)
    leader.get_action.return_value = {"joint.pos": 1.0}
    kwargs = {
        "robot": robot,
        "teleop": leader,
        "fps": 10,
        "display_data": display,
        "teleop_action_processor": lambda pair: pair[0],
        "robot_action_processor": lambda pair: pair[0],
        "robot_observation_processor": lambda obs: obs,
        "temperature_monitor": monitor if enabled else None,
    }
    module = record if loop_name == "record" else teleop
    visualize = MagicMock()
    monkeypatch.setattr(module, "log_visualization_data", visualize)
    if loop_name == "record":
        dataset = SimpleNamespace(fps=10, features={}, add_frame=MagicMock())
        monkeypatch.setattr(
            record, "build_dataset_frame", lambda features, values, prefix: {prefix: dict(values)}
        )
        record.record_loop(
            events={"exit_early": False}, control_time_s=2.2, timer=Timer(), dataset=dataset, **kwargs
        )
        assert dataset.add_frame.call_count == 22
        assert all("temperature" not in str(call.args[0]) for call in dataset.add_frame.call_args_list)
    else:
        monkeypatch.setattr(teleop, "CycleTimer", lambda *args, **kwargs: Timer())
        teleop.teleop_loop(duration=2.2, **kwargs)
    assert bus.sync_read.call_count == (3 if enabled else 0)
    assert visualize.call_count == (22 if display else (3 if enabled else 0))
    if enabled:
        temperature_frames = [
            call.kwargs["observation"]
            for call in visualize.call_args_list
            if "shoulder_pan.temperature" in call.kwargs["observation"]
        ]
        assert len(temperature_frames) == 3
        assert all(len(frame) == (7 if display else 6) for frame in temperature_frames)
        assert monitor._warn.call_count == 0
    bus.connect.assert_not_called()
    bus.write.assert_not_called()


def test_factory_borrows_existing_bus(setup):
    _, bus, _ = setup
    robot = MagicMock(spec=thermal.SOFollower, bus=bus)
    monitor = thermal.make_temperature_monitor(robot)
    assert monitor.bus is bus
    monitor.close()
    robot.connect.assert_not_called()


def test_factory_rejects_unsupported_robot():
    with pytest.raises(ValueError, match="SO-100/SO-101"):
        thermal.make_temperature_monitor(MagicMock())


def test_notification_failure_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(thermal.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(thermal.subprocess, "run", MagicMock(side_effect=OSError("notification denied")))
    thermal._desktop_notification("Heating")
    assert "notification denied" in caplog.text


def test_cli_defaults_to_live_rerun(monkeypatch):
    monkeypatch.setattr("sys.argv", ["lerobot-monitor-temperature", "--port", "unused"])
    run = MagicMock()
    monkeypatch.setattr(cli, "monitor_temperature", run)
    cli.main()
    run.assert_called_once_with("unused", "temperature_monitor", "rerun")
