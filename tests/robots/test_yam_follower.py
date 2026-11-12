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

import math
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.yam_follower import (
    YamFollower,
    YamFollowerRobotConfig,
    mit_control,
    yam_follower as robot_module,
)
from lerobot.robots.yam_follower.config_yam_follower import (
    JOINT_LIMITS_RAD,
    MOTOR_NAMES,
    YAM_FEATURE_NAMES,
)
from lerobot.utils import DeviceNotConnectedError
from tests.mocks.mock_yam import (
    attach_bus,
    calibrate_gripper,
    make_ready,
    mark_running,
    mock_bus,
    mock_hardware,
    raw_states,
)


def make_robot(tmp_path, *, gripper=(0.1, 6.1), **overrides):
    config = {
        "id": "test",
        "calibration_dir": tmp_path,
        "port": "can0",
        "gravity_compensation": False,
        # The control numbers below use 0.3 rad/s and 0.15 rad.
        "max_joint_speed_deg_s": math.degrees(0.3),
        "max_tracking_error_deg": math.degrees(0.15),
        **overrides,
    }
    robot = YamFollower(YamFollowerRobotConfig(**config))
    if gripper is not None:
        calibrate_gripper(robot, *gripper)
    return robot


@pytest.fixture(autouse=True)
def without_hardware_packages(monkeypatch):
    """Build robots without the yam extra; their hardware is mocked."""
    monkeypatch.setattr(robot_module, "require_package", lambda *a, **kw: None)


@pytest.fixture
def robot(tmp_path):
    return make_robot(tmp_path)


def params(robot):
    return robot_module._yam_arm_params(robot.config)


def gripper_params(robot):
    return robot_module._yam_gripper_params(robot.config, robot.calibration["gripper"])


def ready(robot):
    make_ready(robot)
    return dict.fromkeys(robot.action_features, 0.0)


def test_gripper_params_come_from_the_config_and_i2rt(robot):
    grip = gripper_params(robot)
    assert (grip.kp, grip.kd, grip.max_speed) == (20.0, 0.5, 12.0)
    assert (grip.force_limit_n, grip.finger_stroke_m) == (50.0, 0.096)


@pytest.mark.parametrize("friction_compensation", [False, True])
def test_arm_params_come_from_the_config_and_i2rt(robot, friction_compensation):
    robot.config.friction_compensation = friction_compensation
    arm = params(robot)
    np.testing.assert_allclose(arm.kp, [80, 80, 80, 10, 10, 10])
    np.testing.assert_allclose(arm.kd, [5, 5, 5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(arm.gravity_factors, [1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
    assert arm.max_gravity_torque == 25.0
    assert (arm.max_joint_speed, arm.max_tracking_error) == pytest.approx((0.3, 0.15))  # from degrees
    np.testing.assert_allclose(arm.float_kd, [0.1, 0.1, 0.1, 0.3, 0.05, 0.05])
    expected = [0.3, 0.3, 0.3, 0.06, 0.06, 0.06] if friction_compensation else [0.0] * 6
    np.testing.assert_allclose(arm.coulomb_friction, expected)


@pytest.mark.parametrize("use_degrees", [True, False])
def test_actions_are_converted_and_clipped_to_the_limits(robot, use_degrees):
    robot.config.use_degrees = use_degrees
    action = ready(robot)
    action["joint_1.pos"] = math.degrees(0.2) if use_degrees else 0.2
    action["joint_2.pos"] = -1.0  # below the 0 lower limit in either unit
    action["gripper.pos"] = 100.4 if use_degrees else 1.004
    sent = robot.send_action(action)
    np.testing.assert_allclose(robot.controller._target, [0.2, 0, 0, 0, 0, 0, 1.0])
    assert sent == pytest.approx({**action, "joint_2.pos": 0.0, "gripper.pos": 100.0 if use_degrees else 1.0})


def test_measured_position_has_wider_tolerance_than_action(robot):
    states = raw_states(gripper=6.4)
    states.position[0] = JOINT_LIMITS_RAD[0][0] - 0.08
    bus = attach_bus(robot, mock_bus(states))
    measured = mit_control._read_joint_state(bus, params(robot), gripper_params(robot)).position
    assert measured[0] == pytest.approx(states.position[0])
    assert measured[6] == 1.0
    states.position[0] = JOINT_LIMITS_RAD[0][0] - 0.2
    with pytest.raises(ValueError, match="outside"):
        mit_control._read_joint_state(bus, params(robot), gripper_params(robot))


@pytest.fixture
def motorbridge(monkeypatch):
    """Mocked MotorBridge and python-can, to drive a real `_YamMotorsBus`."""
    controller = MagicMock()
    controller.add_damiao_motor.side_effect = lambda *args: MagicMock()
    monkeypatch.setattr(
        robot_module, "MotorBridgeController", MagicMock(return_value=controller), raising=False
    )
    monkeypatch.setattr(robot_module, "can", SimpleNamespace(Bus=MagicMock()), raising=False)
    return controller


def test_motorbridge_mapping_and_mit_radians(motorbridge):
    bus = robot_module._YamMotorsBus("can0", feedback_timeout_s=0.2)
    bus.connect()
    robot_module.MotorBridgeController.assert_called_once_with(channel="can0")
    monitor = robot_module.can.Bus
    # The monitor only receives the seven feedback IDs, without error frames.
    assert [f["can_id"] for f in monitor.call_args.kwargs["can_filters"]] == list(range(17, 24))
    assert monitor.call_args.kwargs["ignore_rx_error_frames"] is True
    assert [call.args for call in motorbridge.add_damiao_motor.call_args_list] == [
        (i + 1, i + 17, "4340" if i < 3 else "4310") for i in range(7)
    ]
    motors = dict(bus.motors)
    bus.enable_torque_at(np.array([0.2, 0.5, 0.3, 0, 0, 0, 6.1]))
    motors["joint_1"].send_mit.assert_called_once_with(0.2, 0, 0, 0, 0)
    motors["gripper"].send_mit.assert_called_once_with(6.1, 0, 0, 0, 0)
    bus.send_mit(
        "joint_1",
        mit_control.MitCommand(position=0.1, velocity=0.2, kp=3.0, kd=4.0, torque=5.0),
    )
    assert motors["joint_1"].send_mit.call_args_list[-1].args == (0.1, 0.2, 3.0, 4.0, 5.0)
    motorbridge.enable_all.assert_called_once()
    bus.disconnect()
    motorbridge.disable_all.assert_called_once()
    motorbridge.close.assert_called_once()


def test_torque_failures_raise_and_still_close_the_bus(motorbridge):
    bus = robot_module._YamMotorsBus("can0", feedback_timeout_s=0.2)
    bus.connect()
    motorbridge.enable_all.side_effect = RuntimeError("CAN enable failed")
    with pytest.raises(RuntimeError, match="enable failed"):
        bus.enable_torque_at(np.zeros(7))
    assert bus.enabled  # some motors may be on, so disconnect still disables them
    motorbridge.disable_all.side_effect = RuntimeError("CAN write failed")
    with pytest.raises(RuntimeError, match="write failed"):
        bus.disconnect()
    motorbridge.disable_all.assert_called_once()
    motorbridge.close.assert_called_once()
    robot_module.can.Bus.return_value.shutdown.assert_called_once()


def frame(motor_id, age=0, status=1):
    """A feedback frame received `age` seconds before `received()` hands it out."""
    return SimpleNamespace(arbitration_id=motor_id + 0x10, data=bytes([status << 4]) + bytes(7), age=age)


def received(msg):
    """Stamp `msg` when the monitor returns it, as SocketCAN does, so a slow test can't age it."""
    if msg is not None:
        msg.timestamp = time.time() - msg.age
    return msg


def feedback_bus(frames):
    """An opened bus whose python-can monitor replays ``frames``."""
    bus = robot_module._YamMotorsBus("can0", feedback_timeout_s=0.2)
    bus.motorbridge_controller = MagicMock()
    queued = deque(frames)
    bus.monitor = SimpleNamespace(recv=lambda timeout: received(queued.popleft()) if queued else None)
    bus.motors = {name: MagicMock() for name in MOTOR_NAMES}
    for motor in bus.motors.values():
        motor.get_state.return_value = SimpleNamespace(pos=0.1, vel=0.2, torq=0.3)
    return bus


@pytest.mark.parametrize(
    ("frames", "enabled", "error"),
    [
        ([frame(i) for i in range(1, 7)], False, "stale"),
        ([frame(i, age=1) for i in range(1, 8)], False, "stale"),
        ([frame(i, status=13 if i == 3 else 1) for i in range(1, 8)], False, "fault 0xd"),
        ([frame(i, status=0 if i == 3 else 1) for i in range(1, 8)], True, "motor 3 unexpectedly disabled"),
    ],
    ids=["a motor never answered", "old frames", "motor fault", "motor disabled after the enable grace"],
)
def test_bad_feedback_fails(frames, enabled, error):
    # MotorBridge's cached states can't show this; the monitor's frames do.
    bus = feedback_bus(frames)
    bus.enabled, bus._enabled_at = enabled, time.monotonic() - 1
    with pytest.raises(ConnectionError, match=error):
        bus.read_states()


@pytest.mark.parametrize(
    ("frames", "enabled", "enabled_ago_s"),
    [
        ([frame(i) for i in range(1, 8)], True, 1.0),
        ([frame(i, status=0) for i in range(1, 8)], False, 1.0),
        ([frame(i, status=0) for i in range(1, 8)], True, 0.0),
        ([frame(3, status=0), *[frame(i) for i in range(1, 8)]], True, 1.0),
    ],
    ids=["fresh", "torque disabled", "torque just enabled", "newer enabled frame"],
)
def test_fresh_feedback_is_accepted(frames, enabled, enabled_ago_s):
    bus = feedback_bus(frames)
    bus.enabled, bus._enabled_at = enabled, time.monotonic() - enabled_ago_s
    states = bus.read_states()
    np.testing.assert_allclose(states.position, np.full(7, 0.1))
    np.testing.assert_allclose(states.velocity, np.full(7, 0.2))
    np.testing.assert_allclose(states.torque, np.full(7, 0.3))


def test_gravity_model_loads_from_the_hub_before_torque(robot, monkeypatch):
    events = []
    model_path = "/cache/yam/v1/linear_4310.urdf"
    download = MagicMock(return_value=model_path)
    gravity_model = MagicMock(side_effect=lambda *args, **kwargs: events.append("model") or MagicMock())
    monkeypatch.setattr(robot_module, "hf_hub_download", download)
    monkeypatch.setattr(robot_module, "GravityCompensation", gravity_model)
    robot.config.gravity_compensation = True
    mock_hardware(robot, monkeypatch)
    robot.bus.enable_torque_at.side_effect = lambda hold: events.append("enable")
    robot.connect()
    robot.disconnect()
    assert events == ["model", "enable"]
    download.assert_called_once_with("lerobot/robot-urdfs", "yam/v1/linear_4310.urdf", repo_type="dataset")
    gravity_model.assert_called_once_with(
        model_path, tuple(f"joint{i}" for i in range(1, 7)), base_frame="base"
    )


def test_features_are_positions_with_optional_velocity_and_torque_per_motor(robot, tmp_path):
    assert tuple(robot.action_features) == YAM_FEATURE_NAMES
    assert robot.observation_features == robot.action_features
    names = tuple(make_robot(tmp_path, use_velocity_and_torque=True).observation_features)
    assert len(names) == 21
    assert names[:3] == ("joint_1.pos", "joint_1.vel", "joint_1.torque")
    assert names[-3:] == ("gripper.pos", "gripper.vel", "gripper.torque")


@pytest.mark.parametrize("use_degrees", [True, False])
def test_observation_uses_public_units(tmp_path, monkeypatch, use_degrees):
    robot = make_robot(tmp_path, use_degrees=use_degrees, use_velocity_and_torque=True)
    ready(robot)
    robot.controller.state = mit_control.JointState(
        position=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5]),
        velocity=np.array([1.0, 0, 0, 0, 0, 0, 0.25]),
        torque=np.array([2.0, 0, 0, 0, 0, 0, 0.3]),
    )
    observation = robot.get_observation()
    scale, gripper_scale = (math.degrees(1), 100.0) if use_degrees else (1.0, 1.0)
    assert observation["joint_1.pos"] == pytest.approx(0.1 * scale)
    assert observation["joint_1.vel"] == pytest.approx(1.0 * scale)
    assert observation["joint_1.torque"] == pytest.approx(2.0)
    assert observation["gripper.pos"] == pytest.approx(0.5 * gripper_scale)
    assert observation["gripper.vel"] == pytest.approx(0.25 * gripper_scale)


def test_single_arm_rejects_partial_action(robot):
    ready(robot)
    with pytest.raises(KeyError, match="joint_2.pos"):
        robot.send_action({"joint_1.pos": 0.2})


def test_connect_keeps_torque_off_until_the_gripper_is_calibrated(tmp_path, monkeypatch):
    robot = make_robot(tmp_path, gripper=None)
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)
    robot.bus.enable_torque_at.assert_not_called()
    assert not robot.controller.active
    robot.disconnect()


def test_connect_calibrates_missing_gripper_limits(tmp_path, monkeypatch):
    robot = make_robot(tmp_path, gripper=None)
    mock_hardware(robot, monkeypatch)

    def prompt(text):
        robot.bus.read_states.return_value.position[6] = 6.1 if "open" in text else 0.1
        return ""

    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(robot_module.time, "sleep", lambda _: None)
    robot.connect()

    assert robot.is_calibrated
    robot.bus.connect.assert_called_once()  # control starts on the calibration connection
    robot.bus.enable_torque_at.assert_called_once()
    robot.disconnect()


def test_connect_enables_torque_only_after_mit_mode(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.connect()
    calls = [
        name for name, *_ in robot.bus.mock_calls if name in ("connect", "set_mit_mode", "enable_torque_at")
    ]
    assert calls == ["connect", "set_mit_mode", "enable_torque_at"]
    assert robot.controller.active
    robot.disconnect()


def test_initial_position_check_uses_degrees(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    states = raw_states()
    states.position[0] = math.radians(robot.config.initial_tolerance_deg + 1)
    robot.bus.read_states.return_value = states
    with pytest.raises(ValueError, match="initial_position_deg"):
        robot.connect()
    robot.bus.enable_torque_at.assert_not_called()


@pytest.mark.parametrize(("timeout", "detail"), [(0, "joint_2=off"), (8001, "joint_2=400.05 ms")])
def test_connect_requires_a_prompt_motor_can_timeout(robot, monkeypatch, timeout, detail):
    mock_hardware(robot, monkeypatch)
    robot.bus.can_timeouts.return_value = {**dict.fromkeys(MOTOR_NAMES, 8000), "joint_2": timeout}
    with pytest.raises(ValueError, match="unsafe motor CAN timeout") as exc_info:
        robot.connect()
    robot.bus.enable_torque_at.assert_not_called()
    assert detail in str(exc_info.value)


def test_bus_reads_the_can_timeout_register():
    bus = robot_module._YamMotorsBus("can0", feedback_timeout_s=0.2)
    bus.motors = {name: MagicMock() for name in MOTOR_NAMES}
    for motor in bus.motors.values():
        motor.get_register_u32.return_value = 8000
    assert bus.can_timeouts() == dict.fromkeys(MOTOR_NAMES, 8000)
    bus.motors["gripper"].get_register_u32.assert_called_once_with(9)


def test_float_mode_requires_gravity_compensation():
    with pytest.raises(ValueError, match="needs gravity_compensation"):
        YamFollowerRobotConfig(port="can0", idle_mode="float", gravity_compensation=False)


def test_configure_needs_a_connection_and_a_stopped_controller(robot):
    with pytest.raises(DeviceNotConnectedError):
        robot.configure()
    robot._connected = True
    mark_running(robot.controller)
    with pytest.raises(RuntimeError, match="before the controller starts"):
        robot.configure()


def test_disconnect_can_retry_after_controller_join_timeout(robot):
    bus = attach_bus(robot, mock_bus())
    robot._connected = True
    mark_running(robot.controller)
    robot.controller.stop = MagicMock(side_effect=RuntimeError("YAM controller did not stop"))
    with pytest.raises(RuntimeError, match="did not stop"):
        robot.disconnect()
    bus.disable_torque.assert_called_once()
    bus.disconnect.assert_not_called()
    assert robot.is_connected
    robot.controller.stop.side_effect = None
    robot.disconnect()
    bus.disconnect.assert_called_once()
    assert not robot.is_connected


def test_disconnect_reports_a_failed_torque_off_after_closing_can(robot):
    bus = attach_bus(robot, mock_bus())
    robot._connected = True
    bus.disable_torque.side_effect = RuntimeError("CAN write failed")
    bus.disconnect.side_effect = RuntimeError("CAN write failed")
    with pytest.raises(RuntimeError, match="CAN write failed"):
        robot.disconnect()
    bus.disconnect.assert_called_once()
    assert not robot.is_connected


def measure_gripper(robot, monkeypatch, *, closed=0.1, opened=6.1, answers=()):
    """Run calibrate() with the gripper motor at `closed` and `opened` radians at the prompts."""
    pending = list(answers)
    prompts = []

    def prompt(text):
        prompts.append(text)
        if pending:
            return pending.pop(0)
        robot.bus.read_states.return_value = raw_states(opened if "open" in text else closed)
        return ""

    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(robot_module.time, "sleep", lambda _: None)
    robot.calibrate()
    return prompts


def test_single_arm_calibration_preserves_factory_zeros(tmp_path, monkeypatch):
    robot = make_robot(tmp_path, gripper=None)
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)
    prompts = measure_gripper(robot, monkeypatch)
    calibration = robot.calibration["gripper"]
    assert (calibration.homing_offset, calibration.drive_mode) == (0, 0)
    # Motor angles of the gripper limits, in whole degrees.
    assert (calibration.range_min, calibration.range_max) == (6, 350)
    assert robot.controller._gripper_params.open == pytest.approx(6.1, abs=0.01)
    assert all("[test]" in text for text in prompts)
    assert robot.calibration_fpath.is_file()
    robot.bus.enable_torque_at.assert_not_called()
    robot.disconnect()


def saved_calibration_robot(tmp_path, monkeypatch, *, closed=0.1, opened=6.1):
    """Calibrate once, then build a new robot that only knows the saved file."""
    first = make_robot(tmp_path, gripper=None)
    mock_hardware(first, monkeypatch)
    first.connect(calibrate=False)
    measure_gripper(first, monkeypatch, closed=closed, opened=opened)
    first.disconnect()
    return make_robot(tmp_path, gripper=None)


@pytest.mark.parametrize("closed,opened", [(0.1, 6.1), (6.1, 0.1)])
def test_saved_calibration_loads_either_polarity(tmp_path, monkeypatch, closed, opened):
    robot = saved_calibration_robot(tmp_path, monkeypatch, closed=closed, opened=opened)
    assert robot.is_calibrated
    assert robot.controller._gripper_params.closed == pytest.approx(math.radians(round(math.degrees(closed))))
    assert robot.controller._gripper_params.open == pytest.approx(math.radians(round(math.degrees(opened))))


def test_existing_calibration_is_kept_unless_recalibrating(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)
    robot.bus.read_states.reset_mock()
    prompts = measure_gripper(robot, monkeypatch, answers=[""])
    assert len(prompts) == 1 and "ENTER" in prompts[0]
    robot.bus.read_states.assert_not_called()
    assert robot.controller.active  # the saved calibration is kept, and so is control

    prompts = measure_gripper(robot, monkeypatch, opened=5.1, answers=["c"])
    assert len(prompts) == 4 and "Support the arm" in prompts[1]  # reuse question, support, closed, open
    assert not robot.controller.active
    robot.bus.disable_torque.assert_called_once()
    assert robot.controller._gripper_params.open == pytest.approx(math.radians(round(math.degrees(5.1))))
    robot.disconnect()


@pytest.mark.parametrize("opened", [0.2, 11.0])  # strokes outside 0.5 to 10 rad
def test_calibration_refuses_an_implausible_stroke(robot, monkeypatch, opened):
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)
    with pytest.raises(ValueError, match="Gripper stroke"):
        measure_gripper(robot, monkeypatch, opened=opened, answers=["c"])
    assert not robot.calibration_fpath.is_file()
    robot.disconnect()


def test_controller_waits_for_asynchronous_feedback(robot):
    attach_bus(robot, feedback_bus([]))
    replies = deque([None, *[frame(i) for i in range(1, 8)], None])

    def receive(timeout):
        value = replies.popleft() if replies else None
        if not replies:
            robot.controller._stop.set()
        return received(value)

    robot.bus.monitor.recv = receive
    zeros = np.zeros(7)
    robot.controller._set_initial_state(mit_control.JointState(position=zeros, velocity=zeros, torque=zeros))
    started_at = robot.controller._updated_at
    robot.controller._run()
    assert robot.controller._failure is None
    assert robot.controller._updated_at > started_at
