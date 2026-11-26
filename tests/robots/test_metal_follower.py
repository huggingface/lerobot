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

import logging
import math
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.motors.damiao.tables import MotorType
from lerobot.robots.metal_follower import (
    MetalFollower,
    MetalFollowerRobotConfig,
    metal_follower as robot_module,
    mit_bus,
)
from lerobot.robots.metal_follower.config_metal_follower import (
    JOINT_LIMITS_DEG,
    METAL_FEATURE_NAMES,
    MOTOR_NAMES,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.yam_follower import mit_control
from tests.mocks.mock_metal import (
    FakeCan,
    attach_bus,
    fake_can,
    make_ready,
    mark_running,
    mock_bus,
    mock_hardware,
    raw_states,
    without_hardware_packages,
)

_built: list[MetalFollower] = []


def make_robot(tmp_path, **overrides):
    config = {"id": "test", "calibration_dir": tmp_path, "port": "can0", "gravity_compensation": False}
    robot = MetalFollower(MetalFollowerRobotConfig(**{**config, **overrides}))
    _built.append(robot)
    return robot


@pytest.fixture(autouse=True)
def no_hardware_packages(monkeypatch):
    without_hardware_packages(monkeypatch)
    yield
    # Robots left marked connected would otherwise disconnect from __del__, after the test's mocks are gone.
    while _built:
        _built.pop()._connected = False


@pytest.fixture
def robot(tmp_path):
    return make_robot(tmp_path)


def ready(robot, position=None):
    make_ready(robot, position)
    return dict.fromkeys(robot.action_features, 0.0)


def test_factory_builds_the_follower(tmp_path):
    config = MetalFollowerRobotConfig(port="can0", calibration_dir=tmp_path, gravity_compensation=False)
    assert isinstance(make_robot_from_config(config), MetalFollower)


def test_arm_params_come_from_the_metal_profile(robot):
    arm = robot_module._metal_arm_params(robot.config)
    np.testing.assert_allclose(arm.joint_limits, np.radians(JOINT_LIMITS_DEG))
    np.testing.assert_allclose(arm.kp, [100, 100, 100, 50, 20, 20])
    np.testing.assert_allclose(arm.kd, [5, 5, 5, 5, 0.6, 0.6])
    np.testing.assert_allclose(arm.gravity_factors, [1.2, 1.15, 1.1, 1.15, 1.0, 1.0])
    assert (arm.control_frequency, arm.feedback_timeout_s) == (100.0, 0.3)


def test_gripper_spans_closed_at_zero_to_open_at_116_degrees(robot):
    grip = robot_module._metal_gripper_params(robot.config)
    assert (grip.closed, math.degrees(grip.open)) == pytest.approx((0.0, 116.4))
    assert (grip.kp, grip.kd) == (20.0, 0.6)


def test_blocked_gripper_holds_the_configured_torque(robot):
    """Close on an object at a third of the stroke: the force limiter settles at the hold torque."""
    grip = robot_module._metal_gripper_params(robot.config)
    limiter = mit_control._GripperForceLimiter(grip)
    position, goal, torque = 1 / 3, grip.to_raw(1 / 3), 0.0
    for step in range(300):
        command = limiter.step(position, 0.0, torque, 0.0, now=step * 0.01)
        goal = command.position
        torque = grip.kp * (goal - grip.to_raw(position))  # the motor's spring, blocked by the object
    assert limiter.blocked
    assert abs(torque) == pytest.approx(robot.config.gripper_hold_torque_nm, rel=0.02)


@pytest.mark.parametrize("use_degrees", [True, False])
def test_actions_are_converted_and_clipped_to_the_limits(robot, use_degrees):
    robot.config.use_degrees = use_degrees
    action = ready(robot)
    action["shoulder_pan.pos"] = math.degrees(0.2) if use_degrees else 0.2
    action["shoulder_lift.pos"] = 10.0  # above the 0 upper limit in either unit
    action["gripper.pos"] = 100.4 if use_degrees else 1.004
    sent = robot.send_action(action)
    np.testing.assert_allclose(robot.controller._target, [0.2, 0, 0, 0, 0, 0, 1.0])
    expected = {**action, "shoulder_lift.pos": 0.0, "gripper.pos": 100.0 if use_degrees else 1.0}
    assert sent == pytest.approx(expected)


def test_non_finite_action_is_rejected(robot):
    action = ready(robot)
    action["elbow_flex.pos"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        robot.send_action(action)


def test_features_match_the_star_leader_keys(robot, tmp_path):
    assert tuple(robot.action_features) == METAL_FEATURE_NAMES
    assert METAL_FEATURE_NAMES[0] == "shoulder_pan.pos" and METAL_FEATURE_NAMES[-1] == "gripper.pos"
    names = tuple(make_robot(tmp_path, use_velocity_and_torque=True).observation_features)
    assert len(names) == 21


class DepthCamera:
    use_rgb = use_depth = True

    def read_latest(self, max_age_ms=500):
        return "rgb"

    def read_latest_depth(self, max_age_ms=500):
        return "depth"


def test_depth_cameras_add_a_depth_feature_and_frame(robot):
    robot.config.cameras = {"front": MagicMock(height=480, width=640, use_rgb=True, use_depth=True)}
    robot.cameras = {"front": DepthCamera()}
    assert robot._cameras_ft == {"front": (480, 640, 3), "front_depth": (480, 640, 1)}
    ready(robot)
    observation = robot.get_observation()
    assert (observation["front"], observation["front_depth"]) == ("rgb", "depth")


@pytest.mark.parametrize("use_degrees", [True, False])
def test_observation_uses_public_units(tmp_path, use_degrees):
    robot = make_robot(tmp_path, use_degrees=use_degrees, use_velocity_and_torque=True)
    ready(robot)
    robot.controller.state = mit_control.JointState(
        position=np.array([0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.5]),
        velocity=np.array([1.0, 0, 0, 0, 0, 0, 0.25]),
        torque=np.array([2.0, 0, 0, 0, 0, 0, 0.3]),
    )
    observation = robot.get_observation()
    scale, gripper_scale = (math.degrees(1), 100.0) if use_degrees else (1.0, 1.0)
    assert observation["shoulder_pan.pos"] == pytest.approx(0.1 * scale)
    assert observation["shoulder_pan.vel"] == pytest.approx(1.0 * scale)
    assert observation["shoulder_pan.torque"] == pytest.approx(2.0)
    assert observation["gripper.pos"] == pytest.approx(0.5 * gripper_scale)


def test_gravity_model_loads_from_the_hub_before_torque(robot, monkeypatch):
    events = []
    download = MagicMock(return_value="/cache/metal/v1/metal_with_gripper.urdf")
    model = MagicMock(side_effect=lambda *args, **kwargs: events.append("model") or MagicMock())
    monkeypatch.setattr(robot_module, "hf_hub_download", download)
    monkeypatch.setattr(robot_module, "GravityCompensation", model)
    robot.config.gravity_compensation = True
    mock_hardware(robot, monkeypatch)
    robot.bus.enable.side_effect = lambda: events.append("enable")
    robot.connect()
    robot.disconnect()
    assert events == ["model", "enable"]
    download.assert_called_once_with(
        "lerobot/robot-urdfs", "metal/v1/metal_with_gripper.urdf", repo_type="dataset"
    )
    model.assert_called_once_with(
        "/cache/metal/v1/metal_with_gripper.urdf",
        tuple(f"JOINT{i}" for i in range(1, 7)),
        base_frame="base_link",
    )


def test_local_urdf_skips_the_download(tmp_path, monkeypatch):
    robot = make_robot(tmp_path, gravity_compensation=True, urdf_path="/models/metal.urdf")
    download = MagicMock()
    model = MagicMock()
    monkeypatch.setattr(robot_module, "hf_hub_download", download)
    monkeypatch.setattr(robot_module, "GravityCompensation", model)
    robot._load_gravity_model()
    download.assert_not_called()
    assert model.call_args.args[0] == "/models/metal.urdf"


def test_gravity_torque_is_capped_per_joint_after_the_factors(robot):
    robot.gravity_model = MagicMock()
    robot.gravity_model.torques.return_value = np.full(6, 100.0)
    torque = robot._gravity_torque(np.zeros(7))
    scaled = mit_control._gravity_feedforward(robot_module._metal_arm_params(robot.config), torque)
    np.testing.assert_allclose(scaled, [1.0, 25.0, 10.0, 3.0, 1.0, 0.5])


def test_failed_gravity_model_never_enables_torque(robot, monkeypatch):
    monkeypatch.setattr(robot_module, "hf_hub_download", MagicMock(side_effect=OSError("offline")))
    robot.config.gravity_compensation = True
    mock_hardware(robot, monkeypatch)
    with pytest.raises(OSError, match="offline"):
        robot.connect()
    robot.bus.enable.assert_not_called()
    robot.bus.disconnect.assert_called_once()
    assert not robot.is_connected


def test_initial_position_check_uses_degrees(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    states = raw_states()
    states.position[2] = math.radians(robot.config.initial_tolerance_deg + 1)
    robot.bus.read_states.return_value = states
    with pytest.raises(ValueError, match="fold it"):
        robot.connect()
    robot.bus.enable.assert_not_called()


def test_connect_starts_control_and_calibrate_changes_nothing(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.connect()
    assert robot.is_calibrated and robot.controller.active
    robot.calibrate()
    assert robot.controller.active
    robot.bus.disable_torque.assert_not_called()
    robot.disconnect()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_joint_speed_deg_s": 300.0}, "at most 280"),
        ({"idle_mode": "float", "gravity_compensation": False}, "needs gravity_compensation"),
        ({"kp": [1.0]}, "six values"),
        ({"max_tracking_error_deg": -1.0}, "max_tracking_error_deg must be positive"),
        ({"gravity_factors": [1, -1, 1, 1, 1, 1]}, "must not be negative"),
        ({"feedback_timeout_s": 0.0}, "feedback_timeout_s must be positive"),
        ({"feedback_timeout_s": float("nan")}, "feedback_timeout_s must be positive"),
    ],
)
def test_invalid_configs_are_rejected(kwargs, error):
    with pytest.raises(ValueError, match=error):
        MetalFollowerRobotConfig(port="can0", **kwargs)


class FakeClock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def state_at(position):
    zeros = np.zeros(7)
    return mit_control.JointState(
        position=np.asarray(position, dtype=float).copy(), velocity=zeros, torque=zeros
    )


def energized(robot, position=None, *, follows=True, speed=0.0):
    """A ready robot with torque on, whose mocked bus reads back the controller's state.

    The controller's measured position jumps to each target (or stays put), joints move at `speed` rad/s, and
    the return runs on a fake clock. Returns the bus and the list of targets.
    """
    bus = attach_bus(robot, mock_bus())
    bus.enabled = True
    ready(robot, position)
    targets = []

    def set_target(target):
        targets.append(target.copy())
        if follows:
            robot.controller.state = state_at(target)

    def read_fresh():
        states = mit_control.MotorStates(
            robot.controller.state.position.copy(), np.full(7, speed), np.zeros(7)
        )
        return states, np.ones(7, dtype=bool), np.ones(7, dtype=bool)

    robot.controller.set_target = set_target
    bus.read_fresh.side_effect = read_fresh
    robot_module.time = FakeClock()
    return bus, targets


@pytest.fixture(autouse=True)
def real_clock(monkeypatch):
    monkeypatch.setattr(robot_module, "time", robot_module.time)


def test_disconnect_returns_to_rest_before_torque_off(robot, caplog):
    bus, targets = energized(robot)
    robot.controller.state = state_at([0.5, -1.0, 1.2, 0, 0, 0, 0.7])
    robot.disconnect()
    np.testing.assert_array_equal(targets[-1], [0, 0, 0, 0, 0, 0, 0.7])  # the gripper stays put
    bus.disconnect.assert_called_once_with(release=True)
    bus.send_confirmed.assert_not_called()
    assert not robot.is_connected


def test_return_is_speed_limited(robot):
    _, targets = energized(robot)
    robot.controller.state = state_at([0, 0, 1.0, 0, 0, 0, 0])
    robot.disconnect()
    steps = np.diff([1.0, *[target[2] for target in targets]])
    assert steps[1:-1] == pytest.approx(np.full(len(steps) - 2, -math.radians(30) * 0.02))


def assert_held(bus, caplog):
    """The arm got a stiff hold at its measured pose, then CAN closed with torque on."""
    commands = bus.send_confirmed.call_args.args[0]
    assert commands["shoulder_lift"].kp == 100 and commands["gripper"].kp == 0.0
    bus.disconnect.assert_called_once_with(release=False)
    assert "still holding" in caplog.text


def test_blocked_return_holds_at_the_measured_pose(robot, caplog):
    bus, _ = energized(robot, follows=False)
    robot.controller.state = state_at([0, 0, 1.0, 0, 0, 0, 0])
    robot.disconnect()
    assert "did not reach its rest pose" in caplog.text
    assert_held(bus, caplog)
    assert bus.send_confirmed.call_args.args[0]["elbow_flex"].position == 1.0
    assert not robot.is_connected and not robot.controller.active


def test_an_arm_moving_through_rest_is_not_released(robot, caplog):
    bus, _ = energized(robot, speed=1.0)
    robot.disconnect()
    assert_held(bus, caplog)


def test_start_pose_away_from_rest_is_not_a_release_pose(robot, caplog):
    bus, _ = energized(robot, [0.5, 0, 0, 0, 0, 0, 0], follows=False)
    robot.disconnect()
    assert_held(bus, caplog)


def test_operator_releases_only_after_the_hold(robot, monkeypatch):
    bus, _ = energized(robot, follows=False)
    robot.controller.state = state_at([0, 0, 1.0, 0, 0, 0, 0])
    events = []
    bus.send_confirmed.side_effect = lambda commands, enabled: events.append("hold") or []
    monkeypatch.setattr(robot_module.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: events.append("prompt") or "RELEASE")
    robot.disconnect()
    assert events == ["hold", "prompt"]
    bus.disconnect.assert_called_once_with(release=True)


def test_interrupted_return_holds_at_the_measured_pose(robot, caplog):
    bus, _ = energized(robot)
    robot.controller.state = state_at([0, 0, 1.0, 0, 0, 0, 0])

    def interrupt(target):
        raise KeyboardInterrupt

    robot.controller.set_target = interrupt
    with pytest.raises(KeyboardInterrupt):
        robot.disconnect()
    assert_held(bus, caplog)
    assert not robot.is_connected and not robot.controller.active


@pytest.mark.parametrize(("position", "released"), [([0] * 7, True), ([0, 0, 1.0, 0, 0, 0, 0], False)])
def test_after_a_fault_torque_goes_off_only_at_rest(robot, position, released):
    bus, _ = energized(robot, position, follows=False)
    robot.controller.fault_event.set()
    robot.disconnect()
    bus.disconnect.assert_called_once_with(release=released)


def test_unsafe_states_turn_torque_off_when_holding_is_switched_off(robot, monkeypatch):
    monkeypatch.setattr(robot_module, "HOLD_WHEN_UNSAFE", False)
    bus, _ = energized(robot, [0, 0, 1.0, 0, 0, 0, 0], follows=False)
    robot.disconnect()
    bus.disconnect.assert_called_once_with(release=True)
    bus.send_confirmed.assert_not_called()


def test_arm_without_torque_skips_the_return(robot):
    bus = attach_bus(robot, mock_bus())
    ready(robot, [0, 0, 1.0, 0, 0, 0, 0])
    robot.controller.set_target = MagicMock()
    robot.disconnect()
    robot.controller.set_target.assert_not_called()
    bus.disconnect.assert_called_once_with(release=True)


def failing_enable(bus):
    def enable():
        bus.enabled = True  # some motors may have torque on
        raise ConnectionError("motors [3] did not confirm torque on")

    bus.enable.side_effect = enable


def test_failed_start_at_rest_turns_torque_off(robot, monkeypatch):
    bus = mock_hardware(robot, monkeypatch)
    failing_enable(bus)
    with pytest.raises(ConnectionError, match="did not confirm"):
        robot.connect()
    bus.disconnect.assert_called_once_with(release=True)


def test_failed_start_without_a_still_rest_pose_holds(robot, monkeypatch, caplog):
    bus = mock_hardware(robot, monkeypatch)
    failing_enable(bus)
    moving = mit_control.MotorStates(np.zeros(7), np.full(7, 1.0), np.zeros(7))
    bus.read_fresh.return_value = (moving, np.ones(7, dtype=bool), np.ones(7, dtype=bool))
    monkeypatch.setattr(robot_module, "time", FakeClock())
    with pytest.raises(ConnectionError, match="did not confirm"):
        robot.connect()
    assert_held(bus, caplog)


def test_disconnect_can_retry_after_controller_join_timeout(robot):
    bus = attach_bus(robot, mock_bus())
    robot._connected = True
    mark_running(robot.controller)
    robot.controller.stop = MagicMock(side_effect=RuntimeError("controller did not stop"))
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


# MetalMotorsBus over a real DamiaoMotorsBus on a fake CAN interface.


@pytest.fixture
def bus():
    metal_bus = new_bus()
    fake_can(metal_bus)
    return metal_bus


def new_bus(port="can0", interface="socketcan"):
    gains = np.array([100, 100, 100, 50, 20, 20.0]), np.array([5, 5, 5, 5, 0.6, 0.6])
    return mit_bus.MetalMotorsBus(port, interface, 0.05, *gains)


def enabled(bus, names=MOTOR_NAMES):
    for name in names:
        bus.bus.canbus.motors[name]["status"] = 1


def test_bus_maps_ids_and_metal_motor_types():
    bus = new_bus("/dev/ttyACM0", "auto")
    motors = bus.bus.motors
    assert [(m.id, m.recv_id) for m in motors.values()] == [(i, i + 0x10) for i in range(1, 8)]
    assert [m.motor_type_str for m in motors.values()] == list(mit_bus.MOTOR_TYPES)
    assert (bus.bus.use_can_fd, bus.bus.bitrate) == (False, 1_000_000)


def test_connect_skips_the_handshake_that_enables_torque():
    bus = new_bus()
    bus.connect()
    sent = bus.bus.canbus.sent
    assert len(sent) == 7 and all(msg.arbitration_id == 0x7FF and msg.data[2] == 0xCC for msg in sent)
    assert not bus.enabled


def test_connect_refuses_an_arm_that_already_holds(monkeypatch):
    bus = new_bus()
    holding = mit_bus.can.interface.Bus()
    holding.motors["elbow_flex"]["status"] = 1
    monkeypatch.setattr(mit_bus.can.interface, "Bus", lambda **kw: holding)
    with pytest.raises(ConnectionError, match=r"motors \[3\] already have torque on"):
        bus.connect()
    assert all(msg.arbitration_id == 0x7FF for msg in holding.sent)  # only refreshes were sent


def test_fresh_feedback_is_converted_to_radians(bus):
    for motor in bus.bus.canbus.motors.values():
        motor["position"] = 90.0
    states = bus.read_states()
    np.testing.assert_allclose(states.position, np.full(7, math.pi / 2), atol=1e-3)
    np.testing.assert_allclose(states.velocity, np.zeros(7), atol=0.01)
    assert bus.bus._last_known_states["wrist_roll"]["position"] == pytest.approx(90.0, abs=0.01)


def test_a_missed_reply_is_retried_within_the_timeout(bus):
    fake = bus.bus.canbus
    fake.silent.add("gripper")
    send = fake.send

    def answer_from_the_second_refresh(msg):
        send(msg)
        if len(fake.sent) == 7:
            fake.silent.clear()

    fake.send = answer_from_the_second_refresh
    bus.read_states()
    assert len(fake.sent) == 14


@pytest.mark.parametrize(
    ("change", "enabled_bus", "error"),
    [
        (lambda fake: fake.silent.add("gripper"), False, "7: missing"),
        (lambda fake: fake.motors["elbow_flex"].update(status=0xD), False, "motor 3 fault 0xd"),
        (
            lambda fake: [fake.motors[name].update(status=int(name != "elbow_flex")) for name in MOTOR_NAMES],
            True,
            "motor 3 unexpectedly disabled",
        ),
    ],
    ids=["a motor never answered", "motor fault", "motor disabled after the enable grace"],
)
def test_bad_feedback_fails(bus, change, enabled_bus, error):
    change(bus.bus.canbus)
    bus.enabled, bus._enabled_at = enabled_bus, time.monotonic() - 1
    with pytest.raises(ConnectionError, match=error):
        bus.read_states()


def test_old_feedback_fails(bus):
    bus.read_states()
    bus._replied_at["wrist_yaw"] -= 1
    bus.bus.canbus.silent.add("wrist_yaw")
    with pytest.raises(ConnectionError, match="stale"):
        bus.read_states()


def test_enable_sends_a_confirmed_stiff_hold_then_confirms(bus):
    fake = bus.bus.canbus
    bus.enable_torque_at(np.array([0.1, 0, 0, 0, 0, 0, math.pi]))
    first = fake.batches[0]
    assert (first["shoulder_pan"].position, first["gripper"].position) == (0.1, math.pi)
    assert [first[name].kp for name in MOTOR_NAMES] == [100, 100, 100, 50, 20, 20, 0]  # the gripper limp
    assert all(c.torque == 0 for c in first.values())
    assert all(motor["status"] == 1 for motor in fake.motors.values()) and bus.enabled
    bus.disable_torque()
    assert all(motor["status"] == 0 for motor in fake.motors.values()) and not bus.enabled


def test_unconfirmed_hold_never_enables(bus):
    fake = bus.bus.canbus
    fake.silent.add("wrist_flex")
    with pytest.raises(ConnectionError, match="wrist_flex did not confirm the hold"):
        bus.enable_torque_at(np.zeros(7))
    assert not any(bytes(msg.data)[:7] == b"\xff" * 7 for msg in fake.sent)
    assert not bus.enabled


@pytest.mark.parametrize(
    ("enable", "stuck", "error"),
    [
        (True, "gripper", r"motors \[7\] did not confirm torque on"),
        (False, "elbow_flex", r"motors \[3\] did not confirm torque off"),
    ],
)
def test_torque_changes_need_every_motor_to_confirm(bus, enable, stuck, error):
    if not enable:
        bus.enable_torque_at(np.zeros(7))
    bus.bus.canbus.stuck.add(stuck)
    with pytest.raises(ConnectionError, match=error):
        bus.enable_torque_at(np.zeros(7)) if enable else bus.disable_torque()
    assert bus.enabled  # a motor may still have torque, so disconnect tries again


def test_queued_replies_are_drained_before_torque_changes(bus):
    """Replies queued before a disable are discarded first, so only later replies can confirm it."""
    fake = bus.bus.canbus
    bus.enable_torque_at(np.zeros(7))
    stale = SimpleNamespace(arbitration_id=0x16, data=bytes([0x06, 0x80, 0, 0x7F, 0xF7, 0xFF, 30, 30]))
    fake.queue += [stale, stale]
    queued_at_disable = []
    disable = bus.bus.disable_torque
    bus.bus.disable_torque = lambda: queued_at_disable.append(len(fake.queue)) or disable()
    bus.disable_torque()
    assert queued_at_disable == [0]


def test_a_faulted_motor_counts_as_off(bus):
    bus.enable_torque_at(np.zeros(7))
    bus.bus.canbus.stuck.add("gripper")
    bus.bus.canbus.motors["gripper"]["status"] = 0xA
    bus.disable_torque()
    assert not bus.enabled


@pytest.mark.parametrize("field", ["position", "velocity", "kp", "kd", "torque"])
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_non_finite_commands_send_nothing(bus, field, value):
    good = mit_control.MitCommand(position=0.1, velocity=0.0, kp=10.0, kd=1.0, torque=0.5)
    for name in MOTOR_NAMES[:-1]:
        bus.send_mit(name, good)
    with pytest.raises(ValueError, match="non-finite"):
        bus.send_mit("gripper", mit_control.MitCommand(**{**good.__dict__, field: value}))
    assert bus.bus.canbus.sent == []


def test_commands_are_sent_as_one_batch_of_metal_frames(bus):
    fake = bus.bus.canbus
    command = mit_control.MitCommand(position=0.5, velocity=0.1, kp=50.0, kd=2.0, torque=1.5)
    for name in MOTOR_NAMES[:-1]:
        bus.send_mit(name, command)
    assert fake.sent == []
    bus.send_mit("gripper", command)
    frames = fake.mit_frames()
    assert [msg.arbitration_id for msg in frames] == list(range(1, 8))
    expected = bus.bus._encode_mit_packet(
        MotorType.METAL_J2, 50.0, 2.0, math.degrees(0.5), math.degrees(0.1), 1.5
    )
    assert list(frames[1].data) == expected
    assert bus._replied_at["gripper"] > -math.inf  # the replies count as feedback


def test_disable_drops_unsent_commands_and_is_idempotent(bus):
    bus.enable_torque_at(np.zeros(7))
    bus.send_mit("shoulder_pan", mit_control.MitCommand(0, 0, 0, 0, 0))
    bus.disable_torque()
    disables = [
        msg for msg in bus.bus.canbus.sent if bytes(msg.data)[:7] == b"\xff" * 7 and msg.data[7] == 0xFD
    ]
    bus.disable_torque()
    assert len(disables) == 7
    bus.send_mit("gripper", mit_control.MitCommand(0, 0, 0, 0, 0))
    assert list(bus.bus.canbus.batches[-1]) == ["gripper"]


def test_failed_first_read_closes_can(robot, monkeypatch):
    dead = FakeCan()
    dead.silent.add("gripper")
    shutdowns = []
    dead.shutdown = lambda: shutdowns.append(True)
    monkeypatch.setattr(mit_bus.can.interface, "Bus", lambda **kw: dead)
    robot.bus.feedback_timeout_s = 0.01
    with pytest.raises(ConnectionError, match="missing"):
        robot.connect()
    assert shutdowns == [True] and not robot.bus.bus.is_connected
    assert not robot.is_connected


def test_disconnect_closes_can_even_if_torque_off_fails(bus):
    bus.enable_torque_at(np.zeros(7))
    bus.bus.canbus.stuck.add("shoulder_lift")
    with pytest.raises(ConnectionError, match="did not confirm torque off"):
        bus.disconnect()
    assert not bus.bus.is_connected
    assert bus.enabled


def test_bus_access_is_serialized(bus):
    """A second thread's read waits for the first to finish, as after a controller join timeout."""
    inside, release, reads = threading.Event(), threading.Event(), []
    refresh = bus._refresh

    def slow_refresh():
        reads.append(threading.current_thread().name)
        if len(reads) == 1:
            inside.set()
            release.wait(1)
        refresh()

    bus._refresh = slow_refresh
    first = threading.Thread(target=bus.read_states, name="control")
    first.start()
    inside.wait(1)
    second = threading.Thread(target=bus.read_states, name="teardown")
    second.start()
    second.join(0.05)
    assert reads == ["control"]  # the teardown read waits for the lock
    release.set()
    first.join(1)
    second.join(1)
    assert reads == ["control", "teardown"]


def test_controller_runs_on_the_metal_bus(robot):
    """One control cycle through MitController reads, then sends seven commands in one batch."""
    fake = fake_can(robot)
    enabled(robot.bus)
    robot.bus.enabled, robot.bus._enabled_at = True, time.monotonic()
    robot.controller._set_initial_state(robot.controller.read_state())
    send_batch = robot.bus._send_batch
    robot.bus._send_batch = lambda commands: (robot.controller._stop.set(), send_batch(commands))[1]
    robot.controller._run()
    assert robot.controller._failure is None
    (commands,) = fake.batches
    assert list(commands) == list(MOTOR_NAMES)
    assert (commands["shoulder_lift"].kp, commands["shoulder_lift"].kd) == (100.0, 5.0)


def test_enable_rechecks_the_initial_position(robot, monkeypatch):
    """The pose is checked again in the read that torque is enabled from, after prepare_controller."""
    mock_hardware(robot, monkeypatch)
    robot.connect_bus()
    robot.prepare_controller()
    robot.bus.read_states.return_value = raw_states()
    robot.bus.read_states.return_value.position[1] = -1.0  # moved after the check
    with pytest.raises(ValueError, match="fold it"):
        robot.start_controller()
    robot.bus.enable.assert_not_called()


# Behaviour after a fault: Metal's motors have no CAN timeout, so the controller leaves them a stiff hold.


def run_until_fault(robot, fault, cycles=2):
    """Run the control loop on a fake CAN interface, and apply `fault` to it after `cycles` control cycles."""
    fake = fake_can(robot, feedback_timeout_s=0.02)
    for motor in fake.motors.values():
        motor.update(status=1, position=3.0)  # degrees
    robot.bus.enabled, robot.bus._enabled_at = True, time.monotonic()
    robot.gravity_model = MagicMock()
    robot.gravity_model.torques.return_value = np.full(6, 0.5)
    robot.controller._set_initial_state(robot.controller.read_state())
    send_batch = robot.bus._send_batch

    def count_cycles(commands):
        replied = send_batch(commands)
        if len(fake.batches) == cycles:
            fault(fake)
        return replied

    robot.bus._send_batch = count_cycles
    robot.controller._run()
    return fake


@pytest.mark.parametrize("idle_mode", ["hold", "float"])
def test_fault_leaves_a_stiff_hold_at_the_measured_pose_and_a_free_gripper(tmp_path, idle_mode, caplog):
    caplog.set_level(logging.INFO)
    robot = make_robot(tmp_path, idle_mode=idle_mode, gravity_compensation=True)
    fake = run_until_fault(robot, lambda fake: fake.motors["gripper"].update(status=0xE))
    assert robot.controller._failure is not None
    hold = fake.batches[-1]
    assert [hold[name].kp for name in MOTOR_NAMES[:6]] == [100, 100, 100, 50, 20, 20]
    assert [hold[name].position for name in MOTOR_NAMES[:6]] == pytest.approx(
        np.full(6, math.radians(3.0)), abs=1e-3
    )
    assert [hold[name].torque for name in MOTOR_NAMES[:6]] == pytest.approx(
        0.5 * np.array([1.2, 1.15, 1.1, 1.15, 1, 1])
    )
    assert (hold["gripper"].kp, hold["gripper"].kd) == (0.0, 0.0)  # limp, best effort
    # The faulted gripper doesn't confirm its limp command; only the arm joints need a confirmed hold.
    assert "got no confirmed hold" not in caplog.text
    assert "the gripper didn't confirm its limp command" in caplog.text


def test_stale_joints_get_no_stiff_hold(robot, caplog):
    fake = run_until_fault(robot, lambda fake: fake.silent.add("elbow_flex"))
    hold = fake.batches[-1]
    assert "elbow_flex" not in hold
    assert (hold["gripper"].kp, hold["gripper"].kd) == (0.0, 0.0) and hold["shoulder_lift"].kp == 100
    assert "elbow_flex got no confirmed hold" in caplog.text


def test_unconfirmed_hold_is_reported(robot, caplog):
    def wrist_stops_answering_mit_frames(fake):
        send = fake.send
        fake.send = lambda msg: msg.arbitration_id == 5 or send(msg)
        robot.controller.fault_event.set()

    fake = run_until_fault(robot, wrist_stops_answering_mit_frames, cycles=1)
    assert "wrist_yaw" in fake.batches[-1]
    assert "wrist_yaw got no confirmed hold" in caplog.text


def test_fault_on_the_other_arm_also_leaves_a_hold(robot):
    fake = run_until_fault(robot, lambda fake: robot.controller.fault_event.set(), cycles=1)
    assert robot.controller._failure is None
    assert len(fake.batches) == 2  # one control cycle, then the hold
    assert (fake.batches[-1]["shoulder_lift"].kp, fake.batches[-1]["gripper"].kp) == (100, 0.0)


def test_fault_turns_torque_off_when_holding_is_switched_off(robot, monkeypatch):
    monkeypatch.setattr(robot_module, "HOLD_WHEN_UNSAFE", False)
    fake = run_until_fault(robot, lambda fake: fake.motors["gripper"].update(status=0xE))
    assert all(motor["status"] != 1 for motor in fake.motors.values())
    assert not robot.bus.enabled


def test_stop_sends_no_hold(robot):
    fake = run_until_fault(robot, lambda fake: robot.controller._stop.set(), cycles=1)
    assert len(fake.batches) == 1


def test_only_valid_state_frames_count_as_feedback(bus):
    fake = bus.bus.canbus
    send = fake.send

    def short_or_foreign_replies(msg):
        send(msg)
        if fake.queue and fake.queue[-1].arbitration_id == 0x11:
            fake.queue[-1] = SimpleNamespace(arbitration_id=0x11, data=bytes([0x11]))  # one byte
        if fake.queue and fake.queue[-1].arbitration_id == 0x12:
            fake.queue[-1] = SimpleNamespace(
                arbitration_id=0x12, data=fake.state_frame("elbow_flex")
            )  # motor 3's

    fake.send = short_or_foreign_replies
    _, fresh, _ = bus.read_fresh()
    assert fresh.tolist() == [False, False, True, True, True, True, True]


def test_reconnecting_forgets_old_feedback(bus, monkeypatch):
    bus.read_states()
    silent = FakeCan()
    silent.silent.add("gripper")
    monkeypatch.setattr(mit_bus.can.interface, "Bus", lambda **kw: silent)
    bus.bus._is_connected = False
    with pytest.raises(ConnectionError, match="7: missing"):
        bus.connect()


def test_a_bus_that_never_goes_quiet_fails_the_drain(bus):
    bus.bus.canbus.recv = lambda timeout=None: object()
    with pytest.raises(ConnectionError, match="didn't go quiet"):
        bus.send_confirmed({"gripper": mit_control.MitCommand(0, 0, 0, 0, 0)}, enabled=True)


@pytest.mark.parametrize(
    ("status", "enabled", "confirmed"),
    [(1, True, True), (0, True, False), (0xE, True, False), (1, False, False)],
)
def test_confirmation_needs_the_expected_status(bus, status, enabled, confirmed):
    bus.bus.canbus.motors["gripper"]["status"] = status
    unconfirmed = bus.send_confirmed({"gripper": mit_control.MitCommand(0, 0, 0, 0, 0)}, enabled=enabled)
    assert unconfirmed == ([] if confirmed else ["gripper"])


def test_a_fault_on_the_other_arm_refuses_the_enable(robot, monkeypatch):
    bus = mock_hardware(robot, monkeypatch)
    robot.connect_bus()
    robot.prepare_controller()
    read = robot.controller.read_state
    robot.controller.read_state = lambda: (robot.controller.fault_event.set(), read())[1]
    with pytest.raises(ConnectionError, match="after a fault"):
        robot.start_controller()
    bus.enable.assert_not_called()


def test_a_fault_hold_goes_out_even_while_stop_waits(robot):
    def fault_then_stop(fake):
        robot.controller.fault_event.set()
        robot.controller._stop.set()

    fake = run_until_fault(robot, fault_then_stop, cycles=1)
    assert len(fake.batches) == 2 and fake.batches[-1]["shoulder_lift"].kp == 100


def test_a_frame_that_fails_to_decode_is_not_feedback(bus, monkeypatch):
    decode = bus.bus._decode_motor_state

    def gripper_fails(data, motor_type):
        if data[0] & 0x0F == 7:
            raise ValueError("Invalid motor state data")
        return decode(data, motor_type)

    monkeypatch.setattr(bus.bus, "_decode_motor_state", gripper_fails)
    _, fresh, _ = bus.read_fresh()
    assert fresh.tolist() == [True] * 6 + [False]


def at_rest_on_fake_can(robot, monkeypatch):
    fake = fake_can(robot)
    for motor in fake.motors.values():
        motor["status"] = 1
    robot.bus.enabled = True
    monkeypatch.setattr(robot_module, "time", FakeClock())
    return fake


def test_rest_is_verified_from_new_still_replies_near_zero(robot, monkeypatch):
    at_rest_on_fake_can(robot, monkeypatch)
    assert robot._verified_at_rest()


@pytest.mark.parametrize("elbow_deg", [4.0, 7.0])
def test_a_joint_off_rest_is_not_verified(robot, monkeypatch, elbow_deg):
    fake = at_rest_on_fake_can(robot, monkeypatch)
    fake.motors["elbow_flex"]["position"] = elbow_deg
    assert not robot._verified_at_rest()


def test_a_cached_rest_reply_is_not_verified(robot, monkeypatch):
    """The elbow said zero once, then stopped answering while it moved: its cached reply doesn't count."""
    fake = at_rest_on_fake_can(robot, monkeypatch)
    robot.bus.read_states()
    fake.silent.add("elbow_flex")
    fake.motors["elbow_flex"]["position"] = 7.0
    assert not robot._verified_at_rest()


def test_an_unconfirmed_hold_is_retried_and_keeps_can_open(robot, caplog):
    bus, _ = energized(robot, [0, 0, 1.0, 0, 0, 0, 0], follows=False)
    bus.send_confirmed.return_value = ["elbow_flex"]
    with pytest.raises(ConnectionError, match="elbow_flex have no confirmed hold"):
        robot.disconnect()
    assert bus.send_confirmed.call_count == 3
    bus.disconnect.assert_not_called()
    assert robot.is_connected and not robot.controller.active
    bus.send_confirmed.return_value = []
    robot.disconnect()  # retried once the hold is confirmed: still away from rest, so it holds
    bus.disconnect.assert_called_once_with(release=False)


def test_an_unconfirmed_gripper_alone_does_not_block(robot, caplog):
    bus, _ = energized(robot, [0, 0, 1.0, 0, 0, 0, 0], follows=False)
    bus.send_confirmed.return_value = ["gripper"]
    robot.disconnect()
    bus.disconnect.assert_called_once_with(release=False)


def test_a_fault_during_the_preload_refuses_the_enable(robot, monkeypatch):
    bus = mock_hardware(robot, monkeypatch)
    bus.preload.side_effect = lambda hold: (
        robot.controller.fault_event.set()
    )  # the other arm faults meanwhile
    robot.connect_bus()
    robot.prepare_controller()
    with pytest.raises(ConnectionError, match="after a fault"):
        robot.start_controller()
    bus.preload.assert_called_once()
    bus.enable.assert_not_called()


def test_an_overtravel_gripper_is_preloaded_limp_where_it_is(bus):
    bus.preload(np.array([0, 0, 0, 0, 0, 0, 2.2]))  # past the 2.03 rad opening
    gripper = bus.bus.canbus.batches[0]["gripper"]
    assert (gripper.position, gripper.kp, gripper.kd) == (2.2, 0.0, 0.0)
