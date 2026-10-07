#!/usr/bin/env python

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

"""Tests of the robot-independent MIT arm control, on a made-up two-joint arm."""

import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.yam_follower import mit_arm


def gripper(closed=1.0, opened=3.0):
    return mit_arm.CalibratedGripper(
        closed=closed, open=opened, kp=4.0, kd=0.1, torque_limit=0.4, max_speed=1.0
    )


def params(**overrides):
    values = {
        "motor_names": ("shoulder", "elbow", "gripper"),
        "joint_limits": np.array([[-1.0, 1.0], [0.0, 2.0]]),
        "joint_signs": np.array([1.0, -1.0]),
        "joint_offsets": np.array([0.0, 0.0]),
        "kp": np.array([50.0, 20.0]),
        "kd": np.array([2.0, 1.0]),
        "max_joint_speed": 0.5,
        "max_tracking_error": 0.1,
        "gravity_factors": np.array([1.0, 2.0]),
        "max_gravity_torque": 5.0,
        "gripper": gripper(),
        "fault_damping_kd": np.array([3.0, 2.0, 0.5]),
        **overrides,
    }
    return mit_arm.MitArmParams(**values)


def motor_states(position, velocity=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)):
    return mit_arm.MotorStates(
        position=np.asarray(position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
        torque=np.asarray(torque, dtype=float),
    )


@dataclass
class Settings:
    control_frequency: float = 100.0
    feedback_timeout_s: float = 0.2
    command_timeout_s: float = 1.0
    freeze_gc: bool = False


class FakeBus:
    """A bus whose motors sit at fixed raw positions; it can stop the servo after some reads."""

    def __init__(self, position=(0.2, -0.5, 2.0), stop_after=None):
        self.enabled = False
        self.states = motor_states(position)
        self.error: Exception | None = None
        self.sent: list[tuple[str, mit_arm.MitCommand]] = []
        self.disable_calls = 0
        self.reads = 0
        self.stop_after = stop_after
        self.servo: mit_arm.MitServo | None = None

    def read_states(self):
        self.reads += 1
        if self.error is not None:
            raise self.error
        if self.stop_after is not None and self.reads >= self.stop_after:
            # Ends the loop as stop() would, from inside the servo thread.
            self.servo._stop_requested.set()
            self.servo.stop_event.set()
        return self.states

    def send_mit(self, motor, command):
        self.sent.append((motor, command))

    def disable(self):
        self.disable_calls += 1
        self.enabled = False


def make_servo(bus=None, settings=None, gravity=(0.0, 0.0)):
    bus = bus or FakeBus()
    servo = mit_arm.MitServo(
        "arm", bus, settings or Settings(), lambda _: np.asarray(gravity), threading.Event()
    )
    servo.params = params()
    bus.servo = servo
    return servo


def wait_for(condition, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


def test_joint_conversion_round_trips_signs_offsets_and_gripper():
    arm = params(joint_offsets=np.array([0.1, 0.2]))
    raw = np.array([0.3, -0.5, 2.5])
    joints = mit_arm.motor_to_joint(raw, arm)
    np.testing.assert_allclose(joints, [0.4, 0.7, 0.75])
    np.testing.assert_allclose(mit_arm.joint_to_motor(joints, arm), raw)
    with pytest.raises(ValueError, match="calibrated stroke"):
        mit_arm.motor_to_joint(np.array([0.0, 0.0, 3.5]), arm)
    with pytest.raises(ConnectionError, match="Non-finite"):
        mit_arm.motor_to_joint(np.array([np.nan, 0.0, 2.0]), arm)


def test_position_validation_and_clipping_are_pure():
    limits = np.array([[-1.0, 1.0], [0.0, 2.0]])
    values = np.array([-1.02, 0.5, 1.2])
    mit_arm.validate_positions(np.r_[values[:2], 0.5], limits, joint_tolerance_rad=0.03)
    clipped = mit_arm.clip_to_limits(values, limits)
    assert values[0] == -1.02
    assert clipped[0] == -1.0
    assert clipped[2] == 1.0
    with pytest.raises(ValueError, match="outside"):
        mit_arm.validate_positions(np.r_[values[:2], 0.5], limits, joint_tolerance_rad=0.0)


def test_control_step_slews_tracks_clips_and_adds_gravity():
    position = np.array([0.2, 0.02, 0.5])
    previous = np.array([0.6, 0.02, 0.5])
    target = np.array([0.6, -0.5, 1.0])
    command, packet = mit_arm.control_step(params(), position, target, previous, np.array([8.0, 1.5]), dt=0.1)
    # shoulder: tracking band; elbow: slew then lower limit; gripper: slew (0.1 stroke per 0.1 s).
    np.testing.assert_allclose(command, [0.3, 0.0, 0.6])
    np.testing.assert_allclose(previous, [0.6, 0.02, 0.5])  # inputs are not mutated
    assert packet["shoulder"] == pytest.approx((0.3, 0, 50, 2, 5.0))  # gravity 8 clipped to 5 Nm
    assert packet["elbow"] == pytest.approx((0.0, 0, 20, 1, -3.0))  # factor 2, sign -1
    assert packet["gripper"] == pytest.approx((2.1, 0, 4, 0.1, 0))  # 0.4 Nm / kp 4 = 0.1 rad band
    assert list(packet) == ["shoulder", "elbow", "gripper"]


@pytest.mark.parametrize("closed,opened", [(1.0, 3.0), (3.0, 1.0)])
def test_gripper_command_caps_its_proportional_torque(closed, opened):
    grip = gripper(closed, opened)
    goal, _, kp, _, _ = grip.command(measured=0.5, commanded=1.0)
    measured_raw = grip.to_raw(0.5)
    assert abs(goal - measured_raw) * kp == pytest.approx(grip.torque_limit)
    assert np.sign(goal - measured_raw) == np.sign(opened - closed)


def test_joint_state_maps_rates_and_torques_to_the_joint_frame():
    arm = params(gripper=gripper(closed=3.0, opened=1.0))
    bus = FakeBus()
    bus.states = motor_states([0.2, -0.5, 2.0], velocity=[0.5, 0.5, 1.0], torque=[2.0, 2.0, 0.3])
    state = mit_arm.read_joint_state(bus, arm)
    np.testing.assert_allclose(state.position, [0.2, 0.5, 0.5])
    np.testing.assert_allclose(state.velocity, [0.5, -0.5, -0.5])  # 1 rad/s over a -2 rad stroke
    np.testing.assert_allclose(state.torque, [2.0, -2.0, -0.3])


def test_servo_holds_then_sends_one_command_per_motor():
    bus = FakeBus(stop_after=3)
    bus.enabled = True
    servo = make_servo(bus)
    servo.seed(mit_arm.read_joint_state(bus, servo.params))
    bus.reads = 0
    servo._run()
    assert servo.failure is None
    assert [name for name, _ in bus.sent[:3]] == ["shoulder", "elbow", "gripper"]
    assert bus.disable_calls == 1


def test_servo_holds_the_measured_pose_once_commands_expire():
    bus = FakeBus()
    servo = make_servo(bus, Settings(command_timeout_s=0.01))
    servo.seed(mit_arm.read_joint_state(bus, servo.params))
    bus.stop_after, bus.reads = 1, 0  # run exactly one servo cycle
    servo.set_target(np.array([0.5, 0.5, 1.0]))
    servo.commanded_at -= 1
    servo._run()
    assert servo.command_timed_out
    np.testing.assert_allclose(servo.target, servo.state.position)


def test_damping_commands_keep_gravity_without_stiffness():
    commands = mit_arm.damping_commands(params(), np.array([0.2, 0.5, 0.5]), np.array([8.0, 1.5]))
    assert commands["shoulder"] == pytest.approx((0.2, 0, 0, 3.0, 5.0))  # gravity 8 clipped to 5 Nm
    assert commands["elbow"] == pytest.approx((-0.5, 0, 0, 2.0, -3.0))
    assert commands["gripper"] == pytest.approx((2.0, 0, 0, 0.5, 0))


def start_enabled(servo):
    servo.seed(mit_arm.read_joint_state(servo.bus, servo.params))
    servo.bus.enabled = True
    servo.start()


def damping_sent(bus):
    return [command for _, command in bus.sent if command[2] == 0]


@pytest.mark.parametrize("cause", ["own fault", "coupled arm"])
def test_fault_keeps_the_arm_damped_until_stop(cause):
    bus = FakeBus()
    servo = make_servo(bus, gravity=(8.0, 1.5))
    start_enabled(servo)
    if cause == "own fault":
        bus.error = ConnectionError("lost")
    else:
        servo.stop_event.set()  # what a fault on the other arm of a bimanual robot does
    wait_for(lambda: len(damping_sent(bus)) >= 3)
    assert {command[3] for command in damping_sent(bus)} == {3.0, 2.0, 0.5}
    assert bus.disable_calls == 0  # torque stays on, damped, until disconnect
    with pytest.raises(ConnectionError, match="stopped"):
        servo.latest()
    servo.stop()
    assert bus.disable_calls == 1
    assert isinstance(servo.failure, ConnectionError) == (cause == "own fault")


def test_normal_stop_disables_without_damping():
    bus = FakeBus()
    servo = make_servo(bus)
    start_enabled(servo)
    wait_for(lambda: len(bus.sent) >= 3)
    servo.stop()
    assert damping_sent(bus) == []
    assert bus.disable_calls == 1


def test_servo_error_stops_the_loop_and_disables_torque():
    bus = FakeBus()
    bus.error = ConnectionError("lost")
    servo = make_servo(bus)
    servo._run()
    assert isinstance(servo.failure, ConnectionError)
    assert servo.stop_event.is_set()
    assert bus.disable_calls == 1
    with pytest.raises(ConnectionError, match="stopped"):
        servo.latest()


def test_servo_start_refuses_a_second_thread():
    servo = make_servo()
    servo.seed(mit_arm.read_joint_state(servo.bus, servo.params))
    servo.start()
    thread = servo._thread
    with pytest.raises(RuntimeError, match="already running"):
        servo.start()
    assert servo._thread is thread
    servo.stop()


def test_servo_needs_params_before_it_starts():
    servo = make_servo()
    servo.params = None
    with pytest.raises(RuntimeError, match="calibrated arm parameters"):
        servo.start()


def test_foreground_watchdog_allows_one_late_feedback_window():
    servo = make_servo()
    servo.updated_at = time.monotonic() - 1.5 * 0.2
    servo.check_healthy()
    servo.updated_at = time.monotonic() - 3 * 0.2
    with pytest.raises(ConnectionError, match="stale"):
        servo.check_healthy()


@pytest.mark.parametrize("enabled,preexisting", [(True, False), (True, True), (False, False)])
def test_gc_freeze_is_shared_and_preserves_callers_state(monkeypatch, enabled, preexisting):
    gc_mock = MagicMock()
    gc_mock.isenabled.return_value = enabled
    gc_mock.get_freeze_count.return_value = int(preexisting)
    monkeypatch.setattr(mit_arm, "gc", gc_mock)
    guard = mit_arm._ControlGC
    assert guard._users == 0
    guard.acquire()
    guard.acquire()
    guard.release()
    gc_mock.unfreeze.assert_not_called()
    guard.release()
    assert gc_mock.collect.call_count == int(enabled)
    assert gc_mock.freeze.call_count == int(enabled)
    assert gc_mock.unfreeze.call_count == int(enabled and not preexisting)
    assert guard._users == 0


@pytest.mark.parametrize("freeze_gc", [True, False])
def test_gc_freeze_lasts_exactly_as_long_as_the_servo(monkeypatch, freeze_gc):
    guard = MagicMock()
    monkeypatch.setattr(mit_arm, "_ControlGC", guard)
    servo = make_servo(settings=Settings(freeze_gc=freeze_gc))
    servo.seed(mit_arm.read_joint_state(servo.bus, servo.params))
    servo.start()
    assert guard.acquire.call_count == int(freeze_gc)
    guard.release.assert_not_called()
    servo.stop()
    assert guard.release.call_count == int(freeze_gc)
