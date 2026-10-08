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

import numpy as np
import pytest

from lerobot.robots.yam_follower import mit_arm


def gripper(closed=1.0, opened=3.0):
    return mit_arm.CalibratedGripper(
        closed=closed, open=opened, kp=4.0, kd=0.1, max_speed=1.0, force_limit_n=10.0, finger_stroke_m=0.1
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
        "float_kd": np.array([0.2, 0.1]),
        "coulomb_friction": np.array([0.5, 0.2]),
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
    idle_mode: str = "hold"


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
    assert list(packet) == ["shoulder", "elbow"]  # the gripper has its own force limiter


def test_control_step_returns_from_past_a_limit_without_jumping():
    position = np.array([0.2, -0.1, 0.5])  # elbow past its 0 lower limit
    command, _ = mit_arm.control_step(params(), position, position, position.copy(), np.zeros(2), dt=0.01)
    np.testing.assert_allclose(command, [0.2, -0.095, 0.5])  # back toward 0 at 0.5 rad/s
    # A command further out than the measured pose is pulled back to it, never beyond.
    previous = np.array([0.2, -0.3, 0.5])
    command, _ = mit_arm.control_step(params(), position, position, previous, np.zeros(2), dt=0.01)
    assert command[1] == pytest.approx(-0.1)


@pytest.mark.parametrize("closed,opened", [(1.0, 3.0), (3.0, 1.0)])
def test_free_gripper_goes_straight_to_its_target(closed, opened):
    limiter = mit_arm.GripperForceLimiter(gripper(closed, opened))
    command = limiter.command(measured=0.5, velocity=1.0, torque=0.1, commanded=0.0, now=0.0)
    assert command == pytest.approx((closed, 0, 4, 0.1, 0))
    assert not limiter.blocked


def blocked_limiter(torque):
    """A closing gripper stopped by an object at mid-stroke (raw 2.0), pressing with ``torque``."""
    limiter = mit_arm.GripperForceLimiter(gripper())
    command = limiter.command(measured=0.5, velocity=0.0, torque=torque, commanded=0.0, now=0.0)
    assert limiter.blocked
    return limiter, command


def test_blocked_gripper_backs_off_to_the_force_limit():
    # Limit torque: 10 N * 0.1 m / 2 rad stroke + 0.3 Nm friction = 0.8 Nm.
    _, command = blocked_limiter(torque=0.8)
    assert command[0] == pytest.approx(2.0)  # already at the limit: keep pressing as is
    # Pressing with 2 Nm: release (2 - 0.8) / kp 4 = 0.3 rad toward open.
    _, command = blocked_limiter(torque=2.0)
    assert command[0] == pytest.approx(2.3)


def test_blocked_gripper_releases_when_asked_to_open():
    limiter, _ = blocked_limiter(torque=2.0)
    command = limiter.command(measured=0.5, velocity=0.0, torque=2.0, commanded=1.0, now=0.01)
    assert not limiter.blocked
    assert command[0] == pytest.approx(3.0)


def test_blocked_gripper_releases_once_its_torque_drops():
    limiter, _ = blocked_limiter(torque=2.0)
    limiter.command(measured=0.5, velocity=0.0, torque=0.0, commanded=0.0, now=0.05)
    assert limiter.blocked  # 0.1 s average still above 0.2 Nm
    command = limiter.command(measured=0.5, velocity=0.0, torque=0.0, commanded=0.0, now=0.2)
    assert not limiter.blocked
    assert command[0] == pytest.approx(1.0)


def test_moving_gripper_is_not_considered_blocked():
    limiter = mit_arm.GripperForceLimiter(gripper())
    limiter.command(measured=0.5, velocity=0.5, torque=2.0, commanded=0.0, now=0.0)  # 1 rad/s raw
    assert not limiter.blocked


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


def test_float_commands_compensate_gravity_and_friction_without_stiffness():
    state = mit_arm.JointState(
        position=np.array([0.2, 0.5, 0.5]), velocity=np.array([1.0, 1.0, 0.0]), torque=np.zeros(3)
    )
    commands = mit_arm.float_commands(params(), state, np.array([8.0, 1.5]))
    # Gravity 8 clipped to 5 Nm, plus friction 0.5 in the direction of motion.
    assert commands["shoulder"] == pytest.approx((0.2, 0, 0, 0.2, 5.5))
    # Gravity -3 Nm and friction -0.2 Nm in the reversed elbow's motor frame.
    assert commands["elbow"] == pytest.approx((-0.5, 0, 0, 0.1, -3.2))
    assert list(commands) == ["shoulder", "elbow"]  # the gripper keeps its own limiter


def run_cycles(servo, cycles):
    """Run ``cycles`` full servo cycles in this thread, then leave the servo ready for more."""
    bus = servo.bus
    bus.enabled, bus.stop_after, bus.reads = True, cycles + 1, 0
    bus.sent.clear()
    servo._run()
    servo._stop_requested.clear()
    servo.stop_event.clear()
    servo.updated_at = time.monotonic()


def joint_stiffness(bus):
    return [command[2] for name, command in bus.sent if name != "gripper"]


def test_float_mode_floats_until_the_first_target_then_tracks_from_the_measured_pose():
    bus = FakeBus()
    servo = make_servo(bus, Settings(idle_mode="float"))
    servo.seed(mit_arm.read_joint_state(bus, servo.params))
    bus.enabled = True
    run_cycles(servo, 2)
    assert joint_stiffness(bus) and set(joint_stiffness(bus)) == {0.0}

    measured = servo.state.position
    servo.set_target(measured + np.array([0.5, 0.5, 0.0]))
    run_cycles(servo, 1)
    assert set(joint_stiffness(bus)) == {50.0, 20.0}
    shoulder_goal = dict(bus.sent)["shoulder"][0]
    assert shoulder_goal == pytest.approx(measured[0], abs=0.5 * 0.05)  # no jump from a stale command


def test_float_mode_floats_again_after_the_command_timeout():
    bus = FakeBus()
    servo = make_servo(bus, Settings(idle_mode="float", command_timeout_s=0.01))
    servo.seed(mit_arm.read_joint_state(bus, servo.params))
    bus.enabled = True
    servo.set_target(servo.state.position)
    servo.commanded_at -= 1
    run_cycles(servo, 1)
    assert servo.idle
    assert set(joint_stiffness(bus)) == {0.0}


@pytest.mark.parametrize("idle_mode", ["hold", "float"])
def test_servo_starts_past_a_limit_without_jumping(idle_mode):
    # Raw elbow 0.1 is joint -0.1 (sign -1): past its 0 lower limit, within the feedback tolerance.
    bus = FakeBus(position=(0.2, 0.1, 2.0))
    servo = make_servo(bus, Settings(idle_mode=idle_mode))
    servo.seed(mit_arm.read_joint_state(bus, servo.params))
    servo.set_target(np.array([0.2, 0.5, 0.5]))
    run_cycles(servo, 2)
    elbow_goals = [-command[0] for name, command in bus.sent if name == "elbow"]
    assert elbow_goals and all(goal == pytest.approx(-0.1, abs=2 * 0.5 * 0.05) for goal in elbow_goals)


def test_hold_mode_keeps_stiffness_while_idle():
    bus = FakeBus()
    servo = make_servo(bus)
    servo.seed(mit_arm.read_joint_state(bus, servo.params))
    bus.enabled = True
    run_cycles(servo, 2)
    assert set(joint_stiffness(bus)) == {50.0, 20.0}


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
