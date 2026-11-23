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

"""Tests of the robot-independent MIT control loop, on a made-up two-joint arm."""

import threading
import time

import numpy as np
import pytest

from lerobot.robots.yam_follower import mit_control


def gripper(closed=1.0, opened=3.0):
    return mit_control.MitGripperParams(
        closed=closed,
        open=opened,
        kp=4.0,
        kd=0.1,
        max_speed=1.0,
        force_limit_n=10.0,
        finger_stroke_m=0.1,
    )


def params(**overrides):
    values = {
        "joint_limits": np.array([[-1.0, 1.0], [0.0, 2.0]]),
        "kp": np.array([50.0, 20.0]),
        "kd": np.array([2.0, 1.0]),
        "max_joint_speed": 0.5,
        "max_tracking_error": 0.1,
        "gravity_factors": np.array([1.0, 2.0]),
        "max_gravity_torque": 5.0,
        "float_kd": np.array([0.2, 0.1]),
        "coulomb_friction": np.array([0.5, 0.2]),
        "control_frequency": 100.0,
        "feedback_timeout_s": 0.2,
        "command_timeout_s": 1.0,
        "idle_mode": "hold",
        **overrides,
    }
    return mit_control.MitArmParams(**values)


def motor_states(position, velocity=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)):
    return mit_control.MotorStates(
        position=np.asarray(position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
        torque=np.asarray(torque, dtype=float),
    )


def assert_mit_command(command, *, position, velocity, kp, kd, torque):
    assert command.position == pytest.approx(position)
    assert command.velocity == pytest.approx(velocity)
    assert command.kp == pytest.approx(kp)
    assert command.kd == pytest.approx(kd)
    assert command.torque == pytest.approx(torque)


class FakeBus:
    """A bus whose motors sit at fixed raw positions; it can stop the controller after some reads."""

    def __init__(self, position=(0.2, 0.5, 2.0), stop_after=None):
        self.motor_names = ("shoulder", "elbow", "gripper")
        self.enabled = False
        self.states = motor_states(position)
        self.error: Exception | None = None
        self.sent: list[tuple[str, mit_control.MitCommand]] = []
        self.disable_calls = 0
        self.hold: np.ndarray | None = None
        self.reads = 0
        self.stop_after = stop_after
        self.controller: mit_control.MitController | None = None

    def read_states(self):
        self.reads += 1
        if self.error is not None:
            raise self.error
        if self.stop_after is not None and self.reads >= self.stop_after:
            self.controller._stop.set()
        return self.states

    def enable_torque_at(self, hold):
        self.hold = hold
        self.enabled = True

    def send_mit(self, motor, command):
        self.sent.append((motor, command))

    def disable_torque(self):
        self.disable_calls += 1
        self.enabled = False


def make_controller(bus=None, gravity=(0.0, 0.0), fault_event=None, **overrides):
    bus = bus or FakeBus()
    controller = mit_control.MitController(
        "arm",
        bus,
        params(**overrides),
        lambda _: np.asarray(gravity),
        fault_event or threading.Event(),
        gripper(),
    )
    bus.controller = controller
    controller._set_initial_state(controller.read_state())
    return controller


@pytest.mark.parametrize("closed,opened,opening", [(1.0, 3.0, 0.75), (3.0, 1.0, 0.25)])
def test_joint_conversion_round_trips_the_gripper(closed, opened, opening):
    raw = np.array([0.3, 0.5, 2.5])
    joints = mit_control._motor_to_joint(raw, gripper(closed, opened))
    np.testing.assert_allclose(joints, [0.3, 0.5, opening])
    np.testing.assert_allclose(mit_control._joint_to_motor(joints, gripper(closed, opened)), raw)
    with pytest.raises(ValueError, match="calibrated stroke"):
        mit_control._motor_to_joint(np.array([0.0, 0.0, 3.5]), gripper(closed, opened))


def test_track_step_slews_tracks_clips_and_adds_gravity():
    position = np.array([0.2, 0.02, 0.5])
    previous = np.array([0.6, 0.02, 0.5])
    target = np.array([0.6, -0.5, 1.0])
    command, packet = mit_control._track_step(
        params(), gripper(), position, target, previous, np.array([8.0, 1.5]), dt=0.1
    )
    # shoulder: tracking band; elbow: slew then lower limit; gripper: slew (0.1 stroke per 0.1 s).
    np.testing.assert_allclose(command, [0.3, 0.0, 0.6])
    np.testing.assert_allclose(previous, [0.6, 0.02, 0.5])  # inputs are not mutated
    assert_mit_command(packet[0], position=0.3, velocity=0, kp=50, kd=2, torque=5.0)
    assert_mit_command(packet[1], position=0.0, velocity=0, kp=20, kd=1, torque=3.0)
    assert len(packet) == 2  # the gripper has its own force limiter


def test_track_step_returns_from_past_a_limit_without_jumping():
    position = np.array([0.2, -0.1, 0.5])  # elbow past its 0 lower limit
    command, _ = mit_control._track_step(
        params(), gripper(), position, position, position.copy(), np.zeros(2), dt=0.01
    )
    np.testing.assert_allclose(command, [0.2, -0.095, 0.5])  # back toward 0 at 0.5 rad/s
    # A command further out than the measured position is pulled back to it, never beyond.
    previous = np.array([0.2, -0.3, 0.5])
    command, _ = mit_control._track_step(
        params(), gripper(), position, position, previous, np.zeros(2), dt=0.01
    )
    assert command[1] == pytest.approx(-0.1)


@pytest.mark.parametrize("closed,opened", [(1.0, 3.0), (3.0, 1.0)])
def test_free_gripper_goes_straight_to_its_target(closed, opened):
    limiter = mit_control._GripperForceLimiter(gripper(closed, opened))
    command = limiter.step(position=0.5, velocity=1.0, torque=0.1, command=0.0, now=0.0)
    assert_mit_command(command, position=closed, velocity=0, kp=4, kd=0.1, torque=0)
    assert not limiter.blocked


def blocked_limiter(torque):
    """A closing gripper stopped by an object at mid-stroke (raw 2.0), pressing with ``torque``."""
    limiter = mit_control._GripperForceLimiter(gripper())
    command = limiter.step(position=0.5, velocity=0.0, torque=torque, command=0.0, now=0.0)
    assert limiter.blocked
    return limiter, command


def test_blocked_gripper_backs_off_to_the_force_limit():
    # Limit torque: 10 N * 0.1 m / 2 rad stroke + 0.3 Nm friction = 0.8 Nm.
    _, command = blocked_limiter(torque=0.8)
    assert command.position == pytest.approx(2.0)  # already at the limit: keep pressing as is
    # Pressing with 2 Nm: release (2 - 0.8) / kp 4 = 0.3 rad toward open.
    _, command = blocked_limiter(torque=2.0)
    assert command.position == pytest.approx(2.3)


def test_blocked_gripper_releases_when_asked_to_open():
    limiter, _ = blocked_limiter(torque=2.0)
    command = limiter.step(position=0.5, velocity=0.0, torque=2.0, command=1.0, now=0.01)
    assert not limiter.blocked
    assert command.position == pytest.approx(3.0)


def test_blocked_gripper_releases_once_its_torque_drops():
    limiter, _ = blocked_limiter(torque=2.0)
    limiter.step(position=0.5, velocity=0.0, torque=0.0, command=0.0, now=0.05)
    assert limiter.blocked  # 0.1 s average still above 0.2 Nm
    command = limiter.step(position=0.5, velocity=0.0, torque=0.0, command=0.0, now=0.2)
    assert not limiter.blocked
    assert command.position == pytest.approx(1.0)


def test_moving_gripper_is_not_considered_blocked():
    limiter = mit_control._GripperForceLimiter(gripper())
    limiter.step(position=0.5, velocity=0.5, torque=2.0, command=0.0, now=0.0)  # 1 rad/s raw
    assert not limiter.blocked


def test_joint_state_maps_the_gripper_rate_and_torque_to_its_stroke():
    bus = FakeBus()
    bus.states = motor_states([0.2, 0.5, 2.0], velocity=[0.5, 0.5, 1.0], torque=[2.0, 2.0, 0.3])
    state = mit_control._read_joint_state(bus, params(), gripper(closed=3.0, opened=1.0))
    np.testing.assert_allclose(state.position, [0.2, 0.5, 0.5])
    np.testing.assert_allclose(state.velocity, [0.5, 0.5, -0.5])  # 1 rad/s over a -2 rad stroke
    np.testing.assert_allclose(state.torque, [2.0, 2.0, -0.3])


def test_controller_holds_then_sends_one_command_per_motor():
    bus = FakeBus(stop_after=3)
    bus.enabled = True
    controller = make_controller(bus)
    bus.reads = 0
    controller._run()
    assert controller._failure is None
    assert [name for name, _ in bus.sent[:3]] == ["shoulder", "elbow", "gripper"]
    assert set(joint_stiffness(bus)) == {50.0, 20.0}  # holds with full stiffness while idle
    assert bus.disable_calls == 0
    controller.stop()
    assert bus.disable_calls == 1


def test_controller_holds_the_measured_position_once_commands_expire():
    bus = FakeBus()
    controller = make_controller(bus, command_timeout_s=0.01)
    bus.stop_after, bus.reads = 1, 0  # run exactly one controller cycle
    controller.set_target(np.array([0.5, 0.5, 1.0]))
    controller._commanded_at -= 1
    controller._run()
    assert controller._command_timed_out
    np.testing.assert_allclose(controller._target, controller.state.position)


def test_controller_copies_submitted_target():
    controller = make_controller()
    target = np.array([0.5, 0.5, 1.0])
    controller.set_target(target)
    target[0] = -0.5
    assert controller._target[0] == 0.5


def test_float_step_compensates_gravity_and_friction_without_stiffness():
    state = mit_control.JointState(
        position=np.array([0.2, 0.5, 0.5]), velocity=np.array([1.0, 1.0, 0.0]), torque=np.zeros(3)
    )
    command, commands = mit_control._float_step(
        params(), gripper(), state, np.array([0.0, 0.0, 0.8]), np.array([8.0, 1.5])
    )
    np.testing.assert_allclose(command, [0.2, 0.5, 0.8])  # the joints follow the arm, the gripper holds
    # Gravity 8 clipped to 5 Nm, plus friction 0.5 in the direction of motion.
    assert_mit_command(commands[0], position=0.2, velocity=0, kp=0, kd=0.2, torque=5.5)
    # Gravity 1.5 Nm doubled by its factor, plus friction 0.2.
    assert_mit_command(commands[1], position=0.5, velocity=0, kp=0, kd=0.1, torque=3.2)
    assert len(commands) == 2  # the gripper keeps its own limiter


def run_cycles(controller, cycles):
    """Run ``cycles`` full controller cycles in this thread, then leave the controller ready for more."""
    bus = controller._bus
    bus.enabled, bus.stop_after, bus.reads = True, cycles + 1, 0
    bus.sent.clear()
    controller._run()
    controller._stop.clear()
    controller._updated_at = time.monotonic()


def joint_stiffness(bus):
    return [command.kp for name, command in bus.sent if name != "gripper"]


def test_float_mode_floats_until_the_first_target_then_tracks_from_the_measured_position():
    bus = FakeBus()
    controller = make_controller(bus, idle_mode="float")
    bus.enabled = True
    run_cycles(controller, 2)
    assert joint_stiffness(bus) and set(joint_stiffness(bus)) == {0.0}

    measured = controller.state.position
    controller.set_target(measured + np.array([0.5, 0.5, 0.0]))
    run_cycles(controller, 1)
    assert set(joint_stiffness(bus)) == {50.0, 20.0}
    shoulder_goal = dict(bus.sent)["shoulder"].position
    assert shoulder_goal == pytest.approx(measured[0], abs=0.5 * 0.05)  # no jump from a stale command


def test_float_mode_floats_again_after_the_command_timeout():
    bus = FakeBus()
    controller = make_controller(bus, idle_mode="float", command_timeout_s=0.01)
    bus.enabled = True
    controller.set_target(controller.state.position)
    controller._commanded_at -= 1
    run_cycles(controller, 1)
    assert controller.idle
    assert set(joint_stiffness(bus)) == {0.0}


def test_fault_stops_commands_until_disconnect():
    bus = FakeBus()
    controller = make_controller(bus)
    run_cycles(controller, 1)
    bus.sent.clear()
    bus.stop_after = None
    bus.error = ConnectionError("lost")
    controller._run()
    assert bus.sent == []
    assert controller.fault_event.is_set()  # a fault here stops the other arm too
    assert bus.disable_calls == 0  # The motors' CAN timeout takes over until disconnect.
    with pytest.raises(ConnectionError, match="stopped") as exc_info:
        controller.latest()
    assert isinstance(exc_info.value.__cause__, ConnectionError)
    controller.stop()
    assert bus.disable_calls == 1


def test_stop_is_per_arm_but_a_fault_stops_both():
    fault_event = threading.Event()
    left, right = arms = [make_controller(fault_event=fault_event) for _ in range(2)]
    left.start()
    np.testing.assert_allclose(left._bus.hold, [0.2, 0.5, 2.0])  # torque on at the measured position
    right.start()
    left.stop()
    assert not left.active and not left._bus.enabled
    right.latest()  # the other arm keeps running
    left.start()  # and a stopped arm restarts without its owner
    fault_event.set()  # what a fault on either arm does
    for arm in arms:
        with pytest.raises(ConnectionError, match="stopped"):
            arm.latest()
        arm.stop()
    with pytest.raises(ConnectionError, match="can't start after a fault"):
        left.start()


def test_failed_start_leaves_torque_off():
    bus = FakeBus()
    controller = make_controller(bus)

    def enable_some_motors_then_fail(hold):
        bus.enabled = True
        raise RuntimeError("enable failed")

    bus.enable_torque_at = enable_some_motors_then_fail
    with pytest.raises(RuntimeError, match="enable failed"):
        controller.start()
    assert not controller.active
    assert not bus.enabled and bus.disable_calls == 1


def test_controller_start_refuses_a_second_thread():
    controller = make_controller()
    controller.start()
    thread = controller._thread
    with pytest.raises(RuntimeError, match="already running"):
        controller.start()
    assert controller._thread is thread
    controller.stop()


def test_controller_needs_the_gripper_calibration_to_start():
    bus = FakeBus()
    controller = mit_control.MitController("arm", bus, params(), lambda _: np.zeros(2), threading.Event())
    with pytest.raises(RuntimeError, match="gripper calibration"):
        controller.start()
    assert bus.hold is None  # torque stays off
    controller.set_gripper_params(gripper())
    controller.start()
    controller.stop()


def test_gripper_params_change_only_while_stopped():
    controller = make_controller()
    controller.start()
    with pytest.raises(RuntimeError, match="before changing the gripper"):
        controller.set_gripper_params(gripper())
    controller.stop()
    controller.set_gripper_params(gripper(closed=1.0, opened=5.0))
    controller.start()
    controller.stop()
    np.testing.assert_allclose(
        controller.state.position, [0.2, 0.5, 0.25]
    )  # raw 2.0 read with the new limits


def test_foreground_watchdog_allows_one_late_feedback_window():
    controller = make_controller()
    controller._updated_at = time.monotonic() - 1.5 * 0.2
    controller.latest()
    controller._updated_at = time.monotonic() - 3 * 0.2
    with pytest.raises(ConnectionError, match="stale"):
        controller.latest()
