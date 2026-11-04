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

"""MIT-mode (impedance) servo loop with gravity compensation, independent of the arm's hardware.

The robot gives its settings as `MitArmParams`, its motors as a `MitArmBus`, and a gravity-torque
function; `MitServo` runs the loop in a background thread. Joints are in radians and the gripper goes
from 0 (closed) to 1 (open).
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Cap on a cycle's time step, so a late cycle can't make the command jump.
_MAX_CONTROL_DT_S = 0.05

# Blocked-gripper detection and force limiting, from I2RT's GripperForceLimiter.
_GRIPPER_EFFORT_WINDOW_S = 0.1
_GRIPPER_BLOCKED_TORQUE_NM = 0.5
_GRIPPER_BLOCKED_SPEED_RAD_S = 0.3
_GRIPPER_RELEASED_TORQUE_NM = 0.2
_GRIPPER_FRICTION_NM = 0.3
_GRIPPER_GOAL_FILTER = 0.1


@dataclass(frozen=True)
class MitCommand:
    """Command for one MIT-mode motor.

    Args:
        position (`float`):
            Target motor position in radians.
        velocity (`float`):
            Target motor velocity in radians per second.
        kp (`float`):
            Position gain.
        kd (`float`):
            Damping gain.
        torque (`float`):
            Feedforward torque in newton-metres.
    """

    position: float
    velocity: float
    kp: float
    kd: float
    torque: float


@dataclass(frozen=True, eq=False)
class MotorStates:
    """Raw feedback in bus order: arm joints, then the gripper.

    Args:
        position (`np.ndarray`):
            Motor positions in radians.
        velocity (`np.ndarray`):
            Motor velocities in radians per second.
        torque (`np.ndarray`):
            Motor torques in newton-metres.
    """

    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray


@dataclass(frozen=True, eq=False)
class JointState:
    """Measured arm state in calibrated joint coordinates.

    Args:
        position (`np.ndarray`):
            Joint positions in radians followed by the normalized gripper opening.
        velocity (`np.ndarray`):
            Joint velocities in radians per second followed by gripper strokes per second.
        torque (`np.ndarray`):
            Joint and gripper motor torques in newton-metres.
    """

    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray


@dataclass(frozen=True, eq=False)
class GripperParams:
    """Calibration and control settings of a linear MIT-mode gripper.

    Args:
        closed (`float`):
            Motor position of the closed gripper, in radians.
        open (`float`):
            Motor position of the open gripper, in radians.
        kp (`float`):
            Position gain.
        kd (`float`):
            Damping gain.
        max_speed (`float`):
            Maximum speed in full strokes per second.
        force_limit_n (`float`):
            Finger force once the gripper is blocked on an object, in newtons.
        finger_stroke_m (`float`):
            Finger stroke used to convert the force limit to motor torque, in metres.
        feedback_tolerance (`float`):
            How far the measured opening may go past `closed` and `open`, as a fraction of the stroke.
    """

    closed: float
    open: float
    kp: float
    kd: float
    max_speed: float
    force_limit_n: float
    finger_stroke_m: float
    feedback_tolerance: float

    @property
    def stroke(self) -> float:
        """Signed motor stroke from closed to open, in radians."""
        return self.open - self.closed

    def to_normalized(self, raw: float) -> float:
        """Convert a raw motor position to an opening from zero to one."""
        return (raw - self.closed) / self.stroke

    def to_raw(self, normalized: float) -> float:
        """Convert a normalized opening to a raw motor position."""
        return self.closed + normalized * self.stroke


class _GripperForceLimiter:
    """Gripper commands that move freely, but limit the force once the gripper is blocked on an object.

    Port of I2RT's `GripperForceLimiter`. The gripper is blocked when its average torque over the last
    0.1 s is high while it barely moves. It then pushes with the torque that gives `force_limit_n` at
    the fingers, until the target opens it or the torque drops.
    """

    def __init__(self, gripper_params: GripperParams) -> None:
        self.gripper_params = gripper_params
        self.blocked = False
        self._efforts: list[tuple[float, float]] = []
        self._limited_goal: float | None = None
        self._last_goal: float | None = None

    def command(
        self, measured: float, velocity: float, torque: float, commanded: float, now: float
    ) -> MitCommand:
        """Return the MIT command that moves the gripper toward `commanded`.

        Openings go from 0 (closed) to 1 (open), and `velocity` is in strokes per second.
        """
        params = self.gripper_params
        current_raw, target_raw = params.to_raw(measured), params.to_raw(commanded)
        self._efforts = [(t, e) for t, e in self._efforts if now - t <= _GRIPPER_EFFORT_WINDOW_S]
        self._efforts.append((now, torque))
        average_effort = abs(float(np.mean([e for _, e in self._efforts])))
        speed_raw = abs(velocity * params.stroke)

        if self.blocked:
            if measured < commanded or average_effort < _GRIPPER_RELEASED_TORQUE_NM:
                self.blocked = False
        elif average_effort > _GRIPPER_BLOCKED_TORQUE_NM and speed_raw < _GRIPPER_BLOCKED_SPEED_RAD_S:
            self.blocked = True

        if self.blocked:
            # Aim past the zero-torque position, so the position gain pushes with the torque that
            # gives force_limit_n at the fingers, plus friction.
            limit_torque = (
                params.force_limit_n * params.finger_stroke_m / abs(params.stroke) + _GRIPPER_FRICTION_NM
            )
            direction = float(np.sign(target_raw - current_raw))
            last_goal = current_raw if self._last_goal is None else self._last_goal
            zero_torque_raw = last_goal - direction * abs(torque) / params.kp
            goal = zero_torque_raw + direction * limit_torque / params.kp
            if self._limited_goal is None:
                self._limited_goal = goal
            self._limited_goal += _GRIPPER_GOAL_FILTER * (goal - self._limited_goal)
            goal = self._limited_goal
        else:
            # The next force limit starts from where the gripper is, as in I2RT.
            self._limited_goal = current_raw
            goal = target_raw
        goal = float(np.clip(goal, min(params.closed, params.open), max(params.closed, params.open)))
        self._last_goal = goal
        return MitCommand(position=goal, velocity=0.0, kp=params.kp, kd=params.kd, torque=0.0)


@dataclass(frozen=True, eq=False)
class MitArmParams:
    """Control settings of one arm, in internal units.

    Args:
        joint_limits (`np.ndarray`):
            Lower and upper arm-joint limits in radians, shaped `(joints, 2)`.
        joint_feedback_tolerance_rad (`float`):
            How far measured joints may go past `joint_limits`, which are not hard stops, in radians.
        joint_signs (`np.ndarray`):
            Direction of each joint relative to its motor, `1` or `-1`.
        joint_offsets (`np.ndarray`):
            Joint positions when the motors read zero, in radians.
        kp (`np.ndarray`):
            Position gains for the arm joints.
        kd (`np.ndarray`):
            Damping gains for the arm joints.
        max_joint_speed (`float`):
            Maximum joint speed in radians per second.
        max_tracking_error (`float`):
            Maximum difference between the measured and commanded positions, in radians.
        gravity_factors (`np.ndarray`):
            Per-joint multipliers of the model's gravity torques.
        max_gravity_torque (`float`):
            Cap on each joint's gravity torque, in newton-metres.
        gripper (`GripperParams`):
            Gripper calibration and control settings.
        float_kd (`np.ndarray`):
            Damping gains in float mode, where the joints have no stiffness.
        coulomb_friction (`np.ndarray`):
            Coulomb friction of each joint, compensated in float mode, in newton-metres; zeros disable it.
        control_frequency (`float`):
            Servo loop frequency in hertz.
        feedback_timeout_s (`float`):
            Maximum time to wait for fresh motor feedback.
        command_timeout_s (`float`):
            Time without a target before the arm goes idle.
        idle_mode (`Literal["hold", "float"]`):
            What the idle arm does: `hold` keeps its position, `float` lets you move it by hand.
    """

    joint_limits: np.ndarray
    joint_feedback_tolerance_rad: float
    joint_signs: np.ndarray
    joint_offsets: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    max_joint_speed: float
    max_tracking_error: float
    gravity_factors: np.ndarray
    max_gravity_torque: float
    gripper: GripperParams
    float_kd: np.ndarray
    coulomb_friction: np.ndarray
    control_frequency: float
    feedback_timeout_s: float
    command_timeout_s: float
    idle_mode: Literal["hold", "float"]

    @property
    def num_joints(self) -> int:
        """Number of arm joints, not counting the gripper."""
        return len(self.kp)


class MitArmBus(Protocol):
    """Motor bus that `MitServo` drives.

    States follow `motor_names`, in radians, radians per second and newton-metres.

    **Attributes**:
        - **motor_names** (`tuple[str, ...]`) -- Motor names, arm joints first and the gripper last.
        - **enabled** (`bool`) -- Whether torque is enabled for the arm.
    """

    motor_names: tuple[str, ...]
    enabled: bool

    def read_states(self) -> MotorStates:
        """Return fresh feedback from every motor in bus order.

        Returns:
            `MotorStates`: The latest position, velocity and torque of every motor.

        Raises:
            ConnectionError: If feedback is missing or stale, or a motor reports a fault.
        """
        ...

    def send_mit(self, motor: str, command: MitCommand) -> None:
        """Send one MIT command to a motor.

        Args:
            motor (`str`):
                Name from `motor_names`.
            command (`MitCommand`):
                Command to send.
        """
        ...

    def disable(self) -> None:
        """Disable torque for the arm; repeated calls must be safe."""
        ...


def _motor_to_joint(raw: np.ndarray, params: MitArmParams) -> np.ndarray:
    """Convert raw motor positions to calibrated joint and gripper positions."""
    n = params.num_joints
    joints = raw[:n] * params.joint_signs + params.joint_offsets
    gripper = params.gripper.to_normalized(float(raw[n]))
    if not -params.gripper.feedback_tolerance <= gripper <= 1 + params.gripper.feedback_tolerance:
        raise ValueError("Gripper feedback is outside the calibrated stroke; check the gripper limits")
    return np.r_[joints, np.clip(gripper, 0, 1)]


def joint_to_motor(positions: np.ndarray, params: MitArmParams) -> np.ndarray:
    """Convert calibrated joint and gripper positions to raw motor positions.

    Args:
        positions (`np.ndarray`):
            Joint positions in radians followed by the normalized gripper opening.
        params (`MitArmParams`):
            Settings of the arm.

    Returns:
        `np.ndarray`: Motor positions in radians, in bus order.
    """
    n = params.num_joints
    raw = (positions[:n] - params.joint_offsets) / params.joint_signs
    return np.r_[raw, params.gripper.to_raw(float(positions[n]))]


def clip_to_limits(values: np.ndarray, joint_limits: np.ndarray) -> np.ndarray:
    """Clip joint positions to `joint_limits`, and the gripper opening to 0 to 1.

    Args:
        values (`np.ndarray`):
            Joint positions in radians followed by the normalized gripper opening.
        joint_limits (`np.ndarray`):
            Lower and upper arm-joint limits in radians, shaped `(joints, 2)`.

    Returns:
        `np.ndarray`: A clipped copy of `values`.
    """
    n = len(joint_limits)
    clipped = values.copy()
    clipped[:n] = np.clip(clipped[:n], *np.asarray(joint_limits).T)
    clipped[n] = np.clip(clipped[n], 0.0, 1.0)
    return clipped


def read_joint_state(bus: MitArmBus, params: MitArmParams) -> JointState:
    """Read fresh feedback and convert it to joint positions and a gripper opening.

    Args:
        bus (`MitArmBus`):
            Motor bus of the arm.
        params (`MitArmParams`):
            Settings of the arm.

    Returns:
        `JointState`: Joint and gripper state.

    Raises:
        ValueError: If a joint or the gripper is outside the allowed feedback range.
    """
    raw = bus.read_states()
    n = params.num_joints
    position = _motor_to_joint(raw.position, params)
    for i, (value, (lower, upper)) in enumerate(zip(position[:n], params.joint_limits, strict=True)):
        if not (
            lower - params.joint_feedback_tolerance_rad
            <= value
            <= upper + params.joint_feedback_tolerance_rad
        ):
            raise ValueError(f"Joint {i} feedback {value} outside [{lower}, {upper}]")
    gripper = params.gripper
    velocity = np.r_[raw.velocity[:n] * params.joint_signs, raw.velocity[n] / gripper.stroke]
    torque = np.r_[raw.torque[:n] * params.joint_signs, raw.torque[n] * np.sign(gripper.stroke)]
    return JointState(position=position, velocity=velocity, torque=torque)


def _track_step(
    params: MitArmParams,
    position: np.ndarray,
    target: np.ndarray,
    command: np.ndarray,
    gravity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, tuple[MitCommand, ...]]:
    """Move the command one step toward the target, and build stiff arm-joint commands that hold it."""
    n = params.num_joints
    # Move toward the target, inside the limits, no faster than the configured speeds.
    speeds = np.r_[np.full(n, params.max_joint_speed), params.gripper.max_speed]
    target = clip_to_limits(target, params.joint_limits)
    command = command + np.clip(target - command, -speeds * dt, speeds * dt)
    # Keep joints near the measured position, so a blocked or pushed arm limits its force.
    band = params.max_tracking_error
    command[:n] = np.clip(command[:n], position[:n] - band, position[:n] + band)
    # An arm past a limit comes back at the speed limit: command no further out than the measured
    # position, instead of jumping to the limit.
    lower, upper = np.asarray(params.joint_limits).T
    command[:n] = np.clip(command[:n], np.minimum(lower, position[:n]), np.maximum(upper, position[:n]))
    command[n] = np.clip(command[n], 0.0, 1.0)

    goal = joint_to_motor(command, params)
    torque = _gravity_feedforward(params, gravity)
    return command, tuple(
        MitCommand(
            position=float(goal[i]),
            velocity=0.0,
            kp=float(params.kp[i]),
            kd=float(params.kd[i]),
            torque=float(torque[i]),
        )
        for i in range(n)
    )


def _float_step(
    params: MitArmParams,
    state: JointState,
    command: np.ndarray,
    gravity: np.ndarray,
) -> tuple[np.ndarray, tuple[MitCommand, ...]]:
    """Let the arm be moved by hand, while the gripper keeps its commanded opening.

    The command follows the measured joints. The arm-joint commands have no stiffness, light damping, the
    gravity torque of the measured position and, when configured, Coulomb friction compensation.
    """
    n = params.num_joints
    command = np.r_[state.position[:n], command[n:]]
    goal = joint_to_motor(state.position, params)
    friction = params.coulomb_friction * np.sign(state.velocity[:n]) * params.joint_signs
    torque = _gravity_feedforward(params, gravity) + friction
    return command, tuple(
        MitCommand(
            position=float(goal[i]),
            velocity=0.0,
            kp=0.0,
            kd=float(params.float_kd[i]),
            torque=float(torque[i]),
        )
        for i in range(n)
    )


def _gravity_feedforward(params: MitArmParams, gravity: np.ndarray) -> np.ndarray:
    """Scale the model's gravity torques by `gravity_factors`, and cap them."""
    torque = gravity * params.gravity_factors * params.joint_signs
    return np.clip(torque, -params.max_gravity_torque, params.max_gravity_torque)


class MitServo:
    """Background servo loop of one arm.

    Each cycle reads feedback, moves the command toward the latest target and sends it. Before the
    first target, and after `command_timeout_s` without one, the arm is idle: `idle_mode="hold"` holds
    the measured position, and `idle_mode="float"` lets you move the arm by hand, with gravity
    compensated.

    After a fault on this arm, or on the other arm of a bimanual robot, the loop stops sending
    commands, so the motors' CAN timeout disables them, and the next `latest` or `set_target` raises.
    `stop` disables torque on a normal shutdown.

    Args:
        name (`str`):
            Name used in thread names and error messages.
        bus (`MitArmBus`):
            Motor bus of the arm.
        gravity (`Callable[[np.ndarray], np.ndarray]`):
            Returns the arm-joint gravity torques for the measured positions.
        stop_event (`threading.Event`):
            Stop signal, shared by both servos of a bimanual robot so a fault on one stops both.
    """

    def __init__(
        self,
        name: str,
        bus: MitArmBus,
        gravity: Callable[[np.ndarray], np.ndarray],
        stop_event: threading.Event,
    ) -> None:
        self.name = name
        self.bus = bus
        # Set by reset(), which every start() needs first.
        self.params: MitArmParams | None = None
        self.gravity = gravity
        self.stop_event = stop_event
        self.state = JointState(position=np.zeros(0), velocity=np.zeros(0), torque=np.zeros(0))
        self.target = np.zeros(0)
        self.command = np.zeros(0)
        self.updated_at = 0.0
        self.commanded_at = 0.0
        self.command_timed_out = False
        self.has_target = False
        self.failure: Exception | None = None
        # True from start() until stop() succeeds, even if the loop already exited on a fault.
        self.active = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._gripper: _GripperForceLimiter | None = None

    def reset(self, state: JointState, params: MitArmParams) -> None:
        """Reset the target and command to a measured state, before the first action.

        Args:
            state (`JointState`):
                Measured state to start from.
            params (`MitArmParams`):
                Settings of the arm.
        """
        self.params = params
        self.state = state
        self.target = state.position.copy()
        self.command = state.position.copy()
        self.updated_at = self.commanded_at = time.monotonic()
        self.command_timed_out = False
        self.has_target = False
        self._gripper = _GripperForceLimiter(params.gripper)

    @property
    def idle(self) -> bool:
        """Whether the arm is waiting for a target: before the first one or after the timeout."""
        return not self.has_target or self.command_timed_out

    def start(self) -> None:
        """Start the background servo thread.

        Raises:
            RuntimeError: If the servo is already running or was never reset.
        """
        if self.active or self._thread is not None:
            raise RuntimeError(f"{self.name} servo is already running")
        if self.params is None:
            raise RuntimeError(f"{self.name} servo needs reset() with the arm settings before it starts")
        self.failure = None
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-servo", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._thread = None
            raise
        self.active = True

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the servo thread and disable the motors.

        Args:
            timeout_s (`float`, *optional*, defaults to `2.0`):
                Maximum time to wait for the servo thread.

        Raises:
            RuntimeError: If the servo thread does not stop before the timeout.
        """
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise RuntimeError(f"{self.name} servo did not stop; use the hardware e-stop")
            self._thread = None
        self.bus.disable()
        self.active = False

    def latest(self) -> JointState:
        """Return the latest measured state.

        Returns:
            `JointState`: A copy of the latest state.

        Raises:
            ConnectionError: If the servo stopped or its feedback is stale.
        """
        with self._lock:
            self._check_healthy()
            state = self.state
        return JointState(
            position=state.position.copy(), velocity=state.velocity.copy(), torque=state.torque.copy()
        )

    def set_target(self, target: np.ndarray) -> None:
        """Set the target the loop moves toward.

        Args:
            target (`np.ndarray`):
                Joint positions in radians followed by the normalized gripper opening.

        Raises:
            ConnectionError: If the servo stopped or its feedback is stale.
        """
        with self._lock:
            params = self._check_healthy()
            if self.idle and params.idle_mode == "float":
                # Resume tracking from wherever the arm was moved by hand.
                self.command = self.state.position.copy()
            self.target = target.copy()
            self.commanded_at = time.monotonic()
            self.command_timed_out = False
            self.has_target = True

    def _check_healthy(self) -> MitArmParams:
        if self.failure is not None or self.stop_event.is_set():
            raise ConnectionError(f"{self.name} servo stopped after a motor/feedback error") from self.failure
        params = self.params
        if params is None:
            raise ConnectionError(f"{self.name} servo was never reset")
        # A healthy loop's own read can take a full feedback timeout, so allow two plus a cycle.
        progress_timeout_s = 2 * params.feedback_timeout_s + 1 / params.control_frequency
        if time.monotonic() - self.updated_at > progress_timeout_s:
            self.stop_event.set()
            raise ConnectionError(f"{self.name} servo feedback is stale; reconnect before commanding motion")
        return params

    def _run(self) -> None:
        previous = started = time.monotonic()
        max_cycle_gap = 0.0
        try:
            params = self.params
            if params is None:
                raise RuntimeError(f"{self.name} servo was never reset")
            while not self.stop_event.is_set():
                started = time.monotonic()
                max_cycle_gap = max(max_cycle_gap, started - previous)
                state = read_joint_state(self.bus, params)
                position = state.position
                with self._lock:
                    self.state = state
                    self.updated_at = time.monotonic()
                    if started - self.commanded_at > params.command_timeout_s and not self.command_timed_out:
                        self.target = position.copy()
                        self.command = position.copy()
                        self.command_timed_out = True
                    joint_commands: tuple[MitCommand, ...] = ()
                    n = params.num_joints
                    if self.bus.enabled and self.idle and params.idle_mode == "float":
                        self.command, joint_commands = _float_step(
                            params, state, self.command, self.gravity(position)
                        )
                    elif self.bus.enabled:
                        self.command, joint_commands = _track_step(
                            params,
                            position,
                            self.target,
                            self.command,
                            self.gravity(position),
                            min(started - previous, _MAX_CONTROL_DT_S),
                        )
                    packet: list[tuple[str, MitCommand]] = []
                    if self.bus.enabled:
                        packet.extend(zip(self.bus.motor_names[:n], joint_commands, strict=True))
                    if self.bus.enabled and self._gripper is not None:
                        packet.append(
                            (
                                self.bus.motor_names[n],
                                self._gripper.command(
                                    float(position[n]),
                                    float(state.velocity[n]),
                                    float(state.torque[n]),
                                    float(self.command[n]),
                                    started,
                                ),
                            )
                        )
                for name, command in packet:
                    if self.stop_event.is_set():
                        break
                    self.bus.send_mit(name, command)
                previous = started
                self.stop_event.wait(max(0, 1 / params.control_frequency - (time.monotonic() - started)))
        except Exception as exc:
            exc.add_note(
                f"{self.name} maximum servo cycle gap: {max_cycle_gap * 1000:.1f} ms; "
                f"current cycle elapsed: {(time.monotonic() - started) * 1000:.1f} ms"
            )
            self.failure = exc
            self.stop_event.set()
            logger.error(
                "%s servo stopped: %s; %s",
                self.name,
                self.failure,
                "; ".join(getattr(self.failure, "__notes__", [])),
            )
