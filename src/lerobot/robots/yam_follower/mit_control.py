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

"""MIT-mode (impedance) control loop with gravity compensation, independent of the arm's hardware.

The robot gives its settings as `MitArmParams`, its motors as a `MitMotorsBus`, and a gravity-torque
function; `MitController` runs the loop in a background thread. Joints are in radians and the gripper goes
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
# How far measured joints may go past their limits, which are not hard stops.
_JOINT_FEEDBACK_TOLERANCE_RAD = 0.15
# How far the measured gripper opening may go past its calibrated stroke, as a fraction of the stroke.
_GRIPPER_FEEDBACK_TOLERANCE = 0.10

# Blocked-gripper detection and force limiting, from I2RT's GripperForceLimiter.
_GRIPPER_TORQUE_WINDOW_S = 0.1
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
    """Measured arm state, in joint coordinates with a normalized gripper opening.

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
class MitGripperParams:
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
    """

    closed: float
    open: float
    kp: float
    kd: float
    max_speed: float
    force_limit_n: float
    finger_stroke_m: float

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
    """Limits the gripper's force once it is blocked on an object, and lets it move freely otherwise."""

    def __init__(self, gripper_params: MitGripperParams) -> None:
        self.gripper_params = gripper_params
        self.blocked = False
        self._torques: list[tuple[float, float]] = []
        self._limited_goal: float | None = None
        self._last_goal: float | None = None

    def step(self, position: float, velocity: float, torque: float, command: float, now: float) -> MitCommand:
        """Return the MIT command that moves the gripper toward `command`.

        Openings go from 0 (closed) to 1 (open), and `velocity` is in strokes per second.
        """
        params = self.gripper_params
        position_raw, command_raw = params.to_raw(position), params.to_raw(command)
        self._torques = [(t, value) for t, value in self._torques if now - t <= _GRIPPER_TORQUE_WINDOW_S]
        self._torques.append((now, torque))
        average_torque = abs(float(np.mean([value for _, value in self._torques])))
        speed_raw = abs(velocity * params.stroke)

        if self.blocked:
            if position < command or average_torque < _GRIPPER_RELEASED_TORQUE_NM:
                self.blocked = False
        elif average_torque > _GRIPPER_BLOCKED_TORQUE_NM and speed_raw < _GRIPPER_BLOCKED_SPEED_RAD_S:
            self.blocked = True

        if self.blocked:
            # Aim past the zero-torque position, so the position gain pushes with the torque that
            # gives force_limit_n at the fingers, plus friction.
            torque_limit = (
                params.force_limit_n * params.finger_stroke_m / abs(params.stroke) + _GRIPPER_FRICTION_NM
            )
            direction = float(np.sign(command_raw - position_raw))
            last_goal = position_raw if self._last_goal is None else self._last_goal
            zero_torque_raw = last_goal - direction * abs(torque) / params.kp
            goal = zero_torque_raw + direction * torque_limit / params.kp
            if self._limited_goal is None:
                self._limited_goal = goal
            self._limited_goal += _GRIPPER_GOAL_FILTER * (goal - self._limited_goal)
            goal = self._limited_goal
        else:
            # The next force limit starts from where the gripper is, as in I2RT.
            self._limited_goal = position_raw
            goal = command_raw
        goal = float(np.clip(goal, min(params.closed, params.open), max(params.closed, params.open)))
        self._last_goal = goal
        return MitCommand(position=goal, velocity=0.0, kp=params.kp, kd=params.kd, torque=0.0)


@dataclass(frozen=True, eq=False)
class MitArmParams:
    """Control settings of one arm, in internal units.

    Args:
        joint_limits (`np.ndarray`):
            Lower and upper arm-joint limits in radians, shaped `(joints, 2)`.
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
        float_kd (`np.ndarray`):
            Damping gains in float mode, where the joints have no stiffness.
        coulomb_friction (`np.ndarray`):
            Coulomb friction of each joint, compensated in float mode, in newton-metres; zeros disable it.
        control_frequency (`float`):
            Control loop frequency in hertz.
        feedback_timeout_s (`float`):
            Maximum time to wait for fresh motor feedback.
        command_timeout_s (`float`):
            Time without a target before the arm goes idle.
        idle_mode (`Literal["hold", "float"]`):
            What the idle arm does: `hold` keeps its position, `float` lets you move it by hand.
    """

    joint_limits: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    max_joint_speed: float
    max_tracking_error: float
    gravity_factors: np.ndarray
    max_gravity_torque: float
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


class MitMotorsBus(Protocol):
    """Motor bus that `MitController` drives.

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

    def enable_torque_at(self, hold: np.ndarray) -> None:
        """Enable torque for the arm, holding the given motor positions.

        Args:
            hold (`np.ndarray`):
                Motor positions in radians, in bus order.
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

    def disable_torque(self) -> None:
        """Disable torque for the arm; repeated calls must be safe."""
        ...


def _motor_to_joint(raw: np.ndarray, gripper_params: MitGripperParams) -> np.ndarray:
    """Convert raw motor positions to joint positions and the normalized gripper opening."""
    gripper = gripper_params.to_normalized(float(raw[-1]))
    if not -_GRIPPER_FEEDBACK_TOLERANCE <= gripper <= 1 + _GRIPPER_FEEDBACK_TOLERANCE:
        raise ValueError("Gripper feedback is outside the calibrated stroke; check the gripper limits")
    return np.r_[raw[:-1], np.clip(gripper, 0, 1)]


def _joint_to_motor(positions: np.ndarray, gripper_params: MitGripperParams) -> np.ndarray:
    """Convert joint positions and the normalized gripper opening to raw motor positions."""
    return np.r_[positions[:-1], gripper_params.to_raw(float(positions[-1]))]


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


def _read_joint_state(
    bus: MitMotorsBus, params: MitArmParams, gripper_params: MitGripperParams
) -> JointState:
    """Read fresh feedback and convert it to joint positions and a gripper opening.

    Raises:
        ValueError: If a joint or the gripper is outside the allowed feedback range.
    """
    raw = bus.read_states()
    n = params.num_joints
    position = _motor_to_joint(raw.position, gripper_params)
    for i, (value, (lower, upper)) in enumerate(zip(position[:n], params.joint_limits, strict=True)):
        if not lower - _JOINT_FEEDBACK_TOLERANCE_RAD <= value <= upper + _JOINT_FEEDBACK_TOLERANCE_RAD:
            raise ValueError(f"Joint {i} feedback {value} outside [{lower}, {upper}]")
    velocity = np.r_[raw.velocity[:n], raw.velocity[n] / gripper_params.stroke]
    torque = np.r_[raw.torque[:n], raw.torque[n] * np.sign(gripper_params.stroke)]
    return JointState(position=position, velocity=velocity, torque=torque)


def _track_step(
    params: MitArmParams,
    gripper_params: MitGripperParams,
    position: np.ndarray,
    target: np.ndarray,
    command: np.ndarray,
    gravity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, tuple[MitCommand, ...]]:
    """Move the command one step toward the target, and build stiff arm-joint commands that hold it."""
    n = params.num_joints
    # Move toward the target, inside the limits, no faster than the configured speeds.
    speeds = np.r_[np.full(n, params.max_joint_speed), gripper_params.max_speed]
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

    goal = _joint_to_motor(command, gripper_params)
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
    gripper_params: MitGripperParams,
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
    goal = _joint_to_motor(state.position, gripper_params)
    friction = params.coulomb_friction * np.sign(state.velocity[:n])
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
    torque = gravity * params.gravity_factors
    return np.clip(torque, -params.max_gravity_torque, params.max_gravity_torque)


class MitController:
    """Background control loop of one arm.

    Each cycle reads feedback, moves the command toward the latest target and sends it. Before the
    first target, and after `command_timeout_s` without one, the arm is idle: `idle_mode="hold"` holds
    the measured position, and `idle_mode="float"` lets you move the arm by hand, with gravity
    compensated.

    `start` enables torque and `stop` disables it. After a fault on this arm, or on the other arm of a
    bimanual robot, the loop stops sending commands, so the motors' CAN timeout disables them, and the
    next `latest` or `set_target` raises.

    Args:
        name (`str`):
            Name used in thread names and error messages.
        bus (`MitMotorsBus`):
            Motor bus of the arm.
        params (`MitArmParams`):
            Settings of the arm joints and the loop.
        gravity (`Callable[[np.ndarray], np.ndarray]`):
            Returns the arm-joint gravity torques for the measured positions.
        stop_event (`threading.Event`):
            Stop signal, shared by both controllers of a bimanual robot so a fault on one stops both.
        gripper_params (`MitGripperParams`, *optional*):
            Gripper calibration and settings; `None` until the gripper is calibrated.
    """

    def __init__(
        self,
        name: str,
        bus: MitMotorsBus,
        params: MitArmParams,
        gravity: Callable[[np.ndarray], np.ndarray],
        stop_event: threading.Event,
        gripper_params: MitGripperParams | None = None,
    ) -> None:
        self.name = name
        self._bus = bus
        self._params = params
        self._gravity = gravity
        self.stop_event = stop_event
        self._gripper_params = gripper_params
        # Empty until start() reads the arm.
        self.state = JointState(position=np.zeros(0), velocity=np.zeros(0), torque=np.zeros(0))
        self._target = np.zeros(0)
        self._command = np.zeros(0)
        self._updated_at = 0.0
        self._commanded_at = 0.0
        self._command_timed_out = False
        self._has_target = False
        self._failure: Exception | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._gripper: _GripperForceLimiter | None = None

    @property
    def active(self) -> bool:
        """Whether the controller runs: from `start` until `stop` succeeds, even after a fault."""
        return self._thread is not None

    def set_gripper_params(self, gripper_params: MitGripperParams) -> None:
        """Replace the gripper calibration and settings, for example after calibrating the gripper.

        Args:
            gripper_params (`MitGripperParams`):
                New gripper calibration and settings.

        Raises:
            RuntimeError: If the controller is running.
        """
        if self.active:
            raise RuntimeError(f"{self.name} controller is running; stop it before changing the gripper")
        self._gripper_params = gripper_params

    def read_state(self) -> JointState:
        """Read fresh feedback, as joint positions and a normalized gripper opening.

        Returns:
            `JointState`: Joint and gripper state.

        Raises:
            RuntimeError: If the gripper is not calibrated.
            ValueError: If a joint or the gripper is outside the allowed feedback range.
        """
        return _read_joint_state(self._bus, self._params, self._require_gripper_params())

    @property
    def idle(self) -> bool:
        """Whether the arm is waiting for a target: before the first one or after the timeout."""
        return not self._has_target or self._command_timed_out

    def start(self) -> None:
        """Read the arm, enable torque at its measured position, then start the background control thread.

        Raises:
            RuntimeError: If the controller is already running or the gripper is not calibrated.
            ValueError: If a joint or the gripper is outside the allowed feedback range.
        """
        if self.active:
            raise RuntimeError(f"{self.name} controller is already running")
        self._set_initial_state(self.read_state())
        self._failure = None
        self._bus.enable_torque_at(_joint_to_motor(self.state.position, self._require_gripper_params()))
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-controller", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._thread = None
            raise

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the control thread and disable the motors.

        Args:
            timeout_s (`float`, *optional*, defaults to `2.0`):
                Maximum time to wait for the control thread.

        Raises:
            RuntimeError: If the control thread does not stop before the timeout.
        """
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise RuntimeError(f"{self.name} controller did not stop; use the hardware e-stop")
            self._thread = None
        self._bus.disable_torque()

    def latest(self) -> JointState:
        """Return the latest measured state.

        Returns:
            `JointState`: A copy of the latest state.

        Raises:
            ConnectionError: If the controller stopped or its feedback is stale.
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
            ConnectionError: If the controller stopped or its feedback is stale.
        """
        with self._lock:
            self._check_healthy()
            if self.idle and self._params.idle_mode == "float":
                # Resume tracking from wherever the arm was moved by hand.
                self._command = self.state.position.copy()
            self._target = target.copy()
            self._commanded_at = time.monotonic()
            self._command_timed_out = False
            self._has_target = True

    def _set_initial_state(self, state: JointState) -> None:
        """Hold the measured state until the first target: the target and command start there."""
        self._gripper = _GripperForceLimiter(self._require_gripper_params())
        self.state = state
        self._target = state.position.copy()
        self._command = state.position.copy()
        self._updated_at = self._commanded_at = time.monotonic()
        self._command_timed_out = False
        self._has_target = False

    def _require_gripper_params(self) -> MitGripperParams:
        if self._gripper_params is None:
            raise RuntimeError(f"{self.name} controller needs the gripper calibration")
        return self._gripper_params

    def _check_healthy(self) -> None:
        if self._failure is not None or self.stop_event.is_set():
            raise ConnectionError(
                f"{self.name} controller stopped after a motor/feedback error"
            ) from self._failure
        # A healthy loop's own read can take a full feedback timeout, so allow two plus a cycle.
        progress_timeout_s = 2 * self._params.feedback_timeout_s + 1 / self._params.control_frequency
        if time.monotonic() - self._updated_at > progress_timeout_s:
            self.stop_event.set()
            raise ConnectionError(
                f"{self.name} controller feedback is stale; reconnect before commanding motion"
            )

    def _run(self) -> None:
        previous = started = time.monotonic()
        max_cycle_gap = 0.0
        try:
            params, gripper_params, gripper = self._params, self._gripper_params, self._gripper
            if gripper_params is None or gripper is None:
                raise RuntimeError(f"{self.name} controller runs only after start() reads the arm")
            while not self.stop_event.is_set():
                started = time.monotonic()
                max_cycle_gap = max(max_cycle_gap, started - previous)
                state = _read_joint_state(self._bus, params, gripper_params)
                position = state.position
                with self._lock:
                    self.state = state
                    self._updated_at = time.monotonic()
                    if (
                        started - self._commanded_at > params.command_timeout_s
                        and not self._command_timed_out
                    ):
                        self._target = position.copy()
                        self._command = position.copy()
                        self._command_timed_out = True
                    packet: list[tuple[str, MitCommand]] = []
                    # Torque can be off while this thread still runs, after a stop() that timed out.
                    if self._bus.enabled:
                        n = params.num_joints
                        if self.idle and params.idle_mode == "float":
                            self._command, joint_commands = _float_step(
                                params, gripper_params, state, self._command, self._gravity(position)
                            )
                        else:
                            self._command, joint_commands = _track_step(
                                params,
                                gripper_params,
                                position,
                                self._target,
                                self._command,
                                self._gravity(position),
                                min(started - previous, _MAX_CONTROL_DT_S),
                            )
                        packet = list(zip(self._bus.motor_names[:n], joint_commands, strict=True))
                        packet.append(
                            (
                                self._bus.motor_names[n],
                                gripper.step(
                                    float(position[n]),
                                    float(state.velocity[n]),
                                    float(state.torque[n]),
                                    float(self._command[n]),
                                    started,
                                ),
                            )
                        )
                for name, command in packet:
                    if self.stop_event.is_set():
                        break
                    self._bus.send_mit(name, command)
                previous = started
                self.stop_event.wait(max(0, 1 / params.control_frequency - (time.monotonic() - started)))
        except Exception as exc:
            exc.add_note(
                f"{self.name} maximum control cycle gap: {max_cycle_gap * 1000:.1f} ms; "
                f"current cycle elapsed: {(time.monotonic() - started) * 1000:.1f} ms"
            )
            self._failure = exc
            self.stop_event.set()
            logger.error(
                "%s controller stopped: %s; %s",
                self.name,
                self._failure,
                "; ".join(getattr(self._failure, "__notes__", [])),
            )
