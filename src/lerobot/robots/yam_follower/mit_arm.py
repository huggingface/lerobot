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

"""Control loop for arms whose motors run in MIT (impedance) mode with gravity feed-forward.

Nothing here is specific to one robot. A robot provides its joint names, limits and gains
(``MitArmParams``), its gripper (``CalibratedGripper``), a bus implementing ``MitArmBus`` and
a gravity model; ``MitServo`` then holds and moves the arm from a background thread.

Internal units: joints in radians, the gripper from 0 (closed) to 1 (open).
"""

import gc
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Measured poses may exceed the model limits, which are not mechanical stops.
FEEDBACK_LIMIT_TOLERANCE_RAD = 0.15
GRIPPER_FEEDBACK_MARGIN = 0.10

# One MIT command per motor: (position_rad, velocity_rad_s, kp, kd, feedforward_torque_nm).
MitCommand = tuple[float, float, float, float, float]


@dataclass(frozen=True, eq=False)
class MotorStates:
    """Raw motor states in bus order (arm joints, then the gripper): rad, rad/s and Nm."""

    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray


@dataclass(frozen=True, eq=False)
class JointState:
    """Measured joint state: arm joints in radians, then the gripper from 0 (closed) to 1 (open)."""

    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray


@dataclass(frozen=True, eq=False)
class CalibratedGripper:
    """A gripper motor in MIT mode, normalized between two measured raw stops.

    Its proportional torque is capped by keeping the commanded position within
    ``torque_limit / kp`` of the measured one.
    """

    closed: float
    open: float
    kp: float
    kd: float
    torque_limit: float
    # Fastest the commanded opening moves, in full strokes per second.
    max_speed: float

    @property
    def stroke(self) -> float:
        return self.open - self.closed

    def to_normalized(self, raw: float) -> float:
        return (raw - self.closed) / self.stroke

    def to_raw(self, normalized: float) -> float:
        return self.closed + normalized * self.stroke

    def command(self, measured: float, commanded: float) -> MitCommand:
        """Build the MIT command moving from the ``measured`` toward the ``commanded`` opening."""
        measured_raw = self.to_raw(measured)
        band = self.torque_limit / self.kp
        goal = float(np.clip(self.to_raw(commanded), measured_raw - band, measured_raw + band))
        return (goal, 0.0, self.kp, self.kd, 0.0)


@dataclass(frozen=True, eq=False)
class MitArmParams:
    """Control settings of one arm in internal units."""

    # Arm joints, then the gripper, in bus order.
    motor_names: tuple[str, ...]
    joint_limits: np.ndarray  # (joints, 2) in rad
    joint_signs: np.ndarray
    joint_offsets: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    max_joint_speed: float
    max_tracking_error: float
    gravity_factors: np.ndarray
    max_gravity_torque: float
    gripper: CalibratedGripper
    # Damping applied to every motor (joints, then the gripper) after a fault, with kp = 0.
    fault_damping_kd: np.ndarray
    # Damping of the arm joints in float mode (kp = 0, live gravity compensation).
    float_kd: np.ndarray
    # Coulomb friction of the arm joints compensated in float mode, in Nm; zeros disable it.
    coulomb_friction: np.ndarray

    @property
    def num_joints(self) -> int:
        return len(self.kp)


class MitArmBus(Protocol):
    """What ``MitServo`` needs from a robot's motor bus."""

    enabled: bool

    def read_states(self) -> MotorStates: ...

    def send_mit(self, motor: str, command: MitCommand) -> None: ...

    def disable(self) -> None: ...


class ServoSettings(Protocol):
    """Timing settings ``MitServo`` reads from a robot config."""

    control_frequency: float
    feedback_timeout_s: float
    command_timeout_s: float
    freeze_gc: bool
    # What the arm does before the first target and after the command timeout: "hold" or "float".
    idle_mode: str


def motor_to_joint(raw: np.ndarray, params: MitArmParams) -> np.ndarray:
    if not np.isfinite(raw).all():
        raise ConnectionError("Non-finite motor feedback")
    n = params.num_joints
    joints = raw[:n] * params.joint_signs + params.joint_offsets
    gripper = params.gripper.to_normalized(raw[n])
    if not -GRIPPER_FEEDBACK_MARGIN <= gripper <= 1 + GRIPPER_FEEDBACK_MARGIN:
        raise ValueError("Gripper feedback is outside the calibrated stroke; check endpoints")
    return np.r_[joints, np.clip(gripper, 0, 1)]


def joint_to_motor(positions: np.ndarray, params: MitArmParams) -> np.ndarray:
    n = params.num_joints
    raw = (positions[:n] - params.joint_offsets) / params.joint_signs
    return np.r_[raw, params.gripper.to_raw(float(positions[n]))]


def validate_positions(values: np.ndarray, joint_limits: np.ndarray, *, joint_tolerance_rad: float) -> None:
    n = len(joint_limits)
    if values.shape != (n + 1,) or not np.isfinite(values).all():
        raise ValueError(f"Expected {n} finite joint radians and one normalized gripper position")
    for i, (value, (lower, upper)) in enumerate(zip(values, (*joint_limits, (0, 1)), strict=True)):
        tolerance = joint_tolerance_rad if i < n else 0.0
        if not lower - tolerance <= value <= upper + tolerance:
            raise ValueError(f"Joint/gripper {i} target {value} outside [{lower}, {upper}]")


def clip_to_limits(values: np.ndarray, joint_limits: np.ndarray) -> np.ndarray:
    """Clip internal positions to the joint limits and the gripper stroke."""
    n = len(joint_limits)
    clipped = values.copy()
    clipped[:n] = np.clip(clipped[:n], *np.asarray(joint_limits).T)
    clipped[n] = np.clip(clipped[n], 0.0, 1.0)
    return clipped


def control_step(
    params: MitArmParams,
    position: np.ndarray,
    target: np.ndarray,
    command: np.ndarray,
    gravity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, dict[str, MitCommand]]:
    """Advance the commanded pose one servo cycle and build the MIT command for each motor.

    Args:
        params (`MitArmParams`): Arm control settings (gains, limits, speed limits, gripper).
        position (`ndarray`): Measured pose (joint radians and a normalized gripper).
        target (`ndarray`): Pose requested by the latest action.
        command (`ndarray`): Pose commanded on the previous cycle.
        gravity (`ndarray`): Model gravity torques of the arm joints for `position`, in Nm.
        dt (`float`): Time since the previous cycle, in seconds.

    Returns:
        The new commanded pose and the MIT command for each motor.
    """
    n = params.num_joints
    # Move toward the target no faster than the configured speeds.
    speeds = np.r_[np.full(n, params.max_joint_speed), params.gripper.max_speed]
    command = command + np.clip(target - command, -speeds * dt, speeds * dt)
    # Keep joints near the measured pose, so a blocked or pushed arm limits its force.
    band = params.max_tracking_error
    command[:n] = np.clip(command[:n], position[:n] - band, position[:n] + band)
    command = clip_to_limits(command, params.joint_limits)

    goal = joint_to_motor(command, params)
    torque = gravity_feedforward(params, gravity)
    commands: dict[str, MitCommand] = {
        name: (float(goal[i]), 0.0, float(params.kp[i]), float(params.kd[i]), float(torque[i]))
        for i, name in enumerate(params.motor_names[:n])
    }
    commands[params.motor_names[n]] = params.gripper.command(float(position[n]), float(command[n]))
    return command, commands


def gravity_feedforward(params: MitArmParams, gravity: np.ndarray) -> np.ndarray:
    """Return the clipped feed-forward torque of each arm joint for model gravity torques."""
    torque = gravity * params.gravity_factors * params.joint_signs
    return np.clip(torque, -params.max_gravity_torque, params.max_gravity_torque)


def damping_commands(
    params: MitArmParams, position: np.ndarray, gravity: np.ndarray
) -> dict[str, MitCommand]:
    """Build commands that only damp motion (kp = 0) while carrying the gravity of ``position``."""
    n = params.num_joints
    goal = joint_to_motor(position, params)
    torque = np.r_[gravity_feedforward(params, gravity), 0.0]
    return {
        name: (float(goal[i]), 0.0, 0.0, float(params.fault_damping_kd[i]), float(torque[i]))
        for i, name in enumerate(params.motor_names[: n + 1])
    }


def float_commands(
    params: MitArmParams, state: JointState, gravity: np.ndarray, gripper_command: float
) -> dict[str, MitCommand]:
    """Build commands that let the arm be moved by hand while compensating gravity and friction.

    The arm joints get no stiffness, light damping, the gravity torque of the measured pose and,
    when configured, Coulomb friction compensation. The gripper keeps holding ``gripper_command``.
    """
    n = params.num_joints
    goal = joint_to_motor(state.position, params)
    friction = params.coulomb_friction * np.sign(state.velocity[:n]) * params.joint_signs
    torque = gravity_feedforward(params, gravity) + friction
    commands: dict[str, MitCommand] = {
        name: (float(goal[i]), 0.0, 0.0, float(params.float_kd[i]), float(torque[i]))
        for i, name in enumerate(params.motor_names[:n])
    }
    commands[params.motor_names[n]] = params.gripper.command(float(state.position[n]), gripper_command)
    return commands


def read_joint_state(bus: MitArmBus, params: MitArmParams) -> JointState:
    """Read fresh feedback and return the validated joint state in internal units."""
    raw = bus.read_states()
    position = motor_to_joint(raw.position, params)
    validate_positions(position, params.joint_limits, joint_tolerance_rad=FEEDBACK_LIMIT_TOLERANCE_RAD)
    n, gripper = params.num_joints, params.gripper
    velocity = np.r_[raw.velocity[:n] * params.joint_signs, raw.velocity[n] / gripper.stroke]
    torque = np.r_[raw.torque[:n] * params.joint_signs, raw.torque[n] * np.sign(gripper.stroke)]
    return JointState(position=position, velocity=velocity, torque=torque)


class _ControlGC:
    """Keep preloaded models out of cyclic scans while servo threads run."""

    _lock = threading.Lock()
    _users = 0
    _owns_freeze = False

    @classmethod
    def acquire(cls) -> None:
        with cls._lock:
            if cls._users == 0 and gc.isenabled():
                cls._owns_freeze = gc.get_freeze_count() == 0
                gc.collect()
                gc.freeze()
            cls._users += 1

    @classmethod
    def release(cls) -> None:
        with cls._lock:
            cls._users -= 1
            if cls._users == 0 and cls._owns_freeze:
                gc.unfreeze()
                cls._owns_freeze = False


class MitServo:
    """Background servo loop of one arm.

    Each cycle reads feedback, advances the command toward the latest target with
    ``control_step`` and sends it. Before the first target and after the command timeout the arm
    is idle: with ``idle_mode="hold"`` it holds the measured pose, with ``"float"`` it can be
    moved by hand while gravity is compensated.

    After a fault, or when a coupled arm faults, the loop keeps the arm under damping (kp = 0
    with the gravity of the last valid pose) instead of cutting torque, so it does not depend on
    what the motor firmware does once commands stop. The next ``latest`` or ``set_target`` call
    raises. ``stop`` ends the loop and disables torque.
    """

    def __init__(
        self,
        name: str,
        bus: MitArmBus,
        config: ServoSettings,
        gravity: Callable[[np.ndarray], np.ndarray],
        stop_event: threading.Event,
    ) -> None:
        self.name = name
        self.bus = bus
        self.config = config
        self.params: MitArmParams | None = None
        self.gravity = gravity
        # Shared between both arms of a bimanual robot so a fault on one stops both.
        self.stop_event = stop_event
        self.state = JointState(position=np.zeros(0), velocity=np.zeros(0), torque=np.zeros(0))
        self.target = np.zeros(0)
        self.command = np.zeros(0)
        self.updated_at = 0.0
        self.commanded_at = 0.0
        self.command_timed_out = False
        # False until the first target arrives; the arm is idle until then.
        self.has_target = False
        self.failure: Exception | None = None
        # True from start() until stop() succeeds, even if the loop already exited on a fault.
        self.active = False
        # Set only by stop(): unlike stop_event, it also ends the post-fault damping.
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._gc_acquired = False

    def seed(self, state: JointState) -> None:
        """Start from a measured state: hold its pose until the first target arrives."""
        self.state = state
        self.target = state.position.copy()
        self.command = state.position.copy()
        self.updated_at = self.commanded_at = time.monotonic()
        self.command_timed_out = False
        self.has_target = False

    @property
    def idle(self) -> bool:
        """Whether the arm is waiting for a target: before the first one or after the timeout."""
        return not self.has_target or self.command_timed_out

    def start(self) -> None:
        if self.active or self._thread is not None:
            raise RuntimeError(f"{self.name} servo is already running")
        if self.params is None:
            raise RuntimeError(f"{self.name} servo needs calibrated arm parameters before it starts")
        self.failure = None
        self._stop_requested.clear()
        if self.config.freeze_gc:
            _ControlGC.acquire()
            self._gc_acquired = True
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-servo", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._thread = None
            if self._gc_acquired:
                _ControlGC.release()
                self._gc_acquired = False
            raise
        self.active = True

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_requested.set()
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise RuntimeError(f"{self.name} servo did not stop; use the hardware e-stop")
            self._thread = None
        self.active = False
        if self._gc_acquired:
            _ControlGC.release()
            self._gc_acquired = False

    def latest(self) -> JointState:
        """Return the last measured state, raising if the servo stopped or its feedback is stale."""
        with self._lock:
            self._check_healthy()
            state = self.state
        return JointState(
            position=state.position.copy(), velocity=state.velocity.copy(), torque=state.torque.copy()
        )

    def set_target(self, target: np.ndarray) -> None:
        with self._lock:
            self._check_healthy()
            if self.idle and self.config.idle_mode == "float":
                # Resume tracking from wherever the arm was moved by hand.
                self.command = self.state.position.copy()
            self.target = target
            self.commanded_at = time.monotonic()
            self.command_timed_out = False
            self.has_target = True

    def check_healthy(self) -> None:
        with self._lock:
            self._check_healthy()

    def _check_healthy(self) -> None:
        if self.failure is not None or self.stop_event.is_set():
            raise ConnectionError(f"{self.name} servo stopped after a motor/feedback error") from self.failure
        # The servo's own read can use the full feedback timeout after the last update.
        progress_timeout_s = 2 * self.config.feedback_timeout_s + 1 / self.config.control_frequency
        if time.monotonic() - self.updated_at > progress_timeout_s:
            self.stop_event.set()
            raise ConnectionError(f"{self.name} servo feedback is stale; reconnect before commanding motion")

    def _run(self) -> None:
        previous = started = time.monotonic()
        max_cycle_gap = 0.0
        try:
            params = self.params
            assert params is not None
            while not self.stop_event.is_set():
                started = time.monotonic()
                max_cycle_gap = max(max_cycle_gap, started - previous)
                state = read_joint_state(self.bus, params)
                position = state.position
                with self._lock:
                    self.state = state
                    self.updated_at = time.monotonic()
                    if (
                        started - self.commanded_at > self.config.command_timeout_s
                        and not self.command_timed_out
                    ):
                        self.target = position.copy()
                        self.command = position.copy()
                        self.command_timed_out = True
                    packet: dict[str, MitCommand] = {}
                    if self.bus.enabled and self.idle and self.config.idle_mode == "float":
                        n = params.num_joints
                        self.command[:n] = position[:n]
                        packet = float_commands(params, state, self.gravity(position), float(self.command[n]))
                    elif self.bus.enabled:
                        self.command, packet = control_step(
                            params,
                            position,
                            self.target,
                            self.command,
                            self.gravity(position),
                            min(started - previous, 0.05),
                        )
                for name, command in packet.items():
                    if self.stop_event.is_set():
                        break
                    self.bus.send_mit(name, command)
                previous = started
                self.stop_event.wait(max(0, 1 / self.config.control_frequency - (time.monotonic() - started)))
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
        try:
            if not self._stop_requested.is_set():
                self._damp_until_stopped()
        finally:
            self.bus.disable()

    def _damp_until_stopped(self) -> None:
        """Keep sending damping commands, as the arm faulted or a coupled arm did, until stop()."""
        if not self.bus.enabled or self.params is None:
            return
        with self._lock:
            position = self.state.position
        if len(position) != self.params.num_joints + 1:
            return
        try:
            commands = damping_commands(self.params, position, self.gravity(position))
        except Exception:
            logger.exception("%s could not build damping commands; disabling torque", self.name)
            return
        logger.warning("%s holding the arm with damping until disconnect", self.name)
        period = 1 / self.config.control_frequency
        while True:
            try:
                for name, command in commands.items():
                    self.bus.send_mit(name, command)
            except Exception:
                logger.exception("%s could not send damping commands; use the hardware e-stop", self.name)
                return
            if self._stop_requested.wait(period):
                return
