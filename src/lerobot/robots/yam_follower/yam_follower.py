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

"""Single-arm YAM v1 with a gravity-compensated impedance loop."""

import gc
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import GravityCompensation
from lerobot.motors import MotorCalibration
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import (
    _can_available,
    _motorbridge_available,
    require_package,
)

from ..robot import Robot
from .config_yam_follower import (
    DM_MIT_POSITION_LIMIT_RAD,
    JOINT_LIMITS_RAD,
    MOTOR_NAMES,
    YAM_FEATURE_NAMES,
    YamFollowerConfig,
    YamFollowerConfigBase,
    motor_feature_names,
)

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import Controller, Mode

if TYPE_CHECKING or _can_available:
    import can

logger = logging.getLogger(__name__)

_MODEL_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
_MODEL_GRIPPER_JOINTS = ("joint7", "joint8")
_MODEL_GRIPPER_STROKE_M = 0.0475
_GRAVITY_MODEL_PATH = Path(__file__).parent / "assets/yam_linear.xml"
_MAX_GRAVITY_TORQUE_NM = 10.0
# Measured poses may exceed the model limits, which are not mechanical stops.
_FEEDBACK_LIMIT_TOLERANCE_RAD = 0.15
_GRIPPER_FEEDBACK_MARGIN = 0.10

# One MIT command per motor: (position_rad, velocity_rad_s, kp, kd, feedforward_torque_nm).
MitCommand = tuple[float, float, float, float, float]


@dataclass(frozen=True, eq=False)
class ArmParams:
    """One arm's control settings in internal units: radians, and the gripper from 0 to 1."""

    joint_signs: np.ndarray
    joint_offsets: np.ndarray
    gripper_closed: float
    gripper_open: float
    kp: np.ndarray
    kd: np.ndarray
    gripper_kp: float
    gripper_kd: float
    gripper_torque_limit: float
    max_joint_speed: float
    max_gripper_speed: float
    max_tracking_error: float
    gravity_factors: np.ndarray

    @classmethod
    def from_config(cls, config: YamFollowerConfigBase) -> "ArmParams":
        if config.gripper_closed_deg is None or config.gripper_open_deg is None:
            raise ValueError("Run lerobot-calibrate with this robot.id to measure the gripper endpoints")
        return cls(
            joint_signs=np.asarray(config.joint_signs, dtype=float),
            joint_offsets=np.deg2rad(config.joint_offsets_deg),
            gripper_closed=math.radians(config.gripper_closed_deg),
            gripper_open=math.radians(config.gripper_open_deg),
            kp=np.asarray(config.kp, dtype=float),
            kd=np.asarray(config.kd, dtype=float),
            gripper_kp=config.gripper_kp,
            gripper_kd=config.gripper_kd,
            gripper_torque_limit=config.gripper_torque_limit,
            max_joint_speed=math.radians(config.max_joint_speed_deg_s),
            max_gripper_speed=config.max_gripper_speed_s,
            max_tracking_error=math.radians(config.max_tracking_error_deg),
            gravity_factors=np.asarray(config.gravity_factors, dtype=float),
        )


@dataclass(frozen=True, eq=False)
class MotorStates:
    """Raw states of the seven motors, in radians, radians per second and newton-metres."""

    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray


@dataclass(frozen=True, eq=False)
class JointState:
    """Measured joint state: six joints in radians and the gripper from 0 (closed) to 1 (open)."""

    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray


def motor_to_joint(raw: np.ndarray, params: ArmParams) -> np.ndarray:
    if not np.isfinite(raw).all():
        raise ConnectionError("Non-finite YAM feedback")
    joints = raw[:6] * params.joint_signs + params.joint_offsets
    gripper = (raw[6] - params.gripper_closed) / (params.gripper_open - params.gripper_closed)
    if not -_GRIPPER_FEEDBACK_MARGIN <= gripper <= 1 + _GRIPPER_FEEDBACK_MARGIN:
        raise ValueError("Gripper feedback is outside the calibrated stroke; check endpoints")
    return np.r_[joints, np.clip(gripper, 0, 1)]


def joint_to_motor(positions: np.ndarray, params: ArmParams) -> np.ndarray:
    raw = (positions[:6] - params.joint_offsets) / params.joint_signs
    return np.r_[
        raw, params.gripper_closed + float(positions[6]) * (params.gripper_open - params.gripper_closed)
    ]


def to_public(values: np.ndarray, use_degrees: bool) -> np.ndarray:
    """Convert joint positions or rates from internal units to the robot's public units."""
    if not use_degrees:
        return values.copy()
    return np.r_[np.rad2deg(values[:6]), values[6] * 100.0]


def from_public(values: np.ndarray, use_degrees: bool) -> np.ndarray:
    """Convert joint positions or rates from the robot's public units to internal units."""
    if not use_degrees:
        return values.copy()
    return np.r_[np.deg2rad(values[:6]), values[6] / 100.0]


def validate_positions(values: np.ndarray, *, joint_tolerance_rad: float) -> None:
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("YAM requires six finite joint radians and one normalized gripper position")
    for i, (value, (lower, upper)) in enumerate(zip(values, (*JOINT_LIMITS_RAD, (0, 1)), strict=True)):
        tolerance = joint_tolerance_rad if i < 6 else 0.0
        if not lower - tolerance <= value <= upper + tolerance:
            raise ValueError(f"YAM joint/gripper {i} target {value} outside [{lower}, {upper}]")


def clip_to_limits(values: np.ndarray) -> np.ndarray:
    """Clip internal positions to the joint limits and the gripper stroke."""
    clipped = values.copy()
    clipped[:6] = np.clip(clipped[:6], *np.asarray(JOINT_LIMITS_RAD).T)
    clipped[6] = np.clip(clipped[6], 0.0, 1.0)
    return clipped


def action_to_target(action: RobotAction, use_degrees: bool) -> np.ndarray:
    """Validate a single-arm action and return its internal target, clipped to the limits."""
    if set(action) != set(YAM_FEATURE_NAMES):
        raise ValueError("YAM requires all seven absolute joint/gripper targets; Cartesian actions need IK")
    values = np.asarray([action[f"{name}.pos"] for name in MOTOR_NAMES], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("YAM actions must be finite")
    return clip_to_limits(from_public(values, use_degrees))


def control_step(
    params: ArmParams,
    position: np.ndarray,
    target: np.ndarray,
    command: np.ndarray,
    gravity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, dict[str, MitCommand]]:
    """Advance the commanded pose one servo cycle and build the MIT command for each motor.

    Args:
        params (`ArmParams`): Arm control settings (gains, speed limits, calibration).
        position (`ndarray`): Measured pose (six joint radians and a normalized gripper).
        target (`ndarray`): Pose requested by the latest action.
        command (`ndarray`): Pose commanded on the previous cycle.
        gravity (`ndarray`): Model gravity torques for `position`, in Nm.
        dt (`float`): Time since the previous cycle, in seconds.

    Returns:
        The new commanded pose and the MIT command for each motor.
    """
    # Move toward the target no faster than the configured speeds.
    speeds = np.r_[np.full(6, params.max_joint_speed), params.max_gripper_speed]
    command = command + np.clip(target - command, -speeds * dt, speeds * dt)
    # Keep joints near the measured pose, so a blocked or pushed arm limits its force.
    band = params.max_tracking_error
    command[:6] = np.clip(command[:6], position[:6] - band, position[:6] + band)
    command = clip_to_limits(command)

    goal = joint_to_motor(command, params)
    # Bound the gripper error so its torque stays within gripper_torque_limit.
    measured_gripper = joint_to_motor(position, params)[6]
    gripper_band = params.gripper_torque_limit / params.gripper_kp
    goal[6] = np.clip(goal[6], measured_gripper - gripper_band, measured_gripper + gripper_band)

    torque = gravity * params.gravity_factors * params.joint_signs
    torque = np.r_[np.clip(torque, -_MAX_GRAVITY_TORQUE_NM, _MAX_GRAVITY_TORQUE_NM), 0.0]
    kp, kd = np.r_[params.kp, params.gripper_kp], np.r_[params.kd, params.gripper_kd]
    return command, {
        name: (float(goal[i]), 0.0, float(kp[i]), float(kd[i]), float(torque[i]))
        for i, name in enumerate(MOTOR_NAMES)
    }


class _YamBus:
    """CAN interface of one YAM arm.

    MotorBridge sends commands and decodes motor states. A receive-only python-can socket on
    the same interface checks that every motor answered recently and reports motor faults,
    because MotorBridge's cached states carry no receive time.
    """

    def __init__(
        self, port: str, feedback_timeout_s: float, expected_adapter_serial: str | None = None
    ) -> None:
        self.port = port
        self.feedback_timeout_s = feedback_timeout_s
        self.expected_adapter_serial = expected_adapter_serial
        self.controller: Controller | None = None
        self.monitor: can.BusABC | None = None
        self.motors: dict[str, Any] = {}
        self.enabled = False
        self._disable_failed = False
        self._enabled_at = 0.0
        self._last_feedback: dict[int, float] = {}  # receive time per feedback CAN ID
        self._last_status: dict[int, int] = {}

    def open(self) -> None:
        if self._disable_failed:
            raise RuntimeError(
                "Previous YAM torque-disable command failed; use the hardware e-stop before reconnecting"
            )
        self.enabled = False
        self._verify_adapter()
        self.monitor = can.Bus(
            channel=self.port,
            interface="socketcan",
            can_filters=[{"can_id": i + 17, "can_mask": 0x7FF, "extended": False} for i in range(7)],
        )
        self.controller = Controller(channel=self.port)
        self.motors = {
            name: self.controller.add_damiao_motor(i + 1, i + 17, "4340" if i < 3 else "4310")
            for i, name in enumerate(MOTOR_NAMES)
        }
        self._last_feedback.clear()
        self._last_status.clear()

    def read_states(self, wait: bool = True) -> MotorStates:
        """Return the raw states of the seven motors, once every motor has fresh feedback."""
        states = self._wait_for_fresh_states(wait)
        return MotorStates(
            position=np.asarray([states[name].pos for name in MOTOR_NAMES], dtype=float),
            velocity=np.asarray([states[name].vel for name in MOTOR_NAMES], dtype=float),
            torque=np.asarray([states[name].torq for name in MOTOR_NAMES], dtype=float),
        )

    def set_mit_mode(self) -> None:
        assert self.controller is not None
        self.controller.disable_all()
        for motor in self.motors.values():
            motor.ensure_mode(Mode.MIT)

    def enable(self, hold: np.ndarray) -> None:
        """Enable torque after sending zero-gain setpoints at the ``hold`` raw positions."""
        assert self.controller is not None
        for name, value in zip(MOTOR_NAMES, hold, strict=True):
            self.motors[name].send_mit(float(value), 0.0, 0.0, 0.0, 0.0)
        self.enabled = True
        self.controller.enable_all()
        self._enabled_at = time.monotonic()

    def send_mit(self, motor: str, command: MitCommand) -> None:
        self.motors[motor].send_mit(*command)

    def disable(self) -> None:
        """Disable torque if enabled; logs instead of raising so shutdown can continue."""
        if not self.enabled or self.controller is None:
            return
        try:
            self.controller.disable_all()
            self.enabled = False
            self._disable_failed = False
        except Exception:
            self._disable_failed = True
            logger.exception("Could not disable YAM torque; use the hardware e-stop")

    def close(self) -> None:
        self.disable()
        for name, motor in self.motors.items():
            try:
                motor.close()
            except Exception:
                logger.exception("Failed to close YAM motor %s", name)
        self.motors.clear()
        if self.controller is not None:
            try:
                self.controller.close()
            except Exception:
                logger.exception("Failed to close YAM MotorBridge controller")
            finally:
                self.controller = None
        if self.monitor is not None:
            try:
                self.monitor.shutdown()
            except Exception:
                logger.exception("Failed to close YAM feedback monitor")
            finally:
                self.monitor = None

    def _verify_adapter(self) -> None:
        """Check the USB adapter serial so can0/can1 enumeration swaps cannot swap arms."""
        if self.expected_adapter_serial is None:
            return
        device = (Path("/sys/class/net") / self.port / "device").resolve()
        for parent in (device, *device.parents):
            serial = parent / "serial"
            if serial.is_file():
                actual = serial.read_text().strip()
                if actual != self.expected_adapter_serial:
                    raise ValueError(
                        f"{self.port} adapter serial {actual!r} does not match the configured arm"
                    )
                return
        raise ValueError(f"Cannot verify USB serial for {self.port}; check the adapter connection")

    def _wait_for_fresh_states(self, wait: bool) -> dict[str, Any]:
        assert self.controller is not None and self.monitor is not None
        for motor in self.motors.values():
            motor.request_feedback()
        deadline = time.monotonic() + self.feedback_timeout_s if wait else time.monotonic()
        while True:
            for _ in range(256):
                msg = self.monitor.recv(timeout=0)
                if msg is None:
                    break
                if (
                    msg.is_error_frame
                    or msg.is_remote_frame
                    or msg.is_extended_id
                    or msg.dlc != 8
                    or msg.arbitration_id not in range(17, 24)
                ):
                    continue
                if msg.data[0] & 0x0F != msg.arbitration_id - 16:
                    continue
                status = msg.data[0] >> 4
                if status not in (0, 1):
                    raise ConnectionError(f"{self.port}: motor {msg.arbitration_id - 16} fault {status:#x}")
                self._last_feedback[msg.arbitration_id] = msg.timestamp
                self._last_status[msg.arbitration_id] = status
            self.controller.poll_feedback_once()
            states = {name: motor.get_state() for name, motor in self.motors.items()}
            # SocketCAN frame timestamps use wall time; monotonic time above bounds the wait.
            now = time.time()
            fresh = all(
                0 <= now - self._last_feedback.get(i + 17, 0) <= self.feedback_timeout_s for i in range(7)
            )
            if fresh and all(state is not None for state in states.values()):
                if self.enabled and time.monotonic() - self._enabled_at > self.feedback_timeout_s:
                    disabled = [i for i in range(1, 8) if self._last_status[i + 16] == 0]
                    if disabled:
                        raise ConnectionError(f"{self.port}: motor {disabled[0]} unexpectedly disabled")
                for name, state in states.items():
                    if state.status_code not in (0, 1) or not math.isfinite(state.pos):
                        raise ConnectionError(f"{self.port}: invalid {name} feedback")
                return states
            if time.monotonic() >= deadline:
                ages = ", ".join(
                    f"{i}: {(now - self._last_feedback[i + 16]) * 1000:.1f} ms"
                    if i + 16 in self._last_feedback
                    else f"{i}: missing"
                    for i in range(1, 8)
                )
                raise ConnectionError(f"{self.port}: missing or stale motor feedback ({ages})")
            time.sleep(0.001)


class _ControlGC:
    """Keep preloaded models out of cyclic scans while YAM servo threads run."""

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


def read_joint_state(bus: _YamBus, params: ArmParams) -> JointState:
    """Read fresh feedback and return the validated joint state in internal units."""
    raw = bus.read_states()
    position = motor_to_joint(raw.position, params)
    validate_positions(position, joint_tolerance_rad=_FEEDBACK_LIMIT_TOLERANCE_RAD)
    stroke = params.gripper_open - params.gripper_closed
    velocity = np.r_[raw.velocity[:6] * params.joint_signs, raw.velocity[6] / stroke]
    torque = np.r_[raw.torque[:6] * params.joint_signs, raw.torque[6] * math.copysign(1.0, stroke)]
    return JointState(position=position, velocity=velocity, torque=torque)


class _Servo:
    """Background servo loop of one arm.

    Each cycle reads feedback, holds the measured pose once actions stop arriving, advances the
    command with ``control_step`` and sends it. Any error stops the loop and disables torque;
    the next ``latest`` or ``set_target`` call then raises.
    """

    def __init__(
        self,
        name: str,
        bus: _YamBus,
        config: YamFollowerConfigBase,
        gravity: Callable[[np.ndarray], np.ndarray],
        stop_event: threading.Event,
    ) -> None:
        self.name = name
        self.bus = bus
        self.config = config
        self.params: ArmParams | None = None
        self.gravity = gravity
        # Shared between both arms of a bimanual robot so a fault on one stops both.
        self.stop_event = stop_event
        self.state = JointState(position=np.zeros(7), velocity=np.zeros(7), torque=np.zeros(7))
        self.target = np.zeros(7)
        self.command = np.zeros(7)
        self.updated_at = 0.0
        self.commanded_at = 0.0
        self.command_timed_out = False
        self.failure: Exception | None = None
        # True from start() until stop() succeeds, even if the loop already exited on a fault.
        self.active = False
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

    def start(self) -> None:
        if self.active or self._thread is not None:
            raise RuntimeError("YAM servo is already running")
        if self.params is None:
            raise RuntimeError("YAM servo needs calibrated arm parameters before it starts")
        self.failure = None
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
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise RuntimeError("YAM servo did not stop; use the hardware e-stop")
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
            self.target = target
            self.commanded_at = time.monotonic()
            self.command_timed_out = False

    def check_healthy(self) -> None:
        with self._lock:
            self._check_healthy()

    def _check_healthy(self) -> None:
        if self.failure is not None or self.stop_event.is_set():
            raise ConnectionError("YAM servo stopped after a motor/feedback error") from self.failure
        # The servo's own read can use the full feedback timeout after the last update.
        progress_timeout_s = 2 * self.config.feedback_timeout_s + 1 / self.config.control_frequency
        if time.monotonic() - self.updated_at > progress_timeout_s:
            self.stop_event.set()
            raise ConnectionError("YAM servo feedback is stale; reconnect before commanding motion")

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
                    if self.bus.enabled:
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
                f"YAM maximum servo cycle gap: {max_cycle_gap * 1000:.1f} ms; "
                f"current cycle elapsed: {(time.monotonic() - started) * 1000:.1f} ms"
            )
            self.failure = exc
            self.stop_event.set()
        finally:
            self.bus.disable()
            if self.failure is not None:
                logger.error(
                    "YAM servo stopped: %s; %s",
                    self.failure,
                    "; ".join(getattr(self.failure, "__notes__", [])),
                )


class YamFollower(Robot):
    """A YAM arm exposing six joint positions and a gripper opening."""

    config_class = YamFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YamFollowerConfig, stop_event: threading.Event | None = None) -> None:
        require_package("motorbridge", extra="yam")
        require_package("python-can", extra="yam", import_name="can")
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.bus = _YamBus(config.port, config.feedback_timeout_s, config.expected_adapter_serial)
        self.gravity_model: GravityCompensation | None = None
        self.params: ArmParams | None = None
        self.servo = _Servo(
            str(self.id), self.bus, config, self._gravity_torque, stop_event or threading.Event()
        )
        self._connected = False
        self._apply_gripper_calibration()
        self._refresh_params()

    @property
    def _motors_ft(self) -> dict[str, type]:
        return dict.fromkeys(motor_feature_names(self.config.use_velocity_and_torque), float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(YAM_FEATURE_NAMES, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self.config.gripper_closed_deg is not None and self.config.gripper_open_deg is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Start control, or open a torque-free gripper calibration session when ``calibrate=False``."""
        if not calibrate:
            self._open_for_calibration()
            logger.info(f"{self} connected for calibration.")
            return
        self.open()
        try:
            self.configure()
            self.start()
        except BaseException:
            self.disconnect()
            raise
        logger.info(f"{self} connected.")

    @check_if_already_connected
    def open(self) -> None:
        """Connect cameras and CAN without torque, check the start pose and seed the servo from it.

        ``connect`` runs ``open``, ``configure`` and ``start``. A bimanual robot opens both arms
        before starting either, so a bad pose on one arm never enables the other.
        """
        if not self.is_calibrated:
            raise ValueError("Run lerobot-calibrate with this robot.id to measure the gripper endpoints")
        try:
            self._refresh_params()
            params = self._require_params()
            self.servo.stop_event.clear()
            self._load_control_model()  # before touching hardware, so a missing dependency fails first
            self.bus.open()
            for camera in self.cameras.values():
                camera.connect()
            state = read_joint_state(self.bus, params)
            if not self.config.read_only:
                self._check_start_pose(state.position)
            self.servo.seed(state)
            self._connected = True
        except BaseException:
            self._close()
            raise

    @check_if_not_connected
    def start(self) -> None:
        """Enable torque at the seeded pose (unless read-only), then start the servo."""
        if self.servo.active or self.servo._thread is not None:
            raise RuntimeError("YAM servo is already running")
        if not self.config.read_only:
            self.bus.enable(joint_to_motor(self.servo.state.position, self._require_params()))
        self.servo.start()

    @check_if_not_connected
    def calibrate(self) -> None:
        """Measure the gripper stops without enabling torque or resetting joint zeros."""
        if self.servo.active:
            raise RuntimeError("Reconnect with calibrate=False before measuring gripper endpoints")
        if self.calibration:
            user_input = input(
                f"Press ENTER to use the gripper calibration saved for {self.id}, "
                "or type 'c' and press ENTER to measure it again: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using the saved gripper calibration of {self}")
                return
        measurements: dict[str, float] = {}
        logger.info(f"{self}: support the arm. Move only the gripper gently by hand; stop if it resists.")
        for endpoint in ("closed", "open"):
            input(f"[{self.id}] Place the gripper fully {endpoint}, release it, then press Enter: ")
            samples = []
            for _ in range(10):
                samples.append(self.bus.read_states().position[6])
                time.sleep(0.02)
            if not np.isfinite(samples).all() or np.ptp(samples) > 0.03:
                raise ValueError("Gripper moved or returned invalid feedback; calibration was not saved")
            measurements[endpoint] = float(np.median(samples))
        closed, opened = measurements["closed"], measurements["open"]
        if not (
            abs(closed) <= DM_MIT_POSITION_LIMIT_RAD
            and abs(opened) <= DM_MIT_POSITION_LIMIT_RAD
            and 0.5 < abs(opened - closed) < 10
        ):
            raise ValueError("Implausible gripper stroke; calibration was not saved")
        # Raw motor angles of the two stops, in whole degrees; drive_mode records which is closed.
        stops_deg = sorted(round(math.degrees(value)) for value in (closed, opened))
        previous: dict[str, MotorCalibration] = self.calibration
        self.calibration = {
            "gripper": MotorCalibration(
                id=7,
                drive_mode=int(opened < closed),
                homing_offset=0,
                range_min=stops_deg[0],
                range_max=stops_deg[1],
            )
        }
        try:
            self._save_calibration()
        except Exception:
            self.calibration = previous
            raise
        self._apply_gripper_calibration(overwrite=True)
        self._refresh_params()
        logger.info("Saved gripper endpoints to %s. Joint zeros were not changed.", self.calibration_fpath)

    def configure(self) -> None:
        """Switch the motors to MIT mode and re-read the pose the servo will start from."""
        if self.servo.active:
            # Changing modes disables torque, which would drop an arm the servo is holding.
            raise RuntimeError("Configure YAM motors before the servo starts, not while it runs")
        if self.config.read_only:
            return
        self.bus.set_mit_mode()
        self.servo.seed(read_joint_state(self.bus, self._require_params()))

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if not self.servo.active:
            raise RuntimeError("Reconnect after calibration before reading policy observations")
        state = self.servo.latest()
        use_degrees = self.config.use_degrees
        values = {"pos": to_public(state.position, use_degrees)}
        if self.config.use_velocity_and_torque:
            values["vel"] = to_public(state.velocity, use_degrees)
            values["torque"] = state.torque
        result: dict[str, Any] = {
            f"{name}.{kind}": float(data[i])
            for i, name in enumerate(MOTOR_NAMES)
            for kind, data in values.items()
        }
        for name, camera in self.cameras.items():
            result[name] = camera.read_latest(max_age_ms=200)
        return result

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.read_only or not self.servo.active:
            raise RuntimeError("YAM read-only/calibration connection forbids motor commands")
        target = action_to_target(action, self.config.use_degrees)
        self.servo.set_target(target)
        sent = to_public(target, self.config.use_degrees)
        return {f"{name}.pos": float(sent[i]) for i, name in enumerate(MOTOR_NAMES)}

    @check_if_not_connected
    def disconnect(self) -> None:
        self._close()
        logger.info(f"{self} disconnected.")

    def _refresh_params(self) -> None:
        """Rebuild the internal control settings from the config, once the gripper is calibrated."""
        self.params = ArmParams.from_config(self.config) if self.is_calibrated else None
        self.servo.params = self.params

    def _require_params(self) -> ArmParams:
        if self.params is None:
            raise ValueError("Run lerobot-calibrate with this robot.id to measure the gripper endpoints")
        return self.params

    def _gravity_torque(self, positions: np.ndarray) -> np.ndarray:
        if self.gravity_model is None:
            return np.zeros(6)
        opening = float(positions[6]) * _MODEL_GRIPPER_STROKE_M
        return self.gravity_model.torques(positions[:6], dict.fromkeys(_MODEL_GRIPPER_JOINTS, opening))

    def _load_control_model(self) -> None:
        if self.config.read_only or not self.config.gravity_compensation or self.gravity_model is not None:
            return
        self.gravity_model = GravityCompensation(
            _GRAVITY_MODEL_PATH, _MODEL_ARM_JOINTS, base_frame="base", mjcf=True
        )

    def _open_for_calibration(self) -> None:
        """Open the CAN interface without cameras, servo or torque."""
        try:
            self.bus.open()
            self.bus.read_states()  # every motor must answer before measuring the gripper
            self._connected = True
        except BaseException:
            self._close()
            raise

    def _check_start_pose(self, position: np.ndarray) -> None:
        cfg = self.config
        if cfg.initial_position_deg is not None and np.any(
            np.abs(np.rad2deg(position[:6]) - cfg.initial_position_deg) > cfg.initial_tolerance_deg
        ):
            raise ValueError("YAM arm is outside the configured initial pose tolerance")
        if (
            cfg.initial_gripper_position is not None
            and abs(position[6] * 100.0 - cfg.initial_gripper_position) > cfg.initial_gripper_tolerance
        ):
            raise ValueError("YAM gripper is outside the initial pose tolerance")

    def _close(self) -> None:
        stop_error: RuntimeError | None = None
        try:
            self.servo.stop()
        except RuntimeError as exc:
            stop_error = exc
            # Leave CAN open while the thread might still be using it, but attempt torque-off.
            self.bus.disable()
            logger.exception("YAM servo did not stop; CAN remains open for a disconnect retry")
        if stop_error is not None:
            raise stop_error
        try:
            self.bus.close()
        except Exception:
            logger.exception("Failed to close YAM arm")
        try:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
        finally:
            self._connected = False

    def _apply_gripper_calibration(self, *, overwrite: bool = False) -> None:
        calibration = self.calibration.get("gripper")
        if calibration is None:
            return
        motor_limit_deg = math.degrees(DM_MIT_POSITION_LIMIT_RAD)
        if not (
            calibration.id == 7
            and calibration.drive_mode in (0, 1)
            and calibration.homing_offset == 0
            and -motor_limit_deg <= calibration.range_min < calibration.range_max <= motor_limit_deg
        ):
            raise ValueError(
                f"Invalid saved gripper calibration in {self.calibration_fpath}; run lerobot-calibrate again"
            )
        stops_deg = (float(calibration.range_min), float(calibration.range_max))
        closed_deg, open_deg = stops_deg[::-1] if calibration.drive_mode else stops_deg
        if overwrite or self.config.gripper_closed_deg is None:
            self.config.gripper_closed_deg, self.config.gripper_open_deg = closed_deg, open_deg
            self.config.__post_init__()

    def _save_calibration(self, fpath: Path | None = None) -> None:
        path = fpath if fpath is not None else self.calibration_fpath
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            super()._save_calibration(temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
