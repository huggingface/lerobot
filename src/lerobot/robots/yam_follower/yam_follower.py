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
from .config_yam_follower import JOINT_LIMITS, MOTOR_NAMES, YAM_FEATURE_NAMES, YamArmConfig, YamFollowerConfig

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
# Measured joints may sit slightly past a limit (quantization, resting on a hard stop).
_LIMIT_TOLERANCE_RAD = 0.03

# One MIT command per motor: (position_rad, velocity_rad_s, kp, kd, feedforward_torque_nm).
MitCommand = tuple[float, float, float, float, float]


def motor_to_joint(raw: np.ndarray, config: YamArmConfig) -> np.ndarray:
    closed, opened = config.gripper_closed_rad, config.gripper_open_rad
    if closed is None or opened is None:
        raise ValueError("Measure and configure gripper_closed_rad and gripper_open_rad before connecting")
    if not np.isfinite(raw).all():
        raise ConnectionError("Non-finite YAM feedback")
    joints = raw[:6] * np.asarray(config.joint_signs) + np.asarray(config.joint_offsets_rad)
    gripper = (raw[6] - closed) / (opened - closed)
    if not -0.05 <= gripper <= 1.05:
        raise ValueError("Gripper feedback is outside the calibrated stroke; check endpoints")
    return np.r_[joints, np.clip(gripper, 0, 1)]


def joint_to_motor(positions: np.ndarray, config: YamArmConfig) -> np.ndarray:
    closed, opened = config.gripper_closed_rad, config.gripper_open_rad
    if closed is None or opened is None:
        raise ValueError("Missing gripper calibration")
    raw = (positions[:6] - np.asarray(config.joint_offsets_rad)) / np.asarray(config.joint_signs)
    return np.r_[raw, closed + float(positions[6]) * (opened - closed)]


def validate_positions(values: np.ndarray, *, joint_tolerance_rad: float) -> None:
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("YAM requires six finite joint radians and one normalized gripper position")
    for i, (value, (lower, upper)) in enumerate(zip(values, (*JOINT_LIMITS, (0, 1)), strict=True)):
        tolerance = joint_tolerance_rad if i < 6 else 0.0
        if not lower - tolerance <= value <= upper + tolerance:
            raise ValueError(f"YAM joint/gripper {i} target {value} outside [{lower}, {upper}]")


def clip_to_limits(values: np.ndarray) -> np.ndarray:
    clipped = values.copy()
    clipped[:6] = np.clip(clipped[:6], *np.asarray(JOINT_LIMITS).T)
    return clipped


def action_to_target(action: RobotAction) -> np.ndarray:
    """Validate a single-arm action and return its joint/gripper target, clipped to the joint limits."""
    if set(action) != set(YAM_FEATURE_NAMES):
        raise ValueError("YAM requires all seven absolute joint/gripper targets; Cartesian actions need IK")
    target = np.asarray([action[f"{name}.pos"] for name in MOTOR_NAMES], dtype=float)
    validate_positions(target, joint_tolerance_rad=_LIMIT_TOLERANCE_RAD)
    return clip_to_limits(target)


def control_step(
    config: YamArmConfig,
    position: np.ndarray,
    target: np.ndarray,
    command: np.ndarray,
    gravity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, dict[str, MitCommand]]:
    """Advance the commanded pose one servo cycle and build the MIT command for each motor.

    Args:
        config (`YamArmConfig`): Arm configuration (gains, speed limits, calibration).
        position (`ndarray`): Measured pose (six joint radians and a normalized gripper).
        target (`ndarray`): Pose requested by the latest action.
        command (`ndarray`): Pose commanded on the previous cycle.
        gravity (`ndarray`): Model gravity torques for `position`, in Nm.
        dt (`float`): Time since the previous cycle, in seconds.

    Returns:
        The new commanded pose and the MIT command for each motor.
    """
    # Move toward the target no faster than the configured speeds.
    speeds = np.r_[np.full(6, config.max_joint_speed_rad_s), config.max_gripper_speed_s]
    command = command + np.clip(target - command, -speeds * dt, speeds * dt)
    # Keep joints near the measured pose, so a blocked or pushed arm limits its force.
    band = config.max_tracking_error_rad
    command[:6] = np.clip(command[:6], position[:6] - band, position[:6] + band)
    command = clip_to_limits(command)

    goal = joint_to_motor(command, config)
    # Bound the gripper error so its torque stays within gripper_torque_limit.
    measured_gripper = joint_to_motor(position, config)[6]
    gripper_band = config.gripper_torque_limit / config.gripper_kp
    goal[6] = np.clip(goal[6], measured_gripper - gripper_band, measured_gripper + gripper_band)

    torque = gravity * np.asarray(config.gravity_factors) * np.asarray(config.joint_signs)
    torque = np.r_[np.clip(torque, -_MAX_GRAVITY_TORQUE_NM, _MAX_GRAVITY_TORQUE_NM), 0.0]
    kp, kd = [*config.kp, config.gripper_kp], [*config.kd, config.gripper_kd]
    return command, {
        name: (float(goal[i]), 0.0, kp[i], kd[i], float(torque[i])) for i, name in enumerate(MOTOR_NAMES)
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
        self._last_feedback: dict[int, float] = {}  # receive time per feedback CAN ID

    def open(self) -> None:
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

    def read_positions(self, wait: bool = True) -> np.ndarray:
        """Return the seven raw motor positions in radians, once every motor has fresh feedback."""
        states = self._read_states(wait)
        return np.asarray([states[name].pos for name in MOTOR_NAMES])

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

    def send_mit(self, motor: str, command: MitCommand) -> None:
        self.motors[motor].send_mit(*command)

    def disable(self) -> None:
        """Disable torque if enabled; logs instead of raising so shutdown can continue."""
        if not self.enabled or self.controller is None:
            return
        try:
            self.controller.disable_all()
            self.enabled = False
        except Exception:
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

    def _read_states(self, wait: bool) -> dict[str, Any]:
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
            self.controller.poll_feedback_once()
            states = {name: motor.get_state() for name, motor in self.motors.items()}
            now = time.time()
            fresh = all(
                0 <= now - self._last_feedback.get(i + 17, 0) <= self.feedback_timeout_s for i in range(7)
            )
            if fresh and all(state is not None for state in states.values()):
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


def read_joint_positions(bus: _YamBus, config: YamArmConfig) -> np.ndarray:
    """Read fresh feedback and return validated joint radians and a normalized gripper position."""
    position = motor_to_joint(bus.read_positions(), config)
    validate_positions(position, joint_tolerance_rad=_LIMIT_TOLERANCE_RAD)
    return position


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
        config: YamFollowerConfig,
        gravity: Callable[[np.ndarray], np.ndarray],
        stop_event: threading.Event,
    ) -> None:
        self.name = name
        self.bus = bus
        self.config = config
        self.gravity = gravity
        # Shared between both arms of a bimanual robot so a fault on one stops both.
        self.stop_event = stop_event
        self.position = np.zeros(7)
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

    def seed(self, position: np.ndarray) -> None:
        """Start from a measured pose: hold it until the first target arrives."""
        self.position = position
        self.target = position.copy()
        self.command = position.copy()
        self.updated_at = self.commanded_at = time.monotonic()
        self.command_timed_out = False

    def start(self) -> None:
        self.failure = None
        _ControlGC.acquire()
        self._gc_acquired = True
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-servo", daemon=True)
        self._thread.start()
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

    def latest(self) -> np.ndarray:
        """Return the last measured pose, raising if the servo stopped or its feedback is stale."""
        with self._lock:
            self._check_healthy()
            return self.position.copy()

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
        if time.monotonic() - self.updated_at > self.config.feedback_timeout_s:
            self.stop_event.set()
            raise ConnectionError("YAM servo feedback is stale; reconnect before commanding motion")

    def _run(self) -> None:
        previous = time.monotonic()
        max_cycle_gap = 0.0
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                max_cycle_gap = max(max_cycle_gap, started - previous)
                position = read_joint_positions(self.bus, self.config)
                with self._lock:
                    self.position = position
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
                            self.config,
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
    """A YAM arm exposing joint radians and a normalized gripper position."""

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
        self.servo = _Servo(
            str(self.id), self.bus, config, self._gravity_torque, stop_event or threading.Event()
        )
        self._connected = False
        self._apply_gripper_calibration()

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(YAM_FEATURE_NAMES, float)

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self.action_features,
            **{name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()},
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self.config.gripper_closed_rad is not None and self.config.gripper_open_rad is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not calibrate:
            self._open_for_calibration()
            return
        self.open()
        try:
            self.configure()
            self.start()
        except BaseException:
            self.disconnect()
            raise

    @check_if_already_connected
    def open(self) -> None:
        """Connect cameras and CAN without torque, check the start pose and seed the servo from it.

        ``connect`` runs ``open``, ``configure`` and ``start``. A bimanual robot opens both arms
        before starting either, so a bad pose on one arm never enables the other.
        """
        if not self.is_calibrated:
            raise ValueError("Run lerobot-calibrate with this robot.id to measure the gripper endpoints")
        try:
            self.servo.stop_event.clear()
            self._load_control_model()  # before touching hardware, so a missing dependency fails first
            self.bus.open()
            for camera in self.cameras.values():
                camera.connect()
            position = read_joint_positions(self.bus, self.config)
            if not self.config.read_only:
                self._check_start_pose(position)
            self.servo.seed(position)
            self._connected = True
        except BaseException:
            self._close()
            raise

    @check_if_not_connected
    def start(self) -> None:
        """Enable torque at the seeded pose (unless read-only), then start the servo."""
        if not self.config.read_only:
            self.bus.enable(joint_to_motor(self.servo.position, self.config))
        self.servo.start()

    @check_if_not_connected
    def calibrate(self) -> None:
        """Measure the gripper stops without enabling torque or resetting joint zeros."""
        if self.servo.active:
            raise RuntimeError("Reconnect with calibrate=False before measuring gripper endpoints")
        measurements: dict[str, float] = {}
        logger.info("Support the arm. Move only the gripper gently by hand; stop if it resists.")
        for endpoint in ("closed", "open"):
            input(f"Place the gripper fully {endpoint}, release it, then press Enter: ")
            samples = []
            for _ in range(10):
                samples.append(self.bus.read_positions()[6])
                time.sleep(0.02)
            if not np.isfinite(samples).all() or np.ptp(samples) > 0.03:
                raise ValueError("Gripper moved or returned invalid feedback; calibration was not saved")
            measurements[endpoint] = float(np.median(samples))
        closed, opened = measurements["closed"], measurements["open"]
        if not (abs(closed) <= 12.5 and abs(opened) <= 12.5 and 0.5 < abs(opened - closed) < 10):
            raise ValueError("Implausible gripper stroke; calibration was not saved")
        counts = [round((value + 12.5) * 65535 / 25.0) for value in (closed, opened)]
        previous = self.calibration
        self.calibration = {
            "gripper": MotorCalibration(
                id=7,
                drive_mode=int(opened < closed),
                homing_offset=0,
                range_min=min(counts),
                range_max=max(counts),
            )
        }
        try:
            self._save_calibration()
        except Exception:
            self.calibration = previous
            raise
        self._apply_gripper_calibration(overwrite=True)
        logger.info("Saved gripper endpoints to %s. Joint zeros were not changed.", self.calibration_fpath)

    def configure(self) -> None:
        """Switch the motors to MIT mode and re-read the pose the servo will start from."""
        if self.servo.active:
            # Changing modes disables torque, which would drop an arm the servo is holding.
            raise RuntimeError("Configure YAM motors before the servo starts, not while it runs")
        if self.config.read_only:
            return
        self.bus.set_mit_mode()
        self.servo.seed(read_joint_positions(self.bus, self.config))

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if not self.servo.active:
            raise RuntimeError("Reconnect after calibration before reading policy observations")
        position = self.servo.latest()
        result: dict[str, Any] = {f"{name}.pos": float(position[i]) for i, name in enumerate(MOTOR_NAMES)}
        for name, camera in self.cameras.items():
            result[name] = camera.read_latest(max_age_ms=200)
        return result

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.read_only or not self.servo.active:
            raise RuntimeError("YAM read-only/calibration connection forbids motor commands")
        target = action_to_target(action)
        self.servo.set_target(target)
        return {f"{name}.pos": float(target[i]) for i, name in enumerate(MOTOR_NAMES)}

    @check_if_not_connected
    def disconnect(self) -> None:
        self._close()

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
            self.bus.read_positions()  # every motor must answer before measuring the gripper
            self._connected = True
        except BaseException:
            self._close()
            raise

    def _check_start_pose(self, position: np.ndarray) -> None:
        cfg = self.config
        if cfg.initial_position_rad is not None and np.any(
            np.abs(position[:6] - cfg.initial_position_rad) > cfg.initial_tolerance_rad
        ):
            raise ValueError("YAM arm is outside the configured initial pose tolerance")
        if (
            cfg.initial_gripper_position is not None
            and abs(position[6] - cfg.initial_gripper_position) > cfg.initial_gripper_tolerance
        ):
            raise ValueError("YAM gripper is outside the initial pose tolerance")

    def _close(self) -> None:
        self.servo.stop()
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
        if not (
            calibration.id == 7
            and calibration.drive_mode in (0, 1)
            and calibration.homing_offset == 0
            and 0 <= calibration.range_min < calibration.range_max <= 65535
        ):
            raise ValueError("Invalid saved gripper calibration")
        endpoints = np.asarray([calibration.range_min, calibration.range_max]) * (25.0 / 65535) - 12.5
        if calibration.drive_mode:
            endpoints = endpoints[::-1]
        if overwrite or self.config.gripper_closed_rad is None:
            self.config.gripper_closed_rad, self.config.gripper_open_rad = map(float, endpoints)
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
