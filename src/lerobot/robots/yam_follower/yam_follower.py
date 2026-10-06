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

from __future__ import annotations

import gc
import logging
import math
import threading
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorCalibration
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import (
    _can_available,
    _motorbridge_available,
    _placo_available,
    require_package,
)

from ..robot import Robot
from .config_yam_follower import JOINT_LIMITS, MOTOR_NAMES, YAM_FEATURE_NAMES, YamArmConfig, YamFollowerConfig

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import Controller, Mode

if TYPE_CHECKING or _can_available:
    import can

if TYPE_CHECKING or _placo_available:
    import placo

logger = logging.getLogger(__name__)

_MODEL_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
_MODEL_GRIPPER_JOINTS = ("joint7", "joint8")
_MODEL_GRIPPER_STROKE_M = 0.0475
_GRAVITY_MODEL_PATH = Path(__file__).parent / "assets/yam_linear.xml"


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


class YamFollower(Robot):
    """A YAM arm exposing joint radians and a normalized gripper position."""

    config_class = YamFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YamFollowerConfig) -> None:
        require_package("motorbridge", extra="yam")
        require_package("python-can", extra="yam", import_name="can")
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.bus: Controller | None = None
        self.monitor: can.BusABC | None = None
        self.motors: dict[str, Any] = {}
        self.position = np.zeros(7)
        self.target = np.zeros(7)
        self.command = np.zeros(7)
        self.updated_at = 0.0
        self.commanded_at = 0.0
        self.command_timed_out = False
        self.gravity_model: placo.RobotWrapper | None = None
        self.enabled = False
        self.last_feedback: dict[int, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: Exception | None = None
        self._connected = False
        self._calibration_session = False
        self._gc_acquired = False
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
        try:
            self._prepare_connect(calibrate)
            if calibrate:
                self._configure_control()
                self._enable_motors()
                self._start_servo()
            self._connected = True
        except BaseException:
            self._close()
            raise

    @check_if_not_connected
    def calibrate(self) -> None:
        """Measure the gripper stops without enabling torque or resetting joint zeros."""
        if not self._calibration_session:
            raise RuntimeError("Reconnect with calibrate=False before measuring gripper endpoints")
        measurements: dict[str, float] = {}
        logger.info("Support the arm. Move only the gripper gently by hand; stop if it resists.")
        for endpoint in ("closed", "open"):
            input(f"Place the gripper fully {endpoint}, release it, then press Enter: ")
            samples = []
            for _ in range(10):
                samples.append(self._read_feedback(wait=True)["gripper"].pos)
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
        self._configure_control()
        self._enable_motors()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self._calibration_session:
            raise RuntimeError("Reconnect after calibration before reading policy observations")
        with self._lock:
            self._check_feedback()
            result: dict[str, Any] = {
                f"{name}.pos": float(self.position[i]) for i, name in enumerate(MOTOR_NAMES)
            }
        for name, camera in self.cameras.items():
            result[name] = camera.read_latest(max_age_ms=200)
        return result

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.read_only or self._calibration_session:
            raise RuntimeError("YAM read-only/calibration connection forbids motor commands")
        target = self._action_target(action)
        with self._lock:
            self._check_feedback()
            self.target = target
            self.commanded_at = time.monotonic()
            self.command_timed_out = False
        return {f"{name}.pos": float(target[i]) for i, name in enumerate(MOTOR_NAMES)}

    @check_if_not_connected
    def disconnect(self) -> None:
        self._close()

    def _action_target(self, action: RobotAction) -> np.ndarray:
        if set(action) != set(YAM_FEATURE_NAMES):
            raise ValueError(
                "YAM requires all seven absolute joint/gripper targets; Cartesian actions need IK"
            )
        target = np.asarray([action[f"{name}.pos"] for name in MOTOR_NAMES], dtype=float)
        validate_positions(target, joint_tolerance_rad=0.03)
        return clip_to_limits(target)

    def _gravity_torque(self, positions: np.ndarray) -> np.ndarray:
        model = self.gravity_model
        if model is None:
            return np.zeros(6)
        for name, position in zip(_MODEL_ARM_JOINTS, positions[:6], strict=True):
            model.set_joint(name, float(position))
        opening = float(positions[6]) * _MODEL_GRIPPER_STROKE_M
        for name in _MODEL_GRIPPER_JOINTS:
            model.set_joint(name, opening)
        model.update_kinematics()
        torques = model.static_gravity_compensation_torques_dict("base")
        return np.asarray([torques[name] for name in _MODEL_ARM_JOINTS])

    def _load_control_model(self) -> None:
        if self.config.read_only or not self.config.gravity_compensation or self.gravity_model is not None:
            return
        require_package("placo", extra="yam")
        self.gravity_model = placo.RobotWrapper(str(_GRAVITY_MODEL_PATH), placo.Flags.mjcf)

    def _verify_adapter(self) -> None:
        if self.config.expected_adapter_serial is None:
            return
        device = (Path("/sys/class/net") / self.config.port / "device").resolve()
        for parent in (device, *device.parents):
            serial = parent / "serial"
            if serial.is_file():
                actual = serial.read_text().strip()
                if actual != self.config.expected_adapter_serial:
                    raise ValueError(
                        f"{self.config.port} adapter serial {actual!r} does not match the configured arm"
                    )
                return
        raise ValueError(f"Cannot verify USB serial for {self.config.port}; check the adapter connection")

    def _open_hardware(self) -> None:
        self.monitor = can.Bus(
            channel=self.config.port,
            interface="socketcan",
            can_filters=[{"can_id": i + 17, "can_mask": 0x7FF, "extended": False} for i in range(7)],
        )
        self.bus = Controller(channel=self.config.port)
        self.motors = {
            name: self.bus.add_damiao_motor(i + 1, i + 17, "4340" if i < 3 else "4310")
            for i, name in enumerate(MOTOR_NAMES)
        }
        self.last_feedback.clear()

    def _read_feedback(self, *, wait: bool = False) -> dict[str, Any]:
        assert self.bus is not None and self.monitor is not None
        for motor in self.motors.values():
            motor.request_feedback()
        deadline = time.monotonic() + self.config.feedback_timeout_s if wait else time.monotonic()
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
                    raise ConnectionError(
                        f"{self.config.port}: motor {msg.arbitration_id - 16} fault {status:#x}"
                    )
                self.last_feedback[msg.arbitration_id] = msg.timestamp
            self.bus.poll_feedback_once()
            states = {name: motor.get_state() for name, motor in self.motors.items()}
            now = time.time()
            fresh = all(
                0 <= now - self.last_feedback.get(i + 17, 0) <= self.config.feedback_timeout_s
                for i in range(7)
            )
            if fresh and all(state is not None for state in states.values()):
                for name, state in states.items():
                    if state.status_code not in (0, 1) or not math.isfinite(state.pos):
                        raise ConnectionError(f"{self.config.port}: invalid {name} feedback")
                return states
            if time.monotonic() >= deadline:
                ages = ", ".join(
                    f"{i}: {(now - self.last_feedback[i + 16]) * 1000:.1f} ms"
                    if i + 16 in self.last_feedback
                    else f"{i}: missing"
                    for i in range(1, 8)
                )
                raise ConnectionError(f"{self.config.port}: missing or stale motor feedback ({ages})")
            time.sleep(0.001)

    def _seed_control_state(self, position: np.ndarray) -> None:
        self.position = position
        self.target = position.copy()
        self.command = position.copy()
        self.updated_at = self.commanded_at = time.monotonic()
        self.command_timed_out = False

    def _prepare_connect(self, calibrate: bool) -> None:
        self._calibration_session = not calibrate
        self._failure = None
        self._stop.clear()
        if calibrate and not self.is_calibrated:
            raise ValueError("Run lerobot-calibrate with this robot.id to measure the gripper endpoints")
        if calibrate:
            self._load_control_model()
            _ControlGC.acquire()
            self._gc_acquired = True
        self._verify_adapter()
        if calibrate:
            for camera in self.cameras.values():
                camera.connect()
        self._open_hardware()
        states = self._read_feedback(wait=True)
        if not calibrate:
            return
        position = motor_to_joint(np.asarray([states[name].pos for name in MOTOR_NAMES]), self.config)
        validate_positions(position, joint_tolerance_rad=0.03)
        if not self.config.read_only:
            if self.config.initial_position_rad is not None and np.any(
                np.abs(position[:6] - self.config.initial_position_rad) > self.config.initial_tolerance_rad
            ):
                raise ValueError("YAM arm is outside the configured initial pose tolerance")
            if (
                self.config.initial_gripper_position is not None
                and abs(position[6] - self.config.initial_gripper_position)
                > self.config.initial_gripper_tolerance
            ):
                raise ValueError("YAM gripper is outside the initial pose tolerance")
        self._seed_control_state(position)

    def _configure_control(self) -> None:
        if self.config.read_only or self._calibration_session:
            return
        assert self.bus is not None
        self.bus.disable_all()
        for motor in self.motors.values():
            motor.ensure_mode(Mode.MIT)
        states = self._read_feedback(wait=True)
        position = motor_to_joint(np.asarray([states[name].pos for name in MOTOR_NAMES]), self.config)
        validate_positions(position, joint_tolerance_rad=0.03)
        self._seed_control_state(position)

    def _enable_motors(self) -> None:
        if self.config.read_only or self._calibration_session:
            return
        assert self.bus is not None
        for name, value in zip(MOTOR_NAMES, joint_to_motor(self.position, self.config), strict=True):
            self.motors[name].send_mit(float(value), 0.0, 0.0, 0.0, 0.0)
        self.enabled = True
        self.bus.enable_all()

    def _start_servo(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"{self.id}-servo", daemon=True)
        self._thread.start()

    def _command_packet(self, position: np.ndarray, dt: float) -> dict[str, tuple[float, ...]]:
        cfg = self.config
        speeds = np.r_[np.full(6, cfg.max_joint_speed_rad_s), cfg.max_gripper_speed_s]
        self.command += np.clip(self.target - self.command, -speeds * dt, speeds * dt)
        self.command[:6] = np.clip(
            self.command[:6],
            position[:6] - cfg.max_tracking_error_rad,
            position[:6] + cfg.max_tracking_error_rad,
        )
        self.command[:6] = np.clip(self.command[:6], *np.asarray(JOINT_LIMITS).T)
        raw_goal = joint_to_motor(self.command, cfg)
        raw_position = joint_to_motor(position, cfg)
        gripper_error_rad = cfg.gripper_torque_limit / cfg.gripper_kp
        raw_goal[6] = np.clip(
            raw_goal[6], raw_position[6] - gripper_error_rad, raw_position[6] + gripper_error_rad
        )
        gravity = self._gravity_torque(position)
        gravity *= np.asarray(cfg.gravity_factors) * np.asarray(cfg.joint_signs)
        gravity = np.clip(gravity, -10.0, 10.0)
        kp, kd = [*cfg.kp, cfg.gripper_kp], [*cfg.kd, cfg.gripper_kd]
        return {
            name: (float(raw_goal[i]), 0.0, kp[i], kd[i], float(gravity[i]) if i < 6 else 0.0)
            for i, name in enumerate(MOTOR_NAMES)
        }

    def _run(self) -> None:
        previous = time.monotonic()
        max_cycle_gap = 0.0
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                max_cycle_gap = max(max_cycle_gap, started - previous)
                states = self._read_feedback(wait=True)
                position = motor_to_joint(np.asarray([states[name].pos for name in MOTOR_NAMES]), self.config)
                validate_positions(position, joint_tolerance_rad=0.03)
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
                    packet = (
                        self._command_packet(position, min(started - previous, 0.05)) if self.enabled else {}
                    )
                for name, command in packet.items():
                    if self._stop.is_set():
                        break
                    self.motors[name].send_mit(*command)
                previous = started
                self._stop.wait(max(0, 1 / self.config.control_frequency - (time.monotonic() - started)))
        except Exception as exc:
            exc.add_note(
                f"YAM maximum servo cycle gap: {max_cycle_gap * 1000:.1f} ms; "
                f"current cycle elapsed: {(time.monotonic() - started) * 1000:.1f} ms"
            )
            self._failure = exc
            self._stop.set()
        finally:
            self._disable_motors()
            if self._failure is not None:
                logger.error(
                    "YAM servo stopped: %s; %s",
                    self._failure,
                    "; ".join(getattr(self._failure, "__notes__", [])),
                )

    def _disable_motors(self) -> None:
        if not self.enabled or self.bus is None:
            return
        try:
            self.bus.disable_all()
            self.enabled = False
        except Exception:
            logger.exception("Could not disable YAM torque; use the hardware e-stop")

    def _check_feedback(self) -> None:
        if self._failure is not None or self._stop.is_set():
            raise ConnectionError("YAM servo stopped after a motor/feedback error") from self._failure
        if time.monotonic() - self.updated_at > self.config.feedback_timeout_s:
            self._stop.set()
            raise ConnectionError("YAM servo feedback is stale; reconnect before commanding motion")

    def _close_hardware(self) -> None:
        self._disable_motors()
        for name, motor in self.motors.items():
            try:
                motor.close()
            except Exception:
                logger.exception("Failed to close YAM motor %s", name)
        self.motors.clear()
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                logger.exception("Failed to close YAM MotorBridge controller")
            finally:
                self.bus = None
        if self.monitor is not None:
            try:
                self.monitor.shutdown()
            except Exception:
                logger.exception("Failed to close YAM feedback monitor")
            finally:
                self.monitor = None

    def _close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("YAM servo did not stop; use the hardware e-stop")
            self._thread = None
        try:
            self._close_hardware()
        except Exception:
            logger.exception("Failed to close YAM arm")
        try:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
        finally:
            self._connected = False
            self._calibration_session = False
            if self._gc_acquired:
                _ControlGC.release()
                self._gc_acquired = False

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
