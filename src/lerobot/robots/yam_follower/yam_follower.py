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

"""Single-arm YAM v1 with a gravity-compensated MIT servo loop."""

import logging
import math
import threading
import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model import GravityCompensation
from lerobot.motors import MotorCalibration
from lerobot.utils import (
    check_if_already_connected,
    check_if_not_connected,
    ensure_bucket_file,
    require_package,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import (
    _can_available,
    _motorbridge_available,
)

from ..robot import Robot
from .config_yam_follower import (
    JOINT_LIMITS_RAD,
    MOTOR_NAMES,
    YAM_FEATURE_NAMES,
    YamFollowerConfig,
    YamFollowerRobotConfig,
)
from .mit_control import (
    GripperParams,
    MitArmParams,
    MitCommand,
    MitServo,
    MotorStates,
    clip_to_limits,
    joint_to_motor,
    read_joint_state,
)

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import Controller, Mode

if TYPE_CHECKING or _can_available:
    import can

logger = logging.getLogger(__name__)

_MODEL_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
_MODEL_GRIPPER_JOINTS = ("joint7", "joint8")
# Stroke of each finger joint in the gravity model, and the finger stroke that converts the linear
# DM4310 gripper's force limit to motor torque.
_MODEL_GRIPPER_STROKE_M = 0.0475
_GRIPPER_FINGER_STROKE_M = 0.096
_GRAVITY_MODEL_BUCKET_ID = "lerobot/robot-urdfs"
_GRAVITY_MODEL_REMOTE_PATH = "yam/v1/linear_4310.urdf"
_GRAVITY_MODEL_LOCAL_PATH = HF_LEROBOT_HOME / "robot-urdfs" / _GRAVITY_MODEL_REMOTE_PATH
_MAX_GRAVITY_TORQUE_NM = 25.0
_JOINT_LIMITS = np.asarray(JOINT_LIMITS_RAD)
_JOINT_FEEDBACK_TOLERANCE_RAD = 0.15
_GRIPPER_FEEDBACK_TOLERANCE = 0.10
# Plausible motor stroke between the gripper limits (0.5 to 10 rad).
_GRIPPER_STROKE_RANGE_DEG = (math.degrees(0.5), math.degrees(10.0))
# Damiao register 9 stores the CAN timeout in 50 us ticks.
_CAN_TIMEOUT_REGISTER = 9
_CAN_TIMEOUT_TICK_US = 50
_MAX_CAN_TIMEOUT_TICKS = 8000  # I2RT's 400 ms YAM timeout.
# YAM's motors use IDs 1 to 7, and each answers on CAN ID motor ID + 0x10.
_MOTOR_IDS = range(1, len(MOTOR_NAMES) + 1)
_FEEDBACK_ID_OFFSET = 0x10


def _yam_arm_params(config: YamFollowerConfig, gripper: MotorCalibration) -> MitArmParams:
    """Build the internal control settings of a YAM arm from its config and gripper calibration."""
    limits = (math.radians(gripper.range_min), math.radians(gripper.range_max))
    closed, opened = limits[::-1] if gripper.drive_mode else limits
    return MitArmParams(
        joint_limits=_JOINT_LIMITS,
        joint_feedback_tolerance_rad=_JOINT_FEEDBACK_TOLERANCE_RAD,
        # YAM's joints read their motors directly, with no flip or offset.
        joint_signs=np.ones(6),
        joint_offsets=np.zeros(6),
        kp=np.asarray(config.kp, dtype=float),
        kd=np.asarray(config.kd, dtype=float),
        max_joint_speed=math.radians(config.max_joint_speed_deg_s),
        max_tracking_error=math.radians(config.max_tracking_error_deg),
        gravity_factors=np.asarray(config.gravity_factors, dtype=float),
        max_gravity_torque=_MAX_GRAVITY_TORQUE_NM,
        gripper=GripperParams(
            closed=closed,
            open=opened,
            kp=config.gripper_kp,
            kd=config.gripper_kd,
            max_speed=config.max_gripper_speed,
            force_limit_n=config.gripper_force_limit_n,
            finger_stroke_m=_GRIPPER_FINGER_STROKE_M,
            feedback_tolerance=_GRIPPER_FEEDBACK_TOLERANCE,
        ),
        float_kd=np.asarray(config.float_kd, dtype=float),
        coulomb_friction=np.asarray(config.coulomb_friction, dtype=float)
        if config.friction_compensation
        else np.zeros(6),
        control_frequency=config.control_frequency,
        feedback_timeout_s=config.feedback_timeout_s,
        command_timeout_s=config.command_timeout_s,
        idle_mode=config.idle_mode,
    )


def _to_public(values: np.ndarray, use_degrees: bool) -> np.ndarray:
    """Convert positions or speeds from internal units to observation and action units."""
    if not use_degrees:
        return values.copy()
    return np.r_[np.rad2deg(values[:6]), values[6] * 100.0]


def _from_public(values: np.ndarray, use_degrees: bool) -> np.ndarray:
    """Convert positions or speeds from observation and action units to internal units."""
    if not use_degrees:
        return values.copy()
    return np.r_[np.deg2rad(values[:6]), values[6] / 100.0]


class _YamBus:
    """CAN interface of one YAM arm.

    MotorBridge sends commands and decodes motor states. A receive-only python-can socket on
    the same interface checks that every motor answered recently and reports motor faults,
    because MotorBridge's cached states carry no receive time.
    """

    motor_names: tuple[str, ...] = MOTOR_NAMES

    def __init__(self, port: str, feedback_timeout_s: float) -> None:
        self.port = port
        self.feedback_timeout_s = feedback_timeout_s
        self.controller: Controller | None = None
        self.monitor: can.BusABC | None = None
        self.motors: dict[str, Any] = {}
        self.enabled = False
        self._enabled_at = 0.0
        self._last_feedback: dict[int, float] = {}  # receive time per motor ID
        self._last_status: dict[int, int] = {}

    def open(self) -> None:
        """Open the SocketCAN monitor and MotorBridge devices."""
        self.enabled = False
        self.monitor = can.Bus(
            channel=self.port,
            interface="socketcan",
            # Only the motors' feedback frames reach the monitor.
            can_filters=[
                {"can_id": motor_id + _FEEDBACK_ID_OFFSET, "can_mask": 0x7FF, "extended": False}
                for motor_id in _MOTOR_IDS
            ],
            ignore_rx_error_frames=True,
        )
        self.controller = Controller(channel=self.port)
        self.motors = {
            name: self.controller.add_damiao_motor(
                motor_id, motor_id + _FEEDBACK_ID_OFFSET, "4340" if motor_id <= 3 else "4310"
            )
            for motor_id, name in zip(_MOTOR_IDS, self.motor_names, strict=True)
        }
        self._last_feedback.clear()
        self._last_status.clear()

    def read_states(self) -> MotorStates:
        """Return the raw states of the seven motors, once every motor has fresh feedback."""
        states = self._wait_for_fresh_states()
        return MotorStates(
            position=np.asarray([states[name].pos for name in self.motor_names], dtype=float),
            velocity=np.asarray([states[name].vel for name in self.motor_names], dtype=float),
            torque=np.asarray([states[name].torq for name in self.motor_names], dtype=float),
        )

    def set_mit_mode(self) -> None:
        """Put every motor into MIT control mode."""
        controller, _ = self._require_open()
        controller.disable_all()
        for motor in self.motors.values():
            motor.ensure_mode(Mode.MIT)

    def can_timeouts(self) -> dict[str, int]:
        """Read each motor's CAN timeout register; 0 means the timeout is off."""
        return {
            name: int(motor.get_register_u32(_CAN_TIMEOUT_REGISTER)) for name, motor in self.motors.items()
        }

    def enable(self, hold: np.ndarray) -> None:
        """Send zero-gain commands at the `hold` motor positions, then enable torque."""
        controller, _ = self._require_open()
        for name, value in zip(self.motor_names, hold, strict=True):
            self.motors[name].send_mit(float(value), 0.0, 0.0, 0.0, 0.0)
        self.enabled = True
        controller.enable_all()
        self._enabled_at = time.monotonic()

    def send_mit(self, motor: str, command: MitCommand) -> None:
        """Send one MIT command to a motor."""
        self.motors[motor].send_mit(
            command.position,
            command.velocity,
            command.kp,
            command.kd,
            command.torque,
        )

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
        """Disable torque and close all CAN resources."""
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

    def _require_open(self) -> "tuple[Controller, can.BusABC]":
        if self.controller is None or self.monitor is None:
            raise RuntimeError(f"{self.port} bus is not open")
        return self.controller, self.monitor

    def _wait_for_fresh_states(self) -> dict[str, Any]:
        controller, monitor = self._require_open()
        for motor in self.motors.values():
            motor.request_feedback()
        deadline = time.monotonic() + self.feedback_timeout_s
        while True:
            for _ in range(256):
                msg = monitor.recv(timeout=0)
                if msg is None:
                    break
                motor_id = msg.arbitration_id - _FEEDBACK_ID_OFFSET
                status = msg.data[0] >> 4
                if status not in (0, 1):
                    raise ConnectionError(f"{self.port}: motor {motor_id} fault {status:#x}")
                self._last_feedback[motor_id] = msg.timestamp
                self._last_status[motor_id] = status
            controller.poll_feedback_once()
            states = {name: motor.get_state() for name, motor in self.motors.items()}
            # SocketCAN frame timestamps use wall time; monotonic time above bounds the wait.
            now = time.time()
            fresh = all(
                0 <= now - self._last_feedback.get(motor_id, 0) <= self.feedback_timeout_s
                for motor_id in _MOTOR_IDS
            )
            if fresh and all(state is not None for state in states.values()):
                if self.enabled and time.monotonic() - self._enabled_at > self.feedback_timeout_s:
                    disabled = [motor_id for motor_id in _MOTOR_IDS if self._last_status[motor_id] == 0]
                    if disabled:
                        raise ConnectionError(f"{self.port}: motor {disabled[0]} unexpectedly disabled")
                return states
            if time.monotonic() >= deadline:
                ages = ", ".join(
                    f"{motor_id}: {(now - self._last_feedback[motor_id]) * 1000:.1f} ms"
                    if motor_id in self._last_feedback
                    else f"{motor_id}: missing"
                    for motor_id in _MOTOR_IDS
                )
                raise ConnectionError(f"{self.port}: missing or stale motor feedback ({ages})")
            time.sleep(0.001)


class YamFollower(Robot):
    """A YAM arm exposing six joint positions and a gripper opening.

    Args:
        config (`YamFollowerRobotConfig`):
            Hardware, calibration and control settings.
        stop_event (`threading.Event | None`, *optional*):
            Stop signal shared by both arms of a bimanual robot.
    """

    config_class = YamFollowerRobotConfig
    name = "yam_follower"
    supports_position_hold = True

    def __init__(self, config: YamFollowerRobotConfig, stop_event: threading.Event | None = None) -> None:
        require_package("motorbridge", extra="yam")
        require_package("python-can", extra="yam", import_name="can")
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.bus = _YamBus(config.port, config.feedback_timeout_s)
        self.gravity_model: GravityCompensation | None = None
        self.servo = MitServo(str(self.id), self.bus, self._gravity_torque, stop_event or threading.Event())
        self._connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        suffixes = (".pos", ".vel", ".torque") if self.config.use_velocity_and_torque else (".pos",)
        return {f"{name}{suffix}": float for name in MOTOR_NAMES for suffix in suffixes}

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

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect the hardware and start control.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Measure missing gripper limits, then start control. When `False`, connect only the CAN
                interface, with torque off, so calibration can be run explicitly.
        """
        self._open_bus()
        if calibrate:
            try:
                if not self.is_calibrated:
                    self.calibrate()
                self.open()
                self.configure()
                self.start()
            except BaseException:
                self.disconnect()
                raise
        logger.info(f"{self} connected." if calibrate else f"{self} connected with torque off.")

    @property
    def is_calibrated(self) -> bool:
        return "gripper" in self.calibration

    @check_if_not_connected
    def calibrate(self) -> None:
        """Measure the gripper limits without enabling torque or changing joint zeros."""
        if self.servo.active:
            raise RuntimeError("Reconnect with calibrate=False before measuring the gripper limits")
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
        for limit in ("closed", "open"):
            input(f"[{self.id}] Place the gripper fully {limit}, release it, then press Enter: ")
            samples = []
            for _ in range(10):
                samples.append(self.bus.read_states().position[6])
                time.sleep(0.02)
            if np.ptp(samples) > 0.03:
                raise ValueError("Gripper moved while it was measured; calibration was not saved")
            measurements[limit] = float(np.median(samples))
        closed, opened = measurements["closed"], measurements["open"]
        # Motor angles of the two limits, in whole degrees; drive_mode records which is closed.
        stops_deg = sorted(round(math.degrees(value)) for value in (closed, opened))
        stroke_deg = stops_deg[1] - stops_deg[0]
        low, high = _GRIPPER_STROKE_RANGE_DEG
        if not low <= stroke_deg <= high:
            raise ValueError(
                f"Gripper stroke of {stroke_deg} motor degrees is outside {low:.0f} to {high:.0f}; "
                "calibration was not saved"
            )
        self.calibration = {
            "gripper": MotorCalibration(
                id=7,
                drive_mode=int(opened < closed),
                homing_offset=0,
                range_min=stops_deg[0],
                range_max=stops_deg[1],
            )
        }
        self._save_calibration()
        logger.info("Saved the gripper limits to %s. Joint zeros were not changed.", self.calibration_fpath)

    def configure(self) -> None:
        """Switch the motors to MIT mode, and read again the position the servo starts from."""
        if self.servo.active:
            # Changing modes disables torque, which would drop an arm the servo is holding.
            raise RuntimeError("Configure YAM motors before the servo starts, not while it runs")
        self.bus.set_mit_mode()
        self._check_can_timeouts()
        params = self._require_params()
        self.servo.reset(read_joint_state(self.bus, params), params)

    @check_if_not_connected
    def open(self) -> None:
        """Connect the cameras, check the initial position, and reset the servo to it, without torque.

        Once the CAN interface is open, `connect` runs `open`, `configure` and `start`.
        """
        params = self._require_params()
        self.servo.stop_event.clear()
        self._load_gravity_model()
        for camera in self.cameras.values():
            camera.connect()
        state = read_joint_state(self.bus, params)
        self._check_initial_position(state.position)
        self.servo.reset(state, params)

    @check_if_not_connected
    def start(self) -> None:
        """Enable torque at the measured position, then start the servo."""
        if self.servo.active:
            raise RuntimeError("YAM servo is already running")
        self.bus.enable(joint_to_motor(self.servo.state.position, self._require_params()))
        self.servo.start()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Return the latest motor state and camera frames.

        Returns:
            `RobotObservation`: Joint state and the latest frame from each configured camera.

        Raises:
            RuntimeError: If the robot was opened only for calibration and the servo is not running.
        """
        if not self.servo.active:
            raise RuntimeError("Reconnect after calibration before reading policy observations")
        state = self.servo.latest()
        use_degrees = self.config.use_degrees
        values = {"pos": _to_public(state.position, use_degrees)}
        if self.config.use_velocity_and_torque:
            values["vel"] = _to_public(state.velocity, use_degrees)
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
        """Clip a joint-position target to the joint limits and submit it to the servo.

        Args:
            action (`RobotAction`):
                Absolute targets for all six joints and the gripper.

        Returns:
            `RobotAction`: The clipped target submitted to the servo.

        Raises:
            RuntimeError: If the servo is not running.
            KeyError: If a joint or gripper target is missing.
        """
        if not self.servo.active:
            raise RuntimeError("YAM commands require an active servo")
        values = np.asarray([action[f"{name}.pos"] for name in MOTOR_NAMES], dtype=float)
        target = clip_to_limits(_from_public(values, self.config.use_degrees), _JOINT_LIMITS)
        self.servo.set_target(target)
        sent = _to_public(target, self.config.use_degrees)
        return {f"{name}.pos": float(sent[i]) for i, name in enumerate(MOTOR_NAMES)}

    @check_if_not_connected
    def disconnect(self) -> None:
        """Stop control and release all hardware resources."""
        self._close()
        logger.info(f"{self} disconnected.")

    def _require_params(self) -> MitArmParams:
        if not self.is_calibrated:
            raise ValueError("Run lerobot-calibrate with this robot.id to measure the gripper limits")
        return _yam_arm_params(self.config, self.calibration["gripper"])

    def _open_bus(self) -> None:
        """Open the CAN interface, with torque off, once every motor answers."""
        try:
            self.bus.open()
            self.bus.read_states()
            self._connected = True
        except BaseException:
            self._close()
            raise

    def _load_gravity_model(self) -> None:
        if not self.config.gravity_compensation or self.gravity_model is not None:
            return
        model_path = ensure_bucket_file(
            _GRAVITY_MODEL_BUCKET_ID,
            _GRAVITY_MODEL_REMOTE_PATH,
            _GRAVITY_MODEL_LOCAL_PATH,
            token=False,
        )
        self.gravity_model = GravityCompensation(model_path, _MODEL_ARM_JOINTS, base_frame="base")

    def _gravity_torque(self, positions: np.ndarray) -> np.ndarray:
        if self.gravity_model is None:
            return np.zeros(6)
        opening = float(positions[6]) * _MODEL_GRIPPER_STROKE_M
        return self.gravity_model.torques(positions[:6], dict.fromkeys(_MODEL_GRIPPER_JOINTS, opening))

    def _check_initial_position(self, position: np.ndarray) -> None:
        if self.config.initial_position_deg is not None and np.any(
            np.abs(np.rad2deg(position[:6]) - self.config.initial_position_deg)
            > self.config.initial_tolerance_deg
        ):
            raise ValueError("YAM arm is not within initial_tolerance_deg of initial_position_deg")

    def _check_can_timeouts(self) -> None:
        """Raise unless every motor disables itself within 400 ms of the last command."""
        unsafe = {
            name: value
            for name, value in self.bus.can_timeouts().items()
            if not 0 < value <= _MAX_CAN_TIMEOUT_TICKS
        }
        if not unsafe:
            return
        details = ", ".join(
            f"{name}=off" if value == 0 else f"{name}={value * _CAN_TIMEOUT_TICK_US / 1000:g} ms"
            for name, value in unsafe.items()
        )
        raise ValueError(
            f"{self}: unsafe motor CAN timeout ({details}); configure every motor to at most 400 ms "
            "with the Damiao/I2RT motor tools"
        )

    def _close(self) -> None:
        stop_error: RuntimeError | None = None
        try:
            self.servo.stop()
        except RuntimeError as exc:
            stop_error = exc
            # Leave CAN open while the thread might still be using it, but still try to disable torque.
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
