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

import logging
import math
import time
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from lerobot.cameras import DepthCamera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorCalibration
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _motorbridge_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rebot_b601_follower import RebotB601FollowerRobotConfig
from .motor_family import (
    GRIPPER_MOTOR,
    MOTOR_PROFILES,
    ArmControlMode,
    GripperControlMode,
    MotorFamily,
    MotorFamilyProfile,
)

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import (
        Controller as MotorBridgeController,
        Mode as MotorBridgeMode,
    )
else:
    MotorBridgeController = None
    MotorBridgeMode = None

logger = logging.getLogger(__name__)

_ZERO_SETTLE_SEC = 0.1

# Impedance gripper tuning.
_GRIPPER_IMPEDANCE_DAMPING = 1.5
_GRIPPER_LPF_ALPHA = 0.3
_GRIPPER_TARGET_VEL_MAX = 3.0
_GRIPPER_HOLD_VEL_THRESHOLD = 0.25


class RebotB601Follower(Robot):
    """Seeed Studio reBot B601 follower arm (6-DOF + gripper, CAN motors).

    Motor communication is handled by the ``motorbridge`` package over a CAN bus,
    reached either through a Damiao serial bridge or MotorBridge's native CAN transport.
    """

    config_class = RebotB601FollowerRobotConfig
    name = "rebot_b601_follower"
    calibration: dict[str, MotorCalibration]

    def __init__(self, config: RebotB601FollowerRobotConfig):
        require_package("motorbridge", extra="rebot")
        super().__init__(config)
        self.config = config
        self.profile: MotorFamilyProfile = MOTOR_PROFILES[config.motor_family]
        # Config normalization expands these values before the robot is built.
        self._motor_can_ids = cast(dict[str, tuple[int, int]], config.motor_can_ids)
        self._joint_limits = cast(dict[str, tuple[float, float]], config.joint_limits)
        self._mit_kp = cast(dict[str, float], config.mit_kp)
        self._mit_kd = cast(dict[str, float], config.mit_kd)
        self._pos_vel_velocity = cast(dict[str, float], config.pos_vel_velocity or {})
        self.bus: MotorBridgeController | None = None
        self.motors: dict = {}
        self.motor_names = list(self._motor_can_ids)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._gripper_prev_target_pos: float | None = None
        self._gripper_prev_target_vel: float | None = None
        self._gripper_prev_state_pos: float | None = None
        self._gripper_prev_time: float | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus is not None and all(cam.is_connected for cam in self.cameras.values())

    def _validate_control_config(self) -> None:
        if self.config.control_mode is ArmControlMode.POS_VEL:
            missing = [
                motor_name
                for motor_name in self.motor_names
                if motor_name != GRIPPER_MOTOR and motor_name not in self._pos_vel_velocity
            ]
            if missing:
                raise ValueError(f"POS_VEL requires pos_vel_velocity for: {', '.join(missing)}.")

        if self.config.gripper_control_mode is GripperControlMode.FORCE_POS:
            if GRIPPER_MOTOR not in self._pos_vel_velocity or self.config.gripper_torque_ratio is None:
                raise ValueError("FORCE_POS requires gripper velocity and torque ratio settings.")
        elif self.config.gripper_control_mode is GripperControlMode.MIT_IMPEDANCE and (
            self.config.gripper_torque_limit is None or self.config.gripper_hold_torque_limit is None
        ):
            raise ValueError("MIT impedance requires moving and holding gripper torque limits.")

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._validate_control_config()
        logger.info(
            f"Connecting {self} on {self.config.port} "
            f"(family={self.config.motor_family.value}, adapter={self.config.can_adapter})..."
        )
        if self.config.can_adapter == "damiao":
            self.bus = MotorBridgeController.from_dm_serial(
                serial_port=self.config.port,
                baud=self.config.dm_serial_baud,
            )
        elif self.config.can_adapter == "socketcan":
            self.bus = MotorBridgeController(channel=self.config.port)
        else:
            raise ValueError(
                f"Unsupported can_adapter '{self.config.can_adapter}'. Use 'damiao' or 'socketcan'."
            )

        add_motor = (
            self.bus.add_damiao_motor
            if self.config.motor_family is MotorFamily.DM
            else self.bus.add_robstride_motor
        )
        for motor_name, (send_id, recv_id) in self._motor_can_ids.items():
            self.motors[motor_name] = add_motor(send_id, recv_id, self.profile.motor_models[motor_name])

        self._reset_gripper_impedance_state()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")
        if self.bus is None:
            raise DeviceNotConnectedError(f"{self} motor bus is not initialized")
        self.bus.disable_all()
        print(
            "\nCalibration: set zero position.\n"
            "Manually move the reBot B601 to its ZERO POSITION and close the gripper.\n"
            "See the B601 manual for the zero pose (the default sit-down position).\n"
        )
        input("Press ENTER when ready...")

        for motor in self.motors.values():
            motor.set_zero_position()
            time.sleep(_ZERO_SETTLE_SEC)
        logger.info("Arm zero position set.")

        self.calibration = {}
        for motor_name, (send_id, _recv_id) in self._motor_can_ids.items():
            range_min, range_max = self._public_to_motor_limits(motor_name, self._joint_limits[motor_name])
            self.calibration[motor_name] = MotorCalibration(
                id=send_id,
                drive_mode=0,
                homing_offset=0,
                range_min=int(range_min),
                range_max=int(range_max),
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Set each motor's control mode before enabling torque."""
        if self.bus is None:
            raise DeviceNotConnectedError(f"{self} motor bus is not initialized")
        self._validate_control_config()
        self.bus.disable_all()
        for motor_name, motor in self.motors.items():
            if motor_name == GRIPPER_MOTOR:
                if self.config.gripper_control_mode is GripperControlMode.FORCE_POS:
                    target_mode = MotorBridgeMode.FORCE_POS
                else:
                    target_mode = MotorBridgeMode.MIT
            elif self.config.control_mode is ArmControlMode.POS_VEL:
                target_mode = MotorBridgeMode.POS_VEL
            else:
                target_mode = MotorBridgeMode.MIT

            motor.ensure_mode(target_mode)
            logger.debug(f"{motor_name} mode set to {target_mode}")
        self.bus.enable_all()

    @check_if_not_connected
    def disable_torque(self) -> None:
        """Disable motor torque so the arm can be moved by hand (read-only debugging)."""
        if self.bus is None:
            raise DeviceNotConnectedError(f"{self} motor bus is not initialized")
        self.bus.disable_all()
        logger.info(f"{self} torque disabled.")

    def _read_feedback(self) -> dict[str, Any]:
        """Request and return the latest motor states from MotorBridge."""
        if self.bus is None:
            raise DeviceNotConnectedError(f"{self} motor bus is not initialized")
        try:
            for motor in self.motors.values():
                motor.request_feedback()
            self.bus.poll_feedback_once()

            states = {motor_name: motor.get_state() for motor_name, motor in self.motors.items()}
            unavailable = [motor_name for motor_name, state in states.items() if state is None]
            if unavailable:
                raise RuntimeError(f"No motor feedback available for: {', '.join(unavailable)}.")
            return states
        except Exception:
            if self.config.gripper_control_mode is GripperControlMode.MIT_IMPEDANCE:
                self._stop_after_feedback_error()
            raise

    def _public_to_motor_position(self, motor_name: str, position_deg: float) -> float:
        return position_deg * self.profile.joint_directions[motor_name]

    def _motor_to_public_position(self, motor_name: str, position_deg: float) -> float:
        return position_deg / self.profile.joint_directions[motor_name]

    def _public_to_motor_limits(self, motor_name: str, limits: tuple[float, float]) -> tuple[float, float]:
        motor_limits = tuple(self._public_to_motor_position(motor_name, value) for value in limits)
        return min(motor_limits), max(motor_limits)

    def _present_pos(self, states: dict[str, Any] | None = None) -> dict[str, float]:
        """Read current positions in the public robot coordinate frame."""
        if states is None:
            states = self._read_feedback()
        return {
            motor_name: self._motor_to_public_position(motor_name, math.degrees(state.pos))
            for motor_name, state in states.items()
        }

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        obs_dict: RobotObservation = {f"{motor}.pos": pos for motor, pos in self._present_pos().items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if isinstance(cam, DepthCamera) and cam.use_depth:
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command the arm to a target joint configuration.

        Positions are expressed in degrees. The relative action magnitude may be
        clipped depending on `max_relative_target`, so the action actually sent is
        always returned.
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap relative target when too far from the present position.
        states = None
        if self.config.max_relative_target is not None:
            states = self._read_feedback()
            present_pos = self._present_pos(states)
            goal_present_pos = {key: (goal, present_pos.get(key, goal)) for key, goal in goal_pos.items()}
            relative_limits = self.config.max_relative_target
            if isinstance(relative_limits, dict):
                relative_limits = {key: relative_limits[key] for key in goal_present_pos}
            goal_pos = ensure_safe_goal_position(goal_present_pos, relative_limits)

        sent_pos = {}
        for motor_name, position_deg in goal_pos.items():
            if motor_name in self.motors:
                state = states.get(motor_name) if states is not None else None
                sent_pos[motor_name] = self._send_joint(motor_name, position_deg, state)
        return {f"{motor}.pos": val for motor, val in sent_pos.items()}

    def _send_joint(self, motor_name: str, position_deg: float, state: Any | None = None) -> float:
        """Send one joint command and return its clipped public position."""
        motor = self.motors[motor_name]

        # Clip against soft joint limits.
        limits = self._joint_limits.get(motor_name)
        if limits is not None:
            range_min, range_max = limits
            clipped = max(range_min, min(range_max, position_deg))
            if clipped != position_deg:
                logger.debug(f"Clipped {motor_name} from {position_deg:.2f} to {clipped:.2f}")
            position_deg = clipped
        motor_position_deg = self._public_to_motor_position(motor_name, position_deg)
        position_rad = math.radians(motor_position_deg)

        if motor_name == GRIPPER_MOTOR:
            self._send_gripper(position_rad, state)
        elif self.config.control_mode is ArmControlMode.POS_VEL:
            motor.send_pos_vel(position_rad, math.radians(self._pos_vel_velocity[motor_name]))
        else:
            motor.send_mit(
                position_rad,
                0.0,
                self._mit_kp[motor_name],
                self._mit_kd[motor_name],
                0.0,
            )
        return position_deg

    def _send_gripper(self, position_rad: float, state: Any | None = None) -> None:
        """Send the gripper command with its configured control mode."""
        motor = self.motors[GRIPPER_MOTOR]
        if self.config.gripper_control_mode is GripperControlMode.FORCE_POS:
            motor.send_force_pos(
                position_rad,
                math.radians(self._pos_vel_velocity[GRIPPER_MOTOR]),
                cast(float, self.config.gripper_torque_ratio),
            )
        elif self.config.gripper_control_mode is GripperControlMode.MIT_IMPEDANCE:
            torque = self._gripper_impedance_torque(position_rad, state)
            motor.send_mit(0.0, 0.0, 0.0, _GRIPPER_IMPEDANCE_DAMPING, torque)
        else:
            motor.send_mit(
                position_rad,
                0.0,
                self._mit_kp[GRIPPER_MOTOR],
                self._mit_kd[GRIPPER_MOTOR],
                0.0,
            )

    def _stop_after_feedback_error(self) -> None:
        logger.warning("Motor feedback failed; disabling all motor torque.")
        motor = self.motors[GRIPPER_MOTOR]
        try:
            motor.send_mit(0.0, 0.0, 0.0, _GRIPPER_IMPEDANCE_DAMPING, 0.0)
        except Exception:
            logger.exception("Failed to command zero RS gripper torque after feedback loss.")
        try:
            if self.bus is not None:
                self.bus.disable_all()
        except Exception:
            logger.exception("Failed to disable reBot torque after RS gripper feedback loss.")

    def _reset_gripper_impedance_state(self) -> None:
        self._gripper_prev_target_pos = None
        self._gripper_prev_target_vel = None
        self._gripper_prev_state_pos = None
        self._gripper_prev_time = None

    def _gripper_impedance_torque(self, target_pos_rad: float, state: Any | None = None) -> float:
        """Compute a force-limited impedance torque for the gripper."""
        if state is None:
            state = self._read_feedback()[GRIPPER_MOTOR]
        now = time.monotonic()
        dt = 0.0 if self._gripper_prev_time is None else now - self._gripper_prev_time
        self._gripper_prev_time = now

        prev_target = self._gripper_prev_target_pos
        target_vel = 0.0 if prev_target is None or dt <= 0.0 else (target_pos_rad - prev_target) / dt
        self._gripper_prev_target_pos = target_pos_rad

        prev_target_vel = self._gripper_prev_target_vel
        if prev_target_vel is not None:
            target_vel = _GRIPPER_LPF_ALPHA * target_vel + (1.0 - _GRIPPER_LPF_ALPHA) * prev_target_vel
        target_vel = max(-_GRIPPER_TARGET_VEL_MAX, min(_GRIPPER_TARGET_VEL_MAX, target_vel))
        self._gripper_prev_target_vel = target_vel

        prev_state_pos = self._gripper_prev_state_pos
        measured_vel = 0.0 if prev_state_pos is None or dt <= 0.0 else (state.pos - prev_state_pos) / dt
        self._gripper_prev_state_pos = state.pos

        torque = self._mit_kp[GRIPPER_MOTOR] * (target_pos_rad - state.pos) + self._mit_kd[GRIPPER_MOTOR] * (
            target_vel - state.vel
        )

        limit = (
            self.config.gripper_torque_limit
            if abs(measured_vel) > _GRIPPER_HOLD_VEL_THRESHOLD
            else self.config.gripper_hold_torque_limit
        )
        limit = cast(float, limit)
        return max(-limit, min(limit, torque))

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect motors and cameras from the robot."""
        if self.bus is None:
            raise DeviceNotConnectedError(f"{self} motor bus is not initialized")
        for motor in self.motors.values():
            if self.config.disable_torque_on_disconnect:
                motor.disable()
            motor.clear_error()
            motor.close()

        self.bus.close()
        self.bus = None
        self.motors = {}

        for cam in self.cameras.values():
            cam.disconnect()

        self._reset_gripper_impedance_state()
        logger.info(f"{self} disconnected.")
