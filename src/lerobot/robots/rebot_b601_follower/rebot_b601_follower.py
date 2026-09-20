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
from typing import TYPE_CHECKING, Any

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
    ARM_MODE_POS_VEL,
    GRIPPER_MODE_FORCE_POS,
    GRIPPER_MODE_MIT_IMPEDANCE,
    GRIPPER_MOTOR,
    MOTOR_PROFILES,
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

_ENSURE_MODE_RETRIES = 9
_SETTLE_SEC = 0.01
_ZERO_SETTLE_SEC = 0.1

# --- Impedance gripper tuning (shared by any family that offers the mode) ---
# Motor-side velocity damping. The position setpoint and kp are both zero, so the
# gripper is driven entirely by the feedforward torque.
_GRIPPER_IMPEDANCE_DAMPING = 1.5
# Low-pass factor and ceiling (rad/s) for the target velocity estimate.
_GRIPPER_LPF_ALPHA = 0.3
_GRIPPER_TARGET_VEL_MAX = 3.0
# Below this |measured velocity| (rad/s) the gentler hold torque limit applies.
_GRIPPER_HOLD_VEL_THRESHOLD = 0.25


class RebotB601Follower(Robot):
    """Seeed Studio reBot B601 follower arm (6-DOF + gripper, CAN motors).

    Supports both builds of the arm through ``config.motor_family``: ``"dm"`` for
    the Damiao B601-DM and ``"rs"`` for the RobStride B601-RS. The two share
    joint topology, names and degree units but differ in geometry, motor models,
    CAN id convention, available control modes, installed joint direction and
    gripper strategy — all carried by the family's
    :class:`MotorFamilyProfile`.

    Motor communication is handled by the ``motorbridge`` package over a CAN bus,
    reached either through a Damiao serial bridge or MotorBridge's native,
    platform-specific CAN transport.

    Observations and actions share one public robot coordinate frame. Family
    profiles convert that frame to and from raw motor coordinates, so a position
    read from an observation can be sent back as an action to hold the joint.
    """

    config_class = RebotB601FollowerRobotConfig
    name = "rebot_b601_follower"

    def __init__(self, config: RebotB601FollowerRobotConfig):
        require_package("motorbridge", extra="rebot")
        super().__init__(config)
        self.config = config
        self.profile: MotorFamilyProfile = MOTOR_PROFILES[config.motor_family]
        self.bus: MotorBridgeController | None = None
        self.motors: dict = {}
        self.motor_names = list(config.motor_can_ids)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._reset_gripper_impedance_state()

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

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(
            f"Connecting {self} on {self.config.port} "
            f"(family={self.config.motor_family.value}, adapter={self.config.can_adapter})..."
        )
        if self.config.can_adapter == "damiao":
            self.bus = MotorBridgeController.from_dm_serial(
                serial_port=self.config.port,
                baud=self.config.dm_serial_baud,
            )
        else:
            self.bus = MotorBridgeController(channel=self.config.port)

        self._register_motors()
        self._reset_gripper_impedance_state()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file "
                "or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def _register_motors(self) -> None:
        """Declare every joint on the bus using this family's motorbridge factory."""
        if self.bus is None:
            raise DeviceNotConnectedError(f"{self} motor bus is not initialized")
        add_motor = (
            self.bus.add_damiao_motor
            if self.config.motor_family is MotorFamily.DM
            else self.bus.add_robstride_motor
        )
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            self.motors[motor_name] = add_motor(send_id, recv_id, self.profile.motor_models[motor_name])

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
        for motor_name, (send_id, _recv_id) in self.config.motor_can_ids.items():
            range_min, range_max = self.config.joint_limits[motor_name]
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
        """Configure every motor while torque is off."""
        if self.bus is None:
            raise DeviceNotConnectedError(f"{self} motor bus is not initialized")
        self.bus.disable_all()
        self._apply_control_modes()
        self.bus.enable_all()

    def _apply_control_modes(self) -> None:
        for motor_name, motor in self.motors.items():
            mode = (
                self.config.gripper_control_mode if motor_name == GRIPPER_MOTOR else self.config.control_mode
            )
            if mode == ARM_MODE_POS_VEL:
                target_mode = MotorBridgeMode.POS_VEL
            elif mode == GRIPPER_MODE_FORCE_POS:
                target_mode = MotorBridgeMode.FORCE_POS
            else:
                target_mode = MotorBridgeMode.MIT
            for attempt in range(_ENSURE_MODE_RETRIES + 1):
                try:
                    motor.ensure_mode(target_mode)
                    break
                except Exception:
                    if attempt == _ENSURE_MODE_RETRIES:
                        raise
                    time.sleep(_SETTLE_SEC)
            logger.debug(f"{motor_name} mode set to {target_mode}")

    @check_if_not_connected
    def disable_torque(self) -> None:
        """Disable motor torque so the arm can be moved by hand (read-only debugging).

        The arm becomes back-drivable and will fall under gravity: hold it first.
        """
        if self.bus is None:
            raise DeviceNotConnectedError(f"{self} motor bus is not initialized")
        self.bus.disable_all()
        logger.info(f"{self} torque disabled.")

    def _read_feedback(self) -> dict[str, Any]:
        """Request and return the latest motor states from MotorBridge."""
        if self.bus is None:
            raise DeviceNotConnectedError(f"{self} motor bus is not initialized")
        for motor in self.motors.values():
            motor.request_feedback()
        self.bus.poll_feedback_once()

        states = {motor_name: motor.get_state() for motor_name, motor in self.motors.items()}
        unavailable = [motor_name for motor_name, state in states.items() if state is None]
        if unavailable:
            raise RuntimeError(f"No motor feedback available for: {', '.join(unavailable)}.")
        return states

    def _public_to_motor_position(self, motor_name: str, position_deg: float) -> float:
        return position_deg * self.profile.joint_directions[motor_name]

    def _motor_to_public_position(self, motor_name: str, position_deg: float) -> float:
        return position_deg / self.profile.joint_directions[motor_name]

    def _present_pos(self) -> dict[str, float]:
        """Read current positions in the public robot coordinate frame."""
        return {
            motor_name: self._motor_to_public_position(motor_name, math.degrees(state.pos))
            for motor_name, state in self._read_feedback().items()
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
        always returned in the same public robot coordinate frame as observations.
        Joints omitted from a partial action are not sent a new command.
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            present_pos = self._present_pos()
            goal_present_pos = {key: (goal, present_pos.get(key, goal)) for key, goal in goal_pos.items()}
            relative_limits = self.config.max_relative_target
            if isinstance(relative_limits, dict):
                relative_limits = {key: relative_limits[key] for key in goal_present_pos}
            goal_pos = ensure_safe_goal_position(goal_present_pos, relative_limits)

        sent_pos = {}
        for motor_name, position_deg in goal_pos.items():
            if motor_name in self.motors:
                sent_pos[motor_name] = self._send_joint(motor_name, position_deg)
        return {f"{motor}.pos": val for motor, val in sent_pos.items()}

    def _send_joint(self, motor_name: str, position_deg: float) -> float:
        """Emit one joint command.

        Every command reaches the bus through here. The requested position enters
        in the public robot frame, then is mapped into the raw motor frame and
        clamped.

        Returns:
            The position actually sent, expressed in the public robot frame.
        """
        motor = self.motors[motor_name]
        motor_position_deg = self._public_to_motor_position(motor_name, position_deg)
        limits = self.config.joint_limits.get(motor_name)
        if limits is not None:
            range_min, range_max = limits
            clipped = max(range_min, min(range_max, motor_position_deg))
            if clipped != motor_position_deg:
                logger.debug(f"Clipped {motor_name} from {motor_position_deg:.2f} to {clipped:.2f}")
            motor_position_deg = clipped
        position_rad = math.radians(motor_position_deg)

        if motor_name == GRIPPER_MOTOR:
            if self.config.gripper_control_mode == GRIPPER_MODE_MIT_IMPEDANCE:
                # Driven purely by a clamped feedforward torque, with the position
                # setpoint and kp both zero. This bounds grip force, where a plain
                # position command would keep pushing toward the target regardless
                # of force and can overcurrent when closing on an object.
                try:
                    tau = self._gripper_impedance_torque(position_rad)
                except Exception:
                    try:
                        motor.send_mit(0.0, 0.0, 0.0, _GRIPPER_IMPEDANCE_DAMPING, 0.0)
                    except Exception:
                        logger.exception("Failed to command zero RS gripper torque after feedback loss.")
                    try:
                        if self.bus is not None:
                            self.bus.disable_all()
                    except Exception:
                        logger.exception("Failed to disable reBot torque after RS gripper feedback loss.")
                    raise
                motor.send_mit(0.0, 0.0, 0.0, _GRIPPER_IMPEDANCE_DAMPING, tau)
            elif self.config.gripper_control_mode == GRIPPER_MODE_FORCE_POS:
                motor.send_force_pos(
                    position_rad,
                    math.radians(self.config.pos_vel_velocity[motor_name]),
                    self.config.gripper_torque_ratio,
                )
            else:
                motor.send_mit(
                    position_rad,
                    0.0,
                    self.config.mit_kp[motor_name],
                    self.config.mit_kd[motor_name],
                    0.0,
                )
        elif self.config.control_mode == ARM_MODE_POS_VEL:
            motor.send_pos_vel(position_rad, math.radians(self.config.pos_vel_velocity[motor_name]))
        else:
            motor.send_mit(
                position_rad,
                0.0,
                self.config.mit_kp[motor_name],
                self.config.mit_kd[motor_name],
                0.0,
            )
        # The impedance gripper sends an internal zero position setpoint, but its
        # effective requested target remains this clipped public position.
        return self._motor_to_public_position(motor_name, motor_position_deg)

    def _reset_gripper_impedance_state(self) -> None:
        self._gripper_prev_target_pos: float | None = None
        self._gripper_prev_target_vel: float | None = None
        self._gripper_prev_state_pos: float | None = None
        self._gripper_prev_time: float | None = None

    def _gripper_impedance_torque(self, target_pos_rad: float) -> float:
        """Force-limited impedance torque for the gripper.

        Computes ``tau = kp*(x_r - x) + kd*(v_r - v)`` from the target and the
        motor's measured state, then clamps it to the moving or holding torque
        limit depending on how fast the gripper is actually travelling. Feedback
        errors propagate to the caller, which commands zero torque and disables
        the bus before re-raising the original error.
        """
        state = self._read_feedback()[GRIPPER_MOTOR]
        now = time.monotonic()
        previous_time = self._gripper_prev_time
        dt = None if previous_time is None else now - previous_time
        self._gripper_prev_time = now

        prev_target = self._gripper_prev_target_pos
        target_vel = (
            0.0 if prev_target is None or dt is None or dt <= 0.0 else (target_pos_rad - prev_target) / dt
        )
        self._gripper_prev_target_pos = target_pos_rad

        prev_target_vel = self._gripper_prev_target_vel
        if prev_target_vel is not None:
            target_vel = _GRIPPER_LPF_ALPHA * target_vel + (1.0 - _GRIPPER_LPF_ALPHA) * prev_target_vel
        target_vel = max(-_GRIPPER_TARGET_VEL_MAX, min(_GRIPPER_TARGET_VEL_MAX, target_vel))
        self._gripper_prev_target_vel = target_vel

        prev_state_pos = self._gripper_prev_state_pos
        measured_vel = (
            0.0 if prev_state_pos is None or dt is None or dt <= 0.0 else (state.pos - prev_state_pos) / dt
        )
        self._gripper_prev_state_pos = state.pos

        torque = self.config.mit_kp[GRIPPER_MOTOR] * (target_pos_rad - state.pos) + self.config.mit_kd[
            GRIPPER_MOTOR
        ] * (target_vel - state.vel)

        limit = (
            self.config.gripper_torque_limit
            if abs(measured_vel) > _GRIPPER_HOLD_VEL_THRESHOLD
            else self.config.gripper_hold_torque_limit
        )
        return max(-limit, min(limit, torque))

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect from the robot.

        With `disable_torque_on_disconnect=True` (the default) the arm becomes
        back-drivable and falls under gravity: hold it or park it in a stable rest
        pose first.
        """
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
