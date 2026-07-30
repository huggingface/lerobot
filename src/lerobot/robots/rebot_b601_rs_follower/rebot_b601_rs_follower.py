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

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _motorbridge_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rebot_b601_rs_follower import RebotB601RSFollowerRobotConfig

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import Controller as MotorBridgeController, Mode as MotorBridgeMode
else:
    MotorBridgeController = None
    MotorBridgeMode = None

logger = logging.getLogger(__name__)

# The gripper is driven by a force-limited MIT impedance torque; every other
# joint is driven by a plain MIT position command. RobStride motors are
# MIT-mode only (no POS_VEL / FORCE_POS).
GRIPPER_MOTOR = "gripper"
# Per-joint RobStride motor models for the B601-RS (passed to motorbridge).
# The three base joints use the larger rs-06; wrists and gripper use the rs-00.
RS_MOTOR_MODELS = {
    "shoulder_pan": "rs-06",
    "shoulder_lift": "rs-06",
    "elbow_flex": "rs-06",
    "wrist_flex": "rs-00",
    "wrist_yaw": "rs-00",
    "wrist_roll": "rs-00",
    "gripper": "rs-00",
}
_ENSURE_MODE_RETRIES = 9
_SETTLE_SEC = 0.01
_ZERO_SETTLE_SEC = 0.1

# --- RS gripper MIT impedance-control parameters (force-limited grasping) ---
# Motor-side velocity damping in the MIT command (the position setpoint and kp
# are 0; the gripper is driven by the feedforward torque from _gripper_mit_torque).
_GRIPPER_MIT_DAMPING = 1.5
# Assumed control period (s) used to estimate target/state velocities.
_GRIPPER_CONTROL_DT_S = 0.02
# Low-pass filter factor for the target velocity estimate.
_GRIPPER_LPF_ALPHA = 0.3
# Cap (rad/s) on the filtered target velocity.
_GRIPPER_TARGET_VEL_MAX = 3.0
# Below this |estimated_state_vel| (rad/s) the gentler hold torque limit applies.
_GRIPPER_HOLD_VEL_THRESHOLD = 0.25


class RebotB601RSFollower(Robot):
    """Seeed Studio reBot B601-RS follower arm (6-DOF + gripper, RobStride RS CAN motors).

    RobStride motors are MIT-mode only. Motor communication is handled by the
    ``motorbridge`` package over a CAN bus, reached either through a SocketCAN
    adapter (the default) or a Damiao serial bridge.
    """

    config_class = RebotB601RSFollowerRobotConfig
    name = "rebot_b601_rs_follower"

    def __init__(self, config: RebotB601RSFollowerRobotConfig):
        require_package("motorbridge", extra="rebot")
        super().__init__(config)
        self.config = config
        self.bus: MotorBridgeController | None = None
        self.motors: dict = {}
        self.motor_names = list(config.motor_can_ids.keys())
        self.cameras = make_cameras_from_configs(config.cameras)
        # State carried across frames for the gripper impedance-torque estimate.
        self._gripper_prev_target_pos: float | None = None
        self._gripper_prev_filtered_target_vel: float | None = None
        self._gripper_prev_state_pos: float | None = None

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
        logger.info(f"Connecting {self} on {self.config.port} (adapter={self.config.can_adapter})...")
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

        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            self.motors[motor_name] = self.bus.add_robstride_motor(
                send_id, recv_id, RS_MOTOR_MODELS[motor_name]
            )

        # Fresh impedance-torque estimate each connection.
        self._gripper_prev_target_pos = None
        self._gripper_prev_filtered_target_vel = None
        self._gripper_prev_state_pos = None

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
        self.bus.disable_all()
        print(
            "\nCalibration: set zero position.\n"
            "Manually move the reBot B601-RS to its ZERO POSITION and close the gripper.\n"
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
        # RobStride motors are MIT-mode only: keep torque off while switching
        # mode, set every motor (incl. gripper) to MIT, then re-enable (matches
        # the Seeed RS reference: avoids jerk while motors change control mode).
        self.bus.disable_all()
        for motor_name, motor in self.motors.items():
            for attempt in range(_ENSURE_MODE_RETRIES + 1):
                try:
                    motor.ensure_mode(MotorBridgeMode.MIT)
                    break
                except Exception:
                    if attempt == _ENSURE_MODE_RETRIES:
                        raise
                    time.sleep(_SETTLE_SEC)
            logger.debug(f"{motor_name} mode set to MIT")
        self.bus.enable_all()

    @check_if_not_connected
    def disable_torque(self) -> None:
        """Disable motor torque so the arm can be moved by hand (read-only debugging)."""
        self.bus.disable_all()
        logger.info(f"{self} torque disabled.")

    def _gripper_mit_torque(self, motor: Any, pos_target_rad: float) -> float | None:
        """Impedance torque for the RS gripper (force-limited grasping).

        Computes ``tau = Kp*(x_r - x) + Kd*(x_dot_r - x_dot)`` from the target
        position and the motor's measured state, then clamps it to a moving or
        holding torque limit. The motor is then driven purely by this feedforward
        torque (plus a small damping term), which bounds grip force — unlike a
        plain position-MIT command that would push to the target regardless of
        force (and can overcurrent when closing on an object).

        ``Kp``/``Kd`` are ``gripper_mit_kp``/``gripper_mit_kd``; the limits are
        ``gripper_mit_torque_limit`` (moving) / ``gripper_mit_hold_torque_limit``
        (near-zero speed).
        """
        if motor is None:
            return None
        self.bus.poll_feedback_once()
        state = motor.get_state()
        if state is None:
            return None

        prev_target = self._gripper_prev_target_pos
        target_vel = (
            0.0 if prev_target is None else (pos_target_rad - prev_target) / _GRIPPER_CONTROL_DT_S
        )
        self._gripper_prev_target_pos = pos_target_rad

        prev_filt_vel = self._gripper_prev_filtered_target_vel
        filtered_vel = (
            target_vel
            if prev_filt_vel is None
            else _GRIPPER_LPF_ALPHA * target_vel + (1.0 - _GRIPPER_LPF_ALPHA) * prev_filt_vel
        )
        filtered_vel = max(-_GRIPPER_TARGET_VEL_MAX, min(_GRIPPER_TARGET_VEL_MAX, filtered_vel))
        self._gripper_prev_filtered_target_vel = filtered_vel

        prev_state_pos = self._gripper_prev_state_pos
        estimated_state_vel = (
            0.0 if prev_state_pos is None else (state.pos - prev_state_pos) / _GRIPPER_CONTROL_DT_S
        )
        self._gripper_prev_state_pos = state.pos

        kp = float(self.config.gripper_mit_kp)
        kd = float(self.config.gripper_mit_kd)
        impedance_torque = kp * (pos_target_rad - state.pos) + kd * (filtered_vel - state.vel)

        max_torque = (
            self.config.gripper_mit_torque_limit
            if abs(estimated_state_vel) > _GRIPPER_HOLD_VEL_THRESHOLD
            else self.config.gripper_mit_hold_torque_limit
        )
        motor.request_feedback()
        return max(-max_torque, min(max_torque, impedance_torque))

    def _present_pos(self) -> dict[str, float]:
        """Read present joint positions in degrees."""
        for motor in self.motors.values():
            motor.request_feedback()
        try:
            self.bus.poll_feedback_once()
        except Exception:
            logger.warning("CAN bus poll feedback failed.")

        present_pos = {}
        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            present_pos[motor_name] = math.degrees(state.pos) if state is not None else 0.0
        return present_pos

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        obs_dict = {f"{motor}.pos": pos for motor, pos in self._present_pos().items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
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

        # Map the shared leader's Damiao-convention action into the RobStride
        # motor's positive-physical range (RobStride motors are mounted opposite
        # to the Damiao variant), before clipping to the physical joint limits.
        # This also puts goal_pos in the same convention as `_present_pos()`, so
        # the `max_relative_target` comparison is sign-consistent.
        for motor_name in list(goal_pos):
            direction = self.config.joint_directions.get(motor_name, 1.0)
            goal_pos[motor_name] = goal_pos[motor_name] * direction

        # Clip against soft joint limits.
        for motor_name in list(goal_pos):
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped = max(min_limit, min(max_limit, goal_pos[motor_name]))
                if clipped != goal_pos[motor_name]:
                    logger.debug(f"Clipped {motor_name} from {goal_pos[motor_name]:.2f} to {clipped:.2f}")
                goal_pos[motor_name] = clipped

        # Tolerate 6-DOF leaders that have no wrist_yaw joint by holding it at zero.
        # This is intentional: it lets a 6-DOF leader such as the SO-100 / SO-101
        # (so100_leader / so101_leader) teleoperate this 7-DOF follower — the missing
        # wrist_yaw command is simply treated as 0.0 instead of raising.
        if "wrist_yaw" not in goal_pos:
            goal_pos["wrist_yaw"] = 0.0

        # Cap relative target when too far from the present position.
        if self.config.max_relative_target is not None:
            present_pos = self._present_pos()
            goal_present_pos = {key: (g, present_pos.get(key, g)) for key, g in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        for motor_name, position_deg in goal_pos.items():
            motor = self.motors.get(motor_name)
            if motor is None:
                continue
            idx = self.motor_names.index(motor_name)
            pos_rad = math.radians(position_deg)
            if motor_name == GRIPPER_MOTOR:
                # Force-limited impedance grasp: drive the gripper purely by a
                # clamped feedforward torque (motor kp=0, pos_des=0) so grip
                # force stays bounded instead of pushing to the target regardless
                # of force (which could overcurrent when closing on an object).
                tau_ff = self._gripper_mit_torque(motor, pos_rad)
                if tau_ff is None:
                    tau_ff = 0.0
                motor.send_mit(0.0, 0.0, 0.0, _GRIPPER_MIT_DAMPING, tau_ff)
            else:
                kp = self.config.mit_kp[idx] if isinstance(self.config.mit_kp, list) else self.config.mit_kp
                kd = self.config.mit_kd[idx] if isinstance(self.config.mit_kd, list) else self.config.mit_kd
                motor.send_mit(pos_rad, 0.0, kp, kd, 0.0)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
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

        logger.info(f"{self} disconnected.")
