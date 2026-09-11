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
from typing import TYPE_CHECKING

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorCalibration
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _motorbridge_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rebot_b601_follower import RebotB601FollowerRobotConfig

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import Controller as MotorBridgeController, Mode as MotorBridgeMode
else:
    MotorBridgeController = None
    MotorBridgeMode = None

logger = logging.getLogger(__name__)

# Joint controlled in FORCE_POS mode; every other joint runs in POS_VEL mode.
GRIPPER_MOTOR = "gripper"
# Per-joint Damiao motor models for the B601-DM (passed to motorbridge).
MOTOR_MODELS = {
    "shoulder_pan": "4340P",
    "shoulder_lift": "4340P",
    "elbow_flex": "4340P",
    "wrist_flex": "4310",
    "wrist_yaw": "4310",
    "wrist_roll": "4310",
    "gripper": "4310",
}
_ENSURE_MODE_RETRIES = 9
_SETTLE_SEC = 0.01
_ZERO_SETTLE_SEC = 0.1


class RebotB601Follower(Robot):
    """Seeed Studio reBot B601-DM follower arm (6-DOF + gripper, Damiao CAN motors).

    Motor communication is handled by the ``motorbridge`` package over a CAN bus,
    reached either through a Damiao serial bridge or a SocketCAN adapter.
    """

    config_class = RebotB601FollowerRobotConfig
    name = "rebot_b601_follower"

    def __init__(self, config: RebotB601FollowerRobotConfig):
        require_package("motorbridge", extra="rebot")
        super().__init__(config)
        self.config = config
        self.bus: MotorBridgeController | None = None
        self.motors: dict = {}
        self.motor_names = list(config.motor_can_ids.keys())
        self.cameras = make_cameras_from_configs(config.cameras)
        # Last complete arm target (degrees) actually sent by `send_action`, used as the
        # command-space start point of the safe-home trajectory.
        self._last_arm_target: dict[str, float] = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def _arm_motor_names(self) -> list[str]:
        return [motor for motor in self.motor_names if motor != GRIPPER_MOTOR]

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
        self._last_arm_target = {}
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
            self.motors[motor_name] = self.bus.add_damiao_motor(send_id, recv_id, MOTOR_MODELS[motor_name])

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
        if self.config.control_mode not in ("pos_vel", "mit"):
            raise ValueError(
                f"Unsupported control_mode '{self.config.control_mode}'. Use 'pos_vel' or 'mit'."
            )
        if self.config.gripper_control_mode not in ("force_pos", "mit"):
            raise ValueError(
                f"Unsupported gripper_control_mode '{self.config.gripper_control_mode}'. "
                "Use 'force_pos' or 'mit'."
            )
        if self.config.safe_home_rate_hz <= 0:
            raise ValueError(f"safe_home_rate_hz must be > 0, got {self.config.safe_home_rate_hz}.")
        if self.config.safe_home_duration_s < 0:
            raise ValueError(f"safe_home_duration_s must be >= 0, got {self.config.safe_home_duration_s}.")
        use_mit = self.config.control_mode == "mit"
        gripper_use_mit = self.config.gripper_control_mode == "mit"
        self.bus.enable_all()
        for motor_name, motor in self.motors.items():
            if motor_name == GRIPPER_MOTOR:
                target_mode = MotorBridgeMode.MIT if gripper_use_mit else MotorBridgeMode.FORCE_POS
            elif use_mit:
                target_mode = MotorBridgeMode.MIT
            else:
                target_mode = MotorBridgeMode.POS_VEL
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
        """Disable motor torque so the arm can be moved by hand (read-only debugging)."""
        self.bus.disable_all()
        logger.info(f"{self} torque disabled.")

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
        goal_pos = self._clip_to_joint_limits(goal_pos)

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

        self._send_goal_pos(goal_pos)
        # A motor keeps its last commanded target until a new one is sent, so remember
        # the joints just commanded without forgetting the ones this action left out.
        self._last_arm_target.update(
            {
                motor_name: goal_pos[motor_name]
                for motor_name in self._arm_motor_names
                if motor_name in goal_pos
            }
        )

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def _clip_to_joint_limits(self, goal_pos: dict[str, float]) -> dict[str, float]:
        """Clip a goal position mapping (degrees) against the configured soft limits."""
        clipped_pos = dict(goal_pos)
        for motor_name, position_deg in goal_pos.items():
            if motor_name not in self.config.joint_limits:
                continue
            min_limit, max_limit = self.config.joint_limits[motor_name]
            clipped = max(min_limit, min(max_limit, position_deg))
            if clipped != position_deg:
                logger.debug(f"Clipped {motor_name} from {position_deg:.2f} to {clipped:.2f}")
            clipped_pos[motor_name] = clipped
        return clipped_pos

    def _send_goal_pos(self, goal_pos: dict[str, float]) -> None:
        """Send a goal position mapping (degrees) to the motors in their control mode."""
        use_mit = self.config.control_mode == "mit"
        for motor_name, position_deg in goal_pos.items():
            motor = self.motors.get(motor_name)
            if motor is None:
                continue
            idx = self.motor_names.index(motor_name)
            pos_rad = math.radians(position_deg)
            if motor_name == GRIPPER_MOTOR:
                if self.config.gripper_control_mode == "mit":
                    motor.send_mit(pos_rad, 0.0, self.config.gripper_mit_kp, self.config.gripper_mit_kd, 0.0)
                else:
                    vel_deg_s = (
                        self.config.pos_vel_velocity[idx]
                        if isinstance(self.config.pos_vel_velocity, list)
                        else self.config.pos_vel_velocity
                    )
                    motor.send_force_pos(pos_rad, math.radians(vel_deg_s), self.config.gripper_torque_ratio)
            elif use_mit:
                kp = self.config.mit_kp[idx] if isinstance(self.config.mit_kp, list) else self.config.mit_kp
                kd = self.config.mit_kd[idx] if isinstance(self.config.mit_kd, list) else self.config.mit_kd
                motor.send_mit(pos_rad, 0.0, kp, kd, 0.0)
            else:
                vel_deg_s = (
                    self.config.pos_vel_velocity[idx]
                    if isinstance(self.config.pos_vel_velocity, list)
                    else self.config.pos_vel_velocity
                )
                motor.send_pos_vel(pos_rad, math.radians(vel_deg_s))

    def _safe_home(self) -> None:
        """Interpolate the arm back to `safe_home_target` before torque is released.

        In MIT mode the position loop holds the arm up through the error between the
        commanded target and the measured position, so the trajectory starts from the
        last commanded arm target rather than from feedback: restarting from feedback
        would zero that supporting error and let the arm dip under gravity. Feedback is
        read for the log lines, and is only used as the trajectory start when no complete
        previous target is available.
        """
        arm_names = self._arm_motor_names
        feedback = self._present_pos()
        if all(motor_name in self._last_arm_target for motor_name in arm_names):
            start = {motor_name: self._last_arm_target[motor_name] for motor_name in arm_names}
        else:
            logger.info("No complete arm target recorded yet, starting safe-home from feedback.")
            start = {motor_name: feedback.get(motor_name, 0.0) for motor_name in arm_names}
        logger.info(f"{self} entering safe-home from {start} (feedback {feedback}).")

        period_s = 1.0 / self.config.safe_home_rate_hz
        hold = dict(start)
        if self.config.safe_home_gripper_pos is not None:
            hold[GRIPPER_MOTOR] = self.config.safe_home_gripper_pos
        hold = self._clip_to_joint_limits(hold)

        # Hold the command-space start point while the gripper pre-open dwell runs, so
        # the arm command stays continuous across the control-to-homing boundary.
        self._send_goal_pos(hold)
        if self.config.safe_home_gripper_pos is not None:
            dwell_end = time.perf_counter() + self.config.safe_home_gripper_dwell_s
            while time.perf_counter() < dwell_end:
                time.sleep(period_s)
                self._send_goal_pos(hold)

        target = {
            motor_name: self.config.safe_home_target.get(motor_name, start[motor_name])
            for motor_name in arm_names
        }
        steps = max(round(self.config.safe_home_duration_s * self.config.safe_home_rate_hz), 1)
        for step in range(1, steps + 1):
            ratio = step / steps
            waypoint = {
                motor_name: start[motor_name] + ratio * (target[motor_name] - start[motor_name])
                for motor_name in arm_names
            }
            time.sleep(period_s)
            self._send_goal_pos(self._clip_to_joint_limits(waypoint))

        self._last_arm_target = dict(target)
        logger.info(f"{self} safe-home done, final feedback {self._present_pos()}.")

    @check_if_not_connected
    def disconnect(self) -> None:
        homed = True
        try:
            if self.config.safe_home_on_disconnect:
                self._safe_home()
        except Exception:
            homed = False
            logger.exception("Safe-home failed, keeping torque enabled for manual recovery.")
        except BaseException:
            # A second Ctrl+C during homing must still release the bus and the cameras.
            homed = False
            raise
        finally:
            for motor in self.motors.values():
                if self.config.disable_torque_on_disconnect and homed:
                    motor.disable()
                motor.clear_error()
                motor.close()

            self.bus.close()
            self.bus = None
            self.motors = {}
            self._last_arm_target = {}

            for cam in self.cameras.values():
                cam.disconnect()

            logger.info(f"{self} disconnected.")
