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
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.import_utils import _motorbridge_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rebot_b601_follower import RebotB601FollowerRobotConfig
from .gravity import B601GravityFeedforward

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
        self._validate_home_config()
        self._in_safe_home = False
        self._safe_home_succeeded = False
        self._emergency_disable_requested = False
        self._startup_home_action: dict[str, float] = {}
        self._gravity_feedforward: B601GravityFeedforward | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    def _validate_home_config(self) -> None:
        arm_joints = [name for name in self.motor_names if name != GRIPPER_MOTOR]
        if self.config.home_action:
            missing = [name for name in arm_joints if name not in self.config.home_action]
            if missing:
                raise ValueError(
                    "home_action must specify every arm joint or be omitted to use the startup pose; "
                    f"missing {missing}."
                )
            for motor_name in arm_joints:
                position_deg = float(self.config.home_action[motor_name])
                if not math.isfinite(position_deg):
                    raise ValueError(f"home_action contains a non-finite target for '{motor_name}'")
                if motor_name in self.config.joint_limits:
                    min_limit, max_limit = self.config.joint_limits[motor_name]
                    if not min_limit <= position_deg <= max_limit:
                        raise ValueError(
                            f"home target for '{motor_name}' ({position_deg}) is outside "
                            f"joint limits [{min_limit}, {max_limit}]"
                        )
        if self.config.home_hz <= 0 or self.config.home_velocity_deg_s <= 0:
            raise ValueError("home_hz and home_velocity_deg_s must be positive")
        if self.config.home_duration_s < 0 or self.config.home_tolerance_deg < 0:
            raise ValueError("home_duration_s and home_tolerance_deg must be non-negative")

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

    def connect(self, calibrate: bool = True) -> None:
        if self.bus is not None:
            raise DeviceAlreadyConnectedError(f"{self.__class__.__name__} motor bus is already connected.")
        self._safe_home_succeeded = False
        self._emergency_disable_requested = False
        self._startup_home_action = {}
        logger.info(f"Connecting {self} on {self.config.port} (adapter={self.config.can_adapter})...")
        try:
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
                self.motors[motor_name] = self.bus.add_damiao_motor(
                    send_id, recv_id, MOTOR_MODELS[motor_name]
                )

            if not self.is_calibrated and calibrate:
                logger.info(
                    "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
                )
                self.calibrate()

            for cam in self.cameras.values():
                cam.connect()

            self.configure()
            self._capture_startup_home_action()
            logger.info(f"{self} connected.")
        except Exception:
            logger.exception("B601 connection failed; emergency-disabling and closing initialized resources.")
            if self.bus is not None:
                self.emergency_disable()
                try:
                    self.disconnect()
                except Exception:
                    logger.exception("Failed cleaning up B601 resources after connection failure.")
            raise

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

    def disable_torque(self) -> None:
        """Disable motor torque so the arm can be moved by hand (read-only debugging)."""
        if self.bus is None:
            raise DeviceNotConnectedError(f"{self.__class__.__name__} motor bus is not connected.")
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

    def _read_home_positions(self, motor_names: list[str]) -> dict[str, float]:
        """Read validated physical joint positions for the safe-home lifecycle."""
        for motor_name in motor_names:
            motor = self.motors.get(motor_name)
            if motor is None:
                raise RuntimeError(f"safe_home failed: motor '{motor_name}' not found")
            motor.request_feedback()
        self.bus.poll_feedback_once()

        positions: dict[str, float] = {}
        for motor_name in motor_names:
            state = self.motors[motor_name].get_state()
            if state is None or not math.isfinite(state.pos):
                raise RuntimeError(f"safe_home failed: invalid feedback for '{motor_name}'")
            positions[motor_name] = math.degrees(state.pos)
        return positions

    def _read_debug_motor_states(self) -> dict[str, dict[str, float | int | None]]:
        """Read the full motorbridge MotorState for B601-DM diagnostics."""
        for motor in self.motors.values():
            motor.request_feedback()
        self.bus.poll_feedback_once()
        states: dict[str, dict[str, float | int | None]] = {}
        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            if state is None:
                states[motor_name] = {"state": None}
                continue
            states[motor_name] = {
                "can_id": state.can_id,
                "arbitration_id": state.arbitration_id,
                "status_code": state.status_code,
                "pos_deg": math.degrees(state.pos),
                "vel_deg_s": math.degrees(state.vel),
                "torq": state.torq,
                "t_mos": state.t_mos,
                "t_rotor": state.t_rotor,
            }
        return states

    def _log_safe_home_debug_command(
        self,
        label: str,
        goal_positions: dict[str, float],
        started_at: float,
        arm_velocity_deg_s: float | None = None,
    ) -> None:
        if not self.config.safe_home_debug:
            return
        commands = {}
        use_mit = self.config.control_mode == "mit"
        for motor_name, position_deg in goal_positions.items():
            send_id, recv_id = self.config.motor_can_ids[motor_name]
            idx = self.motor_names.index(motor_name)
            commands[motor_name] = {
                "can_id": send_id,
                "recv_id": recv_id,
                "target_deg": position_deg,
                "mode": (
                    "MIT"
                    if motor_name != GRIPPER_MOTOR and use_mit
                    else "FORCE_POS"
                    if motor_name == GRIPPER_MOTOR and self.config.gripper_control_mode == "force_pos"
                    else "MIT"
                    if motor_name == GRIPPER_MOTOR
                    else "POS_VEL"
                ),
                "kp": self.config.mit_kp[idx] if isinstance(self.config.mit_kp, list) else self.config.mit_kp,
                "kd": self.config.mit_kd[idx] if isinstance(self.config.mit_kd, list) else self.config.mit_kd,
                "vel_limit_deg_s": arm_velocity_deg_s,
                "tau_ff": 0.0,
            }
        logger.info(
            "B601 safe-home debug command: port=%s elapsed_s=%.3f label=%s "
            "control_mode=%s gripper_control_mode=%s commands=%s",
            self.config.port,
            time.monotonic() - started_at,
            label,
            self.config.control_mode,
            self.config.gripper_control_mode,
            commands,
        )

    def _log_safe_home_debug_feedback(
        self,
        label: str,
        target_positions: dict[str, float],
        started_at: float,
    ) -> None:
        """Log the physical response around safe-home without changing control mode."""
        if not self.config.safe_home_debug:
            return
        feedback = self._read_home_positions(list(target_positions))
        motor_states = self._read_debug_motor_states()
        feedback_minus_target = {name: feedback[name] - target for name, target in target_positions.items()}
        logger.info(
            "B601 safe-home debug: port=%s elapsed_s=%.3f label=%s "
            "target_deg=%s feedback_deg=%s feedback_minus_target_deg=%s",
            self.config.port,
            time.monotonic() - started_at,
            label,
            target_positions,
            feedback,
            feedback_minus_target,
        )
        logger.info(
            "B601 safe-home debug state: port=%s elapsed_s=%.3f label=%s motor_state=%s",
            self.config.port,
            time.monotonic() - started_at,
            label,
            motor_states,
        )
        trace = getattr(self, "_safe_home_debug_trace", None)
        if trace is not None:
            trace_started_at = getattr(self, "_safe_home_debug_trace_started_at", started_at)
            trace.append((time.monotonic() - trace_started_at, label, motor_states))

    def _arm_joint_names(self) -> list[str]:
        return [name for name in self.motor_names if name != GRIPPER_MOTOR]

    def _capture_startup_home_action(self) -> None:
        arm_joints = self._arm_joint_names()
        missing = [name for name in arm_joints if name not in self.config.home_action]
        if not missing:
            logger.info("Using configured safe-home action.")
            return
        if self.config.home_action:
            raise ValueError(
                "home_action must specify every arm joint or be omitted to use the startup pose; "
                f"missing {missing}."
            )
        if not self.config.home_from_start_position:
            logger.info("Startup home capture is disabled; safe_home requires home_action.")
            return

        self._startup_home_action = self._read_home_positions(arm_joints)
        logger.info("Captured startup pose as this session's safe-home action: %s", self._startup_home_action)

    def _safe_home_target(self, arm_joints: list[str]) -> dict[str, float]:
        if all(name in self.config.home_action for name in arm_joints):
            target = {name: float(self.config.home_action[name]) for name in arm_joints}
        elif all(name in self._startup_home_action for name in arm_joints):
            target = {name: float(self._startup_home_action[name]) for name in arm_joints}
        else:
            missing = [name for name in arm_joints if name not in self._startup_home_action]
            raise RuntimeError(
                "safe_home requires an explicit home_action or a startup pose snapshot; "
                f"missing {missing}. Torque remains enabled."
            )

        for motor_name, position_deg in target.items():
            if not math.isfinite(position_deg):
                raise ValueError(f"home_action contains a non-finite target for '{motor_name}'")
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                if not min_limit <= position_deg <= max_limit:
                    raise ValueError(
                        f"home target for '{motor_name}' ({position_deg}) is outside "
                        f"joint limits [{min_limit}, {max_limit}]"
                    )
        return target

    def _send_goal_positions(
        self, goal_pos: dict[str, float], *, arm_velocity_deg_s: float | None = None
    ) -> None:
        """Send physical joint targets that have already been clipped and validated."""
        use_mit = self.config.control_mode == "mit"
        gravity_torque: dict[str, float] = {}
        has_arm_goal = any(motor_name != GRIPPER_MOTOR for motor_name in goal_pos)
        if use_mit and has_arm_goal:
            if self._gravity_feedforward is None:
                self._gravity_feedforward = B601GravityFeedforward()
                logger.info(
                    "Using B601-DM gravity feedforward for MIT control from %s",
                    self._gravity_feedforward.urdf_path,
                )
            gravity_torque = self._gravity_feedforward.torque(
                self._read_home_positions(self._arm_joint_names())
            )
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
                motor.send_mit(pos_rad, 0.0, kp, kd, gravity_torque.get(motor_name, 0.0))
            else:
                vel_deg_s = arm_velocity_deg_s or (
                    self.config.pos_vel_velocity[idx]
                    if isinstance(self.config.pos_vel_velocity, list)
                    else self.config.pos_vel_velocity
                )
                motor.send_pos_vel(pos_rad, math.radians(vel_deg_s))

    def _hold_arm_positions_during_gripper_motion(
        self,
        arm_positions: dict[str, float],
        gripper_position_deg: float,
        duration_s: float,
        *,
        debug_started_at: float | None = None,
        debug_label: str = "gripper-motion",
    ) -> None:
        """Refresh arm hold targets while the gripper moves during safe-home."""
        deadline = time.monotonic() + max(duration_s, 0.0)
        interval_s = 1.0 / self.config.home_hz
        hold_positions = {**arm_positions, GRIPPER_MOTOR: gripper_position_deg}
        next_debug_feedback_at = 0.0
        while True:
            self._send_goal_positions(hold_positions)
            now = time.monotonic()
            if debug_started_at is not None and now >= next_debug_feedback_at:
                self._log_safe_home_debug_feedback(debug_label, hold_positions, debug_started_at)
                next_debug_feedback_at = now + 0.1
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return
            time.sleep(min(interval_s, remaining_s))

    def _safe_home_stages(self, arm_joints: list[str]) -> list[tuple[list[str], list[str]]]:
        """Match Seeed's DM safe-zero order while retaining native joint units."""
        can_id_to_joint = {
            send_id: motor_name
            for motor_name, (send_id, _recv_id) in self.config.motor_can_ids.items()
            if motor_name in arm_joints
        }
        stage_1 = [can_id_to_joint[can_id] for can_id in (1, 4, 5, 6) if can_id in can_id_to_joint]
        stage_2 = [can_id_to_joint[can_id] for can_id in (2, 3) if can_id in can_id_to_joint]
        assigned = set(stage_1) | set(stage_2)
        # Preserve a safe trajectory even for custom CAN maps by moving any
        # nonstandard arm joint in the second stage.
        stage_2.extend(name for name in arm_joints if name not in assigned)
        return [(stage_1, stage_2), (stage_2, stage_1)]

    def _safe_home_stage_frames(
        self,
        current: dict[str, float],
        home: dict[str, float],
        active_joints: list[str],
        total_delta_deg: float,
    ) -> int:
        max_delta_deg = max((abs(home[name] - current[name]) for name in active_joints), default=0.0)
        velocity_frames = math.ceil(max_delta_deg / self.config.home_velocity_deg_s * self.config.home_hz)
        duration_share = max_delta_deg / total_delta_deg if total_delta_deg > 0 else 0.5
        duration_frames = math.ceil(self.config.home_duration_s * self.config.home_hz * duration_share)
        return max(1, velocity_frames, duration_frames)

    def safe_home(self) -> bool:
        """Return to the configured or startup pose before torque is disabled."""
        if self.bus is None:
            raise RuntimeError(f"{self} has no connected motor bus")
        if self._safe_home_succeeded:
            return True
        if self._in_safe_home:
            logger.warning("safe_home skipped: already running.")
            return False
        if self.config.home_hz <= 0 or self.config.home_velocity_deg_s <= 0:
            raise ValueError("home_hz and home_velocity_deg_s must be positive")
        if self.config.home_duration_s < 0 or self.config.home_tolerance_deg < 0:
            raise ValueError("home_duration_s and home_tolerance_deg must be non-negative")

        self._in_safe_home = True
        try:
            safe_home_started_at = time.monotonic()
            arm_joints = self._arm_joint_names()
            home = self._safe_home_target(arm_joints)
            current = self._read_home_positions(arm_joints)
            # Begin exactly at the latest measured follower pose. Reusing an
            # old policy target here can make the arm briefly chase that stale
            # target before its controlled return trajectory begins.
            trajectory_start = dict(current)
            trajectory_start_source = "measured-feedback"
            deltas = {name: home[name] - current[name] for name in arm_joints}
            logger.info(
                "B601 safe-home start: current_deg=%s trajectory_start_deg=%s "
                "trajectory_start_source=%s home_deg=%s delta_deg=%s",
                current,
                trajectory_start,
                trajectory_start_source,
                home,
                deltas,
            )
            if self.config.safe_home_debug:
                logger.info(
                    "B601 safe-home debug config: port=%s control_mode=%s "
                    "gripper_control_mode=%s home_hz=%.3f home_velocity_deg_s=%.3f "
                    "home_duration_s=%.3f open_gripper=%s keep_gripper_open=%s "
                    "close_gripper=%s mit_kp=%s mit_kd=%s motor_can_ids=%s",
                    self.config.port,
                    self.config.control_mode,
                    self.config.gripper_control_mode,
                    self.config.home_hz,
                    self.config.home_velocity_deg_s,
                    self.config.home_duration_s,
                    self.config.open_gripper_before_home,
                    self.config.keep_gripper_open_during_home,
                    self.config.close_gripper_after_home,
                    self.config.mit_kp,
                    self.config.mit_kd,
                    self.config.motor_can_ids,
                )
            self._log_safe_home_debug_feedback("entry", current, safe_home_started_at)
            # The normal control loop has stopped by the time shutdown begins.
            # Refresh MIT/POS hold targets before any gripper dwell so gravity
            # cannot move the loaded arm between the last policy command and
            # the first return-trajectory command.
            # Holding the measured target avoids a transient jump toward an
            # older policy command before the return motion begins.
            self._send_goal_positions(trajectory_start)
            self._log_safe_home_debug_command("arm-hold-command-sent", trajectory_start, safe_home_started_at)
            self._log_safe_home_debug_feedback(
                "arm-hold-command-sent", trajectory_start, safe_home_started_at
            )
            if self.config.open_gripper_before_home:
                self._hold_arm_positions_during_gripper_motion(
                    trajectory_start,
                    self.config.gripper_open_position_deg,
                    self.config.gripper_open_duration_s,
                    debug_started_at=safe_home_started_at,
                    debug_label="gripper-preopen-hold",
                )

            stages = self._safe_home_stages(arm_joints)
            total_delta_deg = sum(
                max(
                    (abs(home[name] - trajectory_start[name]) for name in active_joints),
                    default=0.0,
                )
                for active_joints, _hold_joints in stages
            )
            interval_s = 1.0 / self.config.home_hz
            debug_frame_stride = max(1, math.ceil(0.2 / interval_s))
            for stage_index, (active_joints, hold_joints) in enumerate(stages, start=1):
                if not active_joints:
                    continue
                frames = self._safe_home_stage_frames(trajectory_start, home, active_joints, total_delta_deg)
                logger.info(
                    "B601 safe-home stage %d/%d: move=%s hold=%s frames=%d",
                    stage_index,
                    len(stages),
                    active_joints,
                    hold_joints,
                    frames,
                )
                for frame in range(1, frames + 1):
                    ratio = frame / frames
                    hold_positions = trajectory_start if stage_index == 1 else home
                    positions = {name: hold_positions[name] for name in hold_joints}
                    positions.update(
                        {
                            name: trajectory_start[name] + (home[name] - trajectory_start[name]) * ratio
                            for name in active_joints
                        }
                    )
                    if self.config.open_gripper_before_home and self.config.keep_gripper_open_during_home:
                        positions[GRIPPER_MOTOR] = self.config.gripper_open_position_deg
                    if stage_index == 1 and frame == 1:
                        logger.info("B601 safe-home first return target_deg=%s", positions)
                    self._send_goal_positions(positions, arm_velocity_deg_s=self.config.home_velocity_deg_s)
                    if frame == 1 or frame == frames or frame % debug_frame_stride == 0:
                        self._log_safe_home_debug_feedback(
                            f"stage-{stage_index}-frame-{frame}-of-{frames}",
                            positions,
                            safe_home_started_at,
                        )
                        self._log_safe_home_debug_command(
                            f"stage-{stage_index}-frame-{frame}-of-{frames}",
                            positions,
                            safe_home_started_at,
                            arm_velocity_deg_s=self.config.home_velocity_deg_s,
                        )
                    time.sleep(interval_s)

            final_positions = self._read_home_positions(arm_joints)
            errors = {name: abs(final_positions[name] - home[name]) for name in arm_joints}
            if any(error > self.config.home_tolerance_deg for error in errors.values()):
                raise RuntimeError(
                    f"safe_home final position error exceeds {self.config.home_tolerance_deg} deg: {errors}. "
                    "Torque remains enabled."
                )

            if self.config.close_gripper_after_home:
                self._hold_arm_positions_during_gripper_motion(
                    home,
                    self.config.gripper_closed_position_deg,
                    self.config.gripper_close_duration_s,
                    debug_started_at=safe_home_started_at,
                    debug_label="gripper-close-hold",
                )
            self._safe_home_succeeded = True
            logger.info("safe_home completed successfully.")
            return True
        except Exception:
            logger.exception("safe_home failed; torque intentionally remains enabled for manual recovery.")
            return False
        finally:
            self._in_safe_home = False

    def emergency_disable(self) -> None:
        """Disable torque immediately without attempting a return trajectory."""
        self._emergency_disable_requested = True
        if self.bus is not None:
            self.bus.disable_all()
            logger.warning(f"{self} torque emergency-disabled.")

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
        if not self._in_safe_home:
            self._safe_home_succeeded = False

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

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

        self._send_goal_positions(goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self) -> None:
        # Camera loss must not prevent the live motor bus from returning home.
        if self.bus is None:
            raise RuntimeError(f"{self} has no connected motor bus")
        if (
            self.config.home_on_disconnect
            and not self._in_safe_home
            and not self._emergency_disable_requested
            and not self.safe_home()
        ):
            raise RuntimeError(
                "Refusing to disconnect B601 after failed safe_home; "
                "torque and motor bus remain enabled for manual recovery."
            )

        for motor in self.motors.values():
            if self.config.disable_torque_on_disconnect:
                motor.disable()
            motor.clear_error()
            motor.close()

        self.bus.close()
        self.bus = None
        self.motors = {}

        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

        self._emergency_disable_requested = False
        logger.info(f"{self} disconnected.")
