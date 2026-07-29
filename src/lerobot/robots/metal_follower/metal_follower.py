#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_metal_follower import MetalFollowerConfig

logger = logging.getLogger(__name__)

# Damiao motor variant per joint. Metal's motors are flashed with MIT parameter ranges tailored
# to each joint (see `motors/damiao/tables.py`), so this is fixed by the hardware build.
MOTOR_MODELS = {
    "shoulder_pan": "metal_jlo",
    "shoulder_lift": "metal_j2",
    "elbow_flex": "metal_jlo",
    "wrist_flex": "metal_jlo",
    "wrist_yaw": "metal_jhi",
    "wrist_roll": "metal_jhi",
    "gripper": "metal_jhi",
}

# Startup-sync stall release: a joint counts as making progress when it moved more than
# _SYNC_PROGRESS_EPS_DEG since its last progress mark; after _SYNC_STALL_RELEASE_SEC without
# progress it is released from the slow sync (it physically cannot close the gap at the capped
# step size, e.g. parked past a soft limit or stiction above kp * step).
_SYNC_PROGRESS_EPS_DEG = 0.2
_SYNC_STALL_RELEASE_SEC = 1.0


class MetalFollower(Robot):
    """
    Metal arm follower robot: 6 joints + a permanent gripper, all driven as Damiao motors over
    classic CAN (`use_can_fd=False`) via the stock `DamiaoMotorsBus` in MIT position control.

    All 7 motors are normalized in degrees. The gripper is passed through in raw motor degrees,
    same as the joints (no stroke-to-angle conversion here; that's deferred to a later task).
    """

    config_class = MetalFollowerConfig
    name = "metal_follower"

    def __init__(self, config: MetalFollowerConfig):
        super().__init__(config)
        self.config = config

        if config.can_interface != "socketcan":
            raise ValueError(
                f"metal_follower only supports can_interface='socketcan', got "
                f"'{config.can_interface}'. Bring the USB-CAN adapter up as a socketcan "
                "interface first, e.g. `sudo slcand -o -c -s8 -t hw -S 921600 /dev/ttyACM0 "
                "can0 && sudo ip link set up can0`, then set port='can0'."
            )

        # Build all 7 motors: the 6 joints plus the permanent gripper.
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id) in config.motor_can_ids.items():
            motor_type_str = MOTOR_MODELS[motor_name]
            motor = Motor(send_id, motor_type_str, MotorNormMode.DEGREES)
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self._joint_motor_names = list(motors)

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        # False until the follower has caught up to the leader (slow initial sync), then full speed.
        self._synced = False
        self._synced_motors: set[str] = set()
        self._sync_progress: dict[str, tuple[float, float]] = {}
        self._resolved_gains: dict[str, tuple[float, float]] = {}
        self._reset_velocity_feedforward()

    def _reset_velocity_feedforward(self) -> None:
        self._velocity_ff_previous_goal: dict[str, float] | None = None
        self._velocity_ff_velocities: dict[str, float] = {}
        self._velocity_ff_last_action_time: float | None = None

    def _estimate_command_velocities(self, goal_pos: dict[str, float]) -> dict[str, float]:
        now = time.perf_counter()
        if self._velocity_ff_last_action_time is None or now - self._velocity_ff_last_action_time > 0.5:
            self._reset_velocity_feedforward()

        if self._velocity_ff_previous_goal is None:
            velocities = dict.fromkeys(goal_pos, 0.0)
        else:
            elapsed = now - self._velocity_ff_last_action_time
            dt = max(0.005, min(0.1, elapsed))
            velocities = {}
            for motor, position in goal_pos.items():
                previous_position = self._velocity_ff_previous_goal.get(motor)
                if previous_position is None:
                    velocities[motor] = 0.0
                    continue
                raw_velocity = (position - previous_position) / dt
                max_velocity = self.config.velocity_ff_max_deg_s
                raw_velocity = max(-max_velocity, min(max_velocity, raw_velocity))
                previous_velocity = self._velocity_ff_velocities.get(motor, 0.0)
                velocities[motor] = (
                    1.0 - self.config.velocity_ff_alpha
                ) * previous_velocity + self.config.velocity_ff_alpha * raw_velocity

        self._velocity_ff_previous_goal = dict(goal_pos)
        self._velocity_ff_velocities = velocities
        self._velocity_ff_last_action_time = now
        return dict(velocities)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self._joint_motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam_key, cam in self.cameras.items():
            features[cam_key] = (cam.height, cam.width, 3)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        for cam in self.cameras.values():
            cam.connect()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.bus.enable_torque()
        # Re-arm the slow initial sync on every connect.
        self._synced = False
        self._synced_motors: set[str] = set()
        self._sync_progress: dict[str, tuple[float, float]] = {}
        self._reset_velocity_feedforward()

        # Set firm follow gains (the bus default kp=10 is far too soft to hold the arm
        # against gravity, so the follower sags). Entries for motors this arm does not have
        # are dropped rather than written to a missing id.
        self._resolved_gains = {
            m: (kp, kd) for m, (kp, kd) in self.config.gains.items() if m in self._joint_motor_names
        }
        self.bus.sync_write("Kp", {m: kp for m, (kp, kd) in self._resolved_gains.items()})
        self.bus.sync_write("Kd", {m: kd for m, (kp, kd) in self._resolved_gains.items()})

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # Metal arm uses Damiao absolute encoders — calibration here means the user has
        # confirmed the zero pose once and a calibration file exists on disk.
        return bool(self.calibration)

    def calibrate(self) -> None:
        """Interactive zero-pose calibration for the metal follower arm.

        The Damiao motors use absolute encoders so no range-of-motion recording is needed.
        This procedure just confirms the arm is in its zero position and sets that reference
        on each motor, matching the reBot B601 calibration philosophy.
        """
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        print(
            "\nCalibration: set zero position.\n"
            "Manually move the metal arm to its ZERO POSE and close the gripper.\n"
            "The zero pose is: arm standing upright, all joints at 0 degrees.\n"
        )
        input("Press ENTER when ready...")

        self.bus.set_zero_position()
        logger.info("Arm zero position set.")

        self.calibration = {}
        for motor_name, (send_id, _recv_id) in self.config.motor_can_ids.items():
            range_min, range_max = self.config.joint_limits.get(motor_name, (-360.0, 360.0))
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
        """No-op: follow gains are set in connect() from config.gains."""
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()

        obs_dict: dict[str, Any] = {}

        positions = self.bus.sync_read("Present_Position")
        for motor in self._joint_motor_names:
            obs_dict[f"{motor}.pos"] = positions[motor]

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Clamp every motor, gripper included, to its vendor soft limit.
        for motor_name, position in goal_pos.items():
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f}° to {clipped_position:.2f}°")
                goal_pos[motor_name] = clipped_position

        # The startup sync, the relative-target cap and the torque feedforward all need the
        # arm's present state. Read it at most once per tick and share it between them:
        syncing = self.config.startup_sync_speed_deg is not None and not self._synced
        present_pos: dict[str, float] = {}
        if syncing or self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")

        # Slow initial sync: at teleop start cap each joint's per-step motion until the follower
        # has caught up to the leader, so firm follow gains don't snap it across a large gap.
        # Sync is per joint: a joint that cannot converge (parked past a soft limit, or stiction
        # above what kp * step can overcome) is released after a stall instead of capping the
        # whole arm forever; max_relative_target still bounds its speed after release.
        if syncing:
            step = self.config.startup_sync_speed_deg
            now = time.perf_counter()
            for motor_name, position in goal_pos.items():
                if motor_name in self._synced_motors:
                    continue
                present = present_pos[motor_name]
                err = position - present
                if abs(err) <= self.config.startup_sync_tolerance_deg:
                    self._synced_motors.add(motor_name)
                    continue
                last_pos, last_progress_t = self._sync_progress.get(motor_name, (present, now))
                if abs(present - last_pos) > _SYNC_PROGRESS_EPS_DEG:
                    self._sync_progress[motor_name] = (present, now)
                elif now - last_progress_t > _SYNC_STALL_RELEASE_SEC:
                    logger.warning(
                        f"{self} startup sync stalled on {motor_name} ({abs(err):.1f} deg from goal); "
                        "releasing it to full speed."
                    )
                    self._synced_motors.add(motor_name)
                    continue
                else:
                    self._sync_progress.setdefault(motor_name, (present, now))
                goal_pos[motor_name] = present + max(-step, min(step, err))
            if len(self._synced_motors) >= len(goal_pos):
                self._synced = True
                logger.info(f"{self} synced to leader; tracking at full speed.")

        # Cap goal position when too far away from present position.
        if self.config.max_relative_target is not None:
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        if self.config.velocity_feedforward:
            # sync_write("Goal_Position") pins the MIT velocity field to 0, so the
            # feedforward has to go out through the explicit MIT write. Gains ride along on
            # every frame, read from the resolved store rather than from the bus, matching how
            # the reBot B601 follower passes kp/kd per frame; mutating `_resolved_gains`
            # retunes the arm live.
            velocities = self._estimate_command_velocities(goal_pos)
            commands = {
                motor: (*self._resolved_gains[motor], position, velocities[motor], 0.0)
                for motor, position in goal_pos.items()
            }
            self.bus.sync_write_metal(commands)
        else:
            self.bus.sync_write("Goal_Position", goal_pos)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
