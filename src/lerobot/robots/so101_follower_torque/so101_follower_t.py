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

import numpy as np
import pinocchio as pin

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from .config_so101_follower_t import SO101FollowerTConfig

logger = logging.getLogger(__name__)


class SO101FollowerT(Robot):
    """
    SO-101 Follower Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101FollowerTConfig
    name = "so101_follower_t"

    _CURRENT_STEP_A: float = 6.5e-3  # 6.5 mA per register LSB #http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SMS-STS-emanual-229f4476422d4059abfb1cb0
    _KT_NM_PER_AMP: float = 0.814  # Torque constant Kt [N·m/A] #https://www.feetechrc.com/811177.html
    _MAX_CURRENT_A: float = 3.0  # Safe driver limit for this model

    # Control gains for bilateral teleoperation
    _KP_GAINS = {  # Position gains [Nm/rad] - reduced for bilateral stability
        "shoulder_pan": 5.0,  # Reduced from 7.0
        "shoulder_lift": 15.0,  # Reduced from 15.0
        "elbow_flex": 12.0,  # Reduced from 12.0 (main problem joint)
        "wrist_flex": 4.0,  # Reduced from 6.0
        "wrist_roll": 3.0,  # Reduced from 4.0
        "gripper": 1.5,  # Reduced from 2.0
    }

    _KD_GAINS = {  # Velocity gains [Nm⋅s/rad] - matched to position gains
        "shoulder_pan": 0.4,
        "shoulder_lift": 0.7,
        "elbow_flex": 0.6,
        "wrist_flex": 0.4,
        "wrist_roll": 0.3,
        "gripper": 0.2,
    }

    # Friction model parameters
    _FRICTION_VISCOUS = {  # Viscous friction coefficient [Nm⋅s/rad] per joint
        "shoulder_pan": 0.02,
        "shoulder_lift": 0.08,
        "elbow_flex": 0.05,
        "wrist_flex": 0.02,
        "wrist_roll": 0.015,
        "gripper": 0.01,
    }

    _FRICTION_COULOMB = {  # Coulomb friction [Nm] per joint
        "shoulder_pan": 0.05,
        "shoulder_lift": 0.30,
        "elbow_flex": 0.20,
        "wrist_flex": 0.10,
        "wrist_roll": 0.06,
        "gripper": 0.04,
    }

    def __init__(self, config: SO101FollowerTConfig):
        super().__init__(config)
        self.config = config

        # Ensure calibration is loaded before creating the bus
        if self.calibration_fpath.is_file() and not self.calibration:
            self._load_calibration()

        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "hls3625", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "hls3625", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "hls3625", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "hls3625", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "hls3625", MotorNormMode.DEGREES),
                "gripper": Motor(6, "hls3625", MotorNormMode.DEGREES),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        self.pin_robot = pin.RobotWrapper.BuildFromURDF(
            "src/lerobot/SO101/so101_new_calib.urdf", "src/lerobot/SO101"
        )

        flip = {
            "shoulder_pan": True,
            "shoulder_lift": True,
            "elbow_flex": True,
            "wrist_flex": True,
            "wrist_roll": True,
            "gripper": True,
        }
        self.torque_sign = {n: (-1.0 if flip[n] else 1.0) for n in self.bus.motors}

        self._prev_pos_rad: dict[str, float] | None = None
        self._prev_vel_rad: dict[str, float] | None = None
        self._prev_t: float | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        d: dict[str, type] = {}
        for m in self.bus.motors:
            d[f"{m}.pos"] = float
            d[f"{m}.vel"] = float
            d[f"{m}.acc"] = float  # Add acceleration
            d[f"{m}.tau_meas"] = float  # Changed from tau_res to tau_meas
        return d

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.effort": int for m in self.bus.motors}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def kp_gains(self) -> dict[str, float]:
        """Position control gains [Nm/rad] for bilateral teleoperation"""
        return self._KP_GAINS.copy()

    @property
    def kd_gains(self) -> dict[str, float]:
        """Velocity control gains [Nm⋅s/rad] for bilateral teleoperation"""
        return self._KD_GAINS.copy()

    @property
    def friction_viscous(self) -> dict[str, float]:
        """Viscous friction coefficients [Nm⋅s/rad] for friction compensation"""
        return self._FRICTION_VISCOUS.copy()

    @property
    def friction_coulomb(self) -> dict[str, float]:
        """Coulomb friction coefficients [Nm] for friction compensation"""
        return self._FRICTION_COULOMB.copy()

    def _current_to_torque_nm(self, raw: dict[str, Any]) -> dict[str, float]:
        """Convert "Present_Current" register counts (±2047) → torque [Nm].
        Values are clamped to ±3A before conversion for protection.
        """
        max_cnt = int(round(self._MAX_CURRENT_A / self._CURRENT_STEP_A))  # ≈ 462
        coef = self._CURRENT_STEP_A * self._KT_NM_PER_AMP
        return {k: self.torque_sign[k] * max(-max_cnt, min(max_cnt, v)) * coef for k, v in raw.items()}

    def _torque_nm_to_current(self, torque: dict[str, float]) -> dict[str, int]:
        """Convert torque [Nm] to register counts, clamped to ±3A (2.44 Nm)."""
        inv_coef = 1.0 / (self._CURRENT_STEP_A * self._KT_NM_PER_AMP)
        max_cnt = int(round(self._MAX_CURRENT_A / self._CURRENT_STEP_A))
        counts = {}
        for k, τ in torque.items():
            cnt = τ * self.torque_sign[k] * inv_coef
            cnt = max(-max_cnt, min(max_cnt, cnt))
            counts[k] = int(round(cnt))
        return counts

    def _deg_to_rad(self, deg: dict[str, float | int]) -> dict[str, float]:
        """Degrees to radians."""
        return {m: np.deg2rad(float(v)) for m, v in deg.items()}

    def _gravity_from_q(self, q_rad: dict[str, float]) -> dict[str, float]:
        """
        Compute g(q) [N m] for all joints in the robot.
        The order of joints in the URDF matches self.bus.motors.
        """
        q = np.zeros(self.pin_robot.model.nq)
        for i, motor_name in enumerate(self.bus.motors):
            q[i] = q_rad[motor_name]

        g = pin.computeGeneralizedGravity(self.pin_robot.model, self.pin_robot.data, q)

        return {motor_name: float(g[i]) for i, motor_name in enumerate(self.bus.motors)}

    def _inertia_from_q_dq(
        self, q_rad: dict[str, float], dq_rad: dict[str, float], ddq_rad: dict[str, float]
    ) -> dict[str, float]:
        """
        Compute inertia torques τ_inertia = M(q) * ddq directly from URDF model.
        """
        # Convert joint dictionaries to numpy arrays in correct order
        q = np.zeros(self.pin_robot.model.nq)
        dq = np.zeros(self.pin_robot.model.nv)
        ddq = np.zeros(self.pin_robot.model.nv)

        for i, motor_name in enumerate(self.bus.motors):
            q[i] = q_rad[motor_name]
            dq[i] = dq_rad[motor_name]
            ddq[i] = ddq_rad[motor_name]

        # Compute mass matrix M(q)
        mass_matrix = pin.crba(self.pin_robot.model, self.pin_robot.data, q)

        # Compute inertia torques: τ_inertia = M(q) * ddq
        tau_inertia = mass_matrix @ ddq

        return {motor_name: float(tau_inertia[i]) for i, motor_name in enumerate(self.bus.motors)}

    def _compute_model_based_disturbance(
        self,
        q_rad: dict[str, float],
        dq_rad: dict[str, float],
        ddq_rad: dict[str, float],
        tau_measured: dict[str, float],
        include_friction: bool = True,
    ) -> dict[str, float]:
        """
        Compute disturbance torques using direct model-based approach:
        τ_disturbance = τ_measured - τ_gravity - τ_inertia - τ_friction

        Args:
            include_friction: If True, also removes friction from the disturbance calculation
        """
        # Get gravity compensation
        tau_gravity = self._gravity_from_q(q_rad)

        # Get inertia compensation
        tau_inertia = self._inertia_from_q_dq(q_rad, dq_rad, ddq_rad)

        # Compute disturbance: what's left after removing known dynamics
        tau_disturbance = {}
        for motor_name in self.bus.motors:
            tau_dist = tau_measured[motor_name] - tau_gravity[motor_name] - tau_inertia[motor_name]

            # Optionally remove friction model
            if include_friction:
                # Calculate friction torque using class constants
                omega = dq_rad[motor_name]
                tau_friction = self._FRICTION_VISCOUS[motor_name] * omega + self._FRICTION_COULOMB[
                    motor_name
                ] * (1.0 if omega > 0.01 else -1.0 if omega < -0.01 else 0.0)

                # Apply torque sign correction (same as for gravity/inertia)
                tau_friction = -tau_friction  # Apply torque sign correction

                tau_dist -= tau_friction

            tau_disturbance[motor_name] = tau_dist

        return tau_disturbance

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Ensure calibration is loaded from file if it exists
        if self.calibration_fpath.is_file() and not self.calibration:
            self._load_calibration()
            # Update the bus with the loaded calibration
            self.bus.calibration = self.calibration

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # Check if calibration file exists and is loaded
        return self.calibration_fpath.is_file() and bool(self.calibration)

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=int(homing_offsets[motor]),
                range_min=int(range_mins[motor]),
                range_max=int(range_maxes[motor]),
            )

        # Update the bus calibration with the new values
        self.bus.calibration = self.calibration
        # Save calibration to file only
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, 2, num_retry=2)  # Set to current mode
                self.bus.write("Torque_Limit", motor, 1000, num_retry=2)  # 100%
                self.bus.write("Max_Torque_Limit", motor, 1000, num_retry=2)  # 100%

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        t_now = time.perf_counter()

        # position
        pos_deg = self.bus.sync_read("Present_Position", num_retry=5)
        pos_rad = self._deg_to_rad(pos_deg)

        # velocity - calculate from position differences (HLS3625 motors in torque mode don't respond to Present_Velocity sync_read)
        if self._prev_pos_rad is None or self._prev_t is None:
            vel_rad = dict.fromkeys(pos_rad, 0.0)
        else:
            dt = t_now - self._prev_t
            dt = max(dt, 1e-4)  # Avoid division by zero
            vel_rad = {m: (pos_rad[m] - self._prev_pos_rad[m]) / dt for m in pos_rad}

        # acceleration - calculate from velocity differences
        if self._prev_vel_rad is None or self._prev_t is None:
            acc_rad = dict.fromkeys(pos_rad, 0.0)
        else:
            dt = t_now - self._prev_t
            dt = max(dt, 1e-4)  # Avoid division by zero
            acc_rad = {m: (vel_rad[m] - self._prev_vel_rad[m]) / dt for m in vel_rad}

        # Update previous values
        self._prev_pos_rad = pos_rad.copy()
        self._prev_vel_rad = vel_rad.copy()
        self._prev_t = t_now

        # measured torque (Nm)
        cur_raw = self.bus.sync_read("Present_Current", normalize=False, num_retry=5)
        tau_meas = self._current_to_torque_nm(cur_raw)

        obs_dict = {}
        obs_dict |= {f"{m}.pos": pos_rad[m] for m in self.bus.motors}
        obs_dict |= {f"{m}.vel": vel_rad[m] for m in self.bus.motors}
        obs_dict |= {f"{m}.acc": acc_rad[m] for m in self.bus.motors}
        obs_dict |= {f"{m}.tau_meas": tau_meas[m] for m in self.bus.motors}

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target torque for a joint.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract torque commands
        tau_cmd_nm = {k.removesuffix(".effort"): float(v) for k, v in action.items() if k.endswith(".effort")}
        if not tau_cmd_nm:
            return action

        inv_coef = 1.0 / (self._CURRENT_STEP_A * self._KT_NM_PER_AMP)
        max_cnt = int(round(self._MAX_CURRENT_A / self._CURRENT_STEP_A))
        counts = {}
        for joint, τ in tau_cmd_nm.items():
            # Set wrist_flex commands to 0
            if joint == "wrist_flex":
                counts[joint] = 0
                continue
            cnt = τ * self.torque_sign[joint] * inv_coef  # flip SIGN
            cnt = max(-max_cnt, min(max_cnt, cnt))
            counts[joint] = int(round(cnt))

        self.bus.sync_write("Target_Torque", counts, normalize=False, num_retry=2)
        self._last_cmd_nm = tau_cmd_nm
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
