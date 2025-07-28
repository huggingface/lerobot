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

import collections
import logging
import time
from functools import cached_property
from typing import Any

import numpy as np
import pinocchio as pin
from scipy.signal import butter, lfilter

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
)

from ..robot import Robot
from .config_so101_follower_t import SO101FollowerTConfig

logger = logging.getLogger(__name__)


class SO101FollowerT(Robot):
    """
    SO-101 Arm with HLS3625 motors with current control.
    """

    config_class = SO101FollowerTConfig
    name = "so101_follower_t"

    _CURRENT_STEP_A: float = 6.5e-3  # 6.5 mA per register LSB #http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SMS-STS-emanual-229f4476422d4059abfb1cb0
    _KT_NM_PER_AMP: float = 0.814  # Torque constant Kt [N·m/A] #https://www.feetechrc.com/811177.html
    _MAX_CURRENT_A: float = 4.0  # Safe driver limit

    # Position gains [Nm/rad]
    _KP_GAINS = {
        "shoulder_pan": 5.0,
        "shoulder_lift": 7.0,
        "elbow_flex": 7.0,
        "wrist_flex": 5.0,
        "wrist_roll": 5.0,
        "gripper": 5.0,
    }

    # Velocity gains [Nm⋅s/rad]
    _KD_GAINS = {
        "shoulder_pan": 0.4,
        "shoulder_lift": 0.6,
        "elbow_flex": 0.6,
        "wrist_flex": 0.4,
        "wrist_roll": 0.4,
        "gripper": 0.4,
    }

    # Force gains
    _KF_GAINS = {
        "shoulder_pan": 0.05,
        "shoulder_lift": 0.05,
        "elbow_flex": 0.05,
        "wrist_flex": 0.05,
        "wrist_roll": 0.05,
        "gripper": 0.05,
    }

    # Viscous friction coefficient [Nm⋅s/rad] per joint
    _FRICTION_VISCOUS = {
        "shoulder_pan": 0.05,
        "shoulder_lift": 0.08,
        "elbow_flex": 0.05,
        "wrist_flex": 0.05,
        "wrist_roll": 0.05,
        "gripper": 0.05,
    }

    # Coulomb/static friction [Nm] per joint
    _FRICTION_COULOMB = {
        "shoulder_pan": 0.15,
        "shoulder_lift": 0.25,
        "elbow_flex": 0.25,
        "wrist_flex": 0.20,
        "wrist_roll": 0.20,
        "gripper": 0.20,
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

        self.pin_robot = pin.RobotWrapper.BuildFromURDF("urdf/so101_new_calib.urdf", "urdf")

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

        # Butterworth low-pass filter parameters
        self._cutoff_freq = 10.0  # Hz, cutoff frequency for the filter
        self._filter_order = 2  # Filter order
        self._sampling_freq = 100.0  # Hz, (control loop frequency)

        nyquist_freq = self._sampling_freq / 2
        normalized_cutoff = self._cutoff_freq / nyquist_freq
        self._b, self._a = butter(self._filter_order, normalized_cutoff, btype="low")

        # History buffers
        self._pos_history = {m: collections.deque(maxlen=20) for m in self.bus.motors}
        self._vel_raw_history = {m: collections.deque(maxlen=20) for m in self.bus.motors}
        self._time_history = collections.deque(maxlen=20)

        self._last_observation = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        d: dict[str, type] = {}
        for motor in self.bus.motors:
            d[f"{motor}.pos"] = float
            d[f"{motor}.vel"] = float
            d[f"{motor}.effort"] = float
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
        d: dict[str, type] = {}
        for motor in self.bus.motors:
            d[f"{motor}.pos"] = float
            d[f"{motor}.vel"] = float
            d[f"{motor}.effort"] = float
        return d

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
    def kf_gains(self) -> dict[str, float]:
        """Force control gains for bilateral teleoperation"""
        return self._KF_GAINS.copy()

    @property
    def friction_viscous(self) -> dict[str, float]:
        """Viscous friction coefficients [Nm⋅s/rad] for friction compensation"""
        return self._FRICTION_VISCOUS.copy()

    @property
    def friction_coulomb(self) -> dict[str, float]:
        """Coulomb friction coefficients [Nm] for friction compensation"""
        return self._FRICTION_COULOMB.copy()

    def set_butterworth_params(self, cutoff_freq: float = 10.0, order: int = 2) -> None:
        """Configure Butterworth low-pass filter parameters for velocity/acceleration estimation.

        Args:
            cutoff_freq: Cutoff frequency in Hz (default: 10 Hz)
            order: Filter order (default: 2)
        """
        if cutoff_freq <= 0:
            raise ValueError("Cutoff frequency must be positive")
        if cutoff_freq >= self._sampling_freq / 2:
            raise ValueError(
                f"Cutoff frequency must be less than Nyquist frequency ({self._sampling_freq / 2} Hz)"
            )
        if order < 1:
            raise ValueError("Filter order must be at least 1")

        self._cutoff_freq = cutoff_freq
        self._filter_order = order

        nyquist_freq = self._sampling_freq / 2
        normalized_cutoff = self._cutoff_freq / nyquist_freq
        self._b, self._a = butter(self._filter_order, normalized_cutoff, btype="low")

        # Clear buffers
        for m in self.bus.motors:
            self._pos_history[m].clear()
            self._vel_raw_history[m].clear()
        self._time_history.clear()

        logger.info(f"Butterworth filter updated: cutoff_freq={cutoff_freq} Hz, order={order}")

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
    ) -> dict[str, float]:
        """
        Compute disturbance torques using direct model-based approach:
        τ_disturbance = τ_measured - τ_gravity - τ_inertia - τ_friction

        Args:
            include_friction: If True, also removes friction from the disturbance calculation
        """
        tau_gravity = self._gravity_from_q(q_rad)
        tau_inertia = self._inertia_from_q_dq(q_rad, dq_rad, ddq_rad)

        # Compute disturbance: what's left after removing known dynamics
        tau_disturbance = {}
        tau_friction = {}
        for motor_name in self.bus.motors:
            tau_dist = tau_measured[motor_name] - tau_gravity[motor_name] - tau_inertia[motor_name]

            # Calculate friction torque
            omega = dq_rad[motor_name]
            tau_friction_motor = self._FRICTION_VISCOUS[motor_name] * omega + self._FRICTION_COULOMB[
                motor_name
            ] * (1.0 if omega > 0.01 else -1.0 if omega < -0.01 else 0.0)
            # Apply torque sign correction
            tau_friction_motor = -tau_friction_motor
            tau_friction[motor_name] = tau_friction_motor
            tau_dist -= tau_friction_motor

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
            self.bus.write("Operating_Mode", motor, 2, num_retry=2)  # Set to current mode

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
        self.bus.disable_torque()  # here was issue at startup previously
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, 2, num_retry=2)  # Set to current mode
            self.bus.write("Present_Current", motor, 0, normalize=False, num_retry=5)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        t_now = time.perf_counter()

        # Position
        pos_deg = self.bus.sync_read("Present_Position", num_retry=5)
        pos_rad = self._deg_to_rad(pos_deg)

        # Store position and time history
        for m in pos_rad:
            self._pos_history[m].append(pos_rad[m])
        self._time_history.append(t_now)

        # Calculate raw velocity using finite differences
        vel_rad_raw = {}
        if self._prev_pos_rad is None or self._prev_t is None:
            vel_rad_raw = dict.fromkeys(pos_rad, 0.0)
        else:
            dt = t_now - self._prev_t
            dt = max(dt, 1e-4)  # Avoid division by zero
            vel_rad_raw = {m: (pos_rad[m] - self._prev_pos_rad[m]) / dt for m in pos_rad}

        # Store raw velocity history
        for m in vel_rad_raw:
            self._vel_raw_history[m].append(vel_rad_raw[m])

        # Apply Butterworth low-pass filter to velocity
        vel_rad = {}
        for m in pos_rad:
            if len(self._vel_raw_history[m]) >= 10:
                vel_raw_array = np.array(list(self._vel_raw_history[m]))

                # Apply Butterworth filter
                vel_filtered = lfilter(self._b, self._a, vel_raw_array)
                vel_rad[m] = vel_filtered[-1]
            else:
                vel_rad[m] = vel_rad_raw[m]

        # Calculate acceleration from filtered velocity
        acc_rad = {}
        if self._prev_vel_rad is None or self._prev_t is None:
            acc_rad = dict.fromkeys(pos_rad, 0.0)
        else:
            dt = t_now - self._prev_t
            dt = max(dt, 1e-4)  # Avoid division by zero
            acc_rad = {m: (vel_rad[m] - self._prev_vel_rad[m]) / dt for m in vel_rad}

        self._prev_pos_rad = pos_rad.copy()
        self._prev_vel_rad = vel_rad.copy()
        self._prev_t = t_now

        # Measured torque (Nm)
        cur_raw = self.bus.sync_read("Present_Current", normalize=False, num_retry=5)
        tau_meas = self._current_to_torque_nm(cur_raw)

        # Compute reaction torques using model-based approach
        tau_reaction = self._compute_model_based_disturbance(pos_rad, vel_rad, acc_rad, tau_meas)

        obs_dict = {}
        obs_dict |= {f"{m}.pos": pos_rad[m] for m in self.bus.motors}
        obs_dict |= {f"{m}.vel": vel_rad[m] for m in self.bus.motors}
        obs_dict |= {f"{m}.acc": acc_rad[m] for m in self.bus.motors}
        obs_dict |= {f"{m}.effort": tau_reaction[m] for m in self.bus.motors}

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        # Store observation for feedforward compensation
        self._last_observation = obs_dict.copy()

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

        # Add feedforward compensation if we have a last observation
        if self._last_observation is not None:
            # Extract position, velocity, acceleration from last observation
            pos_rad = {m: self._last_observation[f"{m}.pos"] for m in self.bus.motors}
            vel_rad = {m: self._last_observation[f"{m}.vel"] for m in self.bus.motors}
            acc_rad = {m: self._last_observation[f"{m}.acc"] for m in self.bus.motors}

            # Compute feedforward terms
            tau_gravity = self._gravity_from_q(pos_rad)
            tau_inertia = self._inertia_from_q_dq(pos_rad, vel_rad, acc_rad)

            # Add feedforward compensation to commanded torques
            for motor in tau_cmd_nm:
                # Add gravity compensation
                tau_cmd_nm[motor] += tau_gravity[motor]

                # Add inertia compensation
                tau_cmd_nm[motor] += tau_inertia[motor]

                # Add friction compensation
                omega = vel_rad[motor]
                tau_friction = self._FRICTION_VISCOUS[motor] * omega + self._FRICTION_COULOMB[motor] * (
                    1.0 if omega > 0.01 else -1.0 if omega < -0.01 else 0.0
                )
                tau_friction = -tau_friction  # Apply torque sign correction
                tau_cmd_nm[motor] += tau_friction

        inv_coef = 1.0 / (self._CURRENT_STEP_A * self._KT_NM_PER_AMP)
        max_cnt = int(round(self._MAX_CURRENT_A / self._CURRENT_STEP_A))
        counts = {}
        for joint, τ in tau_cmd_nm.items():
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
