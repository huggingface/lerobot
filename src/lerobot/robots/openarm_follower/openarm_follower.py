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
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import MotorType
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_openarms_follower import OpenArmsFollowerConfig

logger = logging.getLogger(__name__)


class OpenArmsFollower(Robot):
    """
    OpenArms Follower Robot which uses CAN bus communication to control 7 DOF arm with a gripper.
    The arm uses Damiao motors in MIT control mode.
    """

    config_class = OpenArmsFollowerConfig
    name = "openarms_follower"

    def __init__(self, config: OpenArmsFollowerConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES  # Always use degrees for Damiao motors

        # Right arm motors (on port_right)
        # Each arm uses the same CAN IDs since they're on separate buses
        motors_right = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(send_id, motor_type_str, norm_mode_body)
            motor.recv_id = recv_id
            motor.motor_type = getattr(MotorType, motor_type_str.upper().replace("-", "_"))
            motors_right[motor_name] = motor

        # Left arm motors (on port_left, same IDs as right since separate bus)
        motors_left = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(send_id, motor_type_str, norm_mode_body)
            motor.recv_id = recv_id
            motor.motor_type = getattr(MotorType, motor_type_str.upper().replace("-", "_"))
            motors_left[motor_name] = motor

        # Initialize separate Damiao motors buses (one per arm) with CAN FD support
        self.bus_right = DamiaoMotorsBus(
            port=self.config.port_right,
            motors=motors_right,
            calibration={
                k.replace("right_", ""): v
                for k, v in (self.calibration or {}).items()
                if k.startswith("right_")
            },
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        self.bus_left = DamiaoMotorsBus(
            port=self.config.port_left,
            motors=motors_left,
            calibration={
                k.replace("left_", ""): v
                for k, v in (self.calibration or {}).items()
                if k.startswith("left_")
            },
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        # Cache for last valid camera frames (to avoid blocking on slow USB reads)
        self.camera_frame_cache = dict.fromkeys(self.cameras.keys())

        # Initialize Pinocchio robot model for dynamics (optional)
        self.pin_robot = None
        try:
            # Load URDF - try external path first (with meshes), then repository
            import os
            from os.path import dirname, expanduser

            # Try external URDF with meshes first
            external_urdf_path = expanduser("~/Documents/openarm_description/openarm_bimanual_pybullet.urdf")
            if os.path.exists(external_urdf_path):
                urdf_path = external_urdf_path
                urdf_dir = dirname(urdf_path)

                self.pin_robot = pin.RobotWrapper.BuildFromURDF(urdf_path, urdf_dir)
                self.pin_robot.data = self.pin_robot.model.createData()
                logger.info(f"Loaded OpenArms URDF for dynamics computation from {urdf_path}")
        except Exception as e:
            logger.warning(
                f"Could not load URDF for dynamics: {e}. Gravity compensation will not be available."
            )

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features = {}
        # Right arm motors - only positions stored in dataset
        for motor in self.bus_right.motors:
            features[f"right_{motor}.pos"] = float
        # Left arm motors - only positions stored in dataset
        for motor in self.bus_left.motors:
            features[f"left_{motor}.pos"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features (motor positions only)."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return (
            self.bus_right.is_connected
            and self.bus_left.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot and optionally calibrate.

        We assume that at connection time, the arms are in a safe rest position,
        and torque can be safely disabled to run calibration if needed.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to both CAN buses
        logger.info(f"Connecting right arm on {self.config.port_right}...")
        self.bus_right.connect()
        logger.info(f"Connecting left arm on {self.config.port_left}...")
        self.bus_left.connect()

        # Run calibration if needed
        if calibrate:
            logger.info("No calibration found or overwriting calibration. Running calibration...")
            self.calibrate()

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        # Configure motors
        self.configure()

        # Optionally set zero position
        if self.config.zero_position_on_connect:
            logger.info("Setting current position as zero...")
            self.bus_right.set_zero_position()
            self.bus_left.set_zero_position()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self.bus_right.is_calibrated and self.bus_left.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms robot.

        The calibration procedure:
        1. Disable torque
        2. Ask user to position arms in hanging position with grippers closed
        3. Set this as zero position
        4. Record range of motion for each joint
        5. Save calibration
        """
        if self.calibration:
            # Ask user whether to use existing calibration
            user_input = input(
                f"Press ENTER to use existing calibration for {self.id}, "
                f"or type 'c' and press ENTER to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration for {self.id}")
                # Split calibration for each bus
                cal_right = {
                    k.replace("right_", ""): v for k, v in self.calibration.items() if k.startswith("right_")
                }
                cal_left = {
                    k.replace("left_", ""): v for k, v in self.calibration.items() if k.startswith("left_")
                }
                self.bus_right.write_calibration(cal_right)
                self.bus_left.write_calibration(cal_left)
                return

        logger.info(f"\nRunning calibration for {self}")

        # Calibrate each arm separately
        self._calibrate_arm("right", self.bus_right)
        self._calibrate_arm("left", self.bus_left)

        print(f"\nCalibration complete and saved to {self.calibration_fpath}")

    def _calibrate_arm(self, arm_name: str, bus: DamiaoMotorsBus) -> None:
        """Calibrate a single arm."""
        logger.info(f"\n=== Calibrating {arm_name.upper()} arm ===")

        # Disable torque for manual positioning
        bus.disable_torque()
        time.sleep(0.1)

        # Step 1: Set zero position
        input(
            f"\nCalibration: Zero Position ({arm_name.upper()} arm)\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )

        # Set current position as zero for all motors
        bus.set_zero_position()
        logger.info(f"{arm_name.capitalize()} arm zero position set.")

        # Automatically set range to -90° to +90° for all joints
        print("\nAutomatically setting range: -90° to +90° for all joints")

        # Create calibration data with fixed ranges
        if self.calibration is None:
            self.calibration = {}

        for motor_name, motor in bus.motors.items():
            # Prefix motor name with arm name for storage
            prefixed_name = f"{arm_name}_{motor_name}"

            # Use -90 to +90 for all joints and gripper (integers required)
            self.calibration[prefixed_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,  # Normal direction
                homing_offset=0,  # Already set via set_zero_position
                range_min=-90,  # -90 degrees (integer)
                range_max=90,  # +90 degrees (integer)
            )
            logger.info(f"  {prefixed_name}: range set to [-90°, +90°]")

        # Write calibration to this arm's motors
        cal_for_bus = {
            k.replace(f"{arm_name}_", ""): v
            for k, v in self.calibration.items()
            if k.startswith(f"{arm_name}_")
        }
        bus.write_calibration(cal_for_bus)

        # Re-enable torque
        bus.enable_torque()

        # Save calibration after each arm
        self._save_calibration()

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        # Configure right arm
        with self.bus_right.torque_disabled():
            self.bus_right.configure_motors()

        # Configure left arm
        with self.bus_left.torque_disabled():
            self.bus_left.configure_motors()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    def get_observation(self) -> dict[str, Any]:
        """
        Get current observation from robot including position, velocity, and torque.

        OPTIMIZED: Reads all motor states (pos/vel/torque) in one CAN refresh cycle
        instead of 3 separate reads.

        Note: Velocity and torque are read but not stored in dataset (only used for
        internal calculations). Only positions and camera images are stored.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Detailed profiling for bottleneck analysis
        timings = {}

        # OPTIMIZED: Use sync_read_all_states to get pos/vel/torque in one go
        t0 = time.perf_counter()
        right_states = self.bus_right.sync_read_all_states()
        timings["right_motors"] = (time.perf_counter() - t0) * 1000

        for motor in self.bus_right.motors:
            state = right_states.get(motor, {})
            obs_dict[f"right_{motor}.pos"] = state.get("position", 0.0)
            obs_dict[f"right_{motor}.vel"] = state.get("velocity", 0.0)
            obs_dict[f"right_{motor}.torque"] = state.get("torque", 0.0)

        # OPTIMIZED: Use sync_read_all_states to get pos/vel/torque in one go
        t0 = time.perf_counter()
        left_states = self.bus_left.sync_read_all_states()
        timings["left_motors"] = (time.perf_counter() - t0) * 1000

        for motor in self.bus_left.motors:
            state = left_states.get(motor, {})
            obs_dict[f"left_{motor}.pos"] = state.get("position", 0.0)
            obs_dict[f"left_{motor}.vel"] = state.get("velocity", 0.0)
            obs_dict[f"left_{motor}.torque"] = state.get("torque", 0.0)

        # Capture images from cameras (with individual timing)
        # Use async_read with very short timeout to avoid blocking on slow USB cameras
        for cam_key, cam in self.cameras.items():
            t0 = time.perf_counter()
            try:
                # Use 5ms timeout - if frame isn't ready, reuse last frame
                frame = cam.async_read(timeout_ms=5)
                self.camera_frame_cache[cam_key] = frame  # Update cache
                obs_dict[cam_key] = frame
            except TimeoutError:
                # If no new frame available, reuse last valid frame from cache
                # This prevents blocking the entire control loop on slow USB reads
                if self.camera_frame_cache[cam_key] is not None:
                    obs_dict[cam_key] = self.camera_frame_cache[cam_key]
                    logger.debug(f"Camera {cam_key} timeout, reusing cached frame")

            # Store timing with padded name to align output (e.g. "left_wrist    ")
            timings[f"{cam_key:14s}"] = (time.perf_counter() - t0) * 1000

        # Log detailed timings (for debugging slow observations)
        if logger.isEnabledFor(logging.DEBUG):
            total_time = sum(timings.values())
            breakdown = " | ".join([f"{k}: {v:.1f}ms" for k, v in timings.items()])
            logger.debug(f"{self} get_observation: {total_time:.1f}ms total | {breakdown}")

        # Store timings in obs_dict for external profiling
        obs_dict["_timing_breakdown"] = timings

        return obs_dict

    def send_action(
        self,
        action: dict[str, Any],
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Send action command to robot.

        The action magnitude may be clipped based on safety limits.

        Args:
            action: Dictionary with motor positions (e.g., "right_joint_1.pos", "left_joint_2.pos")
            custom_kp: Optional custom kp gains per motor (e.g., {"right_joint_1": 120.0, "left_joint_2": 150.0})
            custom_kd: Optional custom kd gains per motor (e.g., {"right_joint_1": 1.5, "left_joint_2": 2.0})

        Returns:
            The action actually sent (potentially clipped)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract motor positions from action and split by arm
        goal_pos_right = {}
        goal_pos_left = {}

        for key, val in action.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                if motor_name.startswith("right_"):
                    # Remove "right_" prefix for bus access
                    goal_pos_right[motor_name.removeprefix("right_")] = val
                elif motor_name.startswith("left_"):
                    # Remove "left_" prefix for bus access
                    goal_pos_left[motor_name.removeprefix("left_")] = val

        # Apply joint limit clipping to right arm
        for motor_name, position in goal_pos_right.items():
            if motor_name in self.config.joint_limits_right:
                min_limit, max_limit = self.config.joint_limits_right[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(
                        f"Clipped right_{motor_name} from {position:.2f}° to {clipped_position:.2f}°"
                    )
                goal_pos_right[motor_name] = clipped_position

        # Apply joint limit clipping to left arm
        for motor_name, position in goal_pos_left.items():
            if motor_name in self.config.joint_limits_left:
                min_limit, max_limit = self.config.joint_limits_left[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped left_{motor_name} from {position:.2f}° to {clipped_position:.2f}°")
                goal_pos_left[motor_name] = clipped_position

        # Apply safety limits if configured
        if self.config.max_relative_target is not None:
            # Get current positions
            present_pos_right = self.bus_right.sync_read("Present_Position")
            present_pos_left = self.bus_left.sync_read("Present_Position")

            # Apply safety limits to right arm
            if goal_pos_right:
                goal_present_pos_right = {
                    key: (g_pos, present_pos_right.get(key, 0.0)) for key, g_pos in goal_pos_right.items()
                }
                goal_pos_right = ensure_safe_goal_position(
                    goal_present_pos_right, self.config.max_relative_target
                )

            # Apply safety limits to left arm
            if goal_pos_left:
                goal_present_pos_left = {
                    key: (g_pos, present_pos_left.get(key, 0.0)) for key, g_pos in goal_pos_left.items()
                }
                goal_pos_left = ensure_safe_goal_position(
                    goal_present_pos_left, self.config.max_relative_target
                )

        # Motor name to index mapping for gains
        motor_index = {
            "joint_1": 0,
            "joint_2": 1,
            "joint_3": 2,
            "joint_4": 3,
            "joint_5": 4,
            "joint_6": 5,
            "joint_7": 6,
            "gripper": 7,
        }

        # Use batch MIT control for right arm (sends all commands, then collects responses)
        if goal_pos_right:
            commands_right = {}
            for motor_name, position_degrees in goal_pos_right.items():
                idx = motor_index.get(motor_name, 0)

                # Use custom gains if provided, otherwise use config defaults
                full_motor_name = f"right_{motor_name}"
                if custom_kp is not None and full_motor_name in custom_kp:
                    kp = custom_kp[full_motor_name]
                else:
                    kp = (
                        self.config.position_kp[idx]
                        if isinstance(self.config.position_kp, list)
                        else self.config.position_kp
                    )

                if custom_kd is not None and full_motor_name in custom_kd:
                    kd = custom_kd[full_motor_name]
                else:
                    kd = (
                        self.config.position_kd[idx]
                        if isinstance(self.config.position_kd, list)
                        else self.config.position_kd
                    )

                commands_right[motor_name] = (kp, kd, position_degrees, 0.0, 0.0)
            self.bus_right._mit_control_batch(commands_right)

        # Use batch MIT control for left arm (sends all commands, then collects responses)
        if goal_pos_left:
            commands_left = {}
            for motor_name, position_degrees in goal_pos_left.items():
                idx = motor_index.get(motor_name, 0)

                # Use custom gains if provided, otherwise use config defaults
                full_motor_name = f"left_{motor_name}"
                if custom_kp is not None and full_motor_name in custom_kp:
                    kp = custom_kp[full_motor_name]
                else:
                    kp = (
                        self.config.position_kp[idx]
                        if isinstance(self.config.position_kp, list)
                        else self.config.position_kp
                    )

                if custom_kd is not None and full_motor_name in custom_kd:
                    kd = custom_kd[full_motor_name]
                else:
                    kd = (
                        self.config.position_kd[idx]
                        if isinstance(self.config.position_kd, list)
                        else self.config.position_kd
                    )

                commands_left[motor_name] = (kp, kd, position_degrees, 0.0, 0.0)
            self.bus_left._mit_control_batch(commands_left)

        # Return the actions that were actually sent
        result = {}
        for motor, val in goal_pos_right.items():
            result[f"right_{motor}.pos"] = val
        for motor, val in goal_pos_left.items():
            result[f"left_{motor}.pos"] = val
        return result

    def disconnect(self):
        """Disconnect from robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disconnect from CAN buses
        self.bus_right.disconnect(self.config.disable_torque_on_disconnect)
        self.bus_left.disconnect(self.config.disable_torque_on_disconnect)

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def _deg_to_rad(self, deg: dict[str, float | int]) -> dict[str, float]:
        """Convert degrees to radians for all motors."""
        return {m: np.deg2rad(float(v)) for m, v in deg.items()}

    def _gravity_from_q(self, q_rad: dict[str, float]) -> dict[str, float]:
        """
        Compute g(q) [N·m] for all joints in the robot.
        The order of joints in the URDF matches the concatenated motor lists (right then left).

        Args:
            q_rad: Dictionary mapping motor names (with arm prefix) to positions in radians

        Returns:
            Dictionary mapping motor names to gravity torques in N·m

        Raises:
            RuntimeError: If URDF model is not loaded
        """
        if self.pin_robot is None:
            raise RuntimeError(
                "Cannot compute gravity: URDF model not loaded. "
                "Ensure urdf/openarms.urdf exists and is valid."
            )

        # Build position vector in the order of motors (left arm, then right arm)
        # This order must match the URDF joint order
        # URDF has: left_joint1-7, left_finger_joint1-2, right_joint1-7, right_finger_joint1-2
        q = np.zeros(self.pin_robot.model.nq)
        idx = 0

        # Left arm motors (first in URDF) - joints 1-7
        for motor_name in self.bus_left.motors:
            if motor_name == "gripper":
                continue  # Skip gripper, will be handled separately
            full_name = f"left_{motor_name}"
            q[idx] = q_rad.get(full_name, 0.0)
            idx += 1

        # Skip left finger joints (leave as zeros)
        idx += 2

        # Right arm motors (second in URDF) - joints 1-7
        for motor_name in self.bus_right.motors:
            if motor_name == "gripper":
                continue  # Skip gripper, will be handled separately
            full_name = f"right_{motor_name}"
            q[idx] = q_rad.get(full_name, 0.0)
            idx += 1

        # Skip right finger joints (leave as zeros)
        idx += 2

        # Compute generalized gravity vector
        g = pin.computeGeneralizedGravity(self.pin_robot.model, self.pin_robot.data, q)

        # Map back to motor names (only arm joints, not fingers)
        result = {}
        idx = 0

        # Left arm torques (joints 1-7)
        for motor_name in self.bus_left.motors:
            if motor_name == "gripper":
                result["left_gripper"] = 0.0  # No gravity compensation for gripper
                continue
            result[f"left_{motor_name}"] = float(g[idx])
            idx += 1

        # Skip left finger joint torques in output
        idx += 2

        # Right arm torques (joints 1-7)
        for motor_name in self.bus_right.motors:
            if motor_name == "gripper":
                result["right_gripper"] = 0.0  # No gravity compensation for gripper
                continue
            result[f"right_{motor_name}"] = float(g[idx])
            idx += 1

        # Skip right finger joint torques in output
        idx += 2

        return result

    def _friction_from_velocity(
        self,
        velocity_rad_per_sec: dict[str, float],
        friction_scale: float = 1.0,
        amp_tmp: float = 1.0,
        coef_tmp: float = 0.1,
    ) -> dict[str, float]:
        """
        Compute friction torques for all joints in the robot using tanh friction model.

        Args:
            velocity_rad_per_sec: Dictionary mapping motor names (with arm prefix) to velocities in rad/s
            friction_scale: Scale factor for friction compensation (default 1.0, use 0.3 for stability)
            amp_tmp: Amplitude factor for tanh term (default 1.0)
            coef_tmp: Coefficient for tanh steepness (default 0.1)

        Returns:
            Dictionary mapping motor names to friction torques in N·m
        """
        # Motor name to index mapping
        motor_name_to_index = {
            "joint_1": 0,
            "joint_2": 1,
            "joint_3": 2,
            "joint_4": 3,
            "joint_5": 4,
            "joint_6": 5,
            "joint_7": 6,
            "gripper": 7,
        }

        result = {}

        # Process all motors (left and right)
        for motor_full_name, velocity in velocity_rad_per_sec.items():
            # Extract motor name without arm prefix
            if motor_full_name.startswith("right_"):
                motor_name = motor_full_name.removeprefix("right_")
            elif motor_full_name.startswith("left_"):
                motor_name = motor_full_name.removeprefix("left_")
            else:
                result[motor_full_name] = 0.0
                continue

            # Get motor index for friction parameters
            motor_index = motor_name_to_index.get(motor_name, 0)

            # Get friction parameters from config
            f_c = self.config.friction_f_c[motor_index]
            k = self.config.friction_k[motor_index]
            f_v = self.config.friction_f_v[motor_index]
            f_o = self.config.friction_f_o[motor_index]

            # Friction model: τ_fric = amp * F_c * tanh(coef * k * ω) + F_v * ω + F_o
            friction_torque = amp_tmp * f_c * np.tanh(coef_tmp * k * velocity) + f_v * velocity + f_o

            # Apply scale factor
            friction_torque *= friction_scale

            result[motor_full_name] = float(friction_torque)

        return result

    def get_damping_kd(self, motor_name: str) -> float:
        """
        Get damping gain (Kd) for a specific motor.

        Args:
            motor_name: Motor name without arm prefix (e.g., "joint_1", "gripper")

        Returns:
            Damping gain value
        """
        motor_name_to_index = {
            "joint_1": 0,
            "joint_2": 1,
            "joint_3": 2,
            "joint_4": 3,
            "joint_5": 4,
            "joint_6": 5,
            "joint_7": 6,
            "gripper": 7,
        }

        motor_index = motor_name_to_index.get(motor_name, 0)
        return self.config.damping_kd[motor_index]
