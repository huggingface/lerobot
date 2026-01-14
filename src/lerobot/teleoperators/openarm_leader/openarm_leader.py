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
from typing import Any

import numpy as np
import pinocchio as pin

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import MotorType
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_openarms_leader import OpenArmsLeaderConfig

logger = logging.getLogger(__name__)


class OpenArmsLeader(Teleoperator):
    """
    OpenArms Leader/Teleoperator Arm with Damiao motors.

    This teleoperator uses CAN bus communication to read positions from
    Damiao motors that are manually moved (torque disabled).
    """

    config_class = OpenArmsLeaderConfig
    name = "openarms_leader"

    def __init__(self, config: OpenArmsLeaderConfig):
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
    def action_features(self) -> dict[str, type]:
        """Features produced by this teleoperator."""
        features = {}
        # Right arm motors - only positions stored in dataset
        for motor in self.bus_right.motors:
            features[f"right_{motor}.pos"] = float
        # Left arm motors - only positions stored in dataset
        for motor in self.bus_left.motors:
            features[f"left_{motor}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """Feedback features (not implemented for OpenArms)."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self.bus_right.is_connected and self.bus_left.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the teleoperator.

        For manual control, we disable torque after connecting so the
        arm can be moved by hand.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to CAN buses
        logger.info(f"Connecting right arm on {self.config.port_right}...")
        self.bus_right.connect()
        logger.info(f"Connecting left arm on {self.config.port_left}...")
        self.bus_left.connect()

        # Run calibration if needed
        if calibrate:
            logger.info("No calibration found or overwriting calibration. Running calibration...")
            self.calibrate()

        # Configure for manual control
        self.configure()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if teleoperator is calibrated."""
        return self.bus_right.is_calibrated and self.bus_left.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms leader.

        The calibration procedure:
        1. Disable torque (if not already disabled)
        2. Ask user to position arm in zero position (hanging with gripper closed)
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

        # Ensure torque is disabled for manual positioning
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

        # Save calibration after each arm
        self._save_calibration()

    def configure(self) -> None:
        """
        Configure motors for manual teleoperation.

        For manual control, we disable torque so the arm can be moved by hand.
        """
        if self.config.manual_control:
            # Disable torque for manual control
            logger.info("Disabling torque for manual control...")
            self.bus_right.disable_torque()
            self.bus_left.disable_torque()
        else:
            # Configure motors normally
            self.bus_right.configure_motors()
            self.bus_left.configure_motors()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    def get_action(self) -> dict[str, Any]:
        """
        Get current action from the leader arm.

        This is the main method for teleoperators - it reads the current state
        of the leader arm and returns it as an action that can be sent to a follower.

        Reads all motor states (pos/vel/torque) in one CAN refresh cycle.
        Note: Velocity and torque are read but not stored in dataset (only used for
        gravity/friction compensation during recording).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action_dict = {}
        start = time.perf_counter()

        # OPTIMIZED: Use sync_read_all_states to get pos/vel/torque in one go
        right_states = self.bus_right.sync_read_all_states()
        for motor in self.bus_right.motors:
            state = right_states.get(motor, {})
            action_dict[f"right_{motor}.pos"] = state.get("position", 0.0)
            action_dict[f"right_{motor}.vel"] = state.get("velocity", 0.0)
            action_dict[f"right_{motor}.torque"] = state.get("torque", 0.0)

        # OPTIMIZED: Use sync_read_all_states to get pos/vel/torque in one go
        left_states = self.bus_left.sync_read_all_states()
        for motor in self.bus_left.motors:
            state = left_states.get(motor, {})
            action_dict[f"left_{motor}.pos"] = state.get("position", 0.0)
            action_dict[f"left_{motor}.vel"] = state.get("velocity", 0.0)
            action_dict[f"left_{motor}.torque"] = state.get("torque", 0.0)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not yet implemented for OpenArms leader.")

    def disconnect(self) -> None:
        """Disconnect from teleoperator."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # For manual control, ensure torque is disabled before disconnecting
        if self.config.manual_control:
            try:
                self.bus_right.disable_torque()
                self.bus_left.disable_torque()
            except Exception as e:
                logger.warning(f"Failed to disable torque during disconnect: {e}")

        # Disconnect from CAN buses
        self.bus_right.disconnect(disable_torque=False)  # Already disabled above if needed
        self.bus_left.disconnect(disable_torque=False)

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
