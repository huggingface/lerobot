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

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import MotorType
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_openarm_follower import OpenArmFollowerConfig

logger = logging.getLogger(__name__)


class OpenArmFollower(Robot):
    """
    OpenArms Follower Robot which uses CAN bus communication to control 7 DOF arm with a gripper.
    The arm uses Damiao motors in MIT control mode.
    """

    config_class = OpenArmFollowerConfig
    name = "openarm_follower"

    def __init__(self, config: OpenArmFollowerConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES  # Always use degrees for Damiao motors

        # arm motors
        motors = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(send_id, motor_type_str, norm_mode_body)
            motor.recv_id = recv_id
            motor.motor_type = getattr(MotorType, motor_type_str.upper().replace("-", "_"))
            motors[motor_name] = motor

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        # Cache for last valid camera frames (to avoid blocking on slow USB reads)
        self.camera_frame_cache = dict.fromkeys(self.cameras.keys())

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features = {}
        # Arm motors - only positions stored in dataset
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
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
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot and optionally calibrate.

        We assume that at connection time, the arms are in a safe rest position,
        and torque can be safely disabled to run calibration if needed.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to both CAN buses
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

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
            self.bus.set_zero_position()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self.bus.is_calibrated

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
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")

        # Calibrate each arm separately
        self._calibrate_arm(self.bus)

        print(f"\nCalibration complete and saved to {self.calibration_fpath}")

    def _calibrate_arm(self, bus: DamiaoMotorsBus) -> None:
        """Calibrate a single arm."""
        logger.info("\n=== Calibrating arm ===")

        # Disable torque for manual positioning
        bus.disable_torque()
        time.sleep(0.1)

        # Step 1: Set zero position
        input(
            "\nCalibration: Zero Position arm\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )

        # Set current position as zero for all motors
        bus.set_zero_position()
        logger.info("Arm zero position set.")

        # Automatically set range to -90° to +90° for all joints
        print("\nAutomatically setting range: -90° to +90° for all joints")

        # Create calibration data with fixed ranges
        if self.calibration is None:
            self.calibration = {}

        for motor_name, motor in bus.motors.items():
            # Use -90 to +90 for all joints and gripper (integers required)
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,  # Normal direction
                homing_offset=0,  # Already set via set_zero_position
                range_min=-90,  # -90 degrees (integer)
                range_max=90,  # +90 degrees (integer)
            )
            logger.info(f"{motor_name}: range set to [-90°, +90°]")

        # Write calibration to this arm's motors
        bus.write_calibration(self.calibration)

        # Re-enable torque
        bus.enable_torque()

        # Save calibration after each arm
        self._save_calibration()

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        # Configure arm
        with self.bus.torque_disabled():
            self.bus.configure_motors()

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
        states = self.bus.sync_read_all_states()
        timings["motors"] = (time.perf_counter() - t0) * 1000

        for motor in self.bus.motors:
            state = states.get(motor, {})
            obs_dict[f"{motor}.pos"] = state.get("position", 0.0)
            obs_dict[f"{motor}.vel"] = state.get("velocity", 0.0)
            obs_dict[f"{motor}.torque"] = state.get("torque", 0.0)

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

            # Store timing with padded name to align output (e.g. "wrist")
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
            action: Dictionary with motor positions (e.g., "joint_1.pos", "joint_2.pos")
            custom_kp: Optional custom kp gains per motor (e.g., {"joint_1": 120.0, "joint_2": 150.0})
            custom_kd: Optional custom kd gains per motor (e.g., {"joint_1": 1.5, "joint_2": 2.0})

        Returns:
            The action actually sent (potentially clipped)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract motor positions from action and split by arm
        goal_pos = {}

        for key, val in action.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                goal_pos[motor_name] = val

        # Apply joint limit clipping to arm
        for motor_name, position in goal_pos.items():
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f}° to {clipped_position:.2f}°")
                goal_pos[motor_name] = clipped_position

        # Apply safety limits if configured
        if self.config.max_relative_target is not None:
            # Get current positions
            present_pos = self.bus.sync_read("Present_Position")

            # Apply safety limits to arm
            if goal_pos:
                goal_present_pos = {
                    key: (g_pos, present_pos.get(key, 0.0)) for key, g_pos in goal_pos.items()
                }
                goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

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

        # Use batch MIT control for arm (sends all commands, then collects responses)
        if goal_pos:
            commands = {}
            for motor_name, position_degrees in goal_pos.items():
                idx = motor_index.get(motor_name, 0)

                # Use custom gains if provided, otherwise use config defaults
                if custom_kp is not None and motor_name in custom_kp:
                    kp = custom_kp[motor_name]
                else:
                    kp = (
                        self.config.position_kp[idx]
                        if isinstance(self.config.position_kp, list)
                        else self.config.position_kp
                    )

                if custom_kd is not None and motor_name in custom_kd:
                    kd = custom_kd[motor_name]
                else:
                    kd = (
                        self.config.position_kd[idx]
                        if isinstance(self.config.position_kd, list)
                        else self.config.position_kd
                    )

                commands[motor_name] = (kp, kd, position_degrees, 0.0, 0.0)
            self.bus._mit_control_batch(commands)

        # Return the actions that were actually sent
        result = {}
        for motor, val in goal_pos.items():
            result[f"{motor}.pos"] = val
        return result

    def disconnect(self):
        """Disconnect from robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disconnect from CAN buses
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
