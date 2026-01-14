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

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import MotorType
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_openarm_leader import OpenArmLeaderConfig

logger = logging.getLogger(__name__)


class OpenArmLeader(Teleoperator):
    """
    OpenArm Leader/Teleoperator Arm with Damiao motors.

    This teleoperator uses CAN bus communication to read positions from
    Damiao motors that are manually moved (torque disabled).
    """

    config_class = OpenArmLeaderConfig
    name = "openarm_leader"

    def __init__(self, config: OpenArmLeaderConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES  # Always use degrees for Damiao motors

        # Arm motors
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

    @property
    def action_features(self) -> dict[str, type]:
        """Features produced by this teleoperator."""
        features = {}
        # Arm motors - only positions stored in dataset
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """Feedback features (not implemented for OpenArms)."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the teleoperator.

        For manual control, we disable torque after connecting so the
        arm can be moved by hand.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to CAN buses
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

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
        return self.bus.is_calibrated

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
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")

        # Calibrate each arm separately
        self._calibrate_arm(self.bus)

        print(f"\nCalibration complete and saved to {self.calibration_fpath}")

    def _calibrate_arm(self, bus: DamiaoMotorsBus) -> None:
        """Calibrate a single arm."""
        logger.info("\n=== Calibrating arm ===")

        # Ensure torque is disabled for manual positioning
        bus.disable_torque()
        time.sleep(0.1)

        # Step 1: Set zero position
        input(
            "\nCalibration: Zero Position arm)\n"
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

        bus.write_calibration(self.calibration)

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
            self.bus.disable_torque()
        else:
            # Configure motors normally
            self.bus.configure_motors()

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
        states = self.bus.sync_read_all_states()
        for motor in self.bus.motors:
            state = states.get(motor, {})
            action_dict[f"{motor}.pos"] = state.get("position", 0.0)
            action_dict[f"{motor}.vel"] = state.get("velocity", 0.0)
            action_dict[f"{motor}.torque"] = state.get("torque", 0.0)

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
                self.bus.disable_torque()
            except Exception as e:
                logger.warning(f"Failed to disable torque during disconnect: {e}")

        # Disconnect from CAN buses
        self.bus.disconnect(disable_torque=False)

        logger.info(f"{self} disconnected.")
