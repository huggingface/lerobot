#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import torch

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_koch_screwdriver_leader import KochScrewdriverLeaderConfig

logger = logging.getLogger(__name__)


class KochScrewdriverLeader(Teleoperator):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = KochScrewdriverLeaderConfig
    name = "koch_screwdriver_leader"
    
    # Map the leader motor name to the follower motor and action name
    motor_to_action_map = {
        "gripper": "screwdriver"
    }

    def __init__(self, config: KochScrewdriverLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        """
        Map gripper motor to screwdriver action name for follower compatibility.
        """
        features = {}
        for motor in self.bus.motors:
            action_name = self.motor_to_action_map.get(motor, motor)
            if action_name == "screwdriver":
                features[f"{action_name}.vel"] = float
            else:
                features[f"{action_name}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Same as the Koch leader but inverting the elbow flex motor has been removed as I don't think it should be
        hardcoded here, but instead be handled by calibration.
        """
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        drive_modes = dict.fromkeys(self.bus.motors, 0)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            if motor != "gripper":
                # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
                # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while
                # assembling the arm, you could end up with a servo with a position 0 or 4095 at a crucial
                # point
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        # Use 'position control current based' for gripper to be limited by the limit of the current.
        # For the follower gripper, it means it can grasp an object without forcing too much even tho,
        # its goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
        # to make it move, and it will move back to its original target position when we release the force.
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
        # Set gripper's goal pos in current position mode so that we can use it as a trigger.
        self.bus.enable_torque("gripper")
        if self.is_calibrated:
            self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read all joint positions once.
        start = time.perf_counter()
        pos_dict = self.bus.sync_read("Present_Position")

        # Build the action dictionary, converting the screwdriver position into a velocity command.
        action = {}
        for motor, pos in pos_dict.items():
            action_name = self.motor_to_action_map.get(motor, motor)
            
            if action_name == "screwdriver":
                # Map the leader gripper position to follower screwdriver velocity using relative normalization.
                # 
                # MAPPING OVERVIEW:
                # Leader Gripper (CURRENT_POSITION mode):
                #   - Position range: 0-100 (normalized)
                #   - Neutral position: screwdriver_open_pos (e.g., 50)
                #   - Operator can push/pull the handle against resistance
                #   - Returns to neutral when released
                # 
                # Follower Screwdriver (VELOCITY mode):
                #   - Velocity range: -700 to +700 (units)
                #   - Zero velocity = no rotation
                #   - Positive velocity = clockwise rotation
                #   - Negative velocity = counter-clockwise rotation
                # 
                # TRANSFORMATION STEPS:
                # 1. Calculate deviation from neutral: delta = pos - neutral_pos
                # 2. Apply gain scaling: vel_cmd = -delta * GAIN
                # 3. Clamp to velocity limits: vel_cmd = clamp(vel_cmd, -MAX_VEL, +MAX_VEL)
                # 4. Filter small jitters: if |vel_cmd| < threshold, set to 0
                #
                # EXAMPLE MAPPING:
                #   pos=80, neutral=50 → delta=30 → vel_cmd=-300 → clamp(-300, -700, 700) = -300
                #   pos=20, neutral=50 → delta=-30 → vel_cmd=300 → clamp(300, -700, 700) = 300
                #   pos=50, neutral=50 → delta=0 → vel_cmd=0 → no movement
                
                # Step 1: Calculate deviation from neutral position
                delta = pos - self.config.screwdriver_open_pos
                
                # Step 2: Apply gain scaling with sign inversion for intuitive control
                # Negative sign means: open gripper → negative velocity (counter-clockwise)
                GAIN = 10.0
                
                # VELOCITY RANGE RESEARCH (XL330-M077 Dynamixel):
                # - No Load Speed: 383 RPM at 5.0V (most common operating voltage)
                # - Velocity Limit Range: 0 ~ 2,047 raw units (default: 1,620)
                # - Resolution: 0.229 rev/min per unit
                # - Goal Velocity Range: -Velocity Limit ~ +Velocity Limit
                # 
                # Current MAX_VEL = 700 units = 160.3 RPM (700 × 0.229)
                # This is conservative and appropriate because:
                # 1. Safe: 42% of max speed (383 RPM), well within servo capability
                # 2. Controllable: Good precision for screw driving operations
                # 3. Responsive: Fast enough for practical use
                # 4. Conservative: Leaves room for load variations
                # 
                # Note: With GAIN=10.0 and 0-100 position range, max calculated velocity
                # would be 500 units (114.5 RPM), so clamping is still useful for safety.
                MAX_VEL = 700
                vel_cmd = -delta * GAIN
                
                # Step 3: Clamp to velocity limits to prevent excessive speeds
                # With screwdriver open pos of 50 and gain of 10 we should not need clamping
                # But if gain was increased or the open pos was changed we might need to clamp.
                vel_cmd = max(min(vel_cmd, MAX_VEL), -MAX_VEL)

                # Step 4: Filter small jitters around neutral point for stability
                if abs(vel_cmd) < 4.0:
                    vel_cmd = 0.0

                action[f"{action_name}.vel"] = vel_cmd
            else:
                action[f"{action_name}.pos"] = pos

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
