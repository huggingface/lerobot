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

"""
Piper Dual Leader Teleoperator for Hardware-level Teleoperation.

This teleoperator reads Master arm control commands from the Follower's CAN interface.
It is designed for the hardware teleop setup where:
- PC connects to Follower arms via USB
- Leader (Master) arms send control frames to Followers via CAN bus
- This teleoperator reads those control frames to capture operator intent

Delegates CAN reading to PiperMotorsBus from piper_master motor driver.
"""

import logging
import time

from lerobot.motors.piper.piper_master import PiperMotorsBus, PiperMotorsBusConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_piper_dual_leader import PIPERDualLeaderConfig

logger = logging.getLogger(__name__)


class PIPERDualLeader(Teleoperator):
    """Teleoperator that reads Master control commands from Follower's CAN interface.

    In hardware-level teleoperation:
    - Leader arms are connected to Follower arms via CAN bus
    - PC connects to Follower arms via USB
    - PiperMotorsBus (piper_master) reads the Master's control frames
      (GetArmJointCtrl / GetArmGripperCtrl) that Followers receive from Leaders

    This allows recording the operator's intended action (from Master)
    while the robot state (observation) comes from Follower feedback.
    """

    config_class = PIPERDualLeaderConfig
    name = "piper_dual_leader"

    MOTOR_DEFINITIONS = {
        "joint_1": (1, "agilex_piper"),
        "joint_2": (2, "agilex_piper"),
        "joint_3": (3, "agilex_piper"),
        "joint_4": (4, "agilex_piper"),
        "joint_5": (5, "agilex_piper"),
        "joint_6": (6, "agilex_piper"),
        "gripper": (7, "agilex_piper"),
    }

    def __init__(self, config: PIPERDualLeaderConfig):
        super().__init__(config)
        self.config = config

        self.left_bus = PiperMotorsBus(
            PiperMotorsBusConfig(can_name=config.left_port, motors=dict(self.MOTOR_DEFINITIONS))
        )
        self.right_bus = PiperMotorsBus(
            PiperMotorsBusConfig(can_name=config.right_port, motors=dict(self.MOTOR_DEFINITIONS))
        )

        self._is_connected = False

    @property
    def action_features(self) -> dict[str, type]:
        """Action features for dual arm teleoperator."""
        features = {}
        for motor in self.MOTOR_DEFINITIONS:
            features[f"left_{motor}.pos"] = float
            features[f"right_{motor}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """No feedback features for this teleoperator."""
        return {}

    def configure(self, **kwargs):
        pass

    def send_feedback(self, *args, **kwargs):
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True  # No calibration needed - just reading CAN messages

    def connect(self) -> None:
        """Connect to Follower CAN interfaces to read Master commands."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.left_bus.connect()
        self.right_bus.connect()

        self._is_connected = True
        logger.info(f"{self} connected to CAN interfaces for reading Master commands.")

    def calibrate(self):
        """No calibration needed - we're just reading CAN messages."""
        pass

    def get_action(self) -> dict[str, float]:
        """Read Master control commands from both Follower's CAN interfaces.

        Returns the control commands sent by Leader (Master) arms.
        Delegates to PiperMotorsBus.read() which uses
        GetArmJointCtrl() and GetArmGripperCtrl().
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Read left Master control commands via motor bus
        left_state = self.left_bus.read()
        action = {f"left_{k}.pos": v for k, v in left_state.items()}

        # Read right Master control commands via motor bus
        right_state = self.right_bus.read()
        action.update({f"right_{k}.pos": v for k, v in right_state.items()})

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        return action

    def disconnect(self) -> None:
        """Disconnect from CAN interfaces."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.left_bus.disconnect()
        self.right_bus.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")
