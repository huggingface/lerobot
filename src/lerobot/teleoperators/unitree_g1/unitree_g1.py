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
import struct
import time
from functools import cached_property
from typing import Any

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from ..teleoperator import Teleoperator
from .config_unitree_g1 import UnitreeG1TeleoperatorConfig
from .exo_ik import ExoskeletonIKHelper
from .exo_serial import ExoskeletonArm

logger = logging.getLogger(__name__)


class RemoteController:
    """Unitree remote controller data parser for joystick and button state."""

    # ADC parameters for exoskeleton joystick (12-bit ADC)
    ADC_MAX = 4095
    ADC_HALF = ADC_MAX / 2
    JOYSTICK_X_IDX = 11  # X axis in raw ADC array
    JOYSTICK_BTN_IDX = 12  # Button in raw ADC array
    JOYSTICK_Y_IDX = 13  # Y axis in raw ADC array

    # Button indices for Unitree remote (left exo btn -> R2, right exo btn -> R1)
    BTN_R2 = 4  # Lower waist in GR00T
    BTN_R1 = 0  # Raise waist in GR00T

    def __init__(self):
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.button = [0] * 16

        # Joystick center calibration (read at connect time)
        self.left_center_x = self.ADC_HALF
        self.left_center_y = self.ADC_HALF
        self.right_center_x = self.ADC_HALF
        self.right_center_y = self.ADC_HALF

        # Whether to use exo joystick (detected at connect time)
        self.use_left_exo_joystick = False
        self.use_right_exo_joystick = False

    def calibrate_left_center(self, raw16: list[int]) -> None:
        """Calibrate left joystick center from current raw ADC values."""
        if raw16 is None or len(raw16) < 16:
            logger.info("Left exo joystick: no data available")
            return
        btn_val = raw16[self.JOYSTICK_BTN_IDX]
        logger.info(f"Left exo joystick button ADC: {btn_val} (threshold: {self.ADC_HALF})")
        # Check if exo joystick is available (button > ADC_HALF means joystick connected)
        if btn_val > self.ADC_HALF:
            self.use_left_exo_joystick = True
            self.left_center_x = raw16[self.JOYSTICK_X_IDX]
            self.left_center_y = raw16[self.JOYSTICK_Y_IDX]
            logger.info(f"Left exo joystick enabled, center: x={self.left_center_x}, y={self.left_center_y}")
        else:
            logger.info("Left exo joystick not detected (button below threshold)")

    def calibrate_right_center(self, raw16: list[int]) -> None:
        """Calibrate right joystick center from current raw ADC values."""
        if raw16 is None or len(raw16) < 16:
            logger.info("Right exo joystick: no data available")
            return
        btn_val = raw16[self.JOYSTICK_BTN_IDX]
        logger.info(f"Right exo joystick button ADC: {btn_val} (threshold: {self.ADC_HALF})")
        # Check if exo joystick is available (button > ADC_HALF means joystick connected)
        if btn_val > self.ADC_HALF:
            self.use_right_exo_joystick = True
            self.right_center_x = raw16[self.JOYSTICK_X_IDX]
            self.right_center_y = raw16[self.JOYSTICK_Y_IDX]
            logger.info(f"Right exo joystick enabled, center: x={self.right_center_x}, y={self.right_center_y}")
        else:
            logger.info("Right exo joystick not detected (button below threshold)")

    def set(self, data):
        """Parse wireless_remote data from robot lowstate."""
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]

    def _adc_to_joystick(self, raw_value: int, center: float) -> float:
        """Convert ADC value to joystick range (-1 to 1) relative to calibrated center."""
        return (raw_value - center) / self.ADC_HALF

    def set_left_from_exo(self, raw16: list[int]) -> None:
        """Set left joystick from exoskeleton raw ADC if enabled. Button sets R2."""
        if not self.use_left_exo_joystick or raw16 is None or len(raw16) < 16:
            return
        # Read joystick values from exo
        self.lx = self._adc_to_joystick(raw16[self.JOYSTICK_X_IDX], self.left_center_x)
        self.ly = self._adc_to_joystick(raw16[self.JOYSTICK_Y_IDX], self.left_center_y)
        # Set R2 button only when pressed (active low)
        if raw16[self.JOYSTICK_BTN_IDX] < self.ADC_HALF:
            self.button[self.BTN_R2] = 1

    def set_right_from_exo(self, raw16: list[int]) -> None:
        """Set right joystick from exoskeleton raw ADC if enabled. Button sets R1."""
        if not self.use_right_exo_joystick or raw16 is None or len(raw16) < 16:
            return
        # Read joystick values from exo
        self.rx = self._adc_to_joystick(raw16[self.JOYSTICK_X_IDX], self.right_center_x)
        self.ry = self._adc_to_joystick(raw16[self.JOYSTICK_Y_IDX], self.right_center_y)
        # Set R1 button only when pressed (active low)
        if raw16[self.JOYSTICK_BTN_IDX] < self.ADC_HALF:
            self.button[self.BTN_R1] = 1


class UnitreeG1Teleoperator(Teleoperator):
    """
    Bimanual exoskeleton arms teleoperator for Unitree G1 arms.

    Uses inverse kinematics: exoskeleton FK computes end-effector pose,
    G1 IK solves for joint angles.
    """

    config_class = UnitreeG1TeleoperatorConfig
    name = "unitree_g1"

    def __init__(self, config: UnitreeG1TeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self._left_exo_enabled = bool(config.left_arm_config.port.strip())
        self._right_exo_enabled = bool(config.right_arm_config.port.strip())
        self._arm_control_enabled = self._left_exo_enabled and self._right_exo_enabled

        # Setup calibration directory
        self.calibration_dir = (
            config.calibration_dir
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        left_id = f"{config.id}_left" if config.id else "left"
        right_id = f"{config.id}_right" if config.id else "right"

        # Create exoskeleton arm instances
        self.left_arm = ExoskeletonArm(
            port=config.left_arm_config.port,
            baud_rate=config.left_arm_config.baud_rate,
            calibration_fpath=self.calibration_dir / f"{left_id}.json",
            side="left",
        )
        self.right_arm = ExoskeletonArm(
            port=config.right_arm_config.port,
            baud_rate=config.right_arm_config.baud_rate,
            calibration_fpath=self.calibration_dir / f"{right_id}.json",
            side="right",
        )

        self.ik_helper: ExoskeletonIKHelper | None = None
        self.remote_controller = RemoteController()

    @cached_property
    def action_features(self) -> dict[str, type]:
        remote_features = {
            "remote.lx": float,
            "remote.ly": float,
            "remote.rx": float,
            "remote.ry": float,
        }
        if not self._arm_control_enabled:
            return remote_features
        joint_features = {f"{name}.q": float for name in self._g1_arm_joint_names}
        return {**joint_features, **remote_features}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        if not self._arm_control_enabled:
            return True
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        if not self._arm_control_enabled:
            return True
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        import time

        if not self._arm_control_enabled:
            logger.warning("Exo ports not fully configured; teleop will send joystick only (no arm actions)")
            return

        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        # Wait a bit for serial data to be available, then calibrate joystick centers
        time.sleep(0.1)  # Give serial time to populate buffer

        # Try a few times to get valid data
        left_raw = None
        right_raw = None
        for _ in range(10):
            if left_raw is None:
                left_raw = self.left_arm.read_raw()
            if right_raw is None:
                right_raw = self.right_arm.read_raw()
            if left_raw is not None and right_raw is not None:
                break
            time.sleep(0.05)

        self.remote_controller.calibrate_left_center(left_raw)
        self.remote_controller.calibrate_right_center(right_raw)

        frozen_joints = [j.strip() for j in self.config.frozen_joints.split(",") if j.strip()]
        self.ik_helper = ExoskeletonIKHelper(frozen_joints=frozen_joints)
        logger.info("IK helper initialized")

    def calibrate(self) -> None:
        if not self._arm_control_enabled:
            logger.info("Skipping exo calibration: arm control disabled (missing exo ports)")
            return

        logger.info(f"Left arm calibration file: {self.left_arm.calibration_fpath}")
        logger.info(f"Left arm file exists: {self.left_arm.calibration_fpath.is_file()}")
        logger.info(f"Left arm is_calibrated: {self.left_arm.is_calibrated}")

        if not self.left_arm.is_calibrated:
            logger.info("Starting calibration for left arm...")
            self.left_arm.calibrate()
        else:
            logger.info("Left arm already calibrated. Skipping.")

        logger.info(f"Right arm calibration file: {self.right_arm.calibration_fpath}")
        logger.info(f"Right arm file exists: {self.right_arm.calibration_fpath.is_file()}")
        logger.info(f"Right arm is_calibrated: {self.right_arm.is_calibrated}")

        if not self.right_arm.is_calibrated:
            logger.info("Starting calibration for right arm...")
            self.right_arm.calibrate()
        else:
            logger.info("Right arm already calibrated. Skipping.")

        logger.info("Starting visualization to verify calibration...")
        self.run_visualization_loop()

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        joint_action = {}
        left_raw = None
        right_raw = None
        if self._arm_control_enabled:
            # Read raw values from exoskeletons once
            left_raw = self.left_arm.read_raw()
            right_raw = self.right_arm.read_raw()

            # Convert to joint angles (pass raw to avoid re-reading serial)
            left_angles = self.left_arm.get_angles(left_raw)
            right_angles = self.right_arm.get_angles(right_raw)
            joint_action = self.ik_helper.compute_g1_joints_from_exo(left_angles, right_angles)

        # Override with exoskeleton joystick if button pressed
        # Left exo joystick -> left stick (lx, ly)
        # Right exo joystick -> right stick (rx, ry)
        if self._left_exo_enabled:
            self.remote_controller.set_left_from_exo(left_raw)
        if self._right_exo_enabled:
            self.remote_controller.set_right_from_exo(right_raw)

        # Include joystick state in action
        remote_action = {
            "remote.lx": self.remote_controller.lx,
            "remote.ly": self.remote_controller.ly,
            "remote.rx": self.remote_controller.rx,
            "remote.ry": self.remote_controller.ry,
        }

        return {**joint_action, **remote_action}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        wireless_remote = feedback.get("wireless_remote")
        if wireless_remote is not None and len(wireless_remote) >= 24:
            self.remote_controller.set(wireless_remote)

    def disconnect(self) -> None:
        if self._left_exo_enabled:
            self.left_arm.disconnect()
        if self._right_exo_enabled:
            self.right_arm.disconnect()

    def run_visualization_loop(self):
        """Run interactive Meshcat visualization loop to verify tracking."""
        if not self._arm_control_enabled:
            logger.error("Cannot run visualization: arm control disabled (missing exo ports)")
            return
        # Check if both arms are calibrated before starting
        if not self.left_arm.is_calibrated:
            logger.error("Left arm not calibrated, cannot run visualization")
            return
        if not self.right_arm.is_calibrated:
            logger.error("Right arm not calibrated, cannot run visualization")
            return

        logger.info(f"ik_helper is None: {self.ik_helper is None}")
        if self.ik_helper is None:
            frozen_joints = [j.strip() for j in self.config.frozen_joints.split(",") if j.strip()]
            self.ik_helper = ExoskeletonIKHelper(frozen_joints=frozen_joints)

        logger.info("Calling init_visualization...")
        self.ik_helper.init_visualization()
        logger.info("init_visualization done")

        print("\n" + "=" * 60)
        print("Visualization running! Move the exoskeletons to test tracking.")
        print("Press Ctrl+C to exit.")
        print("=" * 60 + "\n")

        try:
            while True:
                try:
                    left_angles = self.left_arm.get_angles()
                    right_angles = self.right_arm.get_angles()

                    self.ik_helper.compute_g1_joints_from_exo(left_angles, right_angles)
                    self.ik_helper.update_visualization()
                except Exception as e:
                    logger.warning(f"Visualization error: {e}")

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\nVisualization stopped.")

    @cached_property
    def _g1_arm_joint_names(self) -> list[str]:
        from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex
        return [joint.name for joint in G1_29_JointArmIndex]
