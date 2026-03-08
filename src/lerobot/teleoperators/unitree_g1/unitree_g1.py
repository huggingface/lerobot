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
import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

from lerobot.robots.unitree_g1.g1_utils import REMOTE_AXES, G1_29_JointArmIndex
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS
from lerobot.utils.import_utils import _unitree_sdk_available

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.utils.joystick import Joystick
else:

    class Joystick:
        def __init__(self):
            raise ImportError(
                "unitree_sdk2py is required for RemoteController. Install with: pip install unitree_sdk2py"
            )


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

    # Map SDK named buttons to positional indices matching the wireless_remote
    # byte layout (little-endian uint16 from bytes 2-3).
    _BUTTON_MAP: list[str] = [
        "RB",
        "LB",
        "start",
        "back",
        "RT",
        "LT",
        "",
        "",
        "A",
        "B",
        "X",
        "Y",
        "up",
        "right",
        "down",
        "left",
    ]

    def __init__(self):
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.button = [0] * 16
        self.remote_action = dict.fromkeys(REMOTE_AXES, 0.0)

        # SDK joystick parser for wireless remote bytes
        self._joystick = Joystick()
        # Disable axis smoothing and deadzone to preserve raw values
        for axis in (self._joystick.lx, self._joystick.ly, self._joystick.rx, self._joystick.ry):
            axis.smooth = 1.0
            axis.deadzone = 0.0

        # Joystick center calibration (read at connect time)
        self.left_center_x = self.ADC_HALF
        self.left_center_y = self.ADC_HALF
        self.right_center_x = self.ADC_HALF
        self.right_center_y = self.ADC_HALF

        # Whether to use exo joystick (detected at connect time)
        self.use_left_exo_joystick = False
        self.use_right_exo_joystick = False

    def _sync_remote_action(self) -> None:
        self.remote_action.update(zip(REMOTE_AXES, (self.lx, self.ly, self.rx, self.ry), strict=True))

    def calibrate_center(self, raw16: list[int] | None, side: str) -> None:
        if raw16 is None or len(raw16) < 16:
            logger.info(f"{side.capitalize()} exo joystick: no data available")
            return

        btn_val = raw16[self.JOYSTICK_BTN_IDX]
        logger.info(f"{side.capitalize()} exo joystick button ADC: {btn_val} (threshold: {self.ADC_HALF})")
        if btn_val <= self.ADC_HALF:
            logger.info(f"{side.capitalize()} exo joystick not detected (button below threshold)")
            return

        x = raw16[self.JOYSTICK_X_IDX]
        y = raw16[self.JOYSTICK_Y_IDX]
        if side == "left":
            self.use_left_exo_joystick = True
            self.left_center_x, self.left_center_y = x, y
        else:
            self.use_right_exo_joystick = True
            self.right_center_x, self.right_center_y = x, y
        logger.info(f"{side.capitalize()} exo joystick enabled, center: x={x}, y={y}")

    def set_from_exo(self, raw16: list[int] | None, side: str) -> None:
        if raw16 is None or len(raw16) < 16:
            return

        if side == "left":
            if not self.use_left_exo_joystick:
                return
            self.lx = (raw16[self.JOYSTICK_X_IDX] - self.left_center_x) / self.ADC_HALF
            self.ly = (raw16[self.JOYSTICK_Y_IDX] - self.left_center_y) / self.ADC_HALF
            self.button[4] = 1 if raw16[self.JOYSTICK_BTN_IDX] < self.ADC_HALF else 0
            return

        if not self.use_right_exo_joystick:
            return
        self.rx = (raw16[self.JOYSTICK_X_IDX] - self.right_center_x) / self.ADC_HALF
        self.ry = (raw16[self.JOYSTICK_Y_IDX] - self.right_center_y) / self.ADC_HALF
        self.button[0] = 1 if raw16[self.JOYSTICK_BTN_IDX] < self.ADC_HALF else 0

    def set_from_wireless(self, wireless_remote: bytes) -> None:
        """Parse Unitree wireless remote raw bytes into joystick + button state."""
        if len(wireless_remote) < 24:
            return
        self._joystick.extract(wireless_remote)

        self.lx = self._joystick.lx.data
        self.ly = self._joystick.ly.data
        self.rx = self._joystick.rx.data
        self.ry = self._joystick.ry.data

        for i, name in enumerate(self._BUTTON_MAP):
            if name:
                self.button[i] = getattr(self._joystick, name).data


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
        left_exo_enabled = bool(config.left_arm_config.port.strip())
        right_exo_enabled = bool(config.right_arm_config.port.strip())
        if left_exo_enabled != right_exo_enabled:
            raise ValueError(
                "Invalid exo config: set both left/right exo ports, or leave both empty for remote-only mode."
            )
        self._arm_control_enabled = left_exo_enabled and right_exo_enabled

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
        remote_features = dict.fromkeys(self.remote_controller.remote_action, float)
        if not self._arm_control_enabled:
            return remote_features
        joint_features = {f"{name}.q": float for name in self._g1_arm_joint_names}
        return {**joint_features, **remote_features}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {"wireless_remote": bytes}

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
        if not self._arm_control_enabled:
            logger.warning("Exo ports not fully configured; teleop will send joystick only (no arm actions)")
            return

        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        frozen_joints = [j.strip() for j in self.config.frozen_joints.split(",") if j.strip()]
        self.ik_helper = ExoskeletonIKHelper(frozen_joints=frozen_joints)
        logger.info("IK helper initialized")

        time.sleep(0.1)  # Give serial time to populate buffer

        left_raw = self.left_arm.read_raw()
        right_raw = self.right_arm.read_raw()
        self.remote_controller.calibrate_center(left_raw, "left")
        self.remote_controller.calibrate_center(right_raw, "right")

    def calibrate(self) -> None:
        if not self.left_arm.is_calibrated:
            logger.info("Starting calibration for left arm...")
            self.left_arm.calibrate()
        else:
            logger.info("Left arm already calibrated. Skipping.")

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
            left_raw = self.left_arm.read_raw()
            right_raw = self.right_arm.read_raw()

            left_angles = self.left_arm.get_angles()
            right_angles = self.right_arm.get_angles()
            joint_action = self.ik_helper.compute_g1_joints_from_exo(left_angles, right_angles)

        # Wireless remote has priority when non-zero; otherwise, use exo joystick.
        rc = self.remote_controller
        wireless_active = (
            abs(rc.lx) > 1e-3 or abs(rc.ly) > 1e-3 or abs(rc.rx) > 1e-3 or abs(rc.ry) > 1e-3
        ) or any(rc.button)
        if self._arm_control_enabled and not wireless_active:
            rc.set_from_exo(left_raw, "left")
            rc.set_from_exo(right_raw, "right")

        rc._sync_remote_action()
        return {**joint_action, **rc.remote_action}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        wireless_remote = feedback.get("wireless_remote")
        if wireless_remote is not None:
            self.remote_controller.set_from_wireless(wireless_remote)

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    def run_visualization_loop(self):
        """Run interactive Meshcat visualization loop to verify tracking."""
        if self.ik_helper is None:
            frozen_joints = [j.strip() for j in self.config.frozen_joints.split(",") if j.strip()]
            self.ik_helper = ExoskeletonIKHelper(frozen_joints=frozen_joints)

        self.ik_helper.init_visualization()

        print("\n" + "=" * 60)
        print("Visualization running! Move the exoskeletons to test tracking.")
        print("Press Ctrl+C to exit.")
        print("=" * 60 + "\n")

        try:
            while True:
                left_angles = self.left_arm.get_angles()
                right_angles = self.right_arm.get_angles()

                self.ik_helper.compute_g1_joints_from_exo(left_angles, right_angles)
                self.ik_helper.update_visualization()

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\nVisualization stopped.")

    @cached_property
    def _g1_arm_joint_names(self) -> list[str]:
        return [joint.name for joint in G1_29_JointArmIndex]
