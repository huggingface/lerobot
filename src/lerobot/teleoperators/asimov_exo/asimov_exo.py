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
from typing import Any

from lerobot.robots.asimov.asimov import ARM_ACTIONS
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS
from lerobot.utils.errors import DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..unitree_g1.exo_serial import ExoskeletonArm
from .config_asimov_exo import AsimovExoTeleoperatorConfig
from .exo_ik import AsimovExoIKHelper

logger = logging.getLogger(__name__)


class AsimovExoTeleoperator(Teleoperator):
    """
    Bimanual exoskeleton arms teleoperator for Asimov 1 arms.

    Uses the same exoskeleton hardware and calibration as the Unitree G1 teleoperator:
    exoskeleton FK computes end-effector poses, Asimov IK solves for joint angles.
    Drive it with ``--robot.control_mode=arms``.
    """

    config_class = AsimovExoTeleoperatorConfig
    name = "asimov_exo"

    def __init__(self, config: AsimovExoTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        if not (config.left_arm_config.port.strip() and config.right_arm_config.port.strip()):
            raise ValueError("Set both --teleop.left_arm_config.port and --teleop.right_arm_config.port.")

        # Same exoskeletons as the G1 teleoperator: reuse its calibration files by default.
        self.calibration_dir = (
            config.calibration_dir
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / "unitree_g1"
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        left_id = f"{config.id}_left" if config.id else "left"
        right_id = f"{config.id}_right" if config.id else "right"
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
        self.ik_helper: AsimovExoIKHelper | None = None
        self.ema_alpha = 2 / (config.ema_window + 1)
        self._ema: dict[str, dict[str, float]] = {"left": {}, "right": {}}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ARM_ACTIONS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def _make_ik_helper(self, visualize: bool = False) -> AsimovExoIKHelper:
        return AsimovExoIKHelper(
            frozen_joints=[j.strip() for j in self.config.frozen_joints.split(",") if j.strip()],
            arm_scale=self.config.arm_scale,
            rotation_weight=self.config.rotation_weight,
            wrist_roll_offset_deg=self.config.wrist_roll_offset_deg,
            visualize=visualize,
        )

    def connect(self, calibrate: bool = True) -> None:
        self._ema = {"left": {}, "right": {}}
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        self.ik_helper = self._make_ik_helper()
        logger.info("IK helper initialized")
        time.sleep(0.1)  # Give serial time to populate buffer

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
        if self.ik_helper is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self.ik_helper.compute_asimov_joints_from_exo(*self._read_smoothed_angles())

    def _smooth(self, side: str, angles: dict[str, float]) -> dict[str, float]:
        """Per-joint EMA; a missed serial read keeps the last smoothed angles."""
        ema = self._ema[side]
        for joint, value in angles.items():
            prev = ema.get(joint)
            ema[joint] = value if prev is None else self.ema_alpha * value + (1 - self.ema_alpha) * prev
        return dict(ema)

    def _read_smoothed_angles(self) -> tuple[dict[str, float], dict[str, float]]:
        return (
            self._smooth("left", self.left_arm.get_angles()),
            self._smooth("right", self.right_arm.get_angles()),
        )

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    def run_visualization_loop(self):
        """Run interactive Meshcat visualization loop to verify retargeting, without the robot."""
        helper = self._make_ik_helper(visualize=True)
        helper.init_visualization()

        print("\n" + "=" * 60)
        print("Visualization running! Move the exoskeletons to test tracking.")
        print("Press Ctrl+C to exit.")
        print("=" * 60 + "\n")

        try:
            while True:
                helper.compute_asimov_joints_from_exo(*self._read_smoothed_angles())
                helper.update_visualization()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n\nVisualization stopped.")
