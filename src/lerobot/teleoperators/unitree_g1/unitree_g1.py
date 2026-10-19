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
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from lerobot.robots.unitree_g1.g1_utils import REMOTE_AXES, REMOTE_BUTTONS, G1_29_JointArmIndex
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _unitree_sdk_available

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.utils.joystick import Joystick
else:

    class Joystick:
        def __init__(self):
            raise ImportError(
                "unitree_sdk2py is required for WirelessRemote. Install with: pip install unitree_sdk2py"
            )


from ..teleoperator import Teleoperator
from .config_unitree_g1 import UnitreeG1TeleoperatorConfig
from .exo_ik import ExoskeletonIKHelper
from .exo_serial import ExoskeletonArm

logger = logging.getLogger(__name__)

# SDK button names in the bit order of the wireless_remote bytes 2-3 (little-endian uint16),
# which is also the index order of the remote.button.* keys.
_WIRELESS_BUTTONS = (
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
)

# remote.button.* indices the exo thumbstick buttons stand in for.
_LEFT_EXO_BUTTON = 4  # RT
_RIGHT_EXO_BUTTON = 0  # RB


@dataclass
class JoystickState:
    """Axes and buttons from the wireless remote or exo thumbsticks; axes become ``remote.*`` actions."""

    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    buttons: list[int] = field(default_factory=lambda: [0] * len(REMOTE_BUTTONS))

    @property
    def is_active(self) -> bool:
        axes = (self.lx, self.ly, self.rx, self.ry)
        return any(abs(value) > 1e-3 for value in axes) or any(self.buttons)

    def to_action(self) -> dict[str, float]:
        return dict(zip(REMOTE_AXES, (self.lx, self.ly, self.rx, self.ry), strict=True))


class WirelessRemote:
    """Decodes the raw ``wireless_remote`` bytes of the G1's lowstate with the Unitree SDK parser."""

    def __init__(self) -> None:
        self._joystick = Joystick()
        # Disable axis smoothing and deadzone to preserve raw values
        for axis in (self._joystick.lx, self._joystick.ly, self._joystick.rx, self._joystick.ry):
            axis.smooth = 1.0
            axis.deadzone = 0.0

    def parse(self, wireless_remote: bytes) -> JoystickState | None:
        if len(wireless_remote) < 24:
            return None
        joystick = self._joystick
        joystick.extract(wireless_remote)
        buttons = [getattr(joystick, name).data if name else 0 for name in _WIRELESS_BUTTONS]
        return JoystickState(
            lx=joystick.lx.data,
            ly=joystick.ly.data,
            rx=joystick.rx.data,
            ry=joystick.ry.data,
            buttons=buttons,
        )


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
        self.wireless_remote = WirelessRemote()
        # Latest reading of each source, kept apart so one never masks the other.
        self._wireless_state = JoystickState()
        self._exo_state = JoystickState()

    @cached_property
    def action_features(self) -> dict[str, type]:
        remote_features: dict[str, type] = dict.fromkeys(REMOTE_AXES, float)
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

        self.left_arm.detect_joystick()
        self.right_arm.detect_joystick()

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
        if not self._arm_control_enabled:
            return self._wireless_state.to_action()

        if self.ik_helper is None:
            raise DeviceNotConnectedError(f"{self} IK helper is not initialized. Run `.connect()` first.")
        # One sample per arm feeds both the IK and the thumbstick.
        left_raw = self.left_arm.read_raw()
        right_raw = self.right_arm.read_raw()
        joint_action = self.ik_helper.compute_g1_joints_from_exo(
            self.left_arm.angles_from_raw(left_raw), self.right_arm.angles_from_raw(right_raw)
        )
        self._update_exo_state(left_raw, right_raw)

        # Wireless remote has priority when touched; otherwise, use exo joystick.
        remote = self._wireless_state if self._wireless_state.is_active else self._exo_state
        return {**joint_action, **remote.to_action()}

    def _update_exo_state(self, left_raw: list[int] | None, right_raw: list[int] | None) -> None:
        # A missing sample keeps the previous reading.
        left = self.left_arm.joystick_from_raw(left_raw)
        if left is not None:
            self._exo_state.lx, self._exo_state.ly = left.x, left.y
            self._exo_state.buttons[_LEFT_EXO_BUTTON] = int(left.pressed)
        right = self.right_arm.joystick_from_raw(right_raw)
        if right is not None:
            self._exo_state.rx, self._exo_state.ry = right.x, right.y
            self._exo_state.buttons[_RIGHT_EXO_BUTTON] = int(right.pressed)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        wireless_remote = feedback.get("wireless_remote")
        if wireless_remote is None:
            return
        remote = self.wireless_remote.parse(wireless_remote)
        if remote is not None:
            self._wireless_state = remote

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
