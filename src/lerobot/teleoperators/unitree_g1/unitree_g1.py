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
        """Placeholder used when `unitree_sdk2py` is not installed.

        Raises `ImportError` on instantiation instead of on import, so the module can still be imported
        (and its non-hardware members inspected) without the SDK present.

        Raises:
            ImportError: Always.
        """

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
        """Initialize joystick axes, button state, and joystick-center calibration to their defaults."""
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
        """Detect and record the center position of one side's exoskeleton-mounted joystick.

        Meant to be called once at connect time. If the joystick's button ADC channel reads above
        half-scale, an exoskeleton joystick is assumed present on that side, and its current X/Y ADC
        reading is stored as the neutral center used by `set_from_exo`.

        Args:
            raw16 (`list[int] | None`):
                The 16 raw ADC channel values read from the exoskeleton's sensor board, or `None` if no
                sample was available.
            side (`str`):
                Which joystick to calibrate, `"left"` or `"right"`.
        """
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
        """Update one side's joystick axes and button from the exoskeleton-mounted joystick, if calibrated.

        No-op if `calibrate_center` did not detect an exoskeleton joystick on that side.

        Args:
            raw16 (`list[int] | None`):
                The 16 raw ADC channel values read from the exoskeleton's sensor board, or `None` if no
                sample was available.
            side (`str`):
                Which joystick to update, `"left"` or `"right"`.
        """
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
    """Bimanual exoskeleton-arm teleoperator for the Unitree G1 humanoid, plus its wireless remote.

    Two exoskeleton arms worn by the operator report joint angles, which are converted to a G1 arm action
    via forward kinematics on the exoskeleton followed by inverse kinematics on the G1 (see
    [`~teleoperators.unitree_g1.exo_ik.ExoskeletonIKHelper`]). A Unitree wireless remote (or an
    exoskeleton-mounted joystick, when the remote is idle) supplies additional axes, typically used for
    locomotion. If neither exoskeleton arm has a configured serial port, the teleoperator falls back to
    remote-controller-only mode and reports no arm joint actions.

    Args:
        config (`UnitreeG1TeleoperatorConfig`): The teleoperator's configuration. Exoskeleton arm
            control is enabled only if both `left_arm_config.port` and `right_arm_config.port` are
            set; leaving both empty runs in remote-controller-only mode.

    Raises:
        ValueError: If exactly one of the two arm ports is configured.

    Example:
        ```python
        >>> from lerobot.teleoperators.unitree_g1 import UnitreeG1Teleoperator, UnitreeG1TeleoperatorConfig
        >>> teleop = UnitreeG1Teleoperator(UnitreeG1TeleoperatorConfig())  # doctest: +SKIP
        >>> with teleop:  # doctest: +SKIP
        ...     action = teleop.get_action()
        ```
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
        """Keys the teleoperator's actions are reported under.

        Includes one `"<joint>.q"` key per G1 arm joint (radians) when both exoskeleton arms are
        configured, plus the remote controller's stick and button axes. See
        [`~teleoperators.Teleoperator.action_features`].

        Returns:
            `dict[str, type]`: Action names mapped to `float`.
        """
        remote_features = dict.fromkeys(self.remote_controller.remote_action, float)
        if not self._arm_control_enabled:
            return remote_features
        joint_features = {f"{name}.q": float for name in self._g1_arm_joint_names}
        return {**joint_features, **remote_features}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """Same as [`~teleoperators.Teleoperator.feedback_features`].

        Returns:
            `dict[str, type]`: A single `"wireless_remote"` key mapped to `bytes`, the raw Unitree
            wireless remote packet to be parsed into joystick and button state.
        """
        return {"wireless_remote": bytes}

    @property
    def is_connected(self) -> bool:
        """Same as [`~teleoperators.Teleoperator.is_connected`].

        Returns:
            `bool`: `True` if exoskeleton arm control is disabled (remote-only mode), or if both
            exoskeleton arms are connected.
        """
        if not self._arm_control_enabled:
            return True
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        """Same as [`~teleoperators.Teleoperator.is_calibrated`].

        Returns:
            `bool`: `True` if exoskeleton arm control is disabled (remote-only mode), or if both
            exoskeleton arms are calibrated.
        """
        if not self._arm_control_enabled:
            return True
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        """Connect both exoskeleton arms, build the IK helper, and calibrate the remote's joystick centers.

        If neither exoskeleton arm has a configured serial port, this is a no-op and the teleoperator
        falls back to reporting only remote-controller actions.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Whether to calibrate each exoskeleton arm that is not yet calibrated.
        """
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
        """Calibrate each exoskeleton arm that is not already calibrated, then verify tracking visually.

        See [`~teleoperators.Teleoperator.calibrate`]. After both arms are calibrated, this opens the
        interactive meshcat visualization (see `run_visualization_loop`) so the operator can confirm the
        G1 arms track the exoskeleton before recording data.
        """
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
        """No-op: the exoskeleton arms require no runtime configuration beyond calibration.

        See [`~teleoperators.Teleoperator.configure`].
        """
        pass

    def get_action(self) -> dict[str, float]:
        """Read both exoskeleton arms and the remote controller, and combine them into one action.

        Exoskeleton joint angles are converted to G1 arm joint angles by forward kinematics on the
        exoskeleton followed by inverse kinematics on the G1, via
        [`~teleoperators.unitree_g1.exo_ik.ExoskeletonIKHelper.compute_g1_joints_from_exo`]. The wireless
        remote takes priority over the exoskeleton-mounted joystick for stick/button axes whenever it
        reports a non-zero stick or a pressed button; otherwise the exoskeleton-mounted joystick (if
        calibrated) is used instead.

        Returns:
            `dict[str, float]`: G1 arm joint angles (`"<joint>.q"`, radians) when exoskeleton control is
            enabled, merged with the remote controller's stick and button axes. Matches
            [`~teleoperators.Teleoperator.action_features`].
        """
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
        """Update the remote controller's parsed state from a raw wireless remote packet.

        Args:
            feedback (`dict[str, Any]`):
                Feedback dict; only the `"wireless_remote"` key (raw bytes) is used, matching
                [`~teleoperators.Teleoperator.feedback_features`]. Ignored if the key is absent.
        """
        wireless_remote = feedback.get("wireless_remote")
        if wireless_remote is not None:
            self.remote_controller.set_from_wireless(wireless_remote)

    def disconnect(self) -> None:
        """Disconnect both exoskeleton arms. See [`~teleoperators.Teleoperator.disconnect`]."""
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
