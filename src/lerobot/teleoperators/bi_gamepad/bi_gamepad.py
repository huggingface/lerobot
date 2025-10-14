#!/usr/bin/env python

import logging
import sys
import time
from functools import cached_property
from typing import Dict, Tuple

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..gamepad.gamepad_utils import GamepadController, GamepadControllerHID
from ..teleoperator import Teleoperator
from .config_bi_gamepad import BiGamepadConfig

logger = logging.getLogger(__name__)


class BiGamepad(Teleoperator):
    """
    Bimanual SO-ARM controller driven by a single gamepad.

    The goal of this teleoperator is to expose the exact same action space as the
    physical bi-manual SO-101 leader arms while sourcing inputs from a consumer
    gamepad. This allows remote teleoperation setups to keep the downstream control
    pipeline untouched.
    """

    config_class = BiGamepadConfig
    name = "bi_gamepad"

    JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    # Joint limits mirror the SO-ARM hardware configuration (range -100..100 except gripper 0..100)
    JOINT_LIMITS = {
        "shoulder_pan": (-100.0, 100.0),
        "shoulder_lift": (-100.0, 100.0),
        "elbow_flex": (-100.0, 100.0),
        "wrist_flex": (-100.0, 100.0),
        "wrist_roll": (-100.0, 100.0),
        "gripper": (0.0, 100.0),
    }
    MAX_TIME_STEP = 0.25  # seconds

    def __init__(self, config: BiGamepadConfig):
        super().__init__(config)
        self.config = config
        self._controller: GamepadController | GamepadControllerHID | None = None
        self._joint_values: Dict[str, float] = self._init_joint_values()
        self._last_update: float | None = None

    @cached_property
    def action_features(self) -> Dict[str, type]:
        return {
            **{f"left_{joint}.pos": float for joint in self.JOINT_ORDER},
            **{f"right_{joint}.pos": float for joint in self.JOINT_ORDER},
        }

    @cached_property
    def feedback_features(self) -> Dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._controller is not None and self._controller.running

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - calibrate unused (gamepad does not calibrate)
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        controller_cls = self._select_controller_backend()
        self._controller = controller_cls(deadzone=self.config.deadzone)
        self._controller.start()
        if not self._controller.running:
            logger.error("Gamepad controller failed to start. Check that the gamepad is connected.")
            self._controller = None
            raise RuntimeError("Unable to start gamepad controller")

        self._last_update = time.perf_counter()
        logger.info("Gamepad controller connected for bi-manual teleoperation.")

    @property
    def is_calibrated(self) -> bool:
        return True  # Gamepads do not require calibration in this context

    def calibrate(self) -> None:
        # No calibration routine needed, but method kept for interface completeness.
        pass

    def configure(self) -> None:
        # No additional configuration is necessary for the gamepad backend.
        pass

    def get_action(self) -> Dict[str, float]:
        if not self.is_connected or self._controller is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        dt = self._compute_dt()
        self._controller.update()

        joystick = getattr(self._controller, "joystick", None)
        if joystick is None:
            logger.warning("Gamepad joystick handle unavailable. Returning last joint values.")
            return dict(self._joint_values)

        left_x, left_y = self._read_stick(joystick, self.config.left_stick_axes)
        right_x, right_y = self._read_stick(joystick, self.config.right_stick_axes)
        hat_x, hat_y = self._read_hat(joystick, self.config.hat_index)

        button_a = self._read_button("a")
        button_b = self._read_button("b")
        button_x = self._read_button("x")
        button_y = self._read_button("y")
        button_lb = self._read_button("lb")
        button_left_press = self._read_button("left_stick")
        button_rb = self._read_button("rb")
        button_right_press = self._read_button("right_stick")

        # Left/right sticks X axis -> shoulder pan (joint 1)
        self._update_joint(
            "left",
            "shoulder_pan",
            self._apply_joint_sensitivity("shoulder_pan", left_x * self.config.axis_speed * dt),
        )
        self._update_joint(
            "right",
            "shoulder_pan",
            self._apply_joint_sensitivity("shoulder_pan", right_x * self.config.axis_speed * dt),
        )

        # Left/right sticks Y axis -> shoulder lift (joint 2) (invert so pushing forward raises arm)
        self._update_joint(
            "left",
            "shoulder_lift",
            self._apply_joint_sensitivity("shoulder_lift", -left_y * self.config.axis_speed * dt),
        )
        self._update_joint(
            "right",
            "shoulder_lift",
            self._apply_joint_sensitivity("shoulder_lift", -right_y * self.config.axis_speed * dt),
        )

        # D-pad up/down -> left elbow flex (joint 3 left)
        self._update_joint(
            "left",
            "elbow_flex",
            self._apply_joint_sensitivity("elbow_flex", -hat_y * self.config.button_speed * dt),
        )

        # B / X -> right elbow flex (joint 3 right)
        self._update_joint(
            "right",
            "elbow_flex",
            self._apply_joint_sensitivity("elbow_flex", (button_b - button_x) * self.config.button_speed * dt),
        )

        # L1 / L3 -> left wrist flex (joint 4 left)
        self._update_joint(
            "left",
            "wrist_flex",
            self._apply_joint_sensitivity("wrist_flex", (button_lb - button_left_press) * self.config.button_speed * dt),
        )

        # R1 / R3 -> right wrist flex (joint 4 right)
        self._update_joint(
            "right",
            "wrist_flex",
            self._apply_joint_sensitivity("wrist_flex", (button_rb - button_right_press) * self.config.button_speed * dt),
        )

        # D-pad left/right -> left wrist roll (joint 5 left)
        self._update_joint(
            "left",
            "wrist_roll",
            self._apply_joint_sensitivity("wrist_roll", hat_x * self.config.button_speed * dt),
        )

        # A / Y -> right wrist roll (joint 5 right)
        self._update_joint(
            "right",
            "wrist_roll",
            self._apply_joint_sensitivity("wrist_roll", (button_a - button_y) * self.config.button_speed * dt),
        )

        # Left / right triggers -> grippers (joint 6)
        left_trigger = self._normalize_trigger(self._get_axis(joystick, self.config.left_trigger_axis))
        right_trigger = self._normalize_trigger(self._get_axis(joystick, self.config.right_trigger_axis))
        gripper_max = self.JOINT_LIMITS["gripper"][1]
        gripper_open_value = max(min(self.config.gripper_open_value, gripper_max), 0.0)
        self._set_joint(
            "left",
            "gripper",
            self._apply_joint_sensitivity(
                "gripper",
                (1.0 - left_trigger) * gripper_open_value,
            ),
        )
        self._set_joint(
            "right",
            "gripper",
            self._apply_joint_sensitivity(
                "gripper",
                (1.0 - right_trigger) * gripper_open_value,
            ),
        )

        return dict(self._joint_values)

    def send_feedback(self, feedback: Dict[str, float]) -> None:  # noqa: ARG002 (unused)
        # No haptic or visual feedback channel available on the standard gamepad interface.
        pass

    def disconnect(self) -> None:
        if self._controller is None:
            return

        self._controller.stop()
        self._controller = None
        logger.info("Gamepad controller disconnected.")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _select_controller_backend(self):
        use_hid = self.config.use_hid_controller
        if use_hid is None:
            use_hid = sys.platform == "darwin"
        return GamepadControllerHID if use_hid else GamepadController

    def _init_joint_values(self) -> Dict[str, float]:
        joint_values: Dict[str, float] = {}
        for prefix in ("left", "right"):
            for joint in self.JOINT_ORDER:
                default = 0.0
                if joint == "gripper":
                    default = self._apply_joint_sensitivity("gripper", self.config.gripper_open_value)
                joint_values[f"{prefix}_{joint}.pos"] = default
        return joint_values

    def _compute_dt(self) -> float:
        now = time.perf_counter()
        if self._last_update is None:
            self._last_update = now
            return 0.0

        dt = now - self._last_update
        self._last_update = now
        return min(dt, self.MAX_TIME_STEP)

    def _read_stick(self, joystick, axes: Tuple[int, int]) -> Tuple[float, float]:
        axis_x = self._apply_deadzone(self._get_axis(joystick, axes[0]))
        axis_y = self._apply_deadzone(self._get_axis(joystick, axes[1]))
        return axis_x, axis_y

    def _read_hat(self, joystick, index: int) -> Tuple[float, float]:
        if hasattr(joystick, "get_numhats") and joystick.get_numhats() > index:
            return joystick.get_hat(index)
        return 0.0, 0.0

    def _read_button(self, name: str) -> float:
        if self._controller is None:
            return 0.0

        button_map = self.config.button_mapping
        if name not in button_map:
            logger.debug("Button '%s' not present in mapping, ignoring.", name)
            return 0.0

        joystick = getattr(self._controller, "joystick", None)
        if joystick is None:
            return 0.0

        index = button_map[name]
        if hasattr(joystick, "get_numbuttons") and index < joystick.get_numbuttons():
            return 1.0 if joystick.get_button(index) else 0.0
        return 0.0

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.config.deadzone:
            return 0.0
        return value

    def _get_axis(self, joystick, index: int) -> float:
        if index is None or not hasattr(joystick, "get_numaxes"):
            return 0.0
        if index >= joystick.get_numaxes():
            return 0.0
        try:
            return float(joystick.get_axis(index))
        except (ValueError, TypeError):
            return 0.0

    def _normalize_trigger(self, raw_value: float) -> float:
        """
        Map trigger readings to a [0, 1] range.

        Xbox controllers report [-1, 1] where -1 means released. Other controllers may
        already report [0, 1]. We normalise both cases to [0, 1].
        """

        raw_value = max(min(raw_value, 1.0), -1.0)
        if raw_value >= 0.0:
            return raw_value
        return (raw_value + 1.0) / 2.0

    def _update_joint(self, arm: str, joint: str, delta: float) -> None:
        key = f"{arm}_{joint}.pos"
        current = self._joint_values[key]
        self._joint_values[key] = self._clip_joint(joint, current + delta)

    def _set_joint(self, arm: str, joint: str, value: float) -> None:
        key = f"{arm}_{joint}.pos"
        self._joint_values[key] = self._clip_joint(joint, value)

    def _clip_joint(self, joint: str, value: float) -> float:
        lower, upper = self.JOINT_LIMITS[joint]
        return max(min(value, upper), lower)

    def _apply_joint_sensitivity(self, joint: str, value: float) -> float:
        sensitivity = self.config.joint_sensitivity.get(joint, 1.0)
        return value * sensitivity
