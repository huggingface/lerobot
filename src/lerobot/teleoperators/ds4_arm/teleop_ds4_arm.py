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

"""DualShock 4 → SO-ARM101 teleoperator.

Wraps the DS4 control logic from ds4_follower1.py into a LeRobot-compatible
Teleoperator subclass.  ``get_action()`` returns a flat dict of joint position
targets (one key per motor, e.g. ``"shoulder_pan"``) that the robot's
``send_action()`` can consume directly.

Controller mapping (USB, macOS/Linux pygame axis indices):
  LS X  → shoulder_pan    (right = +)
  LS Y  → shoulder_lift   (push up = arm rises)
  RS Y  → elbow_flex      (push up = elbow lowers, matches ds4_follower1 inversion)
  RS X  → wrist_roll      (right = +)
  L1/R1 → wrist_flex      (L1 = pitch down, R1 = pitch up)
  D-pad ↑/↓ → gripper     (up = open, down = close)

Speed modes: hold Square = SLOW, hold Circle = FAST, default = NORMAL.
"""

from typing import Any

import pygame

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from .configuration_ds4_arm import DS4ArmConfig

# ── DS4 pygame axis / button indices (USB, macOS/Linux) ──────────────────────
_AXIS_LS_X = 0
_AXIS_LS_Y = 1
_AXIS_RS_X = 2
_AXIS_RS_Y = 3

_BTN_CROSS    = 0
_BTN_CIRCLE   = 1
_BTN_SQUARE   = 2
_BTN_TRIANGLE = 3
_BTN_L1       = 9
_BTN_R1       = 10
_BTN_OPTIONS  = 6
_BTN_DPAD_UP  = 11   # macOS USB: d-pad reported as buttons
_BTN_DPAD_DOWN = 12
# ─────────────────────────────────────────────────────────────────────────────


def _apply_deadzone(value: float, deadzone: float) -> float:
    """Zero small stick values and rescale the remainder to 0–1."""
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def _ema(current: float, target: float, alpha: float) -> float:
    """Exponential moving average: blends *target* toward *current*."""
    return alpha * target + (1.0 - alpha) * current


class DS4ArmTeleop(Teleoperator):
    """LeRobot teleoperator that maps a DualShock 4 controller to SO-ARM101
    joint positions.

    Usage::

        cfg = DS4ArmConfig(id="my_ds4")
        with DS4ArmTeleop(cfg) as teleop:
            while True:
                action = teleop.get_action()   # {motor_name: float, ...}
                robot.send_action(action)
    """

    config_class = DS4ArmConfig
    name = "ds4_arm"

    # ── init ─────────────────────────────────────────────────────────────────

    def __init__(self, config: DS4ArmConfig):
        super().__init__(config)
        self.config = config

        self._joy: pygame.joystick.Joystick | None = None
        self._connected: bool = False
        self._dt: float = 1.0 / config.fps

        # Per-joint position targets (maintained across calls)
        self._targets: dict[str, float] = {m: 0.0 for m in config.motor_names}

        # Edge-detection state for toggle buttons
        self._prev_triangle: bool = False
        self._prev_options: bool = False

    # ── Teleoperator ABC: static metadata ────────────────────────────────────

    @property
    def action_features(self) -> dict:
        """One float per motor, keyed by motor name."""
        return {f"{name}.pos": float for name in self.config.motor_names}

    @property
    def feedback_features(self) -> dict:
        """No haptic/force feedback channel."""
        return {}

    # ── Teleoperator ABC: lifecycle ───────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        """DS4 needs no calibration."""
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Initialise pygame and open the DS4 joystick."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError(
                "No joystick/controller detected. "
                "Connect the DS4 via USB and try again."
            )

        idx = self.config.joystick_index
        if idx >= count:
            raise RuntimeError(
                f"Joystick index {idx} requested but only {count} controller(s) found."
            )

        self._joy = pygame.joystick.Joystick(idx)
        self._joy.init()
        self._connected = True

    def calibrate(self) -> None:
        """No-op — DS4 requires no calibration."""
        pass

    def configure(self) -> None:
        """No-op — no additional runtime configuration needed."""
        pass

    def disconnect(self) -> None:
        """Release pygame joystick resources."""
        if self._joy is not None:
            self._joy.quit()
            self._joy = None
        self._connected = False
        pygame.joystick.quit()

    # ── Core control logic ────────────────────────────────────────────────────

    def _read_inputs(self) -> dict:
        """Pump pygame events and return a clean snapshot of DS4 state.

        Y-axes are inverted so that pushing the stick *up* always yields a
        positive value (matches ds4_follower1.py convention).
        """
        pygame.event.pump()
        joy = self._joy
        dz = self.config.deadzone

        return {
            "ls_x":      _apply_deadzone( joy.get_axis(_AXIS_LS_X), dz),
            "ls_y":      _apply_deadzone(-joy.get_axis(_AXIS_LS_Y), dz),  # inverted
            "rs_x":      _apply_deadzone( joy.get_axis(_AXIS_RS_X), dz),
            "rs_y":      _apply_deadzone(-joy.get_axis(_AXIS_RS_Y), dz),  # inverted
            "dpad_up":   bool(joy.get_button(_BTN_DPAD_UP)),
            "dpad_down": bool(joy.get_button(_BTN_DPAD_DOWN)),
            "cross":     bool(joy.get_button(_BTN_CROSS)),
            "triangle":  bool(joy.get_button(_BTN_TRIANGLE)),
            "square":    bool(joy.get_button(_BTN_SQUARE)),
            "circle":    bool(joy.get_button(_BTN_CIRCLE)),
            "l1":        bool(joy.get_button(_BTN_L1)),
            "r1":        bool(joy.get_button(_BTN_R1)),
            "options":   bool(joy.get_button(_BTN_OPTIONS)),
        }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Read the DS4 and return updated joint position targets.

        Returns:
            dict[str, float]: Flat dict keyed by motor name, values are
                position targets in the same normalised units your
                SO-101 follower was calibrated to (degrees by default).
        """
        cfg = self.config
        inp = self._read_inputs()
        dt  = self._dt

        # ── Speed mode ────────────────────────────────────────────────────────
        if inp["square"]:
            max_speed = cfg.speed_slow
        elif inp["circle"]:
            max_speed = cfg.speed_fast
        else:
            max_speed = cfg.speed_normal

        # ── Raw target deltas (analog axes) ───────────────────────────────────
        raw = dict(self._targets)  # copy current smoothed state as base

        raw["shoulder_pan"]  += inp["ls_x"] * max_speed * dt
        raw["shoulder_lift"] += inp["ls_y"] * max_speed * dt   # up-stick = arm up
        raw["elbow_flex"]    -= inp["rs_y"] * max_speed * dt   # up-stick = elbow down (matches original)
        raw["wrist_roll"]    += inp["rs_x"] * max_speed * dt

        # ── L1/R1 — wrist pitch (digital) ────────────────────────────────────
        wrist_speed = cfg.speed_l1r1 if not inp["square"] and not inp["circle"] else max_speed
        if inp["l1"]:
            raw["wrist_flex"] -= wrist_speed * dt
        if inp["r1"]:
            raw["wrist_flex"] += wrist_speed * dt

        # ── D-pad — gripper (digital) ─────────────────────────────────────────
        gripper_speed = cfg.speed_dpad if not inp["square"] and not inp["circle"] else max_speed * 0.5
        if inp["dpad_up"]:
            raw["gripper"] -= gripper_speed * dt   # open
        if inp["dpad_down"]:
            raw["gripper"] += gripper_speed * dt   # close

        # ── EMA smoothing on analog joints only ───────────────────────────────
        alpha = cfg.smooth_alpha
        for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll"):
            self._targets[m] = _ema(self._targets[m], raw[m], alpha)

        # Digital joints: apply directly (no smoothing = crisp button feel)
        self._targets["wrist_flex"] = raw["wrist_flex"]
        self._targets["gripper"]    = raw["gripper"]

        return {f"{m}.pos": self._targets[m] for m in self.config.motor_names}

    # ── Optional helpers ──────────────────────────────────────────────────────

    def reset_targets(self, targets: dict[str, float]) -> None:
        """Sync internal targets to the robot's current joint positions.

        Call this once after the follower arm is connected to prevent a
        snap to (0, 0, …) on the first ``get_action()`` call::

            obs = robot.get_observation()
            teleop.reset_targets({m: obs[f"{m}.pos"] for m in cfg.motor_names})
        """
        for key, value in targets.items():
            motor = key.replace(".pos", "")
            if motor in self._targets:
                self._targets[motor] = value

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """No-op — DS4 does not support position feedback."""
        pass
