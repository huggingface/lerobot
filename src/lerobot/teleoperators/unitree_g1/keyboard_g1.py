# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discrete joint jogging through the standard G1 teleoperation feedback loop."""

import logging
import math
import threading
import time
from dataclasses import dataclass

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex
from lerobot.utils.keyboard_input import create_key_listener

from ..config import TeleoperatorConfig
from ..teleoperator import Teleoperator

# Radian limits from lerobot/unitree-g1-mujoco's G1-29 model, snapshot
# 68459ed68f6f68e1f661091dfcb6ebce44681aec. These are joint, not collision limits.
ARM_LIMITS = {
    "ShoulderPitch": (-3.0892, 2.6704),
    "ShoulderYaw": (-2.618, 2.618),
    "Elbow": (-1.0472, 2.0944),
    "WristRoll": (-1.97222, 1.97222),
    "WristPitch": (-1.614429558, 1.614429558),
    "WristYaw": (-1.614429558, 1.614429558),
}


def arm_limits(name):
    side = "Left" if name.startswith("kLeft") else "Right"
    joint = name.removeprefix(f"k{side}")
    if joint == "ShoulderRoll":
        return (-1.5882, 2.2515) if side == "Left" else (-2.2515, 1.5882)
    return ARM_LIMITS[joint]


@TeleoperatorConfig.register_subclass("unitree_g1_keyboard")
@dataclass
class UnitreeG1KeyboardConfig(TeleoperatorConfig):
    step_rad: float = 0.02

    def __post_init__(self):
        if not math.isfinite(self.step_rad) or not 0 < self.step_rad <= 0.05:
            raise ValueError("Keyboard step_rad must be in (0, 0.05]")


class UnitreeG1Keyboard(Teleoperator):
    """Select an arm/joint and jog one bounded increment per key event.

    No held-key integration: OS repeat events are capped at 10 Hz and are never
    queued. Enter enables jogging; Space holds the measured pose and disables it.
    This is not an emergency stop or a collision-aware physical controller.
    """

    config_class = UnitreeG1KeyboardConfig
    name = "unitree_g1_keyboard"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.joints = tuple(j.name for j in G1_29_JointArmIndex)
        self.limits = {f"{name}.q": arm_limits(name) for name in self.joints}
        self._lock = threading.Lock()
        self.listener = None
        self._reset()

    def _reset(self):
        self._measured = None
        self._targets = None
        self._feedback_at = -math.inf
        self._jog_at = -math.inf
        self._enabled = False
        self._quit = False
        self._side = "Left"
        self._joint = 0

    @property
    def action_features(self):
        return dict.fromkeys(self.limits, float)

    @property
    def feedback_features(self):
        return self.action_features

    @property
    def is_connected(self):
        return self.listener is not None

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def connect(self, calibrate=True):
        if self.is_connected:
            raise RuntimeError("Keyboard already connected")
        self._reset()
        self.listener = create_key_listener(
            self._on_key,
            controls_help="Enter=enable, Space=hold, L/R=arm, 1-7=joint, +/-=jog, Esc=exit",
        )
        if self.listener is None:
            raise RuntimeError("Keyboard needs an interactive terminal or a supported desktop listener")

    def _on_key(self, key):
        with self._lock:
            key = key.lower()
            if key == "esc":
                self._quit = True
                self._enabled = False
                return
            if key in ("l", "r"):
                self._side = "Left" if key == "l" else "Right"
            elif key in "1234567" and len(key) == 1:
                self._joint = int(key) - 1
            now = time.monotonic()
            if self._measured is None or now - self._feedback_at > 0.5:
                return
            if key in ("space", "enter"):
                self._targets = dict(self._measured)
                self._enabled = key == "enter"
                logging.info("G1 keyboard: %s", "jogging enabled" if self._enabled else "measured-pose hold")
            if key not in ("+", "=", "-") or not self._enabled or now - self._jog_at < 0.1:
                return
            names = [name for name in self.joints if name.startswith(f"k{self._side}")]
            if self._joint >= len(names):
                return
            name = names[self._joint] + ".q"
            lower, upper = self.limits[name]
            delta = self.config.step_rad * (-1 if key == "-" else 1)
            measured = self._measured[name]
            self._targets[name] = min(
                upper, measured + 0.15, max(lower, measured - 0.15, self._targets[name] + delta)
            )
            self._jog_at = now
            logging.info("G1 keyboard jog: %s=%.4f rad", name, self._targets[name])

    def send_feedback(self, feedback):
        if not self.is_connected:
            raise RuntimeError("Keyboard is disconnected")
        with self._lock:
            try:
                measured = {key: float(feedback[key]) for key in self.limits}
                if not all(math.isfinite(value) for value in measured.values()):
                    raise ValueError("nonfinite joint position")
                if any(not lo - 0.01 <= measured[key] <= hi + 0.01 for key, (lo, hi) in self.limits.items()):
                    raise ValueError("joint position outside model limits")
            except (KeyError, ValueError, TypeError) as exc:
                self._enabled = False
                self._feedback_at = -math.inf
                raise ValueError(
                    "Keyboard requires complete, finite G1 arm observations within limits"
                ) from exc
            self._measured = measured
            for key, (lower, upper) in self.limits.items():
                self._measured[key] = min(upper, max(lower, measured[key]))
            self._feedback_at = time.monotonic()
            if self._targets is None:
                self._targets = dict(measured)
                logging.info("G1 keyboard ready: measured-pose hold; press Enter to enable jogging")

    def get_action(self):
        with self._lock:
            if self._quit:
                raise KeyboardInterrupt
            if not self.is_connected:
                raise RuntimeError("Keyboard is disconnected")
            alive = getattr(self.listener, "is_alive", None)
            if (alive is not None and not alive()) or getattr(self.listener, "_running", True) is False:
                raise RuntimeError("Keyboard listener stopped")
            if self._targets is None or time.monotonic() - self._feedback_at > 0.5:
                self._enabled = False
                raise RuntimeError("Missing or stale G1 arm feedback")
            return dict(self._targets)

    def disconnect(self):
        listener, self.listener = self.listener, None
        if listener is not None:
            listener.stop()
        with self._lock:
            self._reset()
