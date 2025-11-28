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

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class BaseKeyboardController:
    """Shared helper to convert pressed keys into XLerobot base actions."""

    def __init__(
        self,
        teleop_keys: Mapping[str, str],
        speed_levels: list[dict[str, float]] | None = None,
    ) -> None:
        self.teleop_keys = teleop_keys
        self.speed_levels = speed_levels or [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0

    def compute_action(self, pressed_keys: Iterable[str]) -> dict[str, float]:
        """Return {'x.vel','y.vel','theta.vel'} based on the pressed key set."""
        keys = {str(key) for key in pressed_keys}

        if self.teleop_keys["speed_up"] in keys:
            self.speed_index = min(self.speed_index + 1, len(self.speed_levels) - 1)
        if self.teleop_keys["speed_down"] in keys:
            self.speed_index = max(self.speed_index - 1, 0)

        speed = self.speed_levels[self.speed_index]
        xy_speed = speed["xy"]
        theta_speed = speed["theta"]

        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0

        if self.teleop_keys["forward"] in keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in keys:
            theta_cmd -= theta_speed

        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    def augment_action(self, action: Mapping[str, Any] | None) -> dict[str, Any]:
        """Merge keyboard presses from ``action`` into base velocity commands."""
        if action is None:
            return {}

        if not isinstance(action, Mapping):
            return action

        action_dict = dict(action.items())

        teleop_values = set(self.teleop_keys.values())
        pressed_keys = {
            key
            for key, value in action_dict.items()
            if isinstance(key, str) and key in teleop_values and value is None
        }

        if not pressed_keys:
            return action_dict

        for key in pressed_keys:
            action_dict.pop(key, None)

        base_velocities = self.compute_action(pressed_keys)

        for key, value in base_velocities.items():
            action_dict.setdefault(key, value)

        return action_dict
