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

"""Thread-safe robot wrapper for concurrent observation/action access."""

from __future__ import annotations

import math
import time
from threading import Lock
from typing import Any

from lerobot.robots import Robot


class ThreadSafeRobot:
    """Lock-protected wrapper around a :class:`Robot` for use with background threads.

    When RTC inference runs in a background thread while the main loop
    executes actions, both threads may access the robot concurrently.
    This wrapper serialises ``get_observation`` and ``send_action`` calls.

    Read-only properties are proxied without the lock since they don't
    mutate hardware state.
    """

    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self._lock = Lock()
        self._position_hold_enabled = False
        self._observed_positions: dict[str, float] = {}
        self._held_action: dict[str, float] | None = None
        self._observation_time: float | None = None

    # -- Lock-protected I/O --------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            sampled_at = time.monotonic()
            observation = self._robot.get_observation()
            self._observation_time = sampled_at
            self._observed_positions = {
                key: float(observation[key])
                for key in self.action_features
                if key.endswith(".pos") and key in observation
            }
            return observation

    @property
    def observation_time(self) -> float | None:
        """Client monotonic sample bound for the last read, before hardware/camera waits."""
        with self._lock:
            return self._observation_time

    def send_action(self, action: dict[str, Any] | Any) -> Any:
        with self._lock:
            self._held_action = None
            return self._robot.send_action(action)

    def configure_position_hold(self) -> None:
        """Enable the explicit, robot-supported position hold; reject mixed control modes."""
        if (
            not self._robot.supports_position_hold
            or not self.action_features
            or any(not key.endswith(".pos") for key in self.action_features)
        ):
            raise ValueError(f"{self.robot_type} has no supported local position-hold contract")
        self._position_hold_enabled = True

    @property
    def supports_hold(self) -> bool:
        return self._position_hold_enabled

    def hold(self) -> None:
        """Hold the last observed pose without camera reads, network waits, or inference.

        The first held pose stays fixed until ordinary dispatch resumes. This is an
        actuator command, not a guarantee that hardware has stopped or achieved it.
        """
        with self._lock:
            if not self._position_hold_enabled:
                raise RuntimeError("Local hold was not configured for this robot")
            if self._held_action is None:
                if set(self._observed_positions) != set(self.action_features) or not all(
                    math.isfinite(value) for value in self._observed_positions.values()
                ):
                    raise RuntimeError("Cannot hold without a finite observed position for every actuator")
                self._held_action = self._observed_positions.copy()
            self._robot.send_action(self._held_action)

    # -- Read-only proxies (no lock needed) -----------------------------------

    @property
    def observation_features(self) -> dict:
        return self._robot.observation_features

    @property
    def action_features(self) -> dict:
        return self._robot.action_features

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def cameras(self):
        return getattr(self._robot, "cameras", {})

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    @property
    def inner(self) -> Robot:
        """Access the underlying robot (e.g. for connect/disconnect)."""
        return self._robot
