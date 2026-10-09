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
        self._command_hold_enabled = False
        self._observed_positions: dict[str, float] = {}
        self._last_applied_action: dict[str, float] | None = None
        self._observation_time: float | None = None
        self._hardware_failure: str | None = None

    # -- Lock-protected I/O --------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            sampled_at = time.monotonic()
            observation = self._robot.get_observation()
            self._observed_positions = {
                key: float(observation[key])
                for key in self.action_features
                if key.endswith(".pos") and key in observation
            }
            self._observation_time = sampled_at
            return observation

    @property
    def observation_time(self) -> float | None:
        """Client monotonic sample bound for the last read, before hardware/camera waits."""
        with self._lock:
            return self._observation_time

    def send_action(self, action: dict[str, Any] | Any) -> Any:
        with self._lock:
            try:
                applied_action = self._robot.send_action(action)
                if self._position_hold_enabled:
                    self._last_applied_action = self._validated_positions(applied_action)
                return applied_action
            except Exception as exc:
                self._record_hardware_failure("send_action", exc)
                raise

    def _record_hardware_failure(self, operation: str, error: Exception) -> None:
        # Caller owns the I/O lock. Failed commands prohibit further shutdown movement.
        if self._hardware_failure is None:
            self._hardware_failure = f"{operation}: {type(error).__name__}: {error}"

    @property
    def hardware_failure(self) -> str | None:
        """First command/hold failure; never cleared by inference reset."""
        with self._lock:
            return self._hardware_failure

    def _validated_positions(self, action: Any) -> dict[str, float]:
        if not isinstance(action, dict) or set(action) != set(self.action_features):
            raise RuntimeError("Position hold requires the applied position target for every actuator")
        positions = {key: float(value) for key, value in action.items()}
        if not all(math.isfinite(value) for value in positions.values()):
            raise RuntimeError("Position hold requires finite position targets for every actuator")
        return positions

    def configure_position_hold(self) -> None:
        """Enable the explicit, robot-supported position hold; reject mixed control modes."""
        if (
            not self._robot.supports_position_hold
            or not self.action_features
            or any(not key.endswith(".pos") for key in self.action_features)
        ):
            raise ValueError(f"{self.robot_type} has no supported local position-hold contract")
        self._position_hold_enabled = True

    def configure_hold(self) -> None:
        if getattr(self._robot, "supports_command_hold", False):
            if not callable(getattr(self._robot, "hold", None)):
                raise ValueError("Command hold requires a driver hold implementation")
            self._command_hold_enabled = True
        else:
            self.configure_position_hold()

    def reset_world(self, task=None, seed=None) -> None:
        from lerobot.robots.remote.world import get_world

        with self._lock:
            world = get_world(self._robot)
            if world is None:
                raise TypeError("This robot does not expose a simulator world")
            world.reset_world(task, seed)
            self._last_applied_action = None
            self._observed_positions.clear()
            self._observation_time = None

    @property
    def supports_hold(self) -> bool:
        return self._position_hold_enabled or self._command_hold_enabled

    @property
    def supports_command_hold(self) -> bool:
        return bool(getattr(self._robot, "supports_command_hold", False))

    @property
    def supports_position_hold(self) -> bool:
        """Whether this robot declares position-target retention support."""
        return self._robot.supports_position_hold

    def hold(self) -> None:
        """Refresh the last applied position targets without camera reads or inference.

        Driver-returned targets include interpolation and safety clipping. Before
        any command, use the last finite measured pose. The robot may still settle
        toward these targets; this is not an instantaneous physical stop.
        """
        with self._lock:
            if self._command_hold_enabled:
                try:
                    self._robot.hold()
                except Exception as exc:
                    self._record_hardware_failure("hold", exc)
                    raise
                return
            if not self._position_hold_enabled:
                raise RuntimeError("Local hold was not configured for this robot")
            try:
                target = self._last_applied_action
                if target is None:
                    target = self._validated_positions(self._observed_positions)
                applied_action = self._robot.send_action(target.copy())
                self._last_applied_action = self._validated_positions(applied_action)
            except Exception as exc:
                self._record_hardware_failure("hold", exc)
                raise

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
