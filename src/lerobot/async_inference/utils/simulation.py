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

"""Simulation helpers for async inference experiments.

Provides mock robot, drop simulation, and latency spike simulation
for testing and experimentation without real hardware.
"""

import random
import re
import time
from typing import Any

import numpy as np


# =============================================================================
# Mock Robot
# =============================================================================


class MockRobot:
    """Mock robot for simulation experiments (no hardware required)."""

    def __init__(self, action_dim: int = 6, state_dim: int = 6):
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._connected = False
        self._step = 0

    @property
    def action_features(self) -> list[str]:
        return [f"joint_{i}" for i in range(self._action_dim)]

    @property
    def state_features(self) -> list[str]:
        return [f"state_{i}" for i in range(self._state_dim)]

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> dict[str, Any]:
        """Return synthetic observation (random state, placeholder images)."""
        self._step += 1
        obs = {}
        # Random joint state (scalar float values)
        for feat in self.state_features:
            obs[feat] = float(np.random.randn())
        # Placeholder image (small random RGB)
        obs["camera1"] = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> None:
        """No-op for mock robot."""
        pass


# =============================================================================
# Drop Simulator
# =============================================================================


class DropSimulator:
    """Simulates random and burst drops for observations/actions."""

    def __init__(
        self,
        random_drop_p: float = 0.0,
        burst_pattern: str | None = None,
    ):
        self._random_drop_p = random_drop_p
        self._burst_duration_s: float = 0.0
        self._burst_period_s: float = 0.0
        self._start_time: float | None = None

        # Parse burst pattern like "1s@20s" -> drop for 1s every 20s
        if burst_pattern:
            match = re.match(r"(\d+(?:\.\d+)?)\s*s?\s*@\s*(\d+(?:\.\d+)?)\s*s?", burst_pattern)
            if match:
                self._burst_duration_s = float(match.group(1))
                self._burst_period_s = float(match.group(2))

    def should_drop(self) -> bool:
        """Check if the current event should be dropped."""
        now = time.time()
        if self._start_time is None:
            self._start_time = now

        # Check burst drop
        if self._burst_period_s > 0:
            elapsed = now - self._start_time
            time_in_period = elapsed % self._burst_period_s
            if time_in_period < self._burst_duration_s:
                return True

        # Check random drop
        if self._random_drop_p > 0 and random.random() < self._random_drop_p:
            return True

        return False


# =============================================================================
# Spike Delay Simulator
# =============================================================================


class SpikeDelaySimulator:
    """Simulates latency spikes for experiments."""

    def __init__(
        self,
        base_delay_ms: float = 0.0,
        spike_pattern: str | None = None,
    ):
        self._base_delay_s = base_delay_ms / 1000.0
        self._spike_extra_s: float = 0.0
        self._spike_period_s: float = 0.0
        self._spike_duration_s: float = 0.0
        self._start_time: float | None = None

        # Parse spike pattern like "+2000ms@30s/1s" -> +2s spike every 30s lasting 1s
        if spike_pattern:
            match = re.match(
                r"\+?(\d+(?:\.\d+)?)\s*ms?\s*@\s*(\d+(?:\.\d+)?)\s*s?\s*/\s*(\d+(?:\.\d+)?)\s*s?",
                spike_pattern,
            )
            if match:
                self._spike_extra_s = float(match.group(1)) / 1000.0
                self._spike_period_s = float(match.group(2))
                self._spike_duration_s = float(match.group(3))

    def get_delay(self) -> float:
        """Get the current delay in seconds (base + any spike)."""
        now = time.time()
        if self._start_time is None:
            self._start_time = now

        delay = self._base_delay_s

        # Check if in a spike window
        if self._spike_period_s > 0:
            elapsed = now - self._start_time
            time_in_period = elapsed % self._spike_period_s
            if time_in_period < self._spike_duration_s:
                delay += self._spike_extra_s

        return delay

    def apply_delay(self) -> None:
        """Sleep for the current delay amount."""
        delay = self.get_delay()
        if delay > 0:
            time.sleep(delay)
