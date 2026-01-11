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
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class SpikeDelayConfig:
    """Configuration for latency spike injection.

    Example usage:
        # Add 2 second spike every 30 seconds, lasting 1 second
        config = SpikeDelayConfig(
            base_delay_ms=100.0,
            spike_delay_ms=2000.0,
            spike_period_s=30.0,
            spike_duration_s=1.0,
        )
    """

    base_delay_ms: float = 0.0  # Base delay in milliseconds
    spike_delay_ms: float = 0.0  # Additional delay during spike (ms)
    spike_period_s: float = 0.0  # Time between spikes (0 = disabled)
    spike_duration_s: float = 0.0  # How long each spike lasts (seconds)


@dataclass
class DropConfig:
    """Configuration for drop injection.

    Example usage:
        # Drop for 1 second every 20 seconds
        config = DropConfig(
            burst_period_s=20.0,
            burst_duration_s=1.0,
        )

        # Random 5% drop rate
        config = DropConfig(random_drop_p=0.05)
    """

    random_drop_p: float = 0.0  # Random drop probability (0.0-1.0)
    burst_period_s: float = 0.0  # Time between burst drops (0 = disabled)
    burst_duration_s: float = 0.0  # How long each burst lasts (seconds)


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
    """Simulates random and burst drops for observations/actions.

    Can be initialized with either a DropConfig dataclass (preferred)
    or individual parameters for backward compatibility.

    Example:
        # Using DropConfig (preferred)
        config = DropConfig(burst_period_s=20.0, burst_duration_s=1.0)
        sim = DropSimulator(config=config)

        # Using individual parameters (backward compatible)
        sim = DropSimulator(random_drop_p=0.05)
    """

    def __init__(
        self,
        config: DropConfig | None = None,
        *,
        random_drop_p: float = 0.0,
        burst_period_s: float = 0.0,
        burst_duration_s: float = 0.0,
    ):
        # Use config if provided, otherwise use individual parameters
        if config is not None:
            self._random_drop_p = config.random_drop_p
            self._burst_period_s = config.burst_period_s
            self._burst_duration_s = config.burst_duration_s
        else:
            self._random_drop_p = random_drop_p
            self._burst_period_s = burst_period_s
            self._burst_duration_s = burst_duration_s

        self._start_time: float | None = None

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

    def reset(self) -> None:
        """Reset the simulator start time."""
        self._start_time = None


# =============================================================================
# Spike Delay Simulator
# =============================================================================


class SpikeDelaySimulator:
    """Simulates latency spikes for experiments.

    Can be initialized with either a SpikeDelayConfig dataclass (preferred)
    or individual parameters for backward compatibility.

    Example:
        # Using SpikeDelayConfig (preferred)
        config = SpikeDelayConfig(
            base_delay_ms=100.0,
            spike_delay_ms=2000.0,
            spike_period_s=30.0,
            spike_duration_s=1.0,
        )
        sim = SpikeDelaySimulator(config=config)

        # Using individual parameters (backward compatible)
        sim = SpikeDelaySimulator(base_delay_ms=100.0)
    """

    def __init__(
        self,
        config: SpikeDelayConfig | None = None,
        *,
        base_delay_ms: float = 0.0,
        spike_delay_ms: float = 0.0,
        spike_period_s: float = 0.0,
        spike_duration_s: float = 0.0,
    ):
        # Use config if provided, otherwise use individual parameters
        if config is not None:
            self._base_delay_s = config.base_delay_ms / 1000.0
            self._spike_extra_s = config.spike_delay_ms / 1000.0
            self._spike_period_s = config.spike_period_s
            self._spike_duration_s = config.spike_duration_s
        else:
            self._base_delay_s = base_delay_ms / 1000.0
            self._spike_extra_s = spike_delay_ms / 1000.0
            self._spike_period_s = spike_period_s
            self._spike_duration_s = spike_duration_s

        self._start_time: float | None = None

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

    def reset(self) -> None:
        """Reset the simulator start time."""
        self._start_time = None

    def is_in_spike(self) -> bool:
        """Check if currently in a spike window (useful for diagnostics)."""
        if self._spike_period_s <= 0:
            return False
        now = time.time()
        if self._start_time is None:
            return False
        elapsed = now - self._start_time
        time_in_period = elapsed % self._spike_period_s
        return time_in_period < self._spike_duration_s
