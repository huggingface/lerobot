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
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class SpikeEvent:
    """A single spike event that fires once at a specific time.

    Attributes:
        start_s: When to trigger the spike (seconds from experiment start)
        delay_ms: How much delay to add when triggered (milliseconds)
    """

    start_s: float  # When to trigger (seconds from start)
    delay_ms: float  # How much delay to add (milliseconds)


@dataclass
class SpikeDelayConfig:
    """Configuration for explicit spike injection.

    Example usage:
        # Add 2s spike at 5s and 1s spike at 15s into the experiment
        config = SpikeDelayConfig(
            spikes=[
                SpikeEvent(start_s=5.0, delay_ms=2000),
                SpikeEvent(start_s=15.0, delay_ms=1000),
            ]
        )

        # Or from dicts (for JSON/CLI compatibility)
        config = SpikeDelayConfig.from_dicts([
            {"start_s": 5.0, "delay_ms": 2000},
            {"start_s": 15.0, "delay_ms": 1000},
        ])
    """

    spikes: list[SpikeEvent] = field(default_factory=list)

    @classmethod
    def from_dicts(cls, spike_dicts: list[dict]) -> "SpikeDelayConfig":
        """Create config from list of dicts (for JSON/CLI compatibility)."""
        spikes = [SpikeEvent(start_s=d["start_s"], delay_ms=d["delay_ms"]) for d in spike_dicts]
        return cls(spikes=spikes)


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
    """Simulates explicit latency spike events for experiments.

    Each spike fires once when the elapsed time crosses its start_s threshold,
    adding the specified delay_ms of latency.

    Example:
        config = SpikeDelayConfig(spikes=[
            SpikeEvent(start_s=5.0, delay_ms=2000),
            SpikeEvent(start_s=15.0, delay_ms=1000),
        ])
        sim = SpikeDelaySimulator(config=config)

        # Or from dicts
        sim = SpikeDelaySimulator.from_dicts([
            {"start_s": 5.0, "delay_ms": 2000},
            {"start_s": 15.0, "delay_ms": 1000},
        ])
    """

    def __init__(self, config: SpikeDelayConfig | None = None):
        self._spikes: list[SpikeEvent] = config.spikes if config else []
        self._fired: set[int] = set()  # Track which spike indices have fired
        self._start_time: float | None = None

    @classmethod
    def from_dicts(cls, spike_dicts: list[dict]) -> "SpikeDelaySimulator":
        """Create simulator from list of dicts (for JSON/CLI compatibility)."""
        config = SpikeDelayConfig.from_dicts(spike_dicts)
        return cls(config=config)

    def get_delay(self) -> float:
        """Get delay if a spike should fire now, else 0.

        Each spike fires exactly once when elapsed time crosses its start_s.
        Returns the delay in seconds.
        """
        if not self._spikes:
            return 0.0

        now = time.time()
        if self._start_time is None:
            self._start_time = now

        elapsed = now - self._start_time

        # Check each spike - fire once when elapsed >= start_s
        for i, spike in enumerate(self._spikes):
            if i not in self._fired and elapsed >= spike.start_s:
                self._fired.add(i)
                return spike.delay_ms / 1000.0

        return 0.0

    def apply_delay(self) -> None:
        """Sleep for any pending spike delay."""
        delay = self.get_delay()
        if delay > 0:
            time.sleep(delay)

    def reset(self) -> None:
        """Reset the simulator (clear start time and fired spikes)."""
        self._start_time = None
        self._fired.clear()

    def pending_spikes(self) -> int:
        """Return count of spikes that haven't fired yet."""
        return len(self._spikes) - len(self._fired)
