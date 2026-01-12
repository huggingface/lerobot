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
class DropEvent:
    """A single drop event with start time and duration.

    Attributes:
        start_s: When to start dropping (seconds from experiment start)
        duration_s: How long to drop (seconds)
    """

    start_s: float  # When to start dropping (seconds from start)
    duration_s: float  # How long to drop (seconds)


@dataclass
class DropConfig:
    """Configuration for drop injection using explicit time-based events.

    Example usage:
        # Drop for 1 second starting at 5s into the experiment
        config = DropConfig(drops=[
            DropEvent(start_s=5.0, duration_s=1.0),
        ])

        # Multiple drop events
        config = DropConfig(drops=[
            DropEvent(start_s=5.0, duration_s=1.0),
            DropEvent(start_s=15.0, duration_s=2.0),
        ])

        # Or from dicts (for JSON/CLI compatibility)
        config = DropConfig.from_dicts([
            {"start_s": 5.0, "duration_s": 1.0},
            {"start_s": 15.0, "duration_s": 2.0},
        ])
    """

    drops: list[DropEvent] = field(default_factory=list)

    @classmethod
    def from_dicts(cls, drop_dicts: list[dict]) -> "DropConfig":
        """Create config from list of dicts (for JSON/CLI compatibility)."""
        drops = [DropEvent(start_s=d["start_s"], duration_s=d["duration_s"]) for d in drop_dicts]
        return cls(drops=drops)


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
    """Simulates explicit time-based drops for observations/actions.

    Drops occur during specified time windows relative to experiment start.

    Example:
        # Using DropConfig (preferred)
        config = DropConfig(drops=[
            DropEvent(start_s=5.0, duration_s=1.0),
            DropEvent(start_s=15.0, duration_s=2.0),
        ])
        sim = DropSimulator(config=config)

        # Or from dicts
        sim = DropSimulator.from_dicts([
            {"start_s": 5.0, "duration_s": 1.0},
            {"start_s": 15.0, "duration_s": 2.0},
        ])
    """

    def __init__(self, config: DropConfig | None = None):
        self._drops: list[DropEvent] = config.drops if config else []
        self._start_time: float | None = None

    @classmethod
    def from_dicts(cls, drop_dicts: list[dict]) -> "DropSimulator":
        """Create simulator from list of dicts (for JSON/CLI compatibility)."""
        config = DropConfig.from_dicts(drop_dicts)
        return cls(config=config)

    def should_drop(self) -> bool:
        """Check if the current event should be dropped.

        Returns True if current time falls within any drop event window.
        """
        if not self._drops:
            return False

        now = time.time()
        if self._start_time is None:
            self._start_time = now

        elapsed = now - self._start_time

        # Check if we're within any drop window
        for drop in self._drops:
            if drop.start_s <= elapsed < drop.start_s + drop.duration_s:
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
