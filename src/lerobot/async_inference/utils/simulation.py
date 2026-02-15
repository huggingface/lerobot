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
                print("DROP OBSERVATION TRUE")
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


# =============================================================================
# Duplicate Simulator
# =============================================================================


@dataclass
class DuplicateEvent:
    """A single duplicate event with start time and duration.

    Attributes:
        start_s: When to start duplicating (seconds from experiment start)
        duration_s: How long to duplicate (seconds)
    """

    start_s: float  # When to start duplicating (seconds from start)
    duration_s: float  # How long to duplicate (seconds)


@dataclass
class DuplicateConfig:
    """Configuration for duplicate injection using explicit time-based events.

    Example usage:
        # Duplicate for 1 second starting at 5s into the experiment
        config = DuplicateConfig(duplicates=[
            DuplicateEvent(start_s=5.0, duration_s=1.0),
        ])

        # Or from dicts (for JSON/CLI compatibility)
        config = DuplicateConfig.from_dicts([
            {"start_s": 5.0, "duration_s": 1.0},
        ])
    """

    duplicates: list[DuplicateEvent] = field(default_factory=list)

    @classmethod
    def from_dicts(cls, dup_dicts: list[dict]) -> "DuplicateConfig":
        """Create config from list of dicts (for JSON/CLI compatibility)."""
        duplicates = [DuplicateEvent(start_s=d["start_s"], duration_s=d["duration_s"]) for d in dup_dicts]
        return cls(duplicates=duplicates)


class DuplicateSimulator:
    """Simulates explicit time-based duplicates for observations/actions.

    During a duplicate window, ``should_duplicate()`` returns True, causing
    the caller to send/handle the same message a second time.  The
    server-side LWW registers and schedule merge absorb the duplicate
    (same ``control_step`` / ``src_control_step``).

    Example:
        config = DuplicateConfig(duplicates=[
            DuplicateEvent(start_s=5.0, duration_s=1.0),
        ])
        sim = DuplicateSimulator(config=config)

        # Or from dicts
        sim = DuplicateSimulator.from_dicts([
            {"start_s": 5.0, "duration_s": 1.0},
        ])
    """

    def __init__(self, config: DuplicateConfig | None = None):
        self._duplicates: list[DuplicateEvent] = config.duplicates if config else []
        self._start_time: float | None = None

    @classmethod
    def from_dicts(cls, dup_dicts: list[dict]) -> "DuplicateSimulator":
        """Create simulator from list of dicts (for JSON/CLI compatibility)."""
        config = DuplicateConfig.from_dicts(dup_dicts)
        return cls(config=config)

    def should_duplicate(self) -> bool:
        """Check if the current message should be duplicated.

        Returns True if current time falls within any duplicate event window.
        """
        if not self._duplicates:
            return False

        now = time.time()
        if self._start_time is None:
            self._start_time = now

        elapsed = now - self._start_time

        for dup in self._duplicates:
            if dup.start_s <= elapsed < dup.start_s + dup.duration_s:
                return True

        return False

    def reset(self) -> None:
        """Reset the simulator start time."""
        self._start_time = None


# =============================================================================
# Reorder Simulator (Hold-and-Swap)
# =============================================================================


@dataclass
class ReorderEvent:
    """A single reorder event with start time and duration.

    During the window the simulator holds one message and lets the next
    pass through, then releases the held message -- creating a single
    pairwise swap.

    Attributes:
        start_s: When to start reordering (seconds from experiment start)
        duration_s: How long the reorder window lasts (seconds)
    """

    start_s: float  # When to start (seconds from start)
    duration_s: float  # Window duration (seconds)


@dataclass
class ReorderConfig:
    """Configuration for reorder injection using explicit time-based events.

    Example usage:
        config = ReorderConfig(reorders=[
            ReorderEvent(start_s=5.0, duration_s=2.0),
        ])

        # Or from dicts (for JSON/CLI compatibility)
        config = ReorderConfig.from_dicts([
            {"start_s": 5.0, "duration_s": 2.0},
        ])
    """

    reorders: list[ReorderEvent] = field(default_factory=list)

    @classmethod
    def from_dicts(cls, reorder_dicts: list[dict]) -> "ReorderConfig":
        """Create config from list of dicts (for JSON/CLI compatibility)."""
        reorders = [ReorderEvent(start_s=d["start_s"], duration_s=d["duration_s"]) for d in reorder_dicts]
        return cls(reorders=reorders)


class ReorderSimulator:
    """Simulates pairwise message reordering (hold-and-swap).

    Design (Option A from the plan):
    - Outside any reorder window: ``process()`` passes items straight through.
    - When a reorder window opens: the *first* item is held, and the *second*
      item passes through immediately.  On the *third* call (or any call
      after the window closes) the held item is released ahead of the new one,
      completing the swap.

    This creates a single pairwise swap per window -- the simplest reordering
    that still exercises the LWW join.

    The caller wraps the send/handle path:

        items = self._reorder_sim.process(item)
        for i in items:
            send(i)   # may yield 0, 1, or 2 items

    Example:
        config = ReorderConfig(reorders=[
            ReorderEvent(start_s=5.0, duration_s=2.0),
        ])
        sim = ReorderSimulator(config=config)
    """

    def __init__(self, config: ReorderConfig | None = None):
        self._reorders: list[ReorderEvent] = config.reorders if config else []
        self._start_time: float | None = None
        # Hold buffer: at most one item held while waiting for the swap partner
        self._held: Any = None
        self._holding: bool = False

    @classmethod
    def from_dicts(cls, reorder_dicts: list[dict]) -> "ReorderSimulator":
        """Create simulator from list of dicts (for JSON/CLI compatibility)."""
        config = ReorderConfig.from_dicts(reorder_dicts)
        return cls(config=config)

    def _in_reorder_window(self) -> bool:
        """Check if current time falls within any reorder window."""
        if not self._reorders:
            return False

        now = time.time()
        if self._start_time is None:
            self._start_time = now

        elapsed = now - self._start_time

        for reorder in self._reorders:
            if reorder.start_s <= elapsed < reorder.start_s + reorder.duration_s:
                return True

        return False

    def process(self, item: Any) -> list:
        """Process an item through the hold-and-swap reorder logic.

        Returns a list of 0, 1, or 2 items to send/handle in order:
        - 0 items: the item is being held (first item in a swap)
        - 1 item: normal pass-through, or the fresh item when the window
          closes while holding (the stale held item is dropped)
        - 2 items: the swap partner followed by the previously held item
                    (completing the swap within the window)
        """
        if not self._reorders:
            return [item]

        in_window = self._in_reorder_window()

        if self._holding:
            # We're holding a message -- release it
            held = self._held
            self._held = None
            self._holding = False

            if in_window:
                # Still in window: complete the swap -- new item first, held second
                return [item, held]
            else:
                # Window closed while holding: drop the stale held item,
                # pass through only the fresh one. Sending both causes
                # server-side inference queuing (the held item was never
                # sent, so the server's LWW accepts it, and the inference
                # producer processes it before the fresh obs).
                return [item]

        if in_window:
            # Enter hold state: hold this item, return nothing
            self._held = item
            self._holding = True
            return []

        # Outside window, nothing held: pass through
        return [item]

    def reset(self) -> None:
        """Reset the simulator state."""
        self._start_time = None
        self._held = None
        self._holding = False


# =============================================================================
# Disconnect Simulator
# =============================================================================


@dataclass
class DisconnectEvent:
    """A single network disconnect event with start time and duration.

    Attributes:
        start_s: When to start the disconnect (seconds from experiment start)
        duration_s: How long the disconnect lasts (seconds)
    """

    start_s: float  # When to start (seconds from start)
    duration_s: float  # How long the disconnect lasts (seconds)


@dataclass
class DisconnectConfig:
    """Configuration for disconnect injection using explicit time-based events.

    A disconnect blocks *both* the observation sender and action receiver
    threads for the configured duration, simulating a full network outage.

    Example usage:
        config = DisconnectConfig(disconnects=[
            DisconnectEvent(start_s=5.0, duration_s=3.0),
        ])

        # Or from dicts (for JSON/CLI compatibility)
        config = DisconnectConfig.from_dicts([
            {"start_s": 5.0, "duration_s": 3.0},
        ])
    """

    disconnects: list[DisconnectEvent] = field(default_factory=list)

    @classmethod
    def from_dicts(cls, disconnect_dicts: list[dict]) -> "DisconnectConfig":
        """Create config from list of dicts (for JSON/CLI compatibility)."""
        disconnects = [
            DisconnectEvent(start_s=d["start_s"], duration_s=d["duration_s"])
            for d in disconnect_dicts
        ]
        return cls(disconnects=disconnects)


class DisconnectSimulator:
    """Simulates network disconnects by blocking caller threads.

    Both the observation sender and action receiver should call
    ``wait_if_disconnected()`` on each iteration.  If the current time
    falls inside a disconnect window the call **sleeps** until the window
    ends and returns the sleep duration so the caller can record a sim
    event.  Outside any window it returns immediately with 0.

    Example:
        config = DisconnectConfig(disconnects=[
            DisconnectEvent(start_s=5.0, duration_s=3.0),
        ])
        sim = DisconnectSimulator(config=config)

        # In a sender/receiver loop:
        slept = sim.wait_if_disconnected()
        if slept > 0:
            record_sim_event("disconnect")
            continue
    """

    def __init__(self, config: DisconnectConfig | None = None):
        self._disconnects: list[DisconnectEvent] = config.disconnects if config else []
        self._start_time: float | None = None

    @classmethod
    def from_dicts(cls, disconnect_dicts: list[dict]) -> "DisconnectSimulator":
        """Create simulator from list of dicts (for JSON/CLI compatibility)."""
        config = DisconnectConfig.from_dicts(disconnect_dicts)
        return cls(config=config)

    def _active_window_end(self) -> float | None:
        """Return the end time (absolute) of the active disconnect window, or None."""
        if not self._disconnects:
            return None

        now = time.time()
        if self._start_time is None:
            self._start_time = now

        elapsed = now - self._start_time

        for dc in self._disconnects:
            if dc.start_s <= elapsed < dc.start_s + dc.duration_s:
                # Return absolute wall-clock time when this window ends
                return self._start_time + dc.start_s + dc.duration_s

        return None

    def is_disconnected(self) -> bool:
        """Check if the network is currently disconnected."""
        return self._active_window_end() is not None

    def wait_if_disconnected(self) -> float:
        """Block until the current disconnect window ends.

        Returns:
            The number of seconds slept (0 if not disconnected).
        """
        window_end = self._active_window_end()
        if window_end is None:
            return 0.0

        sleep_s = max(0.0, window_end - time.time())
        if sleep_s > 0:
            time.sleep(sleep_s)
        return sleep_s

    def reset(self) -> None:
        """Reset the simulator start time."""
        self._start_time = None
