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

"""Action filters for reducing jitter and smoothing robot control signals.

This module provides a class-based hierarchy of action filters that can be
applied to robot control signals to reduce high-frequency noise and jitter
from policy micro-updates without significantly impacting intentional motion.

Filters support optional frozen lookahead, using scheduled actions that are
guaranteed not to change (within the latency window) for improved filtering.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


@dataclass
class FilterContext:
    """Context passed to filters each tick.

    Attributes:
        action: The current action to filter.
        frozen_actions: Actions within the latency window that are guaranteed
            not to be overwritten. Can be used for lookahead filtering.
    """

    action: np.ndarray
    frozen_actions: list[np.ndarray] = field(default_factory=list)


class ActionFilter(ABC):
    """Base class for action filters.

    All filter implementations should inherit from this class and implement
    the apply() method.
    """

    @abstractmethod
    def apply(self, ctx: FilterContext) -> np.ndarray:
        """Apply filter and return filtered action.

        Args:
            ctx: Filter context containing current action and optional
                frozen lookahead actions.

        Returns:
            The filtered action array.
        """
        pass

    def reset(self) -> None:
        """Reset filter state (optional override)."""
        pass


class NoFilter(ActionFilter):
    """Pass-through filter that returns actions unchanged."""

    def apply(self, ctx: FilterContext) -> np.ndarray:
        return ctx.action


class AdaptiveLowpassFilter(ActionFilter):
    """IIR low-pass filter with adaptive alpha based on delta magnitude.

    Uses a high alpha (fast response) for large action deltas and a low alpha
    (heavy smoothing) for small deltas, reducing jitter while maintaining
    responsiveness for intentional motion.
    """

    def __init__(self, alpha_min: float, alpha_max: float, deadband: float):
        """Initialize the adaptive lowpass filter.

        Args:
            alpha_min: Filter alpha for small deltas (heavy smoothing).
            alpha_max: Filter alpha for large deltas (fast response).
            deadband: Threshold for switching between alpha values.
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.deadband = deadband
        self._prev: np.ndarray | None = None

    def apply(self, ctx: FilterContext) -> np.ndarray:
        if self._prev is None:
            self._prev = ctx.action.copy()
            return ctx.action

        delta = float(np.linalg.norm(ctx.action - self._prev))
        alpha = self.alpha_max if delta > self.deadband else self.alpha_min

        filtered = alpha * ctx.action + (1.0 - alpha) * self._prev
        self._prev = filtered.copy()
        return filtered

    def reset(self) -> None:
        self._prev = None


class HoldStableFilter(ActionFilter):
    """Filter that holds the previous action when delta is below threshold.

    Completely eliminates micro-jitter by ignoring small changes, at the cost
    of slight motion quantization.
    """

    def __init__(self, deadband: float):
        """Initialize the hold-stable filter.

        Args:
            deadband: Delta threshold below which actions are held.
        """
        self.deadband = deadband
        self._prev: np.ndarray | None = None

    def apply(self, ctx: FilterContext) -> np.ndarray:
        if self._prev is None:
            self._prev = ctx.action.copy()
            return ctx.action

        delta = float(np.linalg.norm(ctx.action - self._prev))
        if delta <= self.deadband:
            return self._prev.copy()

        self._prev = ctx.action.copy()
        return ctx.action

    def reset(self) -> None:
        self._prev = None


class ButterworthFilter(ActionFilter):
    """Butterworth low-pass filter with optional frozen lookahead.

    Provides frequency-selective filtering to attenuate high-frequency noise
    while passing intentional low-frequency motion with minimal phase lag.

    When lookahead is enabled and frozen actions are available, the filter
    uses a batch filtering approach for better phase response.
    """

    def __init__(
        self,
        cutoff: float,
        order: int,
        fps: float,
        gain: float,
        use_lookahead: bool,
        past_buffer_size: int,
    ):
        """Initialize the Butterworth filter.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order (1-4).
            fps: Control loop frequency in Hz.
            gain: Amplitude gain compensation factor.
            use_lookahead: Whether to use frozen actions for lookahead.
            past_buffer_size: Number of past actions to keep in buffer.
        """
        self.cutoff = cutoff
        self.order = order
        self.fps = fps
        self.gain = gain
        self.use_lookahead = use_lookahead
        self.past_buffer_size = past_buffer_size
        self._sos: np.ndarray | None = None
        self._zi: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._buffer: list[np.ndarray] = []

    def _init_filter(self, action: np.ndarray) -> None:
        """Initialize filter coefficients and state."""
        nyquist = self.fps / 2.0
        normalized = min(max(self.cutoff / nyquist, 0.01), 0.99)
        self._sos = butter(self.order, normalized, btype="low", output="sos")
        zi_single = sosfilt_zi(self._sos)
        self._zi = np.array([zi_single * action[j] for j in range(len(action))])

    def apply(self, ctx: FilterContext) -> np.ndarray:
        if self._sos is None:
            self._init_filter(ctx.action)
            self._prev = ctx.action.copy()
            return ctx.action

        # Add to history buffer
        self._buffer.append(ctx.action.copy())
        if len(self._buffer) > self.past_buffer_size:
            self._buffer.pop(0)

        if self.use_lookahead and ctx.frozen_actions:
            # Use past buffer + frozen future for batch filtering
            batch = list(self._buffer) + ctx.frozen_actions
            stacked = np.stack(batch, axis=0)

            # Apply filter per joint (batch mode, no state)
            filtered_batch = np.zeros_like(stacked)
            for j in range(stacked.shape[1]):
                filtered_batch[:, j] = sosfilt(self._sos, stacked[:, j])

            # Output at current position (end of buffer, before frozen)
            current_idx = len(self._buffer) - 1
            filtered = filtered_batch[current_idx]
        else:
            # Causal filter with state
            filtered = np.zeros_like(ctx.action)
            for j in range(len(ctx.action)):
                out, self._zi[j] = sosfilt(self._sos, [ctx.action[j]], zi=self._zi[j])
                filtered[j] = out[0]

        # Apply gain compensation
        if self.gain != 1.0 and self._prev is not None:
            delta = filtered - self._prev
            filtered = self._prev + delta * self.gain

        self._prev = filtered.copy()
        return filtered

    def reset(self) -> None:
        self._sos = None
        self._zi = None
        self._prev = None
        self._buffer = []


class MedianFilter(ActionFilter):
    """Median filter with optional frozen lookahead.

    Non-linear filter that replaces each action with the median of its
    surrounding window. Excellent at removing impulse noise (spikes) while
    preserving edges and step changes, with no amplitude attenuation.

    When lookahead is enabled, the window includes both past actions from
    the buffer and frozen future actions from the schedule.
    """

    def __init__(self, past_buffer_size: int, use_lookahead: bool):
        """Initialize the median filter.

        Args:
            past_buffer_size: Number of past actions to keep in buffer.
            use_lookahead: Whether to use frozen actions for lookahead.
        """
        self.past_buffer_size = past_buffer_size
        self.use_lookahead = use_lookahead
        self._buffer: list[np.ndarray] = []

    def apply(self, ctx: FilterContext) -> np.ndarray:
        # Add current to buffer
        self._buffer.append(ctx.action.copy())
        if len(self._buffer) > self.past_buffer_size:
            self._buffer.pop(0)

        # Build window: past buffer + frozen future (if enabled)
        if self.use_lookahead and ctx.frozen_actions:
            window = list(self._buffer) + ctx.frozen_actions
        else:
            window = list(self._buffer)

        stacked = np.stack(window, axis=0)
        return np.median(stacked, axis=0)

    def reset(self) -> None:
        self._buffer = []
