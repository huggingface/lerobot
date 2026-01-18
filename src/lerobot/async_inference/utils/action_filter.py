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
    """Butterworth low-pass filter with optional frozen lookahead guidance.

    Provides frequency-selective filtering to attenuate high-frequency noise
    while passing intentional low-frequency motion with minimal phase lag.

    When lookahead is enabled and frozen actions are available, the filter
    uses a guidance-based approach: apply causal filtering first, then blend
    the output toward the mean of frozen actions. This avoids compounding
    smoothing effects from re-filtering the lookahead window.
    """

    def __init__(
        self,
        cutoff: float,
        order: int,
        fps: float,
        gain: float,
        use_lookahead: bool,
        past_buffer_size: int,
        lookahead_blend: float = 0.3,
    ):
        """Initialize the Butterworth filter.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order (1-4).
            fps: Control loop frequency in Hz.
            gain: Amplitude gain compensation factor.
            use_lookahead: Whether to use frozen actions for lookahead guidance.
            past_buffer_size: Number of past actions to keep in buffer.
            lookahead_blend: Blend weight toward frozen action mean (0-1).
                0 = ignore lookahead, 1 = fully trust lookahead mean.
        """
        self.cutoff = cutoff
        self.order = order
        self.fps = fps
        self.gain = gain
        self.use_lookahead = use_lookahead
        self.past_buffer_size = past_buffer_size
        self.lookahead_blend = lookahead_blend
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

        # Always apply causal (stateful) filter first for consistent smoothing
        filtered = np.zeros_like(ctx.action)
        for j in range(len(ctx.action)):
            out, self._zi[j] = sosfilt(self._sos, [ctx.action[j]], zi=self._zi[j])
            filtered[j] = out[0]

        # Apply gain compensation
        if self.gain != 1.0 and self._prev is not None:
            delta = filtered - self._prev
            filtered = self._prev + delta * self.gain

        # Lookahead guidance: blend toward mean of frozen actions
        if self.use_lookahead and ctx.frozen_actions and self.lookahead_blend > 0:
            # Compute guidance target as mean of frozen actions
            stacked = np.stack(ctx.frozen_actions, axis=0)
            target = np.mean(stacked, axis=0)
            # Blend filtered output toward target
            filtered = filtered + self.lookahead_blend * (target - filtered)

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
