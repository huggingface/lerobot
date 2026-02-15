"""Action filters for reducing jitter and smoothing robot control signals.

This module provides a class-based hierarchy of action filters that can be
applied to robot control signals to reduce high-frequency noise and jitter
from policy micro-updates without significantly impacting intentional motion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

@dataclass
class FilterContext:
    """Context passed to filters each tick.

    Attributes:
        action: The current action to filter.
    """

    action: np.ndarray


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

class ButterworthFilter(ActionFilter):
    """Butterworth low-pass filter for action smoothing.

    Provides frequency-selective filtering to attenuate high-frequency noise
    while passing intentional low-frequency motion with minimal phase lag.
    """

    def __init__(
        self,
        cutoff: float,
        order: int,
        fps: float,
        gain: float,
        past_buffer_size: int,
    ):
        """Initialize the Butterworth filter.

        Args:
            cutoff: Cutoff frequency in Hz.
            order: Filter order (1-4).
            fps: Control loop frequency in Hz.
            gain: Amplitude gain compensation factor.
            past_buffer_size: Number of past actions to keep in buffer.
        """
        self.cutoff = cutoff
        self.order = order
        self.fps = fps
        self.gain = gain
        self.past_buffer_size = past_buffer_size
        self._sos: np.ndarray | None = None
        self._zi: np.ndarray | None = None
        self._prev: np.ndarray | None = None

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

        # Apply causal (stateful) filter for consistent smoothing
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
