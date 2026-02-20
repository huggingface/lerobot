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

"""Latency estimation classes for async inference.

Provides abstract base class and implementations for estimating round-trip latency
in the DRTC algorithm.
"""

import math
from abc import ABC, abstractmethod
from collections import deque


class LatencyEstimatorBase(ABC):
    """Abstract base class for latency estimators.

    The estimate_steps property enforces the RTC constraint: d <= H/2
    where d is the inference delay and H is the prediction horizon (action_chunk_size).
    With s = d (maximum soft masking), the constraint d <= H - s becomes d <= H/2.
    """

    def __init__(
        self,
        fps: float,
        action_chunk_size: int | None = None,
        s_min: int = 1,
    ):
        """Initialize the latency estimator.

        Args:
            fps: Control loop frequency for quantizing to action steps.
            action_chunk_size: Prediction horizon H (number of actions per chunk).
                If provided, enables upper bound clamping to H/2.
            s_min: Minimum execution horizon in steps. Used as the pre-measurement
                fallback so estimate_steps returns s_min before any real RTT arrives.
        """
        self._fps = fps
        self._action_chunk_size = action_chunk_size
        self._s_min = s_min

    @property
    def fps(self) -> float:
        return self._fps

    @abstractmethod
    def update(self, measured_rtt: float) -> None:
        """Update the latency estimate with a new RTT measurement."""
        ...

    @property
    @abstractmethod
    def estimate_seconds(self) -> float:
        """Get the latency estimate in seconds."""
        ...

    @property
    def estimate_steps(self) -> int:
        """Get the latency estimate quantized to action steps.

        Upper-bounded by H/2 per RTC constraint: with s = d, d <= H - s becomes d <= H/2.
        This ensures real-time execution is achievable. If the actual delay exceeds
        this bound, the system gracefully degrades to synchronous-with-inpainting behavior.
        """
        raw = max(1, math.ceil(self.estimate_seconds * self._fps))
        if self._action_chunk_size is not None:
            d_max = self._action_chunk_size // 2
            return min(raw, max(1, d_max))
        return raw

    @abstractmethod
    def reset(self) -> None:
        """Reset the estimator state."""
        ...


class JKLatencyEstimator(LatencyEstimatorBase):
    """Jacobson-Karels style latency estimator with exponential smoothing.

    Maintains a smoothed mean and deviation estimate of round-trip latency,
    combining them to produce a conservative estimate that adapts to variance.

    Attributes:
        fps: Control loop frequency for quantizing to action steps.
        alpha: Smoothing factor for mean (default 0.125 per RFC 6298).
        beta: Smoothing factor for deviation (default 0.25 per RFC 6298).
        k: Scaling factor for deviation in estimate (paper suggests K=1 for faster recovery).
    """

    def __init__(
        self,
        fps: float,
        alpha: float = 0.125,
        beta: float = 0.25,
        k: float = 1.0,
        action_chunk_size: int | None = None,
        s_min: int = 1,
    ):
        super().__init__(fps, action_chunk_size, s_min=s_min)
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.smoothed_rtt: float = 0.0
        self.rtt_deviation: float = 0.0
        self._initialized: bool = False

    def update(self, measured_rtt: float) -> None:
        """Update the latency estimate with a new RTT measurement."""
        if not self._initialized:
            self.smoothed_rtt = measured_rtt
            self.rtt_deviation = 0
            self._initialized = True
            return

        error = measured_rtt - self.smoothed_rtt
        self.smoothed_rtt = (1 - self.alpha) * self.smoothed_rtt + self.alpha * measured_rtt
        self.rtt_deviation = (1 - self.beta) * self.rtt_deviation + self.beta * abs(error)

    @property
    def estimate_seconds(self) -> float:
        """Get the latency estimate in seconds: ℓ̂ = ℓ̄ + K·σ"""
        if not self._initialized:
            return self._s_min / self._fps
        return self.smoothed_rtt + self.k * self.rtt_deviation

    def reset(self) -> None:
        """Reset the estimator state."""
        self.smoothed_rtt = 0.0
        self.rtt_deviation = 0.0
        self._initialized = False


class MaxLast10Estimator(LatencyEstimatorBase):
    """Conservative latency estimator using max of last 10 measurements (RTC-style).

    Returns the maximum RTT observed in the last 10 measurements, providing a
    conservative bound that is less adaptive but more stable under spikes.
    """

    def __init__(
        self,
        fps: float,
        window_size: int = 10,
        action_chunk_size: int | None = None,
        s_min: int = 1,
    ):
        super().__init__(fps, action_chunk_size, s_min=s_min)
        self._window_size = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)

    def update(self, measured_rtt: float) -> None:
        """Add a new RTT measurement to the window."""
        self._buffer.append(measured_rtt)

    @property
    def estimate_seconds(self) -> float:
        """Get the latency estimate as max of last N measurements."""
        if not self._buffer:
            return self._s_min / self._fps
        return max(self._buffer)

    def reset(self) -> None:
        """Reset the estimator state."""
        self._buffer.clear()


class FixedLatencyEstimator(LatencyEstimatorBase):
    """Fixed latency estimator for baseline comparisons (SmolVLA-style).

    Returns a fixed, user-specified latency estimate regardless of measurements.
    This represents the behavior of systems that assume a constant network latency
    rather than adapting to actual conditions.

    Note: The estimate is still quantized to at least 1 action step, and
    upper-bounded by H/2 if action_chunk_size is provided.
    """

    def __init__(
        self,
        fps: float,
        fixed_latency_s: float = 0.1,
        action_chunk_size: int | None = None,
        s_min: int = 1,
    ):
        """Initialize with a fixed latency value.

        Args:
            fps: Control loop frequency.
            fixed_latency_s: Fixed latency estimate in seconds (default 100ms).
            action_chunk_size: Prediction horizon H for upper bound clamping to H/2.
            s_min: Minimum execution horizon (unused by fixed estimator, passed to base).
        """
        super().__init__(fps, action_chunk_size, s_min=s_min)
        self._fixed_latency_s = fixed_latency_s

    def update(self, measured_rtt: float) -> None:
        """No-op: fixed estimator ignores measurements."""
        pass

    @property
    def estimate_seconds(self) -> float:
        """Get the fixed latency estimate in seconds."""
        return self._fixed_latency_s

    def reset(self) -> None:
        """No-op: fixed estimator has no state to reset."""
        pass


# Backwards compatibility alias
LatencyEstimator = JKLatencyEstimator


def make_latency_estimator(
    kind: str,
    fps: float,
    alpha: float = 0.125,
    beta: float = 0.25,
    k: float = 1.0,
    fixed_latency_s: float = 0.1,
    action_chunk_size: int | None = None,
    s_min: int = 1,
) -> LatencyEstimatorBase:
    """Factory function to create a latency estimator.

    Args:
        kind: Type of estimator:
            - "jk": Jacobson-Karels (adaptive, fast recovery)
            - "max_last_10": Max of last 10 (conservative, RTC-style)
            - "fixed": Fixed latency (non-adaptive baseline)
        fps: Control loop frequency.
        alpha: JK smoothing factor for mean.
        beta: JK smoothing factor for deviation.
        k: JK scaling factor for deviation.
        fixed_latency_s: Fixed latency in seconds (only used if kind="fixed").
        action_chunk_size: Prediction horizon H. If provided, enables upper bound
            clamping to H/2 per RTC constraint (with s = d, d <= H - s becomes d <= H/2).
        s_min: Minimum execution horizon in steps. Used as the pre-measurement
            fallback so estimate_steps returns s_min before any real RTT arrives.

    Returns:
        A latency estimator instance.
    """
    if kind == "jk":
        return JKLatencyEstimator(
            fps=fps,
            alpha=alpha,
            beta=beta,
            k=k,
            action_chunk_size=action_chunk_size,
            s_min=s_min,
        )
    elif kind == "max_last_10":
        return MaxLast10Estimator(
            fps=fps,
            action_chunk_size=action_chunk_size,
            s_min=s_min,
        )
    elif kind == "fixed":
        return FixedLatencyEstimator(
            fps=fps,
            fixed_latency_s=fixed_latency_s,
            action_chunk_size=action_chunk_size,
            s_min=s_min,
        )
    else:
        raise ValueError(f"Unknown latency estimator type: {kind}. Use 'jk', 'max_last_10', or 'fixed'.")
