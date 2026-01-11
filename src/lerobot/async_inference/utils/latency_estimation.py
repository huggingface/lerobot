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
in the latency-adaptive async inference algorithm.
"""

import math
from abc import ABC, abstractmethod
from collections import deque


class LatencyEstimatorBase(ABC):
    """Abstract base class for latency estimators."""

    def __init__(self, fps: float):
        self._fps = fps

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
        """Get the latency estimate quantized to action steps."""
        return max(1, math.ceil(self.estimate_seconds * self._fps))

    @abstractmethod
    def reset(self) -> None:
        """Reset the estimator state."""
        ...

    @abstractmethod
    def prime(self, samples: list[float]) -> None:
        """Prime the estimator with initial RTT samples.

        This reduces uncertainty at startup by providing real measurements
        before the main control loop begins.

        Args:
            samples: List of RTT measurements in seconds.
        """
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
    ):
        super().__init__(fps)
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
            self.rtt_deviation = measured_rtt / 2.0
            self._initialized = True
            return

        error = measured_rtt - self.smoothed_rtt
        self.smoothed_rtt = (1 - self.alpha) * self.smoothed_rtt + self.alpha * measured_rtt
        self.rtt_deviation = (1 - self.beta) * self.rtt_deviation + self.beta * abs(error)

    @property
    def estimate_seconds(self) -> float:
        """Get the latency estimate in seconds: ℓ̂ = ℓ̄ + K·σ"""
        if not self._initialized:
            return 1.0 / self._fps * 5  # 5 action steps as initial guess
        return self.smoothed_rtt + self.k * self.rtt_deviation

    def reset(self) -> None:
        """Reset the estimator state."""
        self.smoothed_rtt = 0.0
        self.rtt_deviation = 0.0
        self._initialized = False

    def prime(self, samples: list[float]) -> None:
        """Prime the estimator with initial RTT samples.

        Computes the mean and standard deviation of samples to initialize
        the estimator with lower uncertainty than the default first-sample
        initialization.
        """
        if not samples:
            return

        n = len(samples)
        mean_rtt = sum(samples) / n

        if n >= 2:
            # Compute sample standard deviation
            variance = sum((s - mean_rtt) ** 2 for s in samples) / (n - 1)
            std_rtt = math.sqrt(variance)
        else:
            # Single sample: use a conservative deviation estimate
            std_rtt = mean_rtt * 0.25

        self.smoothed_rtt = mean_rtt
        self.rtt_deviation = std_rtt
        self._initialized = True


class MaxLast10Estimator(LatencyEstimatorBase):
    """Conservative latency estimator using max of last 10 measurements (RTC-style).

    Returns the maximum RTT observed in the last 10 measurements, providing a
    conservative bound that is less adaptive but more stable under spikes.
    """

    def __init__(self, fps: float, window_size: int = 10):
        super().__init__(fps)
        self._window_size = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)

    def update(self, measured_rtt: float) -> None:
        """Add a new RTT measurement to the window."""
        self._buffer.append(measured_rtt)

    @property
    def estimate_seconds(self) -> float:
        """Get the latency estimate as max of last N measurements."""
        if not self._buffer:
            return 1.0 / self._fps * 5  # 5 action steps as initial guess
        return max(self._buffer)

    def reset(self) -> None:
        """Reset the estimator state."""
        self._buffer.clear()

    def prime(self, samples: list[float]) -> None:
        """Prime the estimator with initial RTT samples.

        Adds all samples to the buffer, up to the window size limit.
        """
        for sample in samples:
            self._buffer.append(sample)


class FixedLatencyEstimator(LatencyEstimatorBase):
    """Fixed latency estimator for baseline comparisons (SmolVLA-style).

    Returns a fixed, user-specified latency estimate regardless of measurements.
    This represents the behavior of systems that assume a constant network latency
    rather than adapting to actual conditions.

    Note: The estimate is still quantized to at least 1 action step.
    """

    def __init__(self, fps: float, fixed_latency_s: float = 0.1):
        """Initialize with a fixed latency value.

        Args:
            fps: Control loop frequency.
            fixed_latency_s: Fixed latency estimate in seconds (default 100ms).
        """
        super().__init__(fps)
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

    def prime(self, samples: list[float]) -> None:
        """No-op: fixed estimator ignores priming samples."""
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

    Returns:
        A latency estimator instance.
    """
    if kind == "jk":
        return JKLatencyEstimator(fps=fps, alpha=alpha, beta=beta, k=k)
    elif kind == "max_last_10":
        return MaxLast10Estimator(fps=fps)
    elif kind == "fixed":
        return FixedLatencyEstimator(fps=fps, fixed_latency_s=fixed_latency_s)
    else:
        raise ValueError(f"Unknown latency estimator type: {kind}. Use 'jk', 'max_last_10', or 'fixed'.")
