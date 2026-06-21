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

"""Red-cube image-plane speed detector for speed-adaptive replanning."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from .base import DetectorOutput


class RedCubeSpeedDetector:
    """Estimate red cube image-plane speed and suggest a replan threshold.

    This detector uses a simple dependency-free HSV red mask. Red wraps around
    the hue axis, so pixels near both 0 and 360 degrees are accepted.
    """

    def __init__(
        self,
        slow_speed_px_s: float,
        fast_speed_px_s: float,
        min_chunk_size_threshold: float,
        max_chunk_size_threshold: float,
        urgent_speed_px_s: float,
        hue_tolerance_deg: float = 20.0,
        saturation_min: float = 0.45,
        value_min: float = 0.25,
        min_area_ratio: float = 0.001,
    ):
        self.slow_speed_px_s = slow_speed_px_s
        self.fast_speed_px_s = fast_speed_px_s
        self.min_chunk_size_threshold = min_chunk_size_threshold
        self.max_chunk_size_threshold = max_chunk_size_threshold
        self.urgent_speed_px_s = urgent_speed_px_s
        self.hue_tolerance_deg = hue_tolerance_deg
        self.saturation_min = saturation_min
        self.value_min = value_min
        self.min_area_ratio = min_area_ratio

        self._prev_center: tuple[float, float] | None = None
        self._prev_time_s: float | None = None

    def __call__(self, frame: NDArray) -> DetectorOutput:
        return self.detect(frame, now_s=time.perf_counter())

    def detect(self, frame: NDArray, now_s: float) -> DetectorOutput:
        mask = self._red_mask(frame)
        area_ratio = float(mask.mean())
        if area_ratio < self.min_area_ratio:
            self._prev_center = None
            self._prev_time_s = None
            return DetectorOutput(target_visible=False, reason="red_cube_not_visible")

        ys, xs = np.nonzero(mask)
        center = (float(xs.mean()), float(ys.mean()))
        speed_px_s = self._estimate_speed(center, now_s)
        self._prev_center = center
        self._prev_time_s = now_s

        if speed_px_s is None:
            return DetectorOutput(target_visible=True, center_px=center, reason="red_cube_initialized")

        threshold = self._map_speed_to_threshold(speed_px_s)
        replan_now = speed_px_s >= self.urgent_speed_px_s
        reason = "red_cube_urgent_speed" if replan_now else "red_cube_speed"
        return DetectorOutput(
            replan_now=replan_now,
            target_visible=True,
            center_px=center,
            speed_px_s=speed_px_s,
            effective_chunk_size_threshold=threshold,
            reason=reason,
        )

    def _estimate_speed(self, center: tuple[float, float], now_s: float) -> float | None:
        if self._prev_center is None or self._prev_time_s is None:
            return None

        dt = now_s - self._prev_time_s
        if dt <= 0:
            return 0.0

        dx = center[0] - self._prev_center[0]
        dy = center[1] - self._prev_center[1]
        return float(np.hypot(dx, dy) / dt)

    def _map_speed_to_threshold(self, speed_px_s: float) -> float:
        if self.fast_speed_px_s <= self.slow_speed_px_s:
            return self.max_chunk_size_threshold

        alpha = (speed_px_s - self.slow_speed_px_s) / (self.fast_speed_px_s - self.slow_speed_px_s)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return self.min_chunk_size_threshold + alpha * (
            self.max_chunk_size_threshold - self.min_chunk_size_threshold
        )

    def _red_mask(self, frame: NDArray) -> NDArray[np.bool_]:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Expected an RGB image with shape HxWx3, got {frame.shape}")

        rgb = frame[..., :3].astype(np.float32)
        if rgb.max(initial=0) > 1.0:
            rgb = rgb / 255.0

        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc

        hue = np.zeros_like(maxc)
        nonzero_delta = delta > 1e-6

        red_is_max = (maxc == r) & nonzero_delta
        green_is_max = (maxc == g) & nonzero_delta
        blue_is_max = (maxc == b) & nonzero_delta

        hue[red_is_max] = (60.0 * ((g[red_is_max] - b[red_is_max]) / delta[red_is_max])) % 360.0
        hue[green_is_max] = 60.0 * ((b[green_is_max] - r[green_is_max]) / delta[green_is_max] + 2.0)
        hue[blue_is_max] = 60.0 * ((r[blue_is_max] - g[blue_is_max]) / delta[blue_is_max] + 4.0)

        saturation = np.zeros_like(maxc)
        nonzero_value = maxc > 1e-6
        saturation[nonzero_value] = delta[nonzero_value] / maxc[nonzero_value]

        red_hue = (hue <= self.hue_tolerance_deg) | (hue >= 360.0 - self.hue_tolerance_deg)
        return red_hue & (saturation >= self.saturation_min) & (maxc >= self.value_min)
