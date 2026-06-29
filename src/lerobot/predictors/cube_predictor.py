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

"""Red-cube image-plane position/velocity predictor for the overhead camera."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from .base import PredictorOutput


class CubePredictor:
    """Estimate the red cube's image-plane center and velocity from one frame.

    Uses a simple dependency-free HSV red mask (red wraps around the hue axis, so
    pixels near both 0 and 360 degrees are accepted). The center is the mask
    centroid; the velocity is the per-frame centroid displacement divided by the
    measured control-loop dt, returned as a ``(vx, vy)`` vector so a consumer can
    advance the cube along its travel direction.
    """

    def __init__(
        self,
        hue_tolerance_deg: float = 20.0,
        saturation_min: float = 0.45,
        value_min: float = 0.25,
        min_area_ratio: float = 0.001,
    ):
        self.hue_tolerance_deg = hue_tolerance_deg
        self.saturation_min = saturation_min
        self.value_min = value_min
        self.min_area_ratio = min_area_ratio

        self._prev_center: tuple[float, float] | None = None
        self._prev_time_s: float | None = None

    def __call__(self, frame: NDArray) -> PredictorOutput:
        return self.detect(frame, now_s=time.perf_counter())

    def detect(self, frame: NDArray, now_s: float) -> PredictorOutput:
        mask = self.red_mask(frame)
        area_ratio = float(mask.mean())
        if area_ratio < self.min_area_ratio:
            self._prev_center = None
            self._prev_time_s = None
            return PredictorOutput(target_visible=False, reason="red_cube_not_visible")

        ys, xs = np.nonzero(mask)
        center = (float(xs.mean()), float(ys.mean()))
        velocity = self._estimate_velocity(center, now_s)
        self._prev_center = center
        self._prev_time_s = now_s

        if velocity is None:
            return PredictorOutput(target_visible=True, center_px=center, reason="red_cube_initialized")

        return PredictorOutput(
            target_visible=True,
            center_px=center,
            velocity_px_s=velocity,
            reason="red_cube_tracked",
        )

    @staticmethod
    def predict_center(
        center_px: tuple[float, float],
        velocity_px_s: tuple[float, float],
        lead_s: float,
    ) -> tuple[float, float]:
        """Constant-velocity extrapolation: ``center + velocity * lead_s``."""
        return (
            center_px[0] + velocity_px_s[0] * lead_s,
            center_px[1] + velocity_px_s[1] * lead_s,
        )

    def _estimate_velocity(
        self, center: tuple[float, float], now_s: float
    ) -> tuple[float, float] | None:
        if self._prev_center is None or self._prev_time_s is None:
            return None

        dt = now_s - self._prev_time_s
        if dt <= 0:
            return (0.0, 0.0)

        vx = (center[0] - self._prev_center[0]) / dt
        vy = (center[1] - self._prev_center[1]) / dt
        return (float(vx), float(vy))

    def red_mask(self, frame: NDArray) -> NDArray[np.bool_]:
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
