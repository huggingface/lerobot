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

"""Supervisor monitor for event-triggered replanning (Tier 2).

A `SupervisorMonitor` watches a single camera on an independent thread and,
when its `detect_fn` fires, raises a trigger that `RobotClient` ORs into its
observation-sending gate. This forces an early replan the instant a change is
detected, regardless of action-queue level, without touching the static
`chunk_size_threshold`.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .helpers import get_logger


@dataclass(frozen=True)
class DetectorOutput:
    """Structured detector output for event-triggered and speed-adaptive replanning."""

    replan_now: bool = False
    center_px: tuple[float, float] | None = None
    speed_px_s: float | None = None
    effective_chunk_size_threshold: float | None = None
    reason: str = ""


class MotionDetector:
    """Frame-difference motion detector usable as a `detect_fn`.

    Compares consecutive frames in grayscale and fires when the fraction of
    pixels whose intensity changed by more than `PIXEL_DIFF_THRESHOLD` exceeds
    `motion_area_threshold`. Dependency-free (numpy only).
    """

    PIXEL_DIFF_THRESHOLD = 25.0

    def __init__(self, motion_area_threshold: float):
        self.motion_area_threshold = motion_area_threshold
        self._prev_gray: NDArray | None = None

    def __call__(self, frame: NDArray) -> bool:
        gray = frame.mean(axis=2) if frame.ndim == 3 else frame.astype(float)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        diff = np.abs(gray - self._prev_gray)
        self._prev_gray = gray
        motion_ratio = float((diff > self.PIXEL_DIFF_THRESHOLD).mean())
        return motion_ratio > self.motion_area_threshold


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
            return DetectorOutput(reason="red_cube_not_visible")

        ys, xs = np.nonzero(mask)
        center = (float(xs.mean()), float(ys.mean()))
        speed_px_s = self._estimate_speed(center, now_s)
        self._prev_center = center
        self._prev_time_s = now_s

        if speed_px_s is None:
            return DetectorOutput(center_px=center, reason="red_cube_initialized")

        threshold = self._map_speed_to_threshold(speed_px_s)
        replan_now = speed_px_s >= self.urgent_speed_px_s
        reason = "red_cube_urgent_speed" if replan_now else "red_cube_speed"
        return DetectorOutput(
            replan_now=replan_now,
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


class SupervisorMonitor:
    """Watch a camera on an independent thread and trigger early replanning.

    Runs decoupled from the control loop: it peeks the camera's latest frame
    via the non-blocking `read_latest()` and never goes through the policy
    observation path, so it does not steal the control-loop FPS budget.
    """

    MAX_AGE_MS = 500

    def __init__(
        self,
        camera,
        detect_fn: Callable[[NDArray], bool | DetectorOutput],
        poll_fps: int,
        cooldown_s: float,
    ):
        self.camera = camera
        self.detect_fn = detect_fn
        self.poll_dt = 1.0 / poll_fps
        self.cooldown_s = cooldown_s
        self.logger = get_logger("supervisor")

        self._trigger = threading.Event()
        self._latest_output: DetectorOutput | None = None
        self._output_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_fire = -1.0

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.logger.info("Supervisor monitor started")

    def _loop(self):
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            try:
                frame = self.camera.read_latest(max_age_ms=self.MAX_AGE_MS)
                output = self._normalize_output(self.detect_fn(frame))
                with self._output_lock:
                    self._latest_output = output

                if output.replan_now and (loop_start - self._last_fire) > self.cooldown_s:
                    self._last_fire = loop_start
                    self._trigger.set()
                    self.logger.info(f"Re-inference triggered by supervisor: {output.reason}")
            except (TimeoutError, RuntimeError, ValueError) as e:
                self.logger.debug(f"Supervisor frame read skipped: {e}")

            time.sleep(max(0, self.poll_dt - (time.perf_counter() - loop_start)))

    def consume_trigger(self) -> DetectorOutput | None:
        """Return latest output (and clear) if a trigger is pending. One detection fires once."""
        if self._trigger.is_set():
            self._trigger.clear()
            return self.latest_output()
        return None

    def latest_output(self) -> DetectorOutput | None:
        """Return the latest detector output, including non-urgent speed estimates."""
        with self._output_lock:
            return self._latest_output

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @staticmethod
    def _normalize_output(output: bool | DetectorOutput) -> DetectorOutput:
        if isinstance(output, DetectorOutput):
            return output
        return DetectorOutput(replan_now=output, reason="motion" if output else "")
