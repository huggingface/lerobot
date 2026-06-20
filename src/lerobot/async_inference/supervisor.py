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

import numpy as np
from numpy.typing import NDArray

from .helpers import get_logger


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
        detect_fn: Callable[[NDArray], bool],
        poll_fps: int,
        cooldown_s: float,
    ):
        self.camera = camera
        self.detect_fn = detect_fn
        self.poll_dt = 1.0 / poll_fps
        self.cooldown_s = cooldown_s
        self.logger = get_logger("supervisor")

        self._trigger = threading.Event()
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
                if self.detect_fn(frame) and (loop_start - self._last_fire) > self.cooldown_s:
                    self._last_fire = loop_start
                    self._trigger.set()
                    self.logger.info("Re-inference triggered by supervisor")
            except (TimeoutError, RuntimeError) as e:
                self.logger.debug(f"Supervisor frame read skipped: {e}")

            time.sleep(max(0, self.poll_dt - (time.perf_counter() - loop_start)))

    def consume_trigger(self) -> bool:
        """Return True (and clear) if a trigger is pending. One detection fires once."""
        if self._trigger.is_set():
            self._trigger.clear()
            return True
        return False

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
