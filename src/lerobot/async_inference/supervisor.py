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

The detector callables themselves (``MotionDetector``, ``RedCubeSpeedDetector``,
``DetectorOutput``) are transport-agnostic and live in :mod:`lerobot.detectors`;
they are re-exported here for backward compatibility. The RTC rollout engine
reuses the same detectors without this camera-thread monitor (it feeds frames
straight from the control-loop observation).
"""

import threading
import time
from collections.abc import Callable

from numpy.typing import NDArray

from lerobot.detectors import (
    DetectorOutput,
    MotionDetector,
    RedCubeSpeedDetector,
    normalize_detector_output,
)

from .helpers import get_logger

__all__ = [
    "DetectorOutput",
    "MotionDetector",
    "RedCubeSpeedDetector",
    "SupervisorMonitor",
]


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
        return normalize_detector_output(output)
