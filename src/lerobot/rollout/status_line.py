# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""A one-line view of a running rollout, redrawn in place on the terminal.

It shows how the policy is doing while the robot moves: the control loop's rate against its target,
how long policy inference takes, and how much memory the process and the GPU use. Reading the numbers costs tens of microseconds, and the line is redrawn every 0.25 s
from a background thread, so the control loop never waits on the terminal.
"""

from __future__ import annotations

import logging
import os
import statistics
import sys
import threading
import time
from typing import IO

import torch

from .inference import InferenceEngine
from .robot_wrapper import ThreadSafeRobot

REFRESH_S = 0.25
CLEAR_LINE = "\r\033[2K"


def _process_rss_bytes() -> int:
    """Resident memory of this process, or 0 where /proc is not available."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


class StatusLine:
    """Redraws one status line with the loop rate, inference time and memory, in place.

    The loop rate comes from the robot's observation reads, which every rollout strategy makes once
    per control tick. It draws only on an interactive terminal. While active, it is also the stream of
    the console log handler, so a log record and a redraw never interleave: the record clears the
    status line, prints on its own line, and the next redraw brings it back.
    """

    def __init__(
        self,
        robot: ThreadSafeRobot,
        engine: InferenceEngine,
        tick_hz: float,
        device: str | None = None,
        stream: IO[str] | None = None,
    ) -> None:
        self._robot = robot
        self._engine = engine
        self._tick_hz = tick_hz
        self._device = torch.device(device) if device is not None else None
        self._stream = stream or sys.stderr
        self._enabled = self._stream.isatty()
        self._lock = threading.Lock()
        self._shown = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._draw_loop, name="rollout-status", daemon=True)
        self._handlers: list[logging.StreamHandler] = []

    def __enter__(self) -> StatusLine:
        if self._enabled:
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream is self._stream:
                    handler.setStream(self)
                    self._handlers.append(handler)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        for handler in self._handlers:
            handler.setStream(self._stream)
        with self._lock:
            if self._shown:
                self._stream.write("\n")
                self._stream.flush()

    def write(self, text: str) -> None:
        with self._lock:
            if self._shown:
                self._stream.write(CLEAR_LINE)
                self._shown = False
            self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def _draw_loop(self) -> None:
        while not self._stop.wait(REFRESH_S):
            line = self._format()
            with self._lock:
                self._stream.write(CLEAR_LINE + line)
                self._stream.flush()
                self._shown = True

    def _format(self) -> str:
        parts = []
        # Copied in one call each, so the control thread appending meanwhile cannot change them mid-read.
        now = time.perf_counter()
        ticks_last_second = sum(1 for tick in list(self._robot.observation_times) if now - tick <= 1.0)
        inferences = list(self._engine.inference_seconds)
        parts.append(f"loop {ticks_last_second:3d}/{self._tick_hz:g} Hz")
        if inferences:
            median, worst = statistics.median(inferences), max(inferences)
            parts.append(f"infer {median * 1000:5.1f} ms (worst {worst * 1000:5.1f})")
        rss = _process_rss_bytes()
        if rss:
            parts.append(f"process {rss / 2**30:4.2f} GiB")
        if self._device is not None and self._device.type == "cuda" and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info(self._device)
            parts.append(f"GPU {(total - free) / 2**30:4.2f}/{total / 2**30:4.1f} GiB")
        return " · ".join(parts)
