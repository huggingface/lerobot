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
how long policy inference takes, and how much memory the process and the GPU use. The line is a
tqdm bar redrawn every 0.25 s from a background thread, so the control loop never waits on the
terminal, and a stalled loop still shows up as a rate of 0.
"""

from __future__ import annotations

import logging
import os
import statistics
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from lerobot.inference import InferenceEngine

    from .context import RuntimeContext

REFRESH_S = 0.25
# Inference times older than this are not shown, so a stalled or finished policy does not look healthy.
INFERENCE_WINDOW_S = 5.0


def _process_rss_bytes() -> int:
    """Resident memory of this process, or 0 where /proc is not available."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def format_status(runtime: RuntimeContext, engine: InferenceEngine, device: str | None = None) -> str:
    """The status line's text: loop rate, inference time, process and GPU memory, each when known."""
    parts = []
    if (timer := runtime.cycle_timer) is not None:
        parts.append(f"loop {timer.recent_rate():3d}/{timer.fps * timer.multiplier:g} Hz")
    if inferences := engine.recent_inference_times(INFERENCE_WINDOW_S):
        median, worst = statistics.median(inferences), max(inferences)
        parts.append(f"infer {median * 1000:5.1f} ms (worst {worst * 1000:5.1f})")
    if rss := _process_rss_bytes():
        parts.append(f"process {rss / 2**30:4.2f} GiB")
    if device is not None and device.startswith("cuda"):
        # Imported here, so that importing the status line does not load torch.
        import torch

        if torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info(device)
            parts.append(f"GPU {(total - free) / 2**30:4.2f}/{total / 2**30:4.1f} GiB")
    return " · ".join(parts)


class _AboveTheLine:
    """Console stream for log handlers that prints each record above the status line."""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream

    def write(self, text: str) -> None:
        tqdm.write(text, file=self._stream, end="")

    def flush(self) -> None:
        self._stream.flush()


@contextmanager
def show_status_line(
    runtime: RuntimeContext,
    engine: InferenceEngine,
    device: str | None = None,
    stream: IO[str] | None = None,
) -> Iterator[None]:
    """Keep the status line under the logs while the block runs, on an interactive terminal only.

    The loop rate comes from the timer the running strategy publishes on ``runtime.cycle_timer``.
    Meanwhile the console log handlers write through ``tqdm.write``, so a record never lands on the
    line; they are kept rather than replaced, so their level and filters still apply.
    """
    stream = stream or sys.stderr
    # ``disable=None`` turns the bar off when the stream is not a terminal, so redirected logs stay clean.
    bar = tqdm(file=stream, bar_format="{desc}", dynamic_ncols=True, leave=False, disable=None)
    if bar.disable:
        yield
        return
    handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.StreamHandler) and handler.stream is stream
    ]
    for handler in handlers:
        handler.setStream(_AboveTheLine(stream))
    stop = threading.Event()

    def draw() -> None:
        while not stop.wait(REFRESH_S):
            bar.set_description_str(format_status(runtime, engine, device))

    thread = threading.Thread(target=draw, name="rollout-status", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        bar.close()
        for handler in handlers:
            handler.setStream(stream)
