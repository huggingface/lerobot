# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Source-clock windows with overlapping context and nonoverlapping output ownership."""

import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFrame:
    episode_index: int
    frame_index: int
    timestamp: float


@dataclass(frozen=True)
class EpisodeWindow:
    index: int
    # Half-open positions in the original source clock, not renumbered frame IDs.
    context_start: int
    context_end: int
    output_start: int
    output_end: int
    frames: tuple[SourceFrame, ...]

    def __post_init__(self):
        if not (
            0 <= self.context_start <= self.output_start < self.output_end <= self.context_end
            and len(self.frames) == self.context_end - self.context_start
        ):
            raise ValueError("Invalid window context/output ranges")

    @property
    def output_frames(self):
        return self.frames[self.output_start - self.context_start : self.output_end - self.context_start]


def time_windows(
    episode_index: int,
    frame_indices: Sequence[int],
    timestamps: Sequence[float],
    *,
    context_seconds: float,
    overlap_seconds: float = 0,
) -> list[EpisodeWindow]:
    """Tile exact source frames; overlap is context on EACH side of an output range.

    Context is bounded by context_seconds (up to discrete source-clock rounding).
    There is no synthetic frame timestamp, assumed FPS, or duplicated output frame.
    """
    if (
        not math.isfinite(context_seconds)
        or not math.isfinite(overlap_seconds)
        or context_seconds <= 0
        or overlap_seconds < 0
        or 2 * overlap_seconds >= context_seconds
    ):
        raise ValueError("Window context must exceed twice its nonnegative overlap")
    if len(frame_indices) != len(timestamps):
        raise ValueError("Frame indices and timestamps must align")
    if any(not math.isfinite(t) for t in timestamps) or any(
        b <= a for a, b in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    if any(b <= a for a, b in zip(frame_indices, frame_indices[1:], strict=False)):
        raise ValueError("Source frame IDs must be strictly increasing")
    windows: list[EpisodeWindow] = []
    start = 0
    duration = context_seconds - 2 * overlap_seconds
    while start < len(timestamps):
        end = max(start + 1, bisect_left(timestamps, timestamps[start] + duration))
        context_start = bisect_left(timestamps, timestamps[start] - overlap_seconds)
        context_end = bisect_right(timestamps, timestamps[start] + duration + overlap_seconds)
        context_end = max(end, context_end)
        frames = tuple(
            SourceFrame(episode_index, frame_indices[i], timestamps[i])
            for i in range(context_start, context_end)
        )
        windows.append(EpisodeWindow(len(windows), context_start, context_end, start, end, frames))
        start = end
    return windows


def reconcile_frame_rows(window: EpisodeWindow, rows: Sequence[dict]) -> list[dict]:
    """Keep only owned predictions; reject duplicate/misattributed frame references."""
    context = {frame.frame_index: frame for frame in window.frames}
    owned = {frame.frame_index for frame in window.output_frames}
    seen = set()
    output = []
    for row in rows:
        index = row["frame_index"]
        if index not in context or index in seen:
            raise ValueError("Prediction contains a foreign or duplicate source frame")
        frame = context[index]
        if row["episode_index"] != frame.episode_index or row["timestamp"] != frame.timestamp:
            raise ValueError("Prediction altered the source reference")
        seen.add(index)
        if index in owned:
            output.append(row)
    return sorted(output, key=lambda row: row["frame_index"])


def reconcile_spans(window: EpisodeWindow, spans: Sequence[dict], timestamps: Sequence[float]) -> list[dict]:
    """Clip context predictions to this window's output; boundaries snap to source time."""
    lo = timestamps[window.output_start]
    hi = timestamps[min(window.output_end, len(timestamps) - 1)]
    output = []
    for span in spans:
        start = max(lo, min(hi, span["start"]))
        end = max(lo, min(hi, span["end"]))
        if end <= start and not (
            window.output_start == len(timestamps) - 1 and span["start"] <= lo <= span["end"]
        ):
            continue
        start = min(timestamps, key=lambda time: abs(time - start))
        end = min(timestamps, key=lambda time: abs(time - end))
        if end > start or window.output_start == len(timestamps) - 1:
            output.append({**span, "start": start, "end": end})
    return output
