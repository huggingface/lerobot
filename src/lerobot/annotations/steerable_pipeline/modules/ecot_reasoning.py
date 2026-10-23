#!/usr/bin/env python

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
"""``ecot`` module: dense Embodied Chain-of-Thought (ECoT) reasoning traces.

Adapted from *ZR-0: Training Vision-Language-Action Models with Dense
Embodied Chain-of-Thought Supervision* (https://arxiv.org/abs/2606.30552v1),
which supervises a VLA with dense structured reasoning — scene perception,
object identification, task planning, and sub-task decomposition — and shows
this high-level cognitive process transfers across embodiments.

This module ports that *annotation* contribution onto the steerable pipeline
with the same I/O shape as the ``plan`` / ``vqa`` modules: at a configurable
cadence it anchors a timestamp, grounds a short contact-sheet window of
frames around it, asks the shared VLM for a structured ECoT blob, and writes
one ``style="ecot"`` persistent row per anchor whose ``content`` is the
JSON-serialized reasoning. The reasoning describes the scene/task at a
cognitive level rather than in pixel coordinates, so it is camera-agnostic
and lives in ``language_persistent`` — one row stays active from its anchor
until the next, just like the plan/memory rows the ``plan`` module emits.

What is intentionally NOT ported (policy-training concerns, out of scope for
an annotation module): ZR-0's dual-stream architecture (VLM System 2 + a
Diffusion Transformer action expert System 1), flow-matching action chunks,
the cross-attention coupling, and the inference-time ECoT-skip attention
mask. The deliverable here is the dense reasoning *supervision* the dataset
stores; consuming it at training time is a downstream concern.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import EcotConfig
from ..frames import FrameProvider, null_provider, to_contact_sheet_blocks
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord, snap_to_frame
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient

# The four ECoT reasoning stages, canonicalized from ZR-0's schema. The VLM is
# asked for all four; partial traces are kept (missing stages default to "").
_ECOT_FIELDS: tuple[str, ...] = (
    "scene_perception",
    "object_identification",
    "task_plan",
    "subtask_decomposition",
)

# Prepended to every prompt so the VLM treats the images as one ordered
# sequence of timestamped tiles rather than unrelated pictures (same role as
# the plan module's contact-sheet preamble).
_CONTACT_SHEET_NOTE = (
    "CONTACT SHEETS — the images below are grids of sampled video frames "
    "around one segment of the episode, read left-to-right then top-to-bottom. "
    "Each tile has its timestamp burned into the corner; use it to ground each "
    "observation in time.\n\n"
)


@dataclass
class EcotReasoningModule:
    """Emit dense ECoT reasoning traces as persistent ``ecot`` rows."""

    vlm: VlmClient
    config: EcotConfig
    frame_provider: FrameProvider = field(default_factory=null_provider)
    _warned_no_camera: bool = field(default=False, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        if not record.frame_timestamps:
            staging.write("ecot", [])
            return
        if not getattr(self.frame_provider, "camera_keys", []):
            # No camera available — emit nothing rather than reasoning purely
            # from the task text. Surface a one-time warning so this is never
            # silently a no-op.
            if not self._warned_no_camera:
                logging.getLogger(__name__).warning(
                    "ecot module found no cameras on the frame provider — "
                    "every episode will emit zero ECoT rows. Check that the "
                    "dataset declares observation.images.* features in "
                    "meta/info.json; passing --vlm.camera_key=<key> at the "
                    "CLI now also seeds the cameras list as a fallback."
                )
                self._warned_no_camera = True
            staging.write("ecot", [])
            return
        anchors = _anchor_timestamps(record.frame_timestamps, self.config.emission_hz)
        if not anchors:
            staging.write("ecot", [])
            return

        # Build every prompt first (one per anchor), then fan them out as a
        # single batched call so the client can parallelize them.
        per_call: list[tuple[float, list[dict[str, Any]]]] = []
        for ts in anchors:
            messages = self._build_messages(record, ts)
            # Skip anchors that decoded to zero frames: reasoning over an empty
            # window would only hallucinate from the task text.
            if _has_image_block(messages):
                per_call.append((ts, messages))
        if not per_call:
            staging.write("ecot", [])
            return

        results = self.vlm.generate_json([m for _, m in per_call])
        rows: list[dict[str, Any]] = []
        for (ts, _messages), result in zip(per_call, results, strict=True):
            reasoning = _coerce_reasoning(result)
            if reasoning is None:
                continue
            rows.append(
                {
                    "role": "assistant",
                    # No ``sort_keys=True``: ``reasoning`` is already built in
                    # ``_ECOT_FIELDS`` order (perception -> identification ->
                    # planning -> decomposition). Alphabetizing would break the
                    # canonical progression the downstream training recipe
                    # consumes from ``content`` verbatim.
                    "content": json.dumps(reasoning),
                    "style": "ecot",
                    "timestamp": snap_to_frame(ts, record.frame_timestamps),
                    "camera": None,
                    "tool_calls": None,
                }
            )
        staging.write("ecot", rows)

    def _build_messages(self, record: EpisodeRecord, center_ts: float) -> list[dict[str, Any]]:
        timestamps = self._window_timestamps(record, center_ts)
        frames = self.frame_provider.frames_at(record, timestamps)
        sheets = to_contact_sheet_blocks(
            frames,
            timestamps[: len(frames)],
            columns=self.config.contact_sheet_columns,
            frames_per_sheet=self.config.contact_sheet_frames_per_sheet,
            frame_width=self.config.contact_sheet_frame_width,
            quality=self.config.contact_sheet_quality,
        )
        prompt = _CONTACT_SHEET_NOTE + load_prompt("ecot").format(
            episode_task=record.episode_task,
            segment_time=float(center_ts),
        )
        content = [*sheets, {"type": "text", "text": prompt}]
        return [{"role": "user", "content": content}]

    def _window_timestamps(self, record: EpisodeRecord, center_ts: float) -> list[float]:
        """Sample ``frames_per_second`` timestamps in a window around ``center_ts``.

        Clamped to the episode bounds; capped at ``max_frames_per_prompt``.
        """
        t0 = float(record.frame_timestamps[0])
        t_last = float(record.frame_timestamps[-1])
        half = self.config.window_seconds / 2.0
        lo = max(t0, center_ts - half)
        hi = min(t_last, center_ts + half)
        if hi <= lo:
            return [float(center_ts)]
        fps = max(1e-6, float(self.config.frames_per_second))
        n = max(1, int(round((hi - lo) * fps)) + 1)
        n = min(n, self.config.max_frames_per_prompt)
        if n <= 1:
            return [0.5 * (lo + hi)]
        step = (hi - lo) / (n - 1)
        return [lo + i * step for i in range(n)]


def _anchor_timestamps(frame_timestamps: Sequence[float], hz: float) -> list[float]:
    """Episode timestamps to anchor reasoning emissions to.

    One anchor every ``1/hz`` seconds, each snapped to the nearest source
    frame timestamp, deduped so two anchors can't land on the same frame.
    """
    if hz <= 0 or not frame_timestamps:
        return []
    t0 = float(frame_timestamps[0])
    t_last = float(frame_timestamps[-1])
    period = 1.0 / hz
    anchors: list[float] = []
    t = t0
    while t <= t_last + 1e-9:
        nearest = min(frame_timestamps, key=lambda f: abs(float(f) - t))
        nearest = float(nearest)
        if not anchors or anchors[-1] != nearest:
            anchors.append(nearest)
        t += period
    return anchors


def _coerce_reasoning(result: Any) -> dict[str, str] | None:
    """Normalize a VLM ECoT response into a 4-field dict, or ``None`` to skip.

    Tolerates list-valued stages (joined) and missing stages (default ""), but
    drops fully-empty / non-dict responses so a garbled generation never writes
    a content-less row.
    """
    if not isinstance(result, dict):
        return None
    reasoning = {field: _as_text(result.get(field)) for field in _ECOT_FIELDS}
    if not any(reasoning.values()):
        return None
    return reasoning


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _has_image_block(messages: list[dict[str, Any]]) -> bool:
    """Return True if any user content block is a populated image block."""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                return True
    return False
