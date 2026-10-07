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
"""``interjections`` module: interjections + paired speech (EVENT styles + speech atoms).

Two sub-passes:

1. At ``t=0``, emit ONLY a speech tool-call atom (acknowledgement of the
   canonical task). No interjection row — the canonical task is already the
   user utterance from ``meta/tasks.parquet``.

2. For mid-episode interruptions, emit a co-timestamped pair:
       {role:user, style:interjection, content:<text>}
       speech atom (role:assistant, style:None, tool_calls=[say(...)])
   Both rows go in ``language_events`` at the same timestamp.

The ``plan`` module's :meth:`run_plan_updates` reuses this module's
interjection timestamps to refresh the ``plan`` row at the same instant.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import InterjectionsConfig
from ..frames import FrameProvider, null_provider, to_image_blocks
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord, reconstruct_subtask_spans, snap_to_frame
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient
from ..writer import speech_atom


@dataclass
class InterjectionsAndSpeechModule:
    """Generate task-start speech and mid-episode interjection/speech pairs."""

    vlm: VlmClient
    config: InterjectionsConfig
    seed: int = 1729
    frame_provider: FrameProvider = field(default_factory=null_provider)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        rows: list[dict[str, Any]] = []
        if record.frame_timestamps:
            t0 = float(record.frame_timestamps[0])
            initial = self._initial_speech(record)
            if initial:
                rows.append(speech_atom(t0, initial))
        # Pull the ``plan`` module's subtask spans for this episode so the
        # interjection prompt can ground itself in the actual current
        # subtask at each chosen timestamp. The ``plan`` module ran first.
        episode_end_t = float(record.frame_timestamps[-1]) if record.frame_timestamps else None
        subtask_spans = reconstruct_subtask_spans(staging.read("plan"), episode_end_t=episode_end_t)
        rows.extend(self._mid_episode_interjections(record, subtask_spans))
        staging.write("interjections", rows)

    @staticmethod
    def _subtask_at(spans: Sequence[dict[str, Any]], t: float) -> str | None:
        current: str | None = None
        for span in spans:
            if float(span["start"]) <= t:
                current = span.get("text")
            else:
                break
        return current

    def _initial_speech(self, record: EpisodeRecord) -> str | None:
        prompt = load_prompt("interjections_initial_speech").format(
            episode_task=record.episode_task,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict) and isinstance(result.get("text"), str):
            text = result["text"].strip()
            if text:
                return text
        return None

    def _mid_episode_interjections(
        self,
        record: EpisodeRecord,
        subtask_spans: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate interjections aligned with the actual demo trajectory.

        Teleop data is frozen — the robot already executed every step in
        the video. A *counterfactual* interjection like "actually skip
        the wipe" contradicts what then happens in the video, which is
        what qwen36moe-10/11 surfaced as low-quality interjections.

        Instead, anchor every interjection at a subtask boundary and
        write it as a natural user request for the *upcoming* subtask.
        The robot's visible next behavior IS the interjection's effect,
        so the training signal stays consistent: interjection text →
        plan refresh → action stream all line up.
        """
        if self.config.max_interjections_per_episode <= 0:
            return []
        if len(subtask_spans) < 2:
            # Need at least one transition (subtask 0 → subtask 1).
            return []
        # Deterministic per-episode RNG so reruns are stable across SLURM jobs.
        rng = random.Random(f"{self.seed}:{record.episode_index}:interjection")

        # Boundaries: the start time of every subtask except the first
        # (which is just t0 and is covered by the initial-task speech atom).
        boundaries: list[tuple[float, str, str]] = []
        for i in range(1, len(subtask_spans)):
            ts = float(subtask_spans[i]["start"])
            if ts < self.config.interjection_min_t:
                continue
            prev_text = (subtask_spans[i - 1].get("text") or "").strip()
            next_text = (subtask_spans[i].get("text") or "").strip()
            if not next_text:
                continue
            boundaries.append((ts, prev_text, next_text))
        if not boundaries:
            return []

        n = min(self.config.max_interjections_per_episode, len(boundaries))
        chosen = sorted(rng.sample(boundaries, n), key=lambda b: b[0])

        out: list[dict[str, Any]] = []
        for t, prev_subtask, next_subtask in chosen:
            t_snap = snap_to_frame(t, record.frame_timestamps)
            # Window straddles the boundary so the VLM sees the end of the
            # previous subtask and the start of the next one — same
            # conditioning the policy will see at training time.
            window_ts = self._window_timestamps(t_snap, record.frame_timestamps)
            prompt = load_prompt("interjections_interjection").format(
                episode_task=record.episode_task,
                prev_subtask=prev_subtask or "(starting from initial state)",
                next_subtask=next_subtask,
                timestamp=t_snap,
                window_seconds=self.config.interjection_window_seconds,
            )
            images = self.frame_provider.frames_at(record, window_ts)
            content = [*to_image_blocks(images), {"type": "text", "text": prompt}]
            messages = [{"role": "user", "content": content}]
            result = self.vlm.generate_json([messages])[0]
            if not isinstance(result, dict):
                continue
            interjection_text = result.get("interjection")
            speech_text = result.get("speech")
            if not isinstance(interjection_text, str) or not interjection_text.strip():
                continue
            if not isinstance(speech_text, str) or not speech_text.strip():
                continue
            out.append(
                {
                    "role": "user",
                    "content": interjection_text.strip(),
                    "style": "interjection",
                    "timestamp": t_snap,
                    "tool_calls": None,
                }
            )
            out.append(speech_atom(t_snap, speech_text.strip()))
        return out

    def _window_timestamps(self, t_anchor: float, frame_timestamps: Sequence[float]) -> list[float]:
        """Return a small set of frame timestamps centered on ``t_anchor``.

        The window straddles the subtask boundary the interjection sits
        on: roughly half the frames cover the end of the previous
        subtask, half cover the start of the next one. The VLM therefore
        sees BOTH what just finished AND what's about to start, which is
        the conditioning we need to write a natural "now please do X"
        request that matches the visible upcoming behavior.
        """
        if not frame_timestamps:
            return [t_anchor]
        n = max(1, int(self.config.interjection_window_frames))
        if n == 1:
            return [t_anchor]
        window = float(self.config.interjection_window_seconds)
        step = window / max(1, n - 1)
        # Center the window on the anchor so half lands before, half after.
        start_offset = -window / 2.0
        targets = [t_anchor + start_offset + step * i for i in range(n)]
        first_ts = float(frame_timestamps[0])
        last_ts = float(frame_timestamps[-1])
        snapped: list[float] = []
        seen: set[float] = set()
        for tgt in targets:
            clamped = min(last_ts, max(first_ts, tgt))
            t = snap_to_frame(clamped, frame_timestamps)
            if t not in seen:
                seen.add(t)
                snapped.append(t)
        return snapped or [t_anchor]
