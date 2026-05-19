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
"""Module 1: subtask decomposition + plan + memory (PERSISTENT styles)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pathlib import Path

from ..config import Module1Config
from ..frames import (
    FrameProvider,
    VideoFrameProvider,
    episode_clip_path,
    null_provider,
    to_video_block,
    to_video_url_block,
)
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient


def _snap_to_frame(t: float, frame_timestamps: Sequence[float]) -> float:
    """Snap an arbitrary float to the nearest exact source frame timestamp."""
    if not frame_timestamps:
        return float(t)
    nearest = min(frame_timestamps, key=lambda f: abs(f - t))
    return float(nearest)


@dataclass
class PlanSubtasksMemoryModule:
    """Generate subtask spans, plan, and memory rows.

    All output is persistent (lives in ``language_persistent``):

    - ``subtask`` rows: one per span, stamped at the span's *start* timestamp
      (snapped to an exact frame).
    - ``plan`` rows: emitted at ``t=0``; refreshed at every interjection
      timestamp via :meth:`run_plan_updates` (called by the executor after
      Module 2 completes).
    - ``memory`` rows: emitted at each subtask boundary (= subtask start
      timestamp from the second subtask onward).
    """

    vlm: VlmClient
    config: Module1Config
    frame_provider: FrameProvider = field(default_factory=null_provider)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        rows: list[dict[str, Any]] = []
        # Resolve the task that drives every other Module-1 prompt. May be
        # the canonical ``record.episode_task`` (default), or a fresh
        # description derived from the video when the canonical task is
        # empty / placeholder / forced-off (see Module1Config.derive_task_*).
        effective_task = self._resolve_effective_task(record)
        # ``task_aug`` rows at t=0 (role=user), one per rephrasing — the
        # PR 1 renderer rotates ``${task}`` deterministically through them
        # so the policy sees diverse phrasings during training.
        t0 = float(record.frame_timestamps[0]) if record.frame_timestamps else 0.0
        if self.config.n_task_rephrasings > 0 and effective_task:
            rephrasings = self._generate_task_rephrasings(
                effective_task, n=self.config.n_task_rephrasings
            )
            # Always include the effective task itself as the first variant
            # so the rotation is guaranteed to cover the source-of-truth
            # phrasing, not just synthetic alternatives.
            seen: set[str] = set()
            ordered = [effective_task, *rephrasings]
            for phrasing in ordered:
                key = phrasing.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "role": "user",
                        "content": key,
                        "style": "task_aug",
                        "timestamp": t0,
                        "tool_calls": None,
                    }
                )

        subtask_spans = self._generate_subtasks(record, task=effective_task)
        # subtask rows
        for span in subtask_spans:
            rows.append(
                {
                    "role": "assistant",
                    "content": span["text"],
                    "style": "subtask",
                    "timestamp": _snap_to_frame(span["start"], record.frame_timestamps),
                    "tool_calls": None,
                }
            )
        # plan row at t=0
        plan_text = self._generate_plan(record, subtask_spans, task=effective_task)
        if plan_text is not None:
            rows.append(
                {
                    "role": "assistant",
                    "content": plan_text,
                    "style": "plan",
                    "timestamp": float(t0),
                    "tool_calls": None,
                }
            )
        # memory rows at every subtask boundary except the very first start
        prior_memory = ""
        for i, span in enumerate(subtask_spans[1:], start=1):
            completed = subtask_spans[i - 1]["text"]
            remaining = [s["text"] for s in subtask_spans[i:]]
            mem_text = self._generate_memory(
                record, prior_memory, completed, remaining, task=effective_task
            )
            if mem_text:
                ts = _snap_to_frame(span["start"], record.frame_timestamps)
                rows.append(
                    {
                        "role": "assistant",
                        "content": mem_text,
                        "style": "memory",
                        "timestamp": ts,
                        "tool_calls": None,
                    }
                )
                prior_memory = mem_text
        staging.write("module_1", rows)

    # ------------------------------------------------------------------
    # Task derivation + rephrasings
    # ------------------------------------------------------------------

    _PLACEHOLDER_TASKS: frozenset[str] = frozenset(
        {
            "debug",
            "test",
            "tbd",
            "todo",
            "n/a",
            "na",
            "untitled",
            "unnamed",
            "default",
            "placeholder",
        }
    )

    def _resolve_effective_task(self, record: EpisodeRecord) -> str:
        """Decide which task string drives Module 1 for this episode.

        Returns the user-supplied ``record.episode_task`` unless
        ``derive_task_from_video`` says otherwise (see config docstring).
        Falls back gracefully to the canonical task if video derivation
        fails.
        """
        canonical = (record.episode_task or "").strip()
        mode = (self.config.derive_task_from_video or "off").strip().lower()
        if mode == "always":
            derived = self._derive_task_from_video(record)
            return derived or canonical
        if mode == "if_short" and self._task_seems_bad(canonical):
            derived = self._derive_task_from_video(record)
            if derived:
                return derived
        return canonical

    def _task_seems_bad(self, task: str) -> bool:
        if not task:
            return True
        if len(task.split()) < int(self.config.derive_task_min_words):
            return True
        if task.lower() in self._PLACEHOLDER_TASKS:
            return True
        return False

    def _derive_task_from_video(self, record: EpisodeRecord) -> str | None:
        """Ask the VLM "what is this video about" with no task hint at all."""
        prompt = load_prompt("module_1_video_task")
        video_block = self._episode_video_block(record)
        content = [*video_block, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict) and isinstance(result.get("task"), str):
            text = result["task"].strip()
            if text:
                return text
        return None

    def _generate_task_rephrasings(self, base_task: str, *, n: int) -> list[str]:
        """Generate ``n`` text-only paraphrases of ``base_task``."""
        if n <= 0 or not base_task:
            return []
        prompt = load_prompt("module_1_task_rephrasings").format(
            base_task=base_task, n=n
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = self.vlm.generate_json([messages])[0]
        if not isinstance(result, dict):
            return []
        raw = result.get("rephrasings")
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                cleaned = item.strip().strip('"').strip("'")
                if cleaned:
                    out.append(cleaned)
        return out[:n]

    def _episode_video_block(self, record: EpisodeRecord) -> list[dict[str, Any]]:
        """Same video block ``_generate_subtasks`` builds — extracted helper."""
        if not record.frame_timestamps:
            return []
        if self.config.use_video_url and isinstance(self.frame_provider, VideoFrameProvider):
            cache_dir = Path(self.frame_provider.root) / ".annotate_staging" / ".video_clips"
            clip = episode_clip_path(record, self.frame_provider, cache_dir)
            return (
                to_video_url_block(f"file://{clip}", fps=self.config.use_video_url_fps)
                if clip is not None
                else []
            )
        episode_duration = record.frame_timestamps[-1] - record.frame_timestamps[0]
        target_count = max(
            1, int(round(episode_duration * self.config.frames_per_second))
        )
        target_count = min(target_count, self.config.max_video_frames)
        video_frames = self.frame_provider.video_for_episode(record, target_count)
        return to_video_block(video_frames)

    def run_plan_updates(
        self,
        record: EpisodeRecord,
        staging: EpisodeStaging,
        interjection_times: Sequence[float],
        interjection_texts: Sequence[str] | None = None,
    ) -> None:
        """Append additional ``plan`` rows at every interjection timestamp.

        Plans refresh ONLY on user interjections — subtask generation
        runs ~1 Hz at inference, but plan re-emission is event-driven.
        Now also forwards the interjection's own text into the prompt so
        the refreshed plan can actually reflect the user's correction
        (the previous version told the model "an interjection happened"
        without telling it what the user said).
        """
        existing = staging.read("module_1")
        spans = self._reconstruct_subtasks_from_rows(existing)
        already_planned: set[float] = {
            float(r["timestamp"]) for r in existing if r.get("style") == "plan"
        }
        new_rows = list(existing)

        texts: list[str | None] = (
            [None] * len(interjection_times)
            if interjection_texts is None
            else [str(t) if t else None for t in interjection_texts]
        )
        for raw_t, inter_text in zip(interjection_times, texts):
            t = _snap_to_frame(raw_t, record.frame_timestamps)
            if t in already_planned:
                continue
            already_planned.add(t)
            plan_text = self._generate_plan(
                record, spans, refresh_t=t, interjection=inter_text
            )
            if plan_text is not None:
                new_rows.append(
                    {
                        "role": "assistant",
                        "content": plan_text,
                        "style": "plan",
                        "timestamp": t,
                        "tool_calls": None,
                    }
                )
        staging.write("module_1", new_rows)

    @staticmethod
    def _reconstruct_subtasks_from_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        last_t: float | None = None
        for row in sorted(
            (r for r in rows if r.get("style") == "subtask"),
            key=lambda r: float(r["timestamp"]),
        ):
            t = float(row["timestamp"])
            if last_t is not None:
                out[-1]["end"] = t
            out.append({"text": row.get("content") or "", "start": t, "end": t})
            last_t = t
        return out

    def _generate_subtasks(
        self, record: EpisodeRecord, *, task: str | None = None
    ) -> list[dict[str, Any]]:
        if record.row_count == 0 or not record.frame_timestamps:
            return []
        episode_duration = record.frame_timestamps[-1] - record.frame_timestamps[0]
        prompt = load_prompt("module_1_subtasks").format(
            episode_task=(task if task is not None else record.episode_task),
            min_subtask_seconds=self.config.min_subtask_seconds,
            max_steps=self.config.plan_max_steps,
            episode_duration=f"{episode_duration:.3f}",
        )
        if self.config.use_video_url and isinstance(self.frame_provider, VideoFrameProvider):
            cache_dir = Path(self.frame_provider.root) / ".annotate_staging" / ".video_clips"
            clip = episode_clip_path(record, self.frame_provider, cache_dir)
            video_block = (
                to_video_url_block(f"file://{clip}", fps=self.config.use_video_url_fps)
                if clip is not None
                else []
            )
        else:
            target_count = max(
                1,
                int(round(episode_duration * self.config.frames_per_second)),
            )
            target_count = min(target_count, self.config.max_video_frames)
            video_frames = self.frame_provider.video_for_episode(record, target_count)
            video_block = to_video_block(video_frames)
        content = [*video_block, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        result = self.vlm.generate_json([messages])[0]
        spans = result.get("subtasks") if isinstance(result, dict) else None
        if not spans:
            return []
        # clamp to [t0, t_last] and sort
        t0 = record.frame_timestamps[0]
        t_last = record.frame_timestamps[-1]
        cleaned: list[dict[str, Any]] = []
        for span in spans:
            try:
                start = float(span["start"])
                end = float(span["end"])
                text = str(span["text"]).strip()
            except (KeyError, ValueError, TypeError):
                continue
            start = max(t0, min(start, t_last))
            end = max(t0, min(end, t_last))
            if end < start:
                start, end = end, start
            if not text:
                continue
            cleaned.append({"text": text, "start": start, "end": end})
        cleaned.sort(key=lambda s: s["start"])
        return cleaned

    def _generate_plan(
        self,
        record: EpisodeRecord,
        subtask_spans: Sequence[dict[str, Any]],
        *,
        refresh_t: float | None = None,
        interjection: str | None = None,
        task: str | None = None,
    ) -> str | None:
        if not subtask_spans:
            return None
        subtasks_text = "\n".join(f"- {s['text']}" for s in subtask_spans)
        prompt = load_prompt("module_1_plan").format(
            episode_task=(task if task is not None else record.episode_task),
            subtasks_text=subtasks_text,
            plan_max_steps=self.config.plan_max_steps,
        )
        if refresh_t is not None:
            # ``current_subtask`` is the span the refresh time falls into,
            # so the model knows where in the demonstration the planner is
            # standing when it re-emits.
            current_subtask = ""
            for span in subtask_spans:
                if float(span["start"]) <= refresh_t and (
                    "end" not in span or float(span["end"]) > refresh_t
                ):
                    current_subtask = span.get("text", "")
                    break
            if interjection:
                prompt += (
                    f"\n\n(Plan refresh at t={refresh_t:.2f}s after a user "
                    f"interjection: {interjection!r}. Current subtask just "
                    f"before the interjection: {current_subtask!r}. Update "
                    f"the plan so it reflects the interjection — drop or "
                    f"reorder steps as needed; do not just restate.)\n"
                )
            else:
                # Refresh without an interjection text: still tell the model
                # where in the episode the plan stands so the re-emission
                # is grounded. Should be rare — plan refreshes are
                # interjection-driven by design.
                prompt += (
                    f"\n\n(Plan refresh at t={refresh_t:.2f}s. Current "
                    f"subtask: {current_subtask!r}.)\n"
                )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict) and isinstance(result.get("plan"), str):
            return result["plan"].strip()
        return None

    def _generate_memory(
        self,
        record: EpisodeRecord,
        prior_memory: str,
        completed: str,
        remaining: Sequence[str],
        *,
        task: str | None = None,
    ) -> str:
        prompt = load_prompt("module_1_memory").format(
            episode_task=(task if task is not None else record.episode_task),
            prior_memory=prior_memory or "(none)",
            completed_subtask=completed,
            remaining_subtasks=", ".join(remaining) if remaining else "(none)",
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict) and isinstance(result.get("memory"), str):
            return result["memory"].strip()
        return ""
