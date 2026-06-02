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
"""``plan`` module: subtask decomposition + plan + memory (PERSISTENT styles)."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import PlanConfig
from ..frames import (
    FrameProvider,
    VideoFrameProvider,
    null_provider,
    to_image_blocks,
    to_video_block,
    to_video_url_block,
)
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord, reconstruct_subtask_spans, snap_to_frame
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient

logger = logging.getLogger(__name__)


@dataclass
class PlanSubtasksMemoryModule:
    """Generate subtask spans, plan, and memory rows.

    All output is persistent (lives in ``language_persistent``):

    - ``subtask`` rows: one per span, stamped at the span's *start* timestamp
      (snapped to an exact frame).
    - ``plan`` rows: emitted at ``t=0``; refreshed at every interjection
      timestamp via :meth:`run_plan_updates` (called by the executor after
      the ``interjections`` module completes).
    - ``memory`` rows: emitted at each subtask boundary (= subtask start
      timestamp from the second subtask onward).
    """

    vlm: VlmClient
    config: PlanConfig
    frame_provider: FrameProvider = field(default_factory=null_provider)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        rows: list[dict[str, Any]] = []
        # Resolve the task that drives every other ``plan``-module prompt.
        # May be the canonical ``record.episode_task`` (default), or a fresh
        # description derived from the video when the canonical task is
        # empty / placeholder / forced-off (see PlanConfig.derive_task_*).
        effective_task = self._resolve_effective_task(record)
        # ``task_aug`` rows at t=0 (role=user), one per rephrasing — the
        # message renderer rotates ``${task}`` deterministically through
        # them so the policy sees diverse phrasings during training.
        # Two paths:
        #   * ``task_aug_axes.enabled=True`` — structured 5-axis taxonomy
        #     (synonym / omit_arm / omit_orientation / omit_grasp_method
        #     / combined). Replaces the free-form rephrasings flow.
        #   * Otherwise — free-form ``n_task_rephrasings`` (original).
        t0 = float(record.frame_timestamps[0]) if record.frame_timestamps else 0.0
        axes_cfg = self.config.task_aug_axes
        if axes_cfg.enabled and effective_task:
            variants = self._generate_task_aug_by_axes(effective_task, axes_cfg)
            seen: set[str] = set()
            ordered = [effective_task, *variants]
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
        elif self.config.n_task_rephrasings > 0 and effective_task:
            rephrasings = self._generate_task_rephrasings(effective_task, n=self.config.n_task_rephrasings)
            # Always include the effective task itself as the first variant
            # so the rotation is guaranteed to cover the source-of-truth
            # phrasing, not just synthetic alternatives.
            seen = set()
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

        # ----------------------------------------------------------------
        # Phase 1a: structured per-subtask action records (additive)
        # ----------------------------------------------------------------
        # When enabled, for every subtask span we ask the VLM for a typed
        # ActionRecord (verb / object / arm / grasp_type / destination /
        # mistake) and emit it as a separate ``style="action_record"``
        # row for downstream use. This is purely additive — it never
        # touches the VLM's subtask text (reconstructing subtask text
        # from these fields was too easy to hallucinate on tasks that
        # don't fit the manipulation schema).
        records_cfg = self.config.action_records
        action_records: list[dict[str, Any] | None] = [None] * len(subtask_spans)
        if records_cfg.enabled and subtask_spans:
            for i, span in enumerate(subtask_spans):
                rec = self._extract_action_record(record, span, effective_task)
                if rec is not None:
                    action_records[i] = rec

        # subtask rows
        for i, span in enumerate(subtask_spans):
            rows.append(
                {
                    "role": "assistant",
                    "content": span["text"],
                    "style": "subtask",
                    "timestamp": snap_to_frame(span["start"], record.frame_timestamps),
                    "tool_calls": None,
                }
            )
            if records_cfg.enabled and records_cfg.emit_record_row and action_records[i] is not None:
                rows.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action_records[i], sort_keys=True),
                        "style": "action_record",
                        "timestamp": snap_to_frame(span["start"], record.frame_timestamps),
                        "tool_calls": None,
                    }
                )
        # Plan rows at every subtask boundary — including t=0 (start of
        # the first subtask). Because the plan is just a numbered list
        # of *still-todo* subtasks, re-emitting at each boundary makes
        # the active plan shrink as work progresses: at frame t the
        # rendered ``${plan}`` is the most recent emission, which
        # contains exactly the subtasks that started at or after the
        # current span. Saves the runtime from having to derive
        # "what's still left" at inference time.
        for span in subtask_spans:
            boundary_t = snap_to_frame(span["start"], record.frame_timestamps)
            plan_text = self._generate_plan(
                record, subtask_spans, refresh_t=boundary_t, task=effective_task
            )
            if plan_text is not None:
                rows.append(
                    {
                        "role": "assistant",
                        "content": plan_text,
                        "style": "plan",
                        "timestamp": float(boundary_t),
                        "tool_calls": None,
                    }
                )
        # memory rows at every subtask boundary except the very first start
        prior_memory = ""
        for i, span in enumerate(subtask_spans[1:], start=1):
            completed = subtask_spans[i - 1]["text"]
            remaining = [s["text"] for s in subtask_spans[i:]]
            mem_text = self._generate_memory(record, prior_memory, completed, remaining, task=effective_task)
            if mem_text:
                ts = snap_to_frame(span["start"], record.frame_timestamps)
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
        staging.write("plan", rows)

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
        """Decide which task string drives the ``plan`` module for this episode.

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
        return task.lower() in self._PLACEHOLDER_TASKS

    # ------------------------------------------------------------------
    # VLM call helpers (factored out: every ``plan``-module prompt below follows
    # the same "build messages → single VLM call → pull a named field"
    # shape, only differing in field name + post-processing).
    # ------------------------------------------------------------------

    def _vlm_field(self, messages: list[dict[str, Any]], field: str) -> Any:
        """Run a single VLM call and return ``result[field]`` or ``None``.

        Centralizes the ``vlm.generate_json([m])[0]`` + ``isinstance(dict)``
        dance every prompt-call site needs.
        """
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict):
            return result.get(field)
        return None

    @staticmethod
    def _text_message(text: str) -> list[dict[str, Any]]:
        """One-shot text-only user message wrapped for ``generate_json``."""
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    def _video_message(self, record: EpisodeRecord, prompt: str) -> list[dict[str, Any]]:
        """User message combining the episode video block with ``prompt``."""
        content = [*self._episode_video_block(record), {"type": "text", "text": prompt}]
        return [{"role": "user", "content": content}]

    def _derive_task_from_video(self, record: EpisodeRecord) -> str | None:
        """Ask the VLM "what is this video about" with no task hint at all."""
        text = self._vlm_field(self._video_message(record, load_prompt("module_1_video_task")), "task")
        return text.strip() if isinstance(text, str) and text.strip() else None

    def _generate_task_rephrasings(self, base_task: str, *, n: int) -> list[str]:
        """Generate ``n`` text-only paraphrases of ``base_task``."""
        if n <= 0 or not base_task:
            return []
        prompt = load_prompt("module_1_task_rephrasings").format(base_task=base_task, n=n)
        raw = self._vlm_field(self._text_message(prompt), "rephrasings")
        if not isinstance(raw, list):
            return []
        out = [item.strip().strip('"').strip("'") for item in raw if isinstance(item, str)]
        return [s for s in out if s][:n]

    # ------------------------------------------------------------------
    # Phase 1a + 1b: structured per-subtask action records
    # ------------------------------------------------------------------

    def _extract_action_record(
        self,
        record: EpisodeRecord,
        span: dict[str, Any],
        episode_task: str,
    ) -> dict[str, Any] | None:
        """Ask the VLM to extract a typed ``ActionRecord`` from a subtask span.

        Sends ``frames_per_subtask`` frames uniformly sampled from
        ``[span.start, span.end]`` plus the canonical subtask text. The
        VLM is constrained to verb + grasp vocabularies from the config
        — invalid values are silently dropped at this layer (the
        validator catches structural problems pre-write).

        Returns ``None`` when the call fails or the VLM returns something
        unrecognizable; callers fall back to the free-form subtask text.
        """
        cfg = self.config.action_records
        start_t = float(span.get("start", 0.0))
        end_t = float(span.get("end", start_t))
        duration = max(0.0, end_t - start_t)

        # Uniform timestamps within the span; fall back to a single
        # center frame for very short spans.
        n = max(1, int(cfg.frames_per_subtask))
        if n == 1 or duration <= 0.0:
            timestamps = [0.5 * (start_t + end_t)]
        else:
            step = duration / (n - 1)
            timestamps = [start_t + i * step for i in range(n)]
        frames = self.frame_provider.frames_at(record, timestamps)
        if not frames:
            logger.debug(
                "action_record: no frames at span %.2f-%.2f for ep %s; skipping",
                start_t, end_t, record.episode_index,
            )
            return None

        prompt = load_prompt("module_1_action_record").format(
            episode_task=episode_task,
            subtask_text=span.get("text", ""),
            start_time=start_t,
            end_time=end_t,
            duration=duration,
            n_frames=len(frames),
            verb_vocabulary=", ".join(cfg.verb_vocabulary),
            grasp_vocabulary=" | ".join(f'"{g}"' for g in cfg.grasp_vocabulary),
        )
        message = [
            {
                "role": "user",
                "content": [*to_image_blocks(frames), {"type": "text", "text": prompt}],
            }
        ]
        result = self.vlm.generate_json([message])[0]
        if not isinstance(result, dict):
            return None

        # Light validation + normalisation. Verb is required; everything
        # else may be null. Verb / grasp_type are clamped to the
        # vocabularies (out-of-vocab → reject or null).
        verb = (result.get("verb") or "").strip().lower()
        if not verb or verb not in {v.lower() for v in cfg.verb_vocabulary}:
            return None
        obj = (result.get("object") or "").strip()
        if not obj:
            return None
        grasp = result.get("grasp_type")
        if isinstance(grasp, str):
            grasp = grasp.strip().lower()
            if grasp not in {g.lower() for g in cfg.grasp_vocabulary}:
                grasp = None
        else:
            grasp = None
        arm = result.get("arm")
        if isinstance(arm, str):
            arm = arm.strip().lower()
            if arm not in {"left", "right", "both"}:
                arm = None
        else:
            arm = None
        destination = result.get("destination")
        destination = destination.strip() if isinstance(destination, str) and destination.strip() else None
        mistake = result.get("mistake")
        mistake = mistake.strip() if isinstance(mistake, str) and mistake.strip() else None

        return {
            "verb": verb,
            "object": obj,
            "arm": arm,
            "grasp_type": grasp,
            "destination": destination,
            "mistake": mistake,
        }

    # ------------------------------------------------------------------
    # Structured 5-axis task augmentation (EgoMimic-style taxonomy)
    # ------------------------------------------------------------------

    def _generate_task_aug_by_axes(self, base_task: str, axes_cfg: Any) -> list[str]:
        """One VLM call → variants along the 5-axis taxonomy.

        Variants from all axes are flattened into a single list (the
        downstream pipeline doesn't need to know about the per-axis
        bucketing — every variant becomes a ``task_aug`` row). Order
        is preserved for reproducibility: synonym_paraphrase first,
        then omit_arm, then omit_orientation, then omit_grasp_method,
        then combined_omissions.
        """
        if not base_task:
            return []
        prompt = load_prompt("module_1_task_aug_axes").format(
            base_task=base_task,
            n_synonym=axes_cfg.synonym_paraphrase,
            n_omit_arm=axes_cfg.omit_arm,
            n_omit_orientation=axes_cfg.omit_orientation,
            n_omit_grasp_method=axes_cfg.omit_grasp_method,
            n_combined=axes_cfg.combined_omissions,
        )
        result = self.vlm.generate_json([self._text_message(prompt)])[0]
        if not isinstance(result, dict):
            return []
        ordered_axes = (
            "synonym_paraphrase",
            "omit_arm",
            "omit_orientation",
            "omit_grasp_method",
            "combined_omissions",
        )
        flat: list[str] = []
        seen: set[str] = set()
        for axis in ordered_axes:
            entries = result.get(axis)
            if not isinstance(entries, list):
                continue
            for item in entries:
                if not isinstance(item, str):
                    continue
                key = item.strip().strip('"').strip("'")
                if not key or key in seen:
                    continue
                seen.add(key)
                flat.append(key)
        return flat

    def _episode_video_block(self, record: EpisodeRecord) -> list[dict[str, Any]]:
        """Same video block ``_generate_subtasks`` builds — extracted helper."""
        if not record.frame_timestamps:
            return []
        if self.config.use_video_url and isinstance(self.frame_provider, VideoFrameProvider):
            cache_dir = Path(self.frame_provider.root) / ".annotate_staging" / ".video_clips"
            clip = self.frame_provider.episode_clip_path(record, cache_dir)
            return (
                to_video_url_block(f"file://{clip}", fps=self.config.use_video_url_fps)
                if clip is not None
                else []
            )
        episode_duration = record.frame_timestamps[-1] - record.frame_timestamps[0]
        target_count = max(1, int(round(episode_duration * self.config.frames_per_second)))
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
        existing = staging.read("plan")
        # Pass the episode's last frame timestamp so the final subtask
        # span is closed (otherwise its ``end`` equals its ``start``,
        # zero duration, and the "current subtask at refresh_t" lookup
        # in ``_generate_plan`` misses any refresh that lands inside it).
        episode_end_t = float(record.frame_timestamps[-1]) if record.frame_timestamps else None
        spans = reconstruct_subtask_spans(existing, episode_end_t=episode_end_t)
        already_planned: set[float] = {float(r["timestamp"]) for r in existing if r.get("style") == "plan"}
        new_rows = list(existing)

        texts: list[str | None] = (
            [None] * len(interjection_times)
            if interjection_texts is None
            else [str(t) if t else None for t in interjection_texts]
        )
        for raw_t, inter_text in zip(interjection_times, texts, strict=True):
            t = snap_to_frame(raw_t, record.frame_timestamps)
            if t in already_planned:
                continue
            already_planned.add(t)
            plan_text = self._generate_plan(record, spans, refresh_t=t, interjection=inter_text)
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
        staging.write("plan", new_rows)

    def _generate_subtasks(self, record: EpisodeRecord, *, task: str | None = None) -> list[dict[str, Any]]:
        if record.row_count == 0 or not record.frame_timestamps:
            return []
        episode_duration = record.frame_timestamps[-1] - record.frame_timestamps[0]
        prompt = load_prompt("module_1_subtasks").format(
            episode_task=(task if task is not None else record.episode_task),
            min_subtask_seconds=self.config.min_subtask_seconds,
            max_steps=self.config.plan_max_steps,
            episode_duration=f"{episode_duration:.3f}",
        )
        messages = self._video_message(record, prompt)
        spans = self._vlm_field(messages, "subtasks")
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
        cleaned = self._dedupe_starts_to_distinct_frames(cleaned, record)
        return cleaned

    @staticmethod
    def _dedupe_starts_to_distinct_frames(
        spans: list[dict[str, Any]], record: EpisodeRecord
    ) -> list[dict[str, Any]]:
        """Bump same-frame subtask starts onto distinct frames.

        Two consecutive VLM spans whose ``start`` rounds to the same
        source frame (after :func:`snap_to_frame`) would otherwise emit
        two ``style=subtask`` rows at the identical persistent
        timestamp. The training-time renderer's ``active_at(t,
        style=subtask)`` resolver can't disambiguate that and raises
        ``Ambiguous resolver for style='subtask'``.

        Walk the (sorted-by-start) spans, snap each to its frame, and
        if the snapped frame is already taken push the span onto the
        next unused frame so both subtasks survive on distinct
        timestamps. If the episode ends before a free frame is found,
        the trailing span is dropped with a warning — better than
        poisoning the render.
        """
        if not spans:
            return spans
        frames = record.frame_timestamps
        if not frames:
            return spans
        used: set[float] = set()
        out: list[dict[str, Any]] = []
        for span in spans:
            ts = snap_to_frame(span["start"], frames)
            if ts in used:
                next_ts = next((f for f in frames if f > ts and f not in used), None)
                if next_ts is None:
                    logger.warning(
                        "episode %d: subtask %r snapped to occupied frame "
                        "%.3f and no free later frame exists — dropping",
                        record.episode_index,
                        span.get("text"),
                        ts,
                    )
                    continue
                ts = next_ts
            used.add(ts)
            new_span = {**span, "start": ts}
            if float(new_span.get("end", ts)) < ts:
                new_span["end"] = ts
            out.append(new_span)
        return out

    def _generate_plan(
        self,
        record: EpisodeRecord,  # noqa: ARG002  (kept for signature stability)
        subtask_spans: Sequence[dict[str, Any]],
        *,
        refresh_t: float | None = None,
        interjection: str | None = None,  # noqa: ARG002
        task: str | None = None,  # noqa: ARG002
    ) -> str | None:
        """Deterministic plan = numbered list of *still-todo* subtasks.

        Previously this called the VLM with a prompt that asked it to
        compress the subtasks into a "compact hierarchical plan". That
        produced longer-than-necessary plans, cost an extra VLM round-trip
        per episode (plus one per interjection on refresh), and could
        diverge from the actual subtask sequence the model is going to
        execute. Replacing it with a plain summarisation keeps the plan
        tightly aligned with the upcoming subtasks and removes the VLM
        call entirely.

        Layout — short imperative fragments prefixed by "N. ":

            1. <subtask 1>
            2. <subtask 2>
            ...

        On a refresh at ``refresh_t`` (called from ``run_plan_updates``
        on interjection events, and from ``run_episode`` at every subtask
        boundary), only subtasks whose start is at or after ``refresh_t``
        are included — the plan shrinks as work progresses, so it always
        describes what's left.
        """
        if not subtask_spans:
            return None
        remaining = [
            s
            for s in subtask_spans
            if refresh_t is None or float(s.get("start", 0.0)) >= float(refresh_t)
        ]
        if not remaining:
            # Past the last subtask boundary on a late refresh — nothing
            # left to plan; emit None so the caller skips the row.
            return None
        return "\n".join(
            f"{i}. {span.get('text', '').strip()}" for i, span in enumerate(remaining, start=1)
        )

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
        memory = self._vlm_field(self._text_message(prompt), "memory")
        return memory.strip() if isinstance(memory, str) else ""
