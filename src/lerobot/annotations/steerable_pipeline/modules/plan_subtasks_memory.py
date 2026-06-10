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

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import PlanConfig
from ..frames import (
    FrameProvider,
    null_provider,
    to_contact_sheet_blocks,
)
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord, reconstruct_subtask_spans, snap_to_frame
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient

logger = logging.getLogger(__name__)


# Prepended to every describe / segment prompt so the VLM knows the images are
# timestamped contact-sheet grids, not a single video, and reads the burned-in
# per-tile timestamp when choosing boundaries.
def _contact_sheet_preamble(columns: int) -> str:
    return (
        "CONTACT SHEETS — how to read the images below:\n"
        f"- Each image is a grid of sampled video frames, {columns} per row, "
        "with time running left-to-right then top-to-bottom (row-major).\n"
        "- Each frame has its timestamp burned into the top-left corner, e.g. "
        '"012.50s". Use that printed timestamp (not the tile position) when you '
        "choose start/end times; boundaries should land on or near a printed "
        "timestamp.\n"
        "- Frames continue across grids: an action may span the end of one sheet "
        "and the start of the next, so do not place a boundary just because a new "
        "image begins.\n\n"
    )


# Appended to every describe (and segment) prompt. A visual, causal definition
# of where one event ends and the next begins — adapted from macrodata/refiner —
# to sharpen cut points while the existing prompt keeps owning the imperative
# phrasing.
_CAUSAL_BOUNDARY_RULES = (
    "EVENT BOUNDARIES — where one event ends and the next begins:\n"
    "- Start a new event whenever the world state changes: an object becomes "
    "held (the gripper closes on it), an object is released (the gripper opens "
    "and it stays put), an object reaches a new location, a lid/door/drawer "
    "changes open/closed state, a tool starts or stops affecting a surface, or "
    "contents visibly move (e.g. poured).\n"
    "- If a single action changes the same state gradually and continuously, "
    "keep it as ONE event — do not split it.\n"
    "- If the same action repeats on different objects or target locations, "
    "treat each repetition as a separate event.\n"
    "- Do NOT create boundaries for idle time, camera motion, hesitation, or "
    "tiny hand adjustments."
)


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
        # Task driving every plan-module prompt: canonical episode_task, or a
        # video-derived one when it's empty/placeholder (see derive_task_*).
        effective_task = self._resolve_effective_task(record)
        # task_aug rows at t=0: phrasings the renderer rotates ${task} through.
        # Either the structured 5-axis taxonomy (task_aug_axes.enabled) or
        # free-form n_task_rephrasings; the effective task is always emitted
        # first so the rotation covers the source-of-truth phrasing.
        t0 = float(record.frame_timestamps[0]) if record.frame_timestamps else 0.0
        variants: list[str] | None = None
        if self.config.task_aug_axes.enabled and effective_task:
            variants = self._generate_task_aug_by_axes(effective_task, self.config.task_aug_axes)
        elif self.config.n_task_rephrasings > 0 and effective_task:
            variants = self._generate_task_rephrasings(effective_task, n=self.config.n_task_rephrasings)
        if variants is not None:
            rows.extend(self._task_aug_rows([effective_task, *variants], t0))

        subtask_spans = self._generate_subtasks(record, task=effective_task)

        # subtask rows
        for span in subtask_spans:
            rows.append(
                {
                    "role": "assistant",
                    "content": span["text"],
                    "style": "subtask",
                    "timestamp": snap_to_frame(span["start"], record.frame_timestamps),
                    "tool_calls": None,
                }
            )
        # Plan rows at every subtask boundary (incl. t=0). The plan is a
        # numbered list of still-todo subtasks, so re-emitting at each
        # boundary makes it shrink as work progresses — ${plan} at frame t is
        # exactly what's left to do.
        if self.config.emit_plan:
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
        # memory rows at every subtask boundary except the very first start;
        # skipped entirely when ``emit_memory`` is False (subtasks-only / plan-only).
        prior_memory = ""
        memory_boundaries = enumerate(subtask_spans[1:], start=1) if self.config.emit_memory else []
        for i, span in memory_boundaries:
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

    @staticmethod
    def _task_aug_rows(phrasings: Sequence[str], t0: float) -> list[dict[str, Any]]:
        """Build deduplicated ``task_aug`` rows (role=user) at ``t0``."""
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for phrasing in phrasings:
            key = phrasing.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(
                {"role": "user", "content": key, "style": "task_aug", "timestamp": t0, "tool_calls": None}
            )
        return rows

    # ------------------------------------------------------------------
    # VLM call helpers — every plan-module prompt follows the same shape:
    # build messages → single VLM call → pull a named field.
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

    def _video_message(
        self,
        record: EpisodeRecord,
        prompt: str,
        window: tuple[float, float] | None = None,
    ) -> list[dict[str, Any]]:
        """User message combining the (optionally windowed) contact sheets with ``prompt``.

        The prompt is always prefixed with a short explanation of how to read
        the timestamped grids, so the model treats them as one ordered
        sequence of frames rather than unrelated images.
        """
        prompt = _contact_sheet_preamble(self.config.contact_sheet_columns) + prompt
        content = [*self._episode_video_block(record, window=window), {"type": "text", "text": prompt}]
        return [{"role": "user", "content": content}]

    def _derive_task_from_video(self, record: EpisodeRecord) -> str | None:
        """Ask the VLM "what is this video about" with no task hint at all."""
        text = self._vlm_field(self._video_message(record, load_prompt("plan_video_task")), "task")
        return text.strip() if isinstance(text, str) and text.strip() else None

    def _generate_task_rephrasings(self, base_task: str, *, n: int) -> list[str]:
        """Generate ``n`` text-only paraphrases of ``base_task``."""
        if n <= 0 or not base_task:
            return []
        prompt = load_prompt("plan_task_rephrasings").format(base_task=base_task, n=n)
        raw = self._vlm_field(self._text_message(prompt), "rephrasings")
        if not isinstance(raw, list):
            return []
        out = [item.strip().strip('"').strip("'") for item in raw if isinstance(item, str)]
        return [s for s in out if s][:n]

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
        prompt = load_prompt("plan_task_aug_axes").format(
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

    def _episode_video_block(
        self, record: EpisodeRecord, window: tuple[float, float] | None = None
    ) -> list[dict[str, Any]]:
        """Timestamped contact sheets for the describe / segmentation prompts.

        Always renders the (optionally windowed) episode as contact sheets:
        frames sampled at ``frames_per_second`` and packed into timestamped
        JPEG grids. ``max_frames_per_prompt`` caps the frame count; whole
        episodes that exceed it are windowed upstream in
        :meth:`_generate_subtasks` so each call stays within budget while the
        full episode keeps its sampling density.

        When ``window=(w0, w1)`` is given the badges are WINDOW-RELATIVE
        (``ts - w0``) to match the window-relative time frame the
        segmentation prompt works in (spans are offset back to absolute time
        afterwards).
        """
        if not record.frame_timestamps:
            return []
        if window is not None:
            w0, w1 = float(window[0]), float(window[1])
            dur = max(0.0, w1 - w0)
            n = max(1, int(round(dur * self.config.frames_per_second)) + 1)
            n = min(n, self.config.max_frames_per_prompt)
            if n <= 1 or dur <= 0.0:
                timestamps = [0.5 * (w0 + w1)]
            else:
                step = dur / (n - 1)
                timestamps = [w0 + i * step for i in range(n)]
            frames = self.frame_provider.frames_at(record, timestamps)
            rel = [ts - w0 for ts in timestamps[: len(frames)]]
            return self._contact_sheet_blocks(frames, rel)
        episode_duration = record.frame_timestamps[-1] - record.frame_timestamps[0]
        n = max(1, int(round(episode_duration * self.config.frames_per_second)) + 1)
        n = min(n, self.config.max_frames_per_prompt)
        timestamps = self._uniform_episode_timestamps(record, n)
        frames = self.frame_provider.frames_at(record, timestamps)
        return self._contact_sheet_blocks(frames, timestamps[: len(frames)])

    @staticmethod
    def _uniform_episode_timestamps(record: EpisodeRecord, n: int) -> list[float]:
        """``n`` episode-relative timestamps spanning ``[t0, t_last]`` uniformly."""
        ts = record.frame_timestamps
        if n >= len(ts):
            return [float(t) for t in ts]
        t0, t_last = float(ts[0]), float(ts[-1])
        if t_last <= t0 or n <= 1:
            return [t0] * max(1, n)
        step = (t_last - t0) / (n - 1)
        return [t0 + i * step for i in range(n)]

    def _contact_sheet_blocks(self, frames: list[Any], timestamps: list[float]) -> list[dict[str, Any]]:
        """Build timestamped contact-sheet image blocks from decoded frames."""
        return to_contact_sheet_blocks(
            frames,
            timestamps,
            columns=self.config.contact_sheet_columns,
            frames_per_sheet=self.config.contact_sheet_frames_per_sheet,
            frame_width=self.config.contact_sheet_frame_width,
            quality=self.config.contact_sheet_quality,
        )

    def run_plan_updates(
        self,
        record: EpisodeRecord,
        staging: EpisodeStaging,
        interjection_times: Sequence[float],
        interjection_texts: Sequence[str] | None = None,
    ) -> None:
        """Append additional ``plan`` rows at every interjection timestamp.

        Plans refresh ONLY on user interjections (event-driven). The
        interjection text is forwarded into the prompt so the refreshed plan
        reflects the user's correction.
        """
        if not self.config.emit_plan:
            return
        existing = staging.read("plan")
        # Pass the last frame timestamp so the final span is closed (else its
        # end == start, zero duration, and a refresh inside it is missed).
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
        """Generate subtask spans, optionally via a multi-call quality chain.

        Single call (default): watch video → emit subtask JSON.

        Multi-call (opt-in, higher quality, more VLM calls):
          1. ``subtask_describe_first`` — a grounding pass that narrates
             ONLY what is visible (no JSON commitment to subtasks yet);
             its description is injected into the segmentation prompt so
             the model segments its own grounded observations instead of
             pattern-matching the task text.
          2. segmentation — emit subtask JSON (as before).
        """
        if record.row_count == 0 or not record.frame_timestamps:
            return []
        episode_duration = record.frame_timestamps[-1] - record.frame_timestamps[0]
        effective_task = task if task is not None else record.episode_task

        # ---- Auto-windowing (keeps the full sampling density) --------
        # Contact sheets are cheap, but a whole long episode sampled at
        # ``frames_per_second`` can still exceed ``max_frames_per_prompt``.
        # When it does, split into consecutive windows of exactly that many
        # frames (one describe→segment call each, still at the full sampling
        # density), then merge + stitch — so an episode of any length is
        # covered at full density rather than subsampled into one sparse call.
        fps = max(1e-6, float(self.config.frames_per_second))
        n_whole = int(round(episode_duration * fps)) + 1
        if n_whole > self.config.max_frames_per_prompt:
            window_s = self.config.max_frames_per_prompt / fps
            return self._generate_subtasks_windowed(record, effective_task, window_s)

        # ---- Pass 1 (optional): grounding description ----------------
        observation_block = ""
        if getattr(self.config, "subtask_describe_first", False):
            description = self._describe_episode(record, effective_task)
            if description:
                observation_block = (
                    "You watched this video and described, chronologically, "
                    "ONLY what the robot actually does:\n"
                    f'"""{description}"""\n\n'
                    "Segment THAT grounded description (cross-checked against "
                    "the video) into atomic subtasks. Do not introduce any "
                    "action that is not in your description above.\n\n"
                )

        # ---- Pass 2: segmentation ------------------------------------
        prompt = self._with_causal_rules(
            load_prompt("plan_subtasks").format(
                episode_task=effective_task,
                min_subtask_seconds=self.config.min_subtask_seconds,
                max_steps=self.config.plan_max_steps,
                episode_duration=f"{episode_duration:.3f}",
                observation_block=observation_block,
            )
        )
        spans = self._vlm_field(self._video_message(record, prompt), "subtasks")
        cleaned = self._clean_spans(spans, record)
        if not cleaned:
            return []

        # ---- Full-episode coverage stitch ----------------------------
        # The VLM can start after t0 or leave gaps, so frames fall through
        # with no active subtask. Always stitch into a contiguous
        # [t0, t_last] cover.
        cleaned = self._stitch_full_coverage(cleaned, record)

        return cleaned

    def _generate_subtasks_windowed(
        self, record: EpisodeRecord, task: str, window_s: float
    ) -> list[dict[str, Any]]:
        """Subtask generation in fixed-length windows at constant fps.

        Splits ``[t0, t_last]`` into consecutive windows of ``window_s``
        seconds, runs the describe -> segment chain on each window's own
        frames (sampled at ``frames_per_second``), offsets
        each window's spans back to absolute episode time, then merges +
        stitches into a contiguous whole-episode cover.
        """
        t0 = float(record.frame_timestamps[0])
        t_last = float(record.frame_timestamps[-1])
        all_spans: list[dict[str, Any]] = []
        w0 = t0
        n_windows = 0
        while w0 < t_last - 1e-6:
            w1 = min(w0 + window_s, t_last)
            all_spans.extend(self._subtasks_for_window(record, task, w0, w1))
            n_windows += 1
            w0 = w1
        logger.info(
            "episode %d: windowed subtask gen over %d window(s) of %.1fs -> %d raw spans",
            record.episode_index,
            n_windows,
            window_s,
            len(all_spans),
        )
        # Merge across windows: clamp to the absolute episode, sort, and
        # frame-snap to distinct starts (handles any boundary collisions).
        cleaned = self._clean_spans(all_spans, record)
        if not cleaned:
            return []
        return self._stitch_full_coverage(cleaned, record)

    def _subtasks_for_window(
        self, record: EpisodeRecord, task: str, w0: float, w1: float
    ) -> list[dict[str, Any]]:
        """Run describe -> segment on one ``[w0, w1]`` window.

        The model works in window-RELATIVE time ``[0, L]`` (it perceives
        the window as a clip starting at 0); spans are offset back to
        absolute ``[w0, w1]`` before returning.
        """
        window = (w0, w1)
        win_len = max(0.0, w1 - w0)

        observation_block = ""
        if getattr(self.config, "subtask_describe_first", False):
            description = self._describe_episode(record, task, window=window)
            if description:
                observation_block = (
                    "You watched this video clip and described, chronologically, "
                    "ONLY what the robot actually does:\n"
                    f'"""{description}"""\n\n'
                    "Segment THAT grounded description (cross-checked against "
                    "the clip) into atomic subtasks. Do not introduce any "
                    "action that is not in your description above.\n\n"
                )

        prompt = self._with_causal_rules(
            load_prompt("plan_subtasks").format(
                episode_task=task,
                min_subtask_seconds=self.config.min_subtask_seconds,
                max_steps=self.config.plan_max_steps,
                episode_duration=f"{win_len:.3f}",
                observation_block=observation_block,
            )
        )
        spans = self._vlm_field(self._video_message(record, prompt, window=window), "subtasks")
        # Window-relative clamp; no frame-snap dedupe yet (done on the
        # merged absolute set).
        cleaned = self._clean_spans(spans, record, bounds=(0.0, win_len), dedupe=False)
        if not cleaned:
            return []

        # Offset window-relative spans back to absolute episode time.
        for s in cleaned:
            s["start"] = w0 + float(s["start"])
            s["end"] = w0 + float(s["end"])
        return cleaned

    def _stitch_full_coverage(
        self, spans: list[dict[str, Any]], record: EpisodeRecord
    ) -> list[dict[str, Any]]:
        """Make subtask spans tile the full episode with no gaps.

        * The first subtask starts at the episode's first frame ``t0``
          (any idle / approach before the first labelled action is folded
          into it), so every early frame has an active subtask.
        * Each subtask's ``end`` is snapped to the next subtask's
          ``start`` (gaps between spans are closed), and the final
          subtask's ``end`` extends to the last frame ``t_last``.

        Starts are otherwise left as the (already frame-snapped, distinct)
        values the VLM produced — only the FIRST start is pulled
        back to ``t0``, which can't collide with a later span because it
        was already the earliest. Purely deterministic; runs after the
        VLM passes.
        """
        if not spans or not record.frame_timestamps:
            return spans
        t0 = float(record.frame_timestamps[0])
        t_last = float(record.frame_timestamps[-1])
        spans = sorted(spans, key=lambda s: float(s["start"]))
        spans[0]["start"] = t0
        for i in range(len(spans) - 1):
            spans[i]["end"] = float(spans[i + 1]["start"])
        spans[-1]["end"] = t_last
        for s in spans:
            if float(s["end"]) < float(s["start"]):
                s["end"] = float(s["start"])
        return spans

    @staticmethod
    def _with_causal_rules(prompt: str) -> str:
        """Append the causal event-boundary rules to a describe/segment prompt."""
        return f"{prompt}\n\n{_CAUSAL_BOUNDARY_RULES}"

    def _clean_spans(
        self,
        spans: Any,
        record: EpisodeRecord,
        bounds: tuple[float, float] | None = None,
        dedupe: bool = True,
    ) -> list[dict[str, Any]]:
        """Clamp / sort / (optionally) dedupe raw VLM subtask spans into valid rows.

        ``bounds`` overrides the clamp range — pass the window's
        ``(w_lo, w_hi)`` when cleaning window-relative spans, or leave
        ``None`` to clamp to the whole episode ``[t0, t_last]``.
        ``dedupe`` runs the frame-snap distinct-start step; skip it for
        window-relative spans (frame snapping is done once on the merged,
        absolute-time set).
        """
        if not spans:
            return []
        if bounds is not None:
            lo, hi = float(bounds[0]), float(bounds[1])
        else:
            lo = record.frame_timestamps[0]
            hi = record.frame_timestamps[-1]
        cleaned: list[dict[str, Any]] = []
        for span in spans:
            try:
                start = float(span["start"])
                end = float(span["end"])
                text = str(span["text"]).strip()
            except (KeyError, ValueError, TypeError):
                continue
            start = max(lo, min(start, hi))
            end = max(lo, min(end, hi))
            if end < start:
                start, end = end, start
            if not text:
                continue
            cleaned.append({"text": text, "start": start, "end": end})
        cleaned.sort(key=lambda s: s["start"])
        if dedupe:
            return self._dedupe_starts_to_distinct_frames(cleaned, record)
        return cleaned

    def _describe_episode(
        self, record: EpisodeRecord, task: str, window: tuple[float, float] | None = None
    ) -> str:
        """Grounding pass: free-form chronological description of the (windowed) video."""
        prompt = self._with_causal_rules(load_prompt("plan_subtask_describe").format(episode_task=task))
        text = self._vlm_field(self._video_message(record, prompt, window=window), "description")
        return text.strip() if isinstance(text, str) and text.strip() else ""

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

        No VLM call: a plain numbered list keeps the plan aligned with the
        upcoming subtasks (the old VLM "compact hierarchical plan" prompt
        cost a round-trip per episode/refresh and could diverge).

            1. <subtask 1>
            2. <subtask 2>

        On a refresh at ``refresh_t`` (from ``run_plan_updates`` on
        interjections, and ``run_episode`` at each boundary), only subtasks
        starting at or after ``refresh_t`` are included — so it always
        describes what's left.
        """
        if not subtask_spans:
            return None
        remaining = [
            s for s in subtask_spans if refresh_t is None or float(s.get("start", 0.0)) >= float(refresh_t)
        ]
        if not remaining:
            # Past the last subtask boundary on a late refresh — nothing
            # left to plan; emit None so the caller skips the row.
            return None
        return "\n".join(f"{i}. {span.get('text', '').strip()}" for i, span in enumerate(remaining, start=1))

    def _generate_memory(
        self,
        record: EpisodeRecord,
        prior_memory: str,
        completed: str,
        remaining: Sequence[str],
        *,
        task: str | None = None,
    ) -> str:
        prompt = load_prompt("plan_memory").format(
            episode_task=(task if task is not None else record.episode_task),
            prior_memory=prior_memory or "(none)",
            completed_subtask=completed,
            remaining_subtasks=", ".join(remaining) if remaining else "(none)",
        )
        memory = self._vlm_field(self._text_message(prompt), "memory")
        return memory.strip() if isinstance(memory, str) else ""
