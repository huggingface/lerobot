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
from pathlib import Path
from typing import Any

from ..config import PlanConfig
from ..frames import (
    FrameProvider,
    VideoFrameProvider,
    null_provider,
    to_video_block,
    to_video_url_block,
)
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord, reconstruct_subtask_spans, snap_to_frame
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient
from ..vocabulary import Vocabulary

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
    vocabulary: Vocabulary | None = None
    """When set, the module constrains subtask + memory generation to the
    canonical strings in ``vocabulary``. Phase 0 (vocabulary discovery)
    populates this once per dataset; ``None`` falls back to free-form
    generation (original behaviour)."""

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
        t0 = float(record.frame_timestamps[0]) if record.frame_timestamps else 0.0
        if self.config.n_task_rephrasings > 0 and effective_task:
            rephrasings = self._generate_task_rephrasings(effective_task, n=self.config.n_task_rephrasings)
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
            vocabulary_block=self._subtask_vocabulary_block(),
        )
        messages = self._video_message(record, prompt)
        spans = self._vlm_field(messages, "subtasks")
        # When a vocabulary is in force, do a single targeted retry if
        # any returned subtask is off-vocab — strict exact-match only,
        # no fuzzy snapping. The retry includes the offending strings
        # and the full canonical list so the VLM can correct itself.
        if self.vocabulary is not None and self.vocabulary.subtasks and spans:
            invalid = self._invalid_subtasks(spans)
            if invalid:
                logger.info(
                    "episode %d: VLM emitted %d off-vocab subtask(s) (%s); retrying once",
                    record.episode_index,
                    len(invalid),
                    invalid,
                )
                retry_msg = self._build_subtask_retry_message(messages, invalid)
                retried = self._vlm_field(retry_msg, "subtasks")
                if retried:
                    spans = retried

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
            text = self._canonicalize_subtask(text)
            if not text:
                continue
            cleaned.append({"text": text, "start": start, "end": end})
        cleaned.sort(key=lambda s: s["start"])
        cleaned = self._dedupe_starts_to_distinct_frames(cleaned, record)
        if self.vocabulary is not None and self.vocabulary.subtasks and not cleaned:
            logger.warning(
                "episode %d: every VLM subtask was off-vocab even after retry — "
                "episode left empty (extend meta/canonical_vocabulary.json to "
                "cover the missing phase)",
                record.episode_index,
            )
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

    # ------------------------------------------------------------------
    # Canonical-vocabulary helpers
    # ------------------------------------------------------------------

    def _subtask_vocabulary_block(self) -> str:
        """Bullet-list of canonical subtasks the VLM must pick from.

        Returns an empty string when no vocabulary is configured —
        ``module_1_subtasks.txt`` then falls back to its free-form
        rules (original behaviour).
        """
        if self.vocabulary is None or not self.vocabulary.subtasks:
            return ""
        bullets = "\n".join(f"- {s}" for s in self.vocabulary.subtasks)
        return (
            "You MUST choose each subtask label verbatim from this canonical "
            "vocabulary — pick the closest match for each phase of the demo, "
            "and reuse the SAME string every time that phase recurs. The "
            "low-level policy is conditioned on these exact strings; any "
            "novel paraphrase you invent will make its conditioning OOD.\n"
            "Canonical subtask labels:\n"
            f"{bullets}\n\n"
        )

    def _memory_vocabulary_block(self) -> str:
        """Bullet-list of canonical memory milestones the VLM must pick from."""
        if self.vocabulary is None or not self.vocabulary.memory_milestones:
            return ""
        bullets = "\n".join(f"- {m}" for m in self.vocabulary.memory_milestones)
        return (
            "Compose the memory by picking ONLY from this canonical milestone "
            "list — append a milestone (or rewrite the running memory to "
            "compress past ones) using these exact phrases. Do not invent new "
            "wording: every paraphrase weakens the downstream conditioning.\n"
            "Canonical memory milestones:\n"
            f"{bullets}\n\n"
        )

    _NORMALIZE_STRIP_TOKENS: frozenset[str] = frozenset({"the", "a", "an"})

    def _canonicalize_subtask(self, text: str) -> str:
        """Validate ``text`` against the canonical vocabulary; no fuzzy snap.

        Without a vocabulary, the original text passes through. With a
        vocabulary, accept the span only if its normalised form (lower-
        cased, articles stripped, whitespace collapsed) matches a
        canonical entry exactly — the canonical wording is returned so
        the supervised string is byte-identical across episodes.

        Off-vocab spans are dropped (empty string). Upstream
        ``_generate_subtasks`` triggers a targeted retry before reaching
        the drop path; this function never snaps or warps a span into
        a different label.
        """
        if self.vocabulary is None or not self.vocabulary.subtasks:
            return text.strip()
        normalised = self._normalize(text)
        if not normalised:
            return ""
        for candidate in self.vocabulary.subtasks:
            if self._normalize(candidate) == normalised:
                return candidate
        return ""

    @classmethod
    def _normalize(cls, text: str) -> str:
        """Lowercase, strip articles, collapse whitespace, drop punctuation."""
        words = [
            w.strip(".,:;\"'!?()")
            for w in text.lower().replace(",", " ").split()
        ]
        return " ".join(w for w in words if w and w not in cls._NORMALIZE_STRIP_TOKENS)

    def _invalid_subtasks(self, spans: list[dict[str, Any]]) -> list[str]:
        """Return the unique off-vocab subtask strings the VLM produced."""
        seen: list[str] = []
        for span in spans:
            text = str((span or {}).get("text") or "").strip()
            if not text:
                continue
            if self._canonicalize_subtask(text):
                continue
            if text not in seen:
                seen.append(text)
        return seen

    def _build_subtask_retry_message(
        self, original_messages: list[dict[str, Any]], invalid: list[str]
    ) -> list[dict[str, Any]]:
        """Compose a one-shot correction prompt naming the off-vocab strings."""
        assert self.vocabulary is not None
        canonical = "\n".join(f"- {s}" for s in self.vocabulary.subtasks)
        invalid_list = "\n".join(f"- {s!r}" for s in invalid)
        correction = (
            "Your previous response included subtask labels that are NOT in "
            "the canonical vocabulary:\n"
            f"{invalid_list}\n\n"
            "Re-emit the same segmentation (same number of spans, same start/end "
            "timestamps where they were valid) but replace every off-vocab "
            "label with the EXACT canonical string for that phase, copied "
            "verbatim from this list:\n"
            f"{canonical}\n\n"
            "Strict rules:\n"
            "- Output strings must be byte-for-byte identical to entries above.\n"
            "- No articles, no adverbs, no extra words.\n"
            "- If a phase truly has no canonical match, omit that span entirely.\n"
            "Return the same JSON shape as before."
        )
        # Append the correction as an additional user turn; the model
        # sees the original prompt + its prior output is implied by the
        # conversation context (the VLM client is stateless, so we
        # re-send the original content plus this correction).
        retry_messages = [
            {
                "role": m.get("role", "user"),
                "content": (
                    m.get("content")
                    if isinstance(m.get("content"), str)
                    else list(m.get("content") or [])
                ),
            }
            for m in original_messages
        ]
        retry_messages.append({"role": "user", "content": correction})
        return retry_messages

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
            vocabulary_block=self._memory_vocabulary_block(),
        )
        memory = self._vlm_field(self._text_message(prompt), "memory")
        return memory.strip() if isinstance(memory, str) else ""
