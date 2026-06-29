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
"""Pre-write validation against staged outputs.

Runs after all three modules have written their per-episode artifacts but
*before* the writer rewrites parquet shards. The validator never touches
parquet; it only inspects the staging tree and the source frame timestamps
exposed by :class:`EpisodeRecord`.

Checks (per the plan's "Intermediate staging and validation" section):

- exact timestamp alignment against source frame timestamps
- no orphan speech / interjection pairs
- plan / memory emission consistency (events have a paired persistent row)
- VQA assistant ``content`` is valid JSON (one of bbox / keypoint / count /
  attribute / spatial)
- every row maps to its correct column under :func:`column_for_style`
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.datasets.language import (
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
    column_for_style,
    is_view_dependent_style,
    validate_camera_field,
)

from .reader import EpisodeRecord
from .staging import EpisodeStaging

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Outcome of one validation pass across all episodes."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    episodes_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def summary(self) -> str:
        return f"checked={self.episodes_checked} errors={len(self.errors)} warnings={len(self.warnings)}"


VQA_ANSWER_SHAPES: dict[str, set[str]] = {
    "bbox": {"detections"},
    "keypoint": {"label", "point_format", "point"},
    "count": {"label", "count"},
    "attribute": {"label", "attribute", "value"},
    "spatial": {"subject", "relation", "object"},
}


def classify_vqa_answer(payload: Any) -> str | None:
    """Best-effort classification of a VQA answer payload to a question type."""
    if not isinstance(payload, dict):
        return None
    keys = set(payload.keys())
    for kind, required in VQA_ANSWER_SHAPES.items():
        if required.issubset(keys):
            return kind
    return None


@dataclass
class StagingValidator:
    """Walks the staging tree and produces a :class:`ValidationReport`."""

    timestamp_atol: float = 0.0  # exact-match by default
    dataset_camera_keys: tuple[str, ...] | None = None
    """Known ``observation.images.*`` keys on the dataset. When set, the
    validator additionally enforces that every view-dependent row's
    ``camera`` field references one of these keys. Pass ``None`` (default)
    to skip that cross-check (e.g. in unit tests with no real dataset)."""

    def validate(
        self,
        records: Sequence[EpisodeRecord],
        staging_dir: Path,
    ) -> ValidationReport:
        report = ValidationReport()
        for record in records:
            self._validate_episode(record, staging_dir, report)
            report.episodes_checked += 1
        return report

    def _validate_episode(
        self,
        record: EpisodeRecord,
        staging_dir: Path,
        report: ValidationReport,
    ) -> None:
        staging = EpisodeStaging(staging_dir, record.episode_index)
        staged = staging.read_all()
        all_rows: list[dict[str, Any]] = []
        for module_name, rows in staged.items():
            for row in rows:
                row = {**row, "_module": module_name}
                all_rows.append(row)

        frame_ts = set(record.frame_timestamps)

        events: list[dict[str, Any]] = []
        persistent: list[dict[str, Any]] = []
        for row in all_rows:
            self._check_column_routing(row, report, record.episode_index)
            self._check_camera_field(row, report, record.episode_index, self.dataset_camera_keys)
            # ``_check_column_routing`` already recorded any unknown-style error;
            # don't let the same ``column_for_style`` lookup raise here uncaught.
            try:
                column = column_for_style(row.get("style"))
            except ValueError:
                continue
            if column == LANGUAGE_PERSISTENT:
                persistent.append(row)
            else:
                events.append(row)

        for row in events:
            self._check_event_timestamp_alignment(row, frame_ts, report, record.episode_index)

        self._check_speech_interjection_pairs(events, report, record.episode_index)
        self._check_plan_memory_consistency(persistent, events, report, record.episode_index)
        self._check_vqa_json(events, report, record.episode_index)
        self._check_vqa_uniqueness_per_frame_camera(events, report, record.episode_index)

    def _check_camera_field(
        self,
        row: dict[str, Any],
        report: ValidationReport,
        episode_index: int,
        dataset_camera_keys: Sequence[str] | None,
    ) -> None:
        """Enforce the camera invariant + that the key matches the dataset's cameras."""
        style = row.get("style")
        camera = row.get("camera")
        try:
            validate_camera_field(style, camera)
        except ValueError as exc:
            report.add_error(f"ep={episode_index} module={row.get('_module')}: {exc}")
            return
        if is_view_dependent_style(style) and dataset_camera_keys and camera not in dataset_camera_keys:
            report.add_error(
                f"ep={episode_index} module={row.get('_module')}: camera {camera!r} on style "
                f"{style!r} is not one of the dataset's video keys {sorted(dataset_camera_keys)!r}"
            )

    def _check_vqa_uniqueness_per_frame_camera(
        self,
        events: Iterable[dict[str, Any]],
        report: ValidationReport,
        episode_index: int,
    ) -> None:
        """Ensure at most one (vqa, user) and one (vqa, assistant) per (t, camera)."""
        counts: dict[tuple[float, str, str], int] = {}
        for row in events:
            if row.get("style") != "vqa":
                continue
            ts = row.get("timestamp")
            camera = row.get("camera")
            role = row.get("role")
            if ts is None or camera is None or role is None:
                continue  # other validators flag these
            key = (float(ts), str(camera), str(role))
            counts[key] = counts.get(key, 0) + 1
        for (ts, camera, role), n in counts.items():
            if n > 1:
                report.add_error(
                    f"ep={episode_index}: {n} duplicate vqa rows at t={ts} "
                    f"camera={camera!r} role={role!r}; expected at most one per (t, camera, role)"
                )

    def _check_column_routing(
        self,
        row: dict[str, Any],
        report: ValidationReport,
        episode_index: int,
    ) -> None:
        style = row.get("style")
        module = row.get("_module")
        try:
            target_col = column_for_style(style)
        except ValueError:
            report.add_error(f"ep={episode_index} module={module}: unknown style {style!r}")
            return
        if module == "plan" and target_col != LANGUAGE_PERSISTENT:
            report.add_error(
                f"ep={episode_index} module=plan emitted style {style!r} that routes to {target_col} (must be persistent)"
            )
        if module in {"interjections", "vqa"} and target_col != LANGUAGE_EVENTS:
            report.add_error(
                f"ep={episode_index} module={module} emitted style {style!r} that routes to {target_col} (must be events)"
            )

    def _check_event_timestamp_alignment(
        self,
        row: dict[str, Any],
        frame_ts: set[float],
        report: ValidationReport,
        episode_index: int,
    ) -> None:
        ts = row.get("timestamp")
        if ts is None:
            report.add_error(f"ep={episode_index}: event row missing timestamp: {row!r}")
            return
        if self.timestamp_atol == 0.0:
            if float(ts) not in frame_ts:
                report.add_error(
                    f"ep={episode_index}: event row timestamp {ts!r} does not match any source frame timestamp"
                )
        else:
            if not any(abs(float(ts) - f) <= self.timestamp_atol for f in frame_ts):
                report.add_error(
                    f"ep={episode_index}: event row timestamp {ts!r} not within {self.timestamp_atol}s of any frame"
                )

    def _check_speech_interjection_pairs(
        self,
        events: Iterable[dict[str, Any]],
        report: ValidationReport,
        episode_index: int,
    ) -> None:
        speech_ts: dict[float, int] = {}
        interjection_ts: dict[float, int] = {}
        for row in events:
            ts = row.get("timestamp")
            if ts is None:
                continue
            ts_f = float(ts)
            if row.get("style") is None and row.get("role") == "assistant":
                speech_ts[ts_f] = speech_ts.get(ts_f, 0) + 1
            if row.get("style") == "interjection":
                interjection_ts[ts_f] = interjection_ts.get(ts_f, 0) + 1

        for ts in interjection_ts:
            if ts not in speech_ts:
                report.add_error(f"ep={episode_index}: interjection at t={ts} has no paired speech atom")

    def _check_plan_memory_consistency(
        self,
        persistent: Sequence[dict[str, Any]],
        events: Sequence[dict[str, Any]],
        report: ValidationReport,
        episode_index: int,
    ) -> None:
        plan_ts = sorted({float(r["timestamp"]) for r in persistent if r.get("style") == "plan"})
        memory_ts = sorted({float(r["timestamp"]) for r in persistent if r.get("style") == "memory"})
        subtask_ts = sorted({float(r["timestamp"]) for r in persistent if r.get("style") == "subtask"})
        interjection_ts = sorted(
            {
                float(r["timestamp"])
                for r in events
                if r.get("style") == "interjection" and r.get("timestamp") is not None
            }
        )

        if persistent and not plan_ts:
            report.add_warning(f"ep={episode_index}: persistent rows present but no plan emitted")
        # every interjection should have a same-timestamp plan refresh
        for ts in interjection_ts:
            if ts not in set(plan_ts):
                report.add_error(
                    f"ep={episode_index}: interjection at t={ts} has no co-timestamped plan update"
                )
        # memory should be emitted at subtask boundaries (subset relation)
        if memory_ts and subtask_ts:
            mem_set = set(memory_ts)
            sub_set = set(subtask_ts)
            stray = sorted(mem_set - sub_set)
            if stray:
                report.add_warning(f"ep={episode_index}: memory rows at {stray} not at any subtask boundary")

    def _check_vqa_json(
        self,
        events: Iterable[dict[str, Any]],
        report: ValidationReport,
        episode_index: int,
    ) -> None:
        for row in events:
            if row.get("style") != "vqa" or row.get("role") != "assistant":
                continue
            content = row.get("content")
            if content is None:
                report.add_error(
                    f"ep={episode_index}: VQA assistant row at t={row.get('timestamp')} has null content"
                )
                continue
            try:
                payload = json.loads(content)
            except (TypeError, ValueError) as exc:
                report.add_error(
                    f"ep={episode_index}: VQA assistant content not valid JSON at t={row.get('timestamp')}: {exc}"
                )
                continue
            shape = classify_vqa_answer(payload)
            if shape is None:
                report.add_error(
                    f"ep={episode_index}: VQA assistant payload at t={row.get('timestamp')} does not match any known shape: keys={list(payload) if isinstance(payload, dict) else type(payload).__name__}"
                )
