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
"""Validator behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ``lerobot.annotations`` imports pull in ``lerobot.datasets`` (-> the HF
# ``datasets`` library), which only ships under the ``dataset`` extra. Skip
# this module in tiers without it instead of erroring at import.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

from lerobot.annotations.steerable_pipeline.reader import iter_episodes  # noqa: E402
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging  # noqa: E402
from lerobot.annotations.steerable_pipeline.validator import StagingValidator  # noqa: E402
from lerobot.annotations.steerable_pipeline.writer import speech_atom  # noqa: E402


def _validate(root: Path, staging_dir: Path):
    records = list(iter_episodes(root))
    return StagingValidator().validate(records, staging_dir)


def test_validator_catches_misaligned_timestamps(fixture_dataset_root: Path, tmp_path: Path) -> None:
    staging_dir = tmp_path / "stage"
    EpisodeStaging(staging_dir, 0).write(
        "vqa",
        [
            {
                "role": "assistant",
                "content": json.dumps({"label": "cup", "count": 2}, sort_keys=True),
                "style": "vqa",
                "timestamp": 9.999,  # not on any 10 fps frame
                "tool_calls": None,
            }
        ],
    )
    report = _validate(fixture_dataset_root, staging_dir)
    assert not report.ok
    assert any("does not match any source frame timestamp" in e for e in report.errors)


def test_validator_catches_orphan_speech(fixture_dataset_root: Path, tmp_path: Path) -> None:
    staging_dir = tmp_path / "stage"
    EpisodeStaging(staging_dir, 0).write(
        "interjections",
        [
            speech_atom(0.0, "Got it."),
            # interjection at 0.3s with NO paired speech
            {
                "role": "user",
                "content": "skip it",
                "style": "interjection",
                "timestamp": 0.3,
                "tool_calls": None,
            },
        ],
    )
    report = _validate(fixture_dataset_root, staging_dir)
    assert not report.ok
    assert any("paired speech" in e for e in report.errors)


def test_validator_catches_inconsistent_plan_memory(fixture_dataset_root: Path, tmp_path: Path) -> None:
    staging_dir = tmp_path / "stage"
    EpisodeStaging(staging_dir, 0).write(
        "plan",
        [
            {
                "role": "assistant",
                "content": "1. do x",
                "style": "plan",
                "timestamp": 0.0,
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": "do x",
                "style": "subtask",
                "timestamp": 0.0,
                "tool_calls": None,
            },
        ],
    )
    EpisodeStaging(staging_dir, 0).write(
        "interjections",
        [
            speech_atom(0.0, "Got it."),
            speech_atom(0.4, "Replanning."),
            {
                "role": "user",
                "content": "replan",
                "style": "interjection",
                "timestamp": 0.4,
                "tool_calls": None,
            },
        ],
    )
    report = _validate(fixture_dataset_root, staging_dir)
    # missing co-timestamped plan refresh at 0.4s → error
    assert not report.ok
    assert any("co-timestamped plan update" in e for e in report.errors)


def test_validator_catches_wrong_column(fixture_dataset_root: Path, tmp_path: Path) -> None:
    staging_dir = tmp_path / "stage"
    EpisodeStaging(staging_dir, 0).write(
        "plan",
        [
            {"role": "user", "content": "where?", "style": "vqa", "timestamp": 0.0, "tool_calls": None},
        ],
    )
    report = _validate(fixture_dataset_root, staging_dir)
    assert not report.ok
    assert any("plan emitted style 'vqa'" in e or "must be persistent" in e for e in report.errors)
