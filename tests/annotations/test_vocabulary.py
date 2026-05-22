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
"""Vocabulary-discovery phase (phase 0) tests."""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.annotations.steerable_pipeline.config import (
    PlanConfig,
    VocabularyConfig,
)
from lerobot.annotations.steerable_pipeline.modules import PlanSubtasksMemoryModule
from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.vocabulary import (
    Vocabulary,
    VocabularyDiscoveryModule,
    load_vocabulary,
    save_vocabulary,
    vocabulary_path,
)

from ._helpers import make_canned_responder


_CANONICAL_SUBTASKS = (
    "grasp blue cube",
    "place blue cube in box",
    "retract arm",
)
_CANONICAL_MEMORY = (
    "I picked up the blue cube.",
    "I placed the blue cube in the box.",
)


# ---------------------------------------------------------------------------
# Vocabulary dataclass + on-disk round-trip
# ---------------------------------------------------------------------------


def test_vocabulary_roundtrip(tmp_path: Path) -> None:
    vocab = Vocabulary(
        subtasks=_CANONICAL_SUBTASKS, memory_milestones=_CANONICAL_MEMORY
    )
    save_path = save_vocabulary(tmp_path, vocab)
    assert save_path == vocabulary_path(tmp_path)
    assert save_path.exists()

    loaded = load_vocabulary(tmp_path)
    assert loaded is not None
    assert loaded.subtasks == _CANONICAL_SUBTASKS
    assert loaded.memory_milestones == _CANONICAL_MEMORY


def test_vocabulary_load_missing_returns_none(tmp_path: Path) -> None:
    assert load_vocabulary(tmp_path) is None


def test_vocabulary_load_malformed_returns_none(tmp_path: Path) -> None:
    path = vocabulary_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json", encoding="utf-8")
    assert load_vocabulary(tmp_path) is None


def test_vocabulary_load_empty_payload_returns_none(tmp_path: Path) -> None:
    path = vocabulary_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"subtasks": [], "memory_milestones": []}), encoding="utf-8")
    assert load_vocabulary(tmp_path) is None


# ---------------------------------------------------------------------------
# Discovery module
# ---------------------------------------------------------------------------


def test_vocabulary_discovery_calls_vlm_and_returns_vocab(
    fixture_dataset_root: Path,
) -> None:
    vlm = make_canned_responder(
        {
            "canonical vocabulary": {
                "subtasks": list(_CANONICAL_SUBTASKS),
                "memory_milestones": list(_CANONICAL_MEMORY),
            }
        }
    )
    module = VocabularyDiscoveryModule(vlm=vlm, config=VocabularyConfig(sample_episodes=2))
    records = list(iter_episodes(fixture_dataset_root))
    vocab = module.discover(records)
    assert vocab is not None
    assert vocab.subtasks == _CANONICAL_SUBTASKS
    assert vocab.memory_milestones == _CANONICAL_MEMORY


def test_vocabulary_discovery_reuses_existing(fixture_dataset_root: Path) -> None:
    """``reuse_existing=True`` short-circuits the VLM call entirely."""

    def _explode(_messages):  # pragma: no cover - must not be called
        raise AssertionError("VLM should not be invoked when reusing existing vocabulary")

    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

    vlm = StubVlmClient(responder=_explode)
    module = VocabularyDiscoveryModule(
        vlm=vlm, config=VocabularyConfig(reuse_existing=True)
    )
    records = list(iter_episodes(fixture_dataset_root))
    existing = Vocabulary(subtasks=("a", "b"), memory_milestones=("I a.",))
    vocab = module.discover(records, existing=existing)
    assert vocab is existing


def test_vocabulary_discovery_empty_payload_returns_none(
    fixture_dataset_root: Path,
) -> None:
    vlm = make_canned_responder({"canonical vocabulary": {"subtasks": [], "memory_milestones": []}})
    module = VocabularyDiscoveryModule(vlm=vlm, config=VocabularyConfig())
    records = list(iter_episodes(fixture_dataset_root))
    assert module.discover(records) is None


# ---------------------------------------------------------------------------
# PlanSubtasksMemoryModule consumes the vocabulary
# ---------------------------------------------------------------------------


def test_plan_module_inlines_vocab_into_subtask_prompt(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    captured: list[str] = []

    def responder(messages):
        # Find the last user text block and stash it for inspection.
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        captured.append(block.get("text", ""))
        # Return canned subtasks; pick the first two canonical strings so
        # the validator accepts them.
        return {
            "subtasks": [
                {"text": "grasp blue cube", "start": 0.0, "end": 0.4},
                {"text": "place blue cube in box", "start": 0.4, "end": 0.9},
            ]
        }

    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

    vlm = StubVlmClient(responder=responder)
    vocab = Vocabulary(subtasks=_CANONICAL_SUBTASKS, memory_milestones=_CANONICAL_MEMORY)
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=PlanConfig(n_task_rephrasings=0),
        vocabulary=vocab,
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    # The subtask prompt (and the memory prompt) carries the canonical
    # bullet list so the VLM can't paraphrase them away.
    assert any("Canonical subtask labels:" in t for t in captured)
    assert any("grasp blue cube" in t for t in captured)


def test_plan_module_canonicalizes_paraphrased_subtask(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """Off-vocab paraphrase with high token overlap snaps to canonical form."""
    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

    def responder(_messages):
        return {
            "subtasks": [
                # paraphrase of "grasp blue cube" — overlapping tokens
                {"text": "grasp the blue cube", "start": 0.0, "end": 0.4},
                # paraphrase of "place blue cube in box" — high overlap
                {"text": "place the blue cube into the box", "start": 0.4, "end": 0.9},
            ]
        }

    vlm = StubVlmClient(responder=responder)
    vocab = Vocabulary(subtasks=_CANONICAL_SUBTASKS, memory_milestones=_CANONICAL_MEMORY)
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=PlanConfig(n_task_rephrasings=0),
        vocabulary=vocab,
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("plan")
    subtask_texts = [r["content"] for r in rows if r["style"] == "subtask"]
    # Both paraphrases snapped onto canonical strings.
    assert subtask_texts == ["grasp blue cube", "place blue cube in box"]


def test_plan_module_drops_off_vocab_subtask(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """A subtask with low overlap to every canonical label is dropped."""
    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

    def responder(_messages):
        return {
            "subtasks": [
                # in-vocab
                {"text": "grasp blue cube", "start": 0.0, "end": 0.4},
                # off-vocab hallucination — no token overlap above the
                # Jaccard floor; should be dropped.
                {"text": "perform a fancy macarena dance", "start": 0.4, "end": 0.9},
            ]
        }

    vlm = StubVlmClient(responder=responder)
    vocab = Vocabulary(subtasks=_CANONICAL_SUBTASKS, memory_milestones=_CANONICAL_MEMORY)
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=PlanConfig(n_task_rephrasings=0),
        vocabulary=vocab,
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("plan")
    subtask_texts = [r["content"] for r in rows if r["style"] == "subtask"]
    assert subtask_texts == ["grasp blue cube"]


def test_plan_module_without_vocab_passes_through(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """No vocabulary configured → original free-form behavior is preserved."""
    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

    def responder(_messages):
        return {
            "subtasks": [
                {"text": "any free-form text the VLM wants", "start": 0.0, "end": 1.0},
            ]
        }

    vlm = StubVlmClient(responder=responder)
    module = PlanSubtasksMemoryModule(
        vlm=vlm, config=PlanConfig(n_task_rephrasings=0)
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("plan")
    subtask_texts = [r["content"] for r in rows if r["style"] == "subtask"]
    assert subtask_texts == ["any free-form text the VLM wants"]
