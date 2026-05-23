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


def test_plan_module_accepts_article_only_difference(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """Articles like 'the'/'a'/'an' are stripped during validation."""
    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

    def responder(_messages):
        return {
            "subtasks": [
                # Same canonical phrase modulo "the" — should be accepted.
                {"text": "grasp the blue cube", "start": 0.0, "end": 0.4},
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


def test_plan_module_retries_when_subtask_off_vocab(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """One-shot retry replaces an off-vocab paraphrase with the canonical form."""
    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

    call_count = {"n": 0}

    def responder(messages):
        call_count["n"] += 1
        # First call: returns an off-vocab paraphrase.
        if call_count["n"] == 1:
            return {
                "subtasks": [
                    # paraphrase, not in vocab
                    {"text": "pick up blue cube", "start": 0.0, "end": 0.4},
                ]
            }
        # Second call (the retry): should contain the correction prompt;
        # respond with the canonical phrase exactly.
        last_user_text = ""
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                last_user_text = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        last_user_text = block.get("text", "")
        assert "NOT in the canonical vocabulary" in last_user_text
        return {
            "subtasks": [
                {"text": "grasp blue cube", "start": 0.0, "end": 0.4},
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
    # The retry must have fired exactly once.
    assert call_count["n"] == 2


def test_plan_module_drops_off_vocab_subtask_after_retry(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """If the VLM stays off-vocab even after the retry, the bad span is dropped."""
    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

    call_count = {"n": 0}

    def responder(_messages):
        call_count["n"] += 1
        # Both calls return the same off-vocab span — the model can't
        # be corrected. The second call also returns one in-vocab span
        # so the episode isn't empty; this lets us check that the
        # off-vocab span is dropped without affecting the in-vocab one.
        if call_count["n"] == 1:
            return {
                "subtasks": [
                    {"text": "perform a fancy macarena dance", "start": 0.0, "end": 0.4},
                    {"text": "grasp blue cube", "start": 0.4, "end": 0.9},
                ]
            }
        return {
            "subtasks": [
                {"text": "perform a fancy macarena dance", "start": 0.0, "end": 0.4},
                {"text": "grasp blue cube", "start": 0.4, "end": 0.9},
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
    # Retry fired exactly once; bad span dropped, good span kept.
    assert call_count["n"] == 2
    assert subtask_texts == ["grasp blue cube"]


def test_plan_module_bumps_collocated_subtasks_to_distinct_frames(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """Two subtasks whose starts snap to the same frame get split onto two frames.

    Without this guard, both spans would emit ``style=subtask`` rows at the
    identical persistent timestamp; the training-time renderer's
    ``active_at(t, style=subtask)`` then raises an ambiguity error.
    """
    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

    def responder(_messages):
        # Two canonical labels with starts within one frame of each other —
        # both snap to the same source frame, so the dedupe pass must bump
        # the later one to the next frame.
        return {
            "subtasks": [
                {"text": "grasp blue cube", "start": 0.40, "end": 0.42},
                {"text": "place blue cube in box", "start": 0.41, "end": 0.50},
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
    subtask_rows = [r for r in rows if r["style"] == "subtask"]
    # Both subtasks present, both on distinct timestamps.
    assert len(subtask_rows) == 2
    timestamps = [r["timestamp"] for r in subtask_rows]
    assert len(set(timestamps)) == 2, f"subtask timestamps collide: {timestamps}"
    # Order preserved: the chronologically earlier span keeps the earlier
    # frame, the later one was bumped onto the next available frame.
    assert subtask_rows[0]["content"] == "grasp blue cube"
    assert subtask_rows[1]["content"] == "place blue cube in box"
    assert subtask_rows[1]["timestamp"] > subtask_rows[0]["timestamp"]


def test_plan_module_empty_when_all_off_vocab_after_retry(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """All-off-vocab spans → episode comes out empty (no silent fuzzy snap)."""
    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

    def responder(_messages):
        # Returns the same off-vocab spans on both attempts.
        return {
            "subtasks": [
                {"text": "make a smoothie", "start": 0.0, "end": 0.4},
                {"text": "consult the wizard", "start": 0.4, "end": 0.9},
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
    # No subtask gets fabricated — better to leave the episode empty
    # so the operator notices the vocabulary gap than to silently
    # warp the labels.
    assert subtask_texts == []


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
