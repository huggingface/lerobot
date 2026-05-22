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
"""Dataset-level canonical vocabulary discovery (Phase 0).

The downstream consumer of these annotations is a low-level action expert
conditioned on the ``subtask`` string. Free-form per-episode LLM rephrasing
gives near-unique strings per occurrence, which collapses the action
expert's conditioning to noise and makes runtime subtask-paraphrase drift
catastrophic. The Hi-Robot / π0.6-MEM recipe ships a small canonical
vocabulary per environment (~10 strings) that every episode reuses; this
module derives that vocabulary automatically from the first few episode
videos and persists it next to the dataset.

Pipeline-level flow:

    Phase 0 (here): watch N sample episodes → produce vocabulary.json
    Phase 1 (plan module): reuse vocabulary on every episode, both as
                           prompt-side constraint *and* post-VLM validation

The vocabulary is JSON, lives at ``<root>/meta/canonical_vocabulary.json``,
and is human-inspectable / hand-editable — if the discovered set is wrong,
operators edit the file and re-run the pipeline without phase 0.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import VocabularyConfig
from .frames import FrameProvider, null_provider, to_video_block
from .prompts import load as load_prompt
from .reader import EpisodeRecord
from .vlm_client import VlmClient

logger = logging.getLogger(__name__)

VOCABULARY_FILENAME = "canonical_vocabulary.json"


@dataclass
class Vocabulary:
    """Canonical phrasings shared across every episode of one dataset.

    Both lists are strict: per-episode subtask + memory generation pick
    from these strings only; the downstream policy then has a small,
    repeatable target distribution to learn instead of thousands of
    LLM paraphrases.
    """

    subtasks: tuple[str, ...]
    """Imperative subtask labels — what the low-level policy is conditioned
    on. Verb-first, telegraphic, consistent object nouns. Example:
    ``("move to blue cube", "grasp blue cube", "lift blue cube",
       "place blue cube in box", "retract arm")``.
    """

    memory_milestones: tuple[str, ...]
    """First-person past-tense milestone sentences — building blocks for
    the running memory string. Example: ``("I picked up the blue cube.",
    "I placed the blue cube in the green box.")``. Each milestone maps
    1:1 onto a completed subtask phase; ``memory_at_step_k`` is the
    concatenation of milestones for completed phases.
    """

    def to_json(self) -> dict[str, list[str]]:
        return {
            "subtasks": list(self.subtasks),
            "memory_milestones": list(self.memory_milestones),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Vocabulary:
        subtasks = tuple(
            str(s).strip() for s in (payload.get("subtasks") or []) if str(s).strip()
        )
        memory_milestones = tuple(
            str(s).strip() for s in (payload.get("memory_milestones") or []) if str(s).strip()
        )
        return cls(subtasks=subtasks, memory_milestones=memory_milestones)

    def is_empty(self) -> bool:
        return not self.subtasks and not self.memory_milestones


def vocabulary_path(root: Path) -> Path:
    """Return the canonical on-disk location for the vocabulary file."""
    return root / "meta" / VOCABULARY_FILENAME


def load_vocabulary(root: Path) -> Vocabulary | None:
    """Read ``<root>/meta/canonical_vocabulary.json`` if present.

    Returns ``None`` when the file does not exist — callers fall back to
    free-form (unconstrained) subtask + memory generation, preserving the
    pipeline's behaviour on datasets that never ran phase 0.
    """
    path = vocabulary_path(root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s — proceeding without vocabulary", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("%s is not a JSON object — ignoring", path)
        return None
    vocab = Vocabulary.from_json(payload)
    if vocab.is_empty():
        return None
    return vocab


def save_vocabulary(root: Path, vocab: Vocabulary) -> Path:
    """Atomically persist ``vocab`` to ``<root>/meta/canonical_vocabulary.json``."""
    path = vocabulary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(vocab.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


@dataclass
class VocabularyDiscoveryModule:
    """Derive a dataset-level canonical vocabulary from sample episodes.

    Phase 0 of the executor: pulls ``config.sample_episodes`` episode
    videos, packs them into one Qwen-VL multi-video prompt, and asks the
    model to enumerate the small set of canonical subtask labels +
    memory milestones that recur across them. The output is persisted
    to ``meta/canonical_vocabulary.json`` and consumed by phase 1.
    """

    vlm: VlmClient
    config: VocabularyConfig
    frame_provider: FrameProvider = field(default_factory=null_provider)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def discover(
        self,
        records: Sequence[EpisodeRecord],
        *,
        existing: Vocabulary | None = None,
    ) -> Vocabulary | None:
        """Run vocabulary discovery against the first N sample episodes.

        ``existing`` short-circuits the VLM call when ``config.reuse_existing``
        is True and an on-disk vocabulary is already present — keeps re-runs
        cheap and lets operators hand-edit the file without it getting
        overwritten.
        """
        if existing is not None and self.config.reuse_existing:
            logger.info(
                "vocabulary: reusing existing (%d subtasks, %d memory milestones)",
                len(existing.subtasks),
                len(existing.memory_milestones),
            )
            return existing

        sample = list(records[: max(1, int(self.config.sample_episodes))])
        if not sample:
            return None

        task_hint = next((r.episode_task for r in sample if r.episode_task), "")
        prompt = load_prompt("module_0_vocabulary").format(
            episode_task=task_hint or "(unspecified)",
            n_episodes=len(sample),
        )
        # Pack one video block per sample episode so the VLM sees the
        # variation across episodes (different starting poses, different
        # object placements) rather than overfitting to one trajectory.
        content: list[dict[str, Any]] = []
        for record in sample:
            video_frames = self.frame_provider.video_for_episode(
                record, int(self.config.max_video_frames_per_episode)
            )
            if video_frames:
                content.extend(to_video_block(video_frames))
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        result = self.vlm.generate_json([messages])[0]
        if not isinstance(result, dict):
            logger.warning("vocabulary: VLM did not return a JSON object — skipping")
            return None

        vocab = Vocabulary.from_json(result)
        if vocab.is_empty():
            logger.warning("vocabulary: VLM returned an empty vocabulary — skipping")
            return None
        logger.info(
            "vocabulary: discovered %d subtask labels + %d memory milestones from %d episodes",
            len(vocab.subtasks),
            len(vocab.memory_milestones),
            len(sample),
        )
        return vocab
