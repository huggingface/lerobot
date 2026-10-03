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
"""Integration tests for the ``ecot`` dense Embodied Chain-of-Thought module.

These exercise the wiring through non-new modules: the module writes
``style="ecot"`` rows to staging, and the full ``Executor`` phase ->
``StagingValidator`` -> ``LanguageColumnsWriter`` path lands them in the
parquet ``language_persistent`` column.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import PIL.Image
import pytest

# ``lerobot.annotations`` imports pull in ``lerobot.datasets`` (-> the HF
# ``datasets`` library), which only ships under the ``dataset`` extra.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

import pyarrow.parquet as pq  # noqa: E402

from lerobot.annotations.steerable_pipeline.config import (  # noqa: E402
    AnnotationPipelineConfig,
    EcotConfig,
    InterjectionsConfig,
    PlanConfig,
    VqaConfig,
)
from lerobot.annotations.steerable_pipeline.executor import Executor  # noqa: E402
from lerobot.annotations.steerable_pipeline.modules import (  # noqa: E402
    EcotReasoningModule,
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot.annotations.steerable_pipeline.reader import iter_episodes  # noqa: E402
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging  # noqa: E402
from lerobot.annotations.steerable_pipeline.validator import StagingValidator  # noqa: E402
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter  # noqa: E402
from lerobot.datasets.language import LANGUAGE_PERSISTENT  # noqa: E402

from ._helpers import make_canned_responder  # noqa: E402


@dataclass
class _StubFrameProvider:
    """Returns one tiny PIL image per requested timestamp (contact-sheet safe)."""

    sentinel: Any = field(default_factory=lambda: PIL.Image.new("RGB", (32, 24)))
    cameras: tuple[str, ...] = ("observation.images.top",)

    @property
    def camera_keys(self) -> list[str]:
        return list(self.cameras)

    @property
    def camera_key(self) -> str | None:
        return self.cameras[0] if self.cameras else None

    def frames_at(self, record, timestamps, camera_key=None):  # noqa: ARG002
        return [self.sentinel] * len(timestamps)

    def video_for_episode(self, record, max_frames, camera_key=None):  # noqa: ARG002
        n = min(max_frames, len(record.frame_timestamps))
        return [self.sentinel] * n


_ECOT_PAYLOAD: dict[str, str] = {
    "scene_perception": "a counter with a sponge near a sink",
    "object_identification": "yellow rectangular sponge",
    "task_plan": "1. grasp sponge 2. wipe counter 3. place sponge in sink",
    "subtask_decomposition": "reach and grasp the sponge",
}


def test_ecot_module_emits_persistent_reasoning_rows(single_episode_root: Path, tmp_path: Path) -> None:
    vlm = make_canned_responder({"Embodied Chain-of-Thought": _ECOT_PAYLOAD})
    module = EcotReasoningModule(
        vlm=vlm,
        config=EcotConfig(emission_hz=1.0, window_seconds=1.0),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("ecot")

    assert rows, "ecot module emitted zero rows"
    frame_set = set(record.frame_timestamps)
    for row in rows:
        assert row["style"] == "ecot"
        assert row["role"] == "assistant"
        assert row["camera"] is None
        assert row["tool_calls"] is None
        # persistent rows are stamped on real source frames
        assert row["timestamp"] in frame_set
        decoded = json.loads(row["content"])
        assert decoded["scene_perception"] == _ECOT_PAYLOAD["scene_perception"]
        assert decoded["subtask_decomposition"] == _ECOT_PAYLOAD["subtask_decomposition"]
        # Serialized keys must appear in the canonical ECoT progression
        # (perception -> identification -> planning -> decomposition). The
        # downstream training recipe consumes ``content`` verbatim, so an
        # alphabetized dump would teach models the wrong order.
        assert list(decoded.keys()) == [
            "scene_perception",
            "object_identification",
            "task_plan",
            "subtask_decomposition",
        ]
        assert row["content"].startswith('{"scene_perception":')


def test_ecot_no_cameras_warns_and_emits_zero_rows(
    single_episode_root: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A frame provider with no cameras must emit zero rows and warn once
    (matching ``GeneralVqaModule``'s pattern) — silent no-op is unacceptable
    when the module is enabled by default."""
    vlm = make_canned_responder({"Embodied Chain-of-Thought": _ECOT_PAYLOAD})
    module = EcotReasoningModule(
        vlm=vlm,
        config=EcotConfig(emission_hz=1.0, window_seconds=1.0),
        frame_provider=_StubFrameProvider(cameras=()),
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)

    with caplog.at_level("WARNING"):
        module.run_episode(record, staging)

    assert staging.read("ecot") == []
    assert any("ecot module found no cameras" in r.message for r in caplog.records), (
        "no-camera warning not surfaced"
    )

    # Second run stays quiet: the one-time flag suppresses the repeat.
    caplog.clear()
    with caplog.at_level("WARNING"):
        module.run_episode(record, staging)
    assert not any("ecot module found no cameras" in r.message for r in caplog.records), (
        "no-camera warning fired more than once"
    )


def test_ecot_disabled_is_skipped_by_executor(single_episode_root: Path) -> None:
    """A disabled ``ecot`` config makes the executor skip the phase (the
    ``enabled`` gate lives on the executor, mirroring plan/vqa/interjections)."""
    vlm = make_canned_responder({"Embodied Chain-of-Thought": _ECOT_PAYLOAD})
    cfg = AnnotationPipelineConfig(
        plan=PlanConfig(enabled=False),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
        ecot=EcotConfig(enabled=False),
    )
    provider = _StubFrameProvider()
    executor = Executor(
        config=cfg,
        plan=PlanSubtasksMemoryModule(vlm=vlm, config=cfg.plan, frame_provider=provider),
        interjections=InterjectionsAndSpeechModule(vlm=vlm, config=cfg.interjections, seed=cfg.seed),
        vqa=GeneralVqaModule(vlm=vlm, config=cfg.vqa, seed=cfg.seed, frame_provider=provider),
        writer=LanguageColumnsWriter(),
        validator=StagingValidator(),
        ecot=EcotReasoningModule(vlm=vlm, config=cfg.ecot, frame_provider=provider),
    )
    summary = executor.run(single_episode_root)

    ecot_phases = [p for p in summary.phases if p.name == "ecot"]
    assert ecot_phases, "executor never ran an ecot phase"
    assert ecot_phases[0].episodes_processed == 0
    assert ecot_phases[0].episodes_skipped == 1

    table = pq.read_table(single_episode_root / "data" / "chunk-000" / "file-000.parquet")
    persistent_lists = table.column(LANGUAGE_PERSISTENT).to_pylist()
    ecot_rows = [r for rows in persistent_lists for r in rows if r.get("style") == "ecot"]
    assert ecot_rows == []


def test_ecot_wired_into_executor_writes_persistent_column(single_episode_root: Path) -> None:
    """End-to-end: the executor's ecot phase -> validator -> writer lands
    ECoT traces in the parquet ``language_persistent`` column."""
    vlm = make_canned_responder({"Embodied Chain-of-Thought": _ECOT_PAYLOAD})
    # Disable the sibling modules so the only persistent rows produced are
    # the ecot ones — keeps the assertion focused on the new wiring.
    cfg = AnnotationPipelineConfig(
        plan=PlanConfig(enabled=False),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
        ecot=EcotConfig(emission_hz=1.0, window_seconds=1.0),
    )
    provider = _StubFrameProvider()
    executor = Executor(
        config=cfg,
        plan=PlanSubtasksMemoryModule(vlm=vlm, config=cfg.plan, frame_provider=provider),
        interjections=InterjectionsAndSpeechModule(vlm=vlm, config=cfg.interjections, seed=cfg.seed),
        vqa=GeneralVqaModule(vlm=vlm, config=cfg.vqa, seed=cfg.seed, frame_provider=provider),
        writer=LanguageColumnsWriter(),
        validator=StagingValidator(),
        ecot=EcotReasoningModule(vlm=vlm, config=cfg.ecot, frame_provider=provider),
    )
    summary = executor.run(single_episode_root)
    assert summary.validation_report.ok, summary.validation_report.summary()

    # the ecot phase actually ran and processed the episode
    ecot_phases = [p for p in summary.phases if p.name == "ecot"]
    assert ecot_phases, "executor never ran an ecot phase"
    assert ecot_phases[0].episodes_processed == 1

    table = pq.read_table(single_episode_root / "data" / "chunk-000" / "file-000.parquet")
    persistent_lists = table.column(LANGUAGE_PERSISTENT).to_pylist()
    ecot_rows = [r for rows in persistent_lists for r in rows if r.get("style") == "ecot"]
    assert ecot_rows, "no ecot rows reached the language_persistent column"
    decoded = json.loads(ecot_rows[0]["content"])
    assert "scene_perception" in decoded
    assert "subtask_decomposition" in decoded
