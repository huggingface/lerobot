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
"""End-to-end smoke: pipeline output → PR 1 canonical recipe rendering."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from lerobot.annotations.steerable_pipeline.config import (
    AnnotationPipelineConfig,
    Module1Config,
    Module2Config,
    Module3Config,
)
from lerobot.annotations.steerable_pipeline.executor import Executor
from lerobot.annotations.steerable_pipeline.modules import (
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot.annotations.steerable_pipeline.validator import StagingValidator
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter
from lerobot.configs.recipe import TrainingRecipe
from lerobot.datasets.language_render import render_sample

from ._helpers import make_canned_responder

_RECIPE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "lerobot"
    / "configs"
    / "recipes"
    / "subtask_mem_vqa_speech.yaml"
)


def _build_executor() -> Executor:
    vlm = make_canned_responder(
        {
            "atomic subtasks": {
                "subtasks": [
                    {"text": "grasp the bottle", "start": 0.0, "end": 0.5},
                    {"text": "pour into the cup", "start": 0.5, "end": 1.0},
                    {"text": "place the bottle down", "start": 1.0, "end": 1.5},
                ]
            },
            "concise hierarchical PLAN": {"plan": "1. grasp\n2. pour\n3. place"},
            "Update the memory": {"memory": "poured once"},
            "acknowledgement the robot": {"text": "Sure."},
            "ONE realistic interruption": {
                "interjection": "use less water",
                "speech": "Using less water.",
            },
            "frame-grounded visual question": {
                "question": "How many cups?",
                "answer": {"label": "cup", "count": 1},
            },
        },
    )
    config = AnnotationPipelineConfig(
        module_1=Module1Config(),
        module_2=Module2Config(max_interjections_per_episode=1, interjection_min_t=0.5),
        module_3=Module3Config(vqa_emission_hz=1.0, K=2),
    )
    return Executor(
        config=config,
        module_1=PlanSubtasksMemoryModule(vlm=vlm, config=config.module_1),
        module_2=InterjectionsAndSpeechModule(vlm=vlm, config=config.module_2, seed=config.seed),
        module_3=GeneralVqaModule(vlm=vlm, config=config.module_3, seed=config.seed),
        writer=LanguageColumnsWriter(),
        validator=StagingValidator(),
    )


def test_pr1_canonical_recipe_renders_nonempty_from_pipeline_output(
    single_episode_root: Path,
) -> None:
    executor = _build_executor()
    summary = executor.run(single_episode_root)
    # validator may emit warnings but no errors for the synthetic fixture
    assert summary.validation_report.ok, summary.validation_report.summary()

    table = pq.read_table(single_episode_root / "data" / "chunk-000" / "file-000.parquet")
    persistent_lists = table.column("language_persistent").to_pylist()
    events_lists = table.column("language_events").to_pylist()
    timestamps = table.column("timestamp").to_pylist()

    recipe = TrainingRecipe.from_yaml(_RECIPE_PATH) if hasattr(TrainingRecipe, "from_yaml") else None
    if recipe is None:
        # PR 1 may not expose from_yaml; load via PyYAML and TrainingRecipe(**...)
        import yaml

        loaded = yaml.safe_load(_RECIPE_PATH.read_text(encoding="utf-8"))
        recipe = TrainingRecipe(**loaded)

    rendered_any = False
    for sample_idx, (ts, persistent, events) in enumerate(
        zip(timestamps, persistent_lists, events_lists, strict=True)
    ):
        result = render_sample(
            recipe=recipe,
            persistent=persistent,
            events=events,
            t=float(ts),
            sample_idx=sample_idx,
            dataset_ctx={"task": "Pour water from the bottle into the cup."},
        )
        if result is None:
            continue
        if result["messages"]:
            rendered_any = True
            # A valid render supervises something: a text-CE target turn
            # OR a flow-only ``low_level``-stream turn (action loss).
            assert (
                result["target_message_indices"]
                or "low_level" in result["message_streams"]
            )
            break
    assert rendered_any, "recipe rendered no messages from pipeline output"

    # Sanity: speech atom appears in events column intact
    flat_events = [r for ev in events_lists for r in ev]
    speech_rows = [r for r in flat_events if r.get("style") is None and r.get("role") == "assistant"]
    assert speech_rows
    say = speech_rows[0]["tool_calls"][0]
    assert say["function"]["name"] == "say"
    assert isinstance(say["function"]["arguments"]["text"], str)
    # PR 2 no longer writes a ``tools`` column — the say schema lives as a
    # constant (``SAY_TOOL_SCHEMA``) so PR 1's row struct is the single
    # source of truth for the v3.1 schema.
    assert "tools" not in table.column_names
