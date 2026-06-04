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
"""End-to-end smoke: pipeline output → canonical recipe rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

# ``pyarrow`` and the ``lerobot.datasets`` chain (-> the HF ``datasets``
# library) only ship under the ``dataset`` extra. Skip this module in
# tiers without it instead of erroring at import.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

import pyarrow.parquet as pq  # noqa: E402

from lerobot.annotations.steerable_pipeline.config import (  # noqa: E402
    AnnotationPipelineConfig,
    InterjectionsConfig,
    PlanConfig,
    VqaConfig,
)
from lerobot.annotations.steerable_pipeline.executor import Executor  # noqa: E402
from lerobot.annotations.steerable_pipeline.modules import (  # noqa: E402
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot.annotations.steerable_pipeline.validator import StagingValidator  # noqa: E402
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter  # noqa: E402
from lerobot.configs.recipe import MessageTurn, TrainingRecipe  # noqa: E402
from lerobot.datasets.language_render import render_sample  # noqa: E402

from ._helpers import make_canned_responder  # noqa: E402


def _build_style_blend_recipe() -> TrainingRecipe:
    """Inline blend recipe that consumes every style this pipeline produces.

    The language schema/DSL work used to ship
    ``src/lerobot/configs/recipes/pi05_hirobot.yaml`` as a canonical
    example, but that file was dropped during review. The contract this
    test guards is "the recipe DSL can render non-empty messages from
    pipeline output", which doesn't require a specific YAML — so we build
    the equivalent blend in code.
    """
    return TrainingRecipe(
        blend={
            "low_level_execution": TrainingRecipe(
                weight=0.35,
                messages=[
                    MessageTurn(
                        role="user",
                        content="${task}\nPlan: ${plan}\nMemory: ${memory}",
                        stream="high_level",
                    ),
                    MessageTurn(role="assistant", content="${subtask}", stream="low_level", target=True),
                ],
            ),
            "user_interjection_response": TrainingRecipe(
                weight=0.16,
                bindings={
                    "speech": "emitted_at(t, role=assistant, tool_name=say)",
                    "interjection": "emitted_at(t, style=interjection)",
                },
                messages=[
                    MessageTurn(role="user", content="${task}", stream="high_level"),
                    MessageTurn(
                        role="user",
                        content="${interjection}",
                        stream="high_level",
                        if_present="interjection",
                    ),
                    MessageTurn(
                        role="assistant",
                        content="${plan}",
                        stream="high_level",
                        target=True,
                        if_present="plan",
                        tool_calls_from="speech",
                    ),
                ],
            ),
        }
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
            "compressed semantic memory": {"memory": "poured once"},
            "acknowledgement the robot": {"text": "Sure."},
            "compact interjection": {
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
        plan=PlanConfig(),
        interjections=InterjectionsConfig(max_interjections_per_episode=1, interjection_min_t=0.5),
        vqa=VqaConfig(vqa_emission_hz=1.0, K=2),
    )
    return Executor(
        config=config,
        plan=PlanSubtasksMemoryModule(vlm=vlm, config=config.plan),
        interjections=InterjectionsAndSpeechModule(vlm=vlm, config=config.interjections, seed=config.seed),
        vqa=GeneralVqaModule(vlm=vlm, config=config.vqa, seed=config.seed),
        writer=LanguageColumnsWriter(),
        validator=StagingValidator(),
    )


def test_canonical_recipe_renders_nonempty_from_pipeline_output(
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

    recipe = _build_style_blend_recipe()

    rendered_any = False
    for ts, persistent, events in zip(timestamps, persistent_lists, events_lists, strict=True):
        result = render_sample(
            recipe=recipe,
            persistent=persistent,
            events=events,
            t=float(ts),
            sample_idx=0,
            dataset_ctx={"task": "Pour water from the bottle into the cup."},
        )
        if result is None:
            continue
        if result["messages"]:
            rendered_any = True
            assert result["target_message_indices"]
            break
    assert rendered_any, "recipe rendered no messages from pipeline output"

    # Sanity: speech atom appears in events column intact
    flat_events = [r for ev in events_lists for r in ev]
    speech_rows = [r for r in flat_events if r.get("style") is None and r.get("role") == "assistant"]
    assert speech_rows
    say = speech_rows[0]["tool_calls"][0]
    assert say["function"]["name"] == "say"
    assert isinstance(say["function"]["arguments"]["text"], str)
    # The pipeline does not write a ``tools`` column — the say schema lives
    # as a constant (``SAY_TOOL_SCHEMA``) so the language row struct is the
    # single source of truth for the v3.1 schema.
    assert "tools" not in table.column_names
