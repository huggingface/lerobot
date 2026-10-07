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
"""Opt-in E2E smoke run for ``make annotation-e2e``.

Builds the shared annotation fixture (:func:`build_annotation_dataset`),
runs the full annotation pipeline against it with a stub VLM, and prints a
short report. This is intentionally not a pytest test — it exercises the
CLI plumbing — but it reuses the same on-disk dataset builder as the pytest
fixtures so there is no duplicated fixture code.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig
from lerobot.annotations.steerable_pipeline.executor import Executor
from lerobot.annotations.steerable_pipeline.modules import (
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot.annotations.steerable_pipeline.validator import StagingValidator
from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter
from tests.fixtures.dataset_factories import build_annotation_dataset


def _stub_responder(messages):
    text = ""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
            elif isinstance(content, str):
                text = content
    if "atomic subtasks" in text:
        return {
            "subtasks": [
                {"text": "grasp the bottle", "start": 0.0, "end": 1.0},
                {"text": "pour into the cup", "start": 1.0, "end": 2.0},
                {"text": "place the bottle down", "start": 2.0, "end": 3.0},
            ]
        }
    if "compressed semantic memory" in text:
        return {"memory": "poured once"}
    if "acknowledgement the robot" in text:
        return {"text": "Sure."}
    if "compact interjection" in text:
        return {"interjection": "use less water", "speech": "Using less water."}
    if "frame-grounded visual question" in text:
        return {"question": "How many cups?", "answer": {"label": "cup", "count": 1}}
    return None


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = build_annotation_dataset(
            Path(tmp) / "ds",
            episode_specs=[(0, 30, "Pour water into the cup.")],
            fps=10,
        )
        vlm = StubVlmClient(responder=_stub_responder)
        cfg = AnnotationPipelineConfig()
        executor = Executor(
            config=cfg,
            plan=PlanSubtasksMemoryModule(vlm=vlm, config=cfg.plan),
            interjections=InterjectionsAndSpeechModule(vlm=vlm, config=cfg.interjections, seed=cfg.seed),
            vqa=GeneralVqaModule(vlm=vlm, config=cfg.vqa, seed=cfg.seed),
            writer=LanguageColumnsWriter(),
            validator=StagingValidator(),
        )
        summary = executor.run(root)
        print(f"phases={[(p.name, p.episodes_processed) for p in summary.phases]}")
        print(f"validation: {summary.validation_report.summary()}")
        print(f"shards rewritten: {len(summary.written_paths)}")

        # Assert the interjection code path actually fired — otherwise a stale
        # canned-VLM marker would silently produce zero interjections and this
        # smoke run would still "pass" by only printing.
        import pyarrow.parquet as pq  # noqa: PLC0415

        events = [
            r
            for shard in summary.written_paths
            for ev in pq.read_table(shard).column("language_events").to_pylist()
            for r in ev
        ]
        n_interjections = sum(1 for r in events if r.get("style") == "interjection")
        n_speech = sum(1 for r in events if r.get("style") is None and r.get("role") == "assistant")
        print(f"interjections={n_interjections} speech_atoms={n_speech}")
        assert n_interjections > 0, "no interjection rows produced — check the interjection prompt marker"
        assert n_speech > 0, "no speech tool-call atoms produced — check the speech prompt marker"
    return 0


if __name__ == "__main__":
    sys.exit(main())
