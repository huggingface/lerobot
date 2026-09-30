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
"""``human_video`` module tests with a stubbed VLM and a fake video generator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import PIL.Image
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lerobot.annotations.steerable_pipeline.config import (  # noqa: E402
    AnnotationPipelineConfig,
    HumanVideoConfig,
    InterjectionsConfig,
    PlanConfig,
    VqaConfig,
)
from lerobot.annotations.steerable_pipeline.executor import Executor  # noqa: E402
from lerobot.annotations.steerable_pipeline.modules import (  # noqa: E402
    GeneralVqaModule,
    HumanVideoModule,
    InferenceProvidersGenerator,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot.annotations.steerable_pipeline.modules.human_video import subtask_spans  # noqa: E402
from lerobot.annotations.steerable_pipeline.reader import iter_episodes  # noqa: E402
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging  # noqa: E402
from lerobot.annotations.steerable_pipeline.validator import StagingValidator  # noqa: E402
from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient  # noqa: E402
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter  # noqa: E402

PLAN = {
    "human_task": "Pick up the bottle with the right hand and pour into the cup.",
    "object": "green bottle",
    "target": "white cup",
    "hand": "right",
    "steps": ["grasp the bottle", "tilt it over the cup"],
    "final_state": "water in the cup",
    "edit_prompt": "Remove the robot arms and add two human hands at the bottom edge.",
}
VIDEO = {
    "video_prompt": "From a fixed first-person camera perspective, ...",
    "caption": "A hand pours water.",
}


def responder(messages: list[dict[str, Any]]) -> Any:
    text = " ".join(b.get("text", "") for m in messages for b in m["content"] if isinstance(b, dict))
    if "preparing a human demonstration" in text:
        return PLAN
    if "image-to-video model will animate it" in text:
        return VIDEO
    return {}


@dataclass
class _Frames:
    cameras: tuple[str, ...] = ("observation.images.top",)
    calls: list[tuple[int, tuple[float, ...], str | None]] = field(default_factory=list)

    @property
    def camera_keys(self) -> list[str]:
        return list(self.cameras)

    def frames_at(self, record, timestamps, camera_key=None):
        self.calls.append((record.episode_index, tuple(timestamps), camera_key))
        return [PIL.Image.new("RGB", (64, 48), (i * 20 % 255, 0, 0)) for i in range(len(timestamps))]


@dataclass
class _Generator:
    fail_on: set[str] = field(default_factory=set)
    edits: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)

    def edit_image(self, image, prompt):
        self.edits.append(prompt)
        return image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    def image_to_video(self, image, prompt):
        self.videos.append(prompt)
        if any(s in prompt for s in self.fail_on):
            raise RuntimeError("provider error")
        return b"\x00\x00\x00\x18ftypmp42fake-video"


def _stage_subtasks(staging: EpisodeStaging, spans: list[tuple[str, float]]) -> None:
    staging.write(
        "plan",
        [{"role": "assistant", "content": text, "style": "subtask", "timestamp": t} for text, t in spans],
    )


def _module(root: Path, generator=None, **config) -> HumanVideoModule:
    return HumanVideoModule(
        vlm=StubVlmClient(responder=responder),
        config=HumanVideoConfig(enabled=True, **config),
        root=root,
        generator=generator or _Generator(),
        frame_provider=_Frames(),
    )


def test_generates_one_video_per_subtask(single_episode_root: Path, tmp_path: Path) -> None:
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    _stage_subtasks(staging, [("grasp the bottle", 0.0), ("pour into the cup", 1.0), ("put it down", 2.0)])
    generator = _Generator()
    module = _module(single_episode_root, generator)
    module.run_episode(record, staging)

    entries = json.loads((staging.episode_dir / "human_video.json").read_text())
    assert [e["status"] for e in entries] == ["ok", "ok", "ok"]
    assert [e["subtask"] for e in entries] == ["grasp the bottle", "pour into the cup", "put it down"]
    assert entries[0]["start_timestamp"] == 0.0 and entries[0]["end_timestamp"] == 1.0
    assert entries[2]["end_timestamp"] == record.frame_timestamps[-1]
    for e in entries:
        assert (single_episode_root / e["video_path"]).read_bytes().startswith(b"\x00\x00\x00\x18ftyp")
        assert (single_episode_root / e["first_frame_path"]).exists()
        assert e["video_model"] == "MiniMaxAI/MiniMax-H3" and e["resolution"] == "480P"
    assert generator.edits == [PLAN["edit_prompt"]] * 3
    assert generator.videos == [VIDEO["video_prompt"]] * 3
    # The first frame is taken at the subtask start, followed by the context frames.
    first_call = module.frame_provider.calls[0]
    assert first_call[1][0] == 0.0 and len(first_call[1]) == 1 + module.config.context_frames


def test_failed_segment_is_recorded_and_others_continue(single_episode_root: Path, tmp_path: Path) -> None:
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    _stage_subtasks(staging, [("grasp the bottle", 0.0), ("pour into the cup", 1.0)])

    calls = {"n": 0}

    class _FlakyGenerator(_Generator):
        def image_to_video(self, image, prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("provider error")
            return b"video"

    _module(single_episode_root, _FlakyGenerator(), max_concurrency=1).run_episode(record, staging)
    entries = json.loads((staging.episode_dir / "human_video.json").read_text())
    assert sorted(e["status"] for e in entries) == ["failed", "ok"]
    assert "provider error" in next(e for e in entries if e["status"] == "failed")["error"]


def test_max_segments_and_frame_offset(single_episode_root: Path, tmp_path: Path) -> None:
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    _stage_subtasks(staging, [("a", 0.0), ("b", 1.0), ("c", 2.0)])
    module = _module(single_episode_root, max_segments_per_episode=2, frame_offset_s=0.5)
    module.run_episode(record, staging)
    entries = json.loads((staging.episode_dir / "human_video.json").read_text())
    assert len(entries) == 2
    assert module.frame_provider.calls[1][1][0] == pytest.approx(1.5)


def test_spans_fall_back_to_existing_dataset_subtasks(
    single_episode_root: Path, tmp_path: Path, monkeypatch
) -> None:
    record = next(iter_episodes(single_episode_root))
    rows = [
        {"role": "assistant", "content": "second", "style": "subtask", "timestamp": 1.5},
        {"role": "assistant", "content": "first", "style": "subtask", "timestamp": 0.0},
        {"role": "assistant", "content": "a plan", "style": "plan", "timestamp": 0.0},
    ]
    # Parquet round-trips the per-frame list as a numpy object array.
    frames = pd.DataFrame(
        {"language_persistent": [np.array(rows, dtype=object)] * len(record.frame_timestamps)}
    )
    monkeypatch.setattr(type(record), "frames_df", lambda self: frames)
    spans = subtask_spans(record, EpisodeStaging(tmp_path / "stage", record.episode_index))
    assert [(s["text"], s["start"], s["end"]) for s in spans] == [
        ("first", 0.0, 1.5),
        ("second", 1.5, record.frame_timestamps[-1]),
    ]


def test_executor_indexes_videos_without_touching_language_columns(single_episode_root: Path) -> None:
    subtasks = {
        "subtasks": [
            {"text": "grasp the bottle", "start": 0.0, "end": 1.5},
            {"text": "pour", "start": 1.5, "end": 2.9},
        ]
    }

    def pipeline_responder(messages):
        text = " ".join(b.get("text", "") for m in messages for b in m["content"] if isinstance(b, dict))
        if '"subtasks"' in text:
            return subtasks
        if '"description"' in text:
            return {"description": "0.0s a hand grasps the bottle; 1.5s it pours into the cup."}
        return responder(messages)

    vlm = StubVlmClient(responder=pipeline_responder)
    config = AnnotationPipelineConfig(
        plan=PlanConfig(emit_plan=False, emit_memory=False, n_task_rephrasings=0),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
        human_video=HumanVideoConfig(enabled=True),
    )
    executor = Executor(
        config=config,
        plan=PlanSubtasksMemoryModule(vlm=vlm, config=config.plan),
        interjections=InterjectionsAndSpeechModule(vlm=vlm, config=config.interjections, seed=config.seed),
        vqa=GeneralVqaModule(vlm=vlm, config=config.vqa, seed=config.seed),
        writer=LanguageColumnsWriter(),
        validator=StagingValidator(),
        human_video=HumanVideoModule(
            vlm=vlm,
            config=config.human_video,
            root=single_episode_root,
            generator=_Generator(),
            frame_provider=_Frames(),
        ),
    )
    summary = executor.run(single_episode_root)
    assert summary.validation_report.ok, summary.validation_report.summary()
    assert [p.name for p in summary.phases][-1] == "human_video"

    lines = (single_episode_root / "meta" / "human_videos.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in lines]
    assert entries and all(e["status"] == "ok" for e in entries)
    assert {e["subtask"] for e in entries} <= {"grasp the bottle", "pour"}
    persistent = (
        pq.read_table(single_episode_root / "data" / "chunk-000" / "file-000.parquet")
        .column("language_persistent")
        .to_pylist()[0]
    )
    assert {r["style"] for r in persistent} == {"subtask"}


def test_generator_routes_image_text_to_video_models_through_fal(monkeypatch) -> None:
    import huggingface_hub

    class _Client:
        def __init__(self, provider=None, api_key=None):
            self.provider = provider

        def image_to_video(self, *args, **kwargs):
            raise ValueError(
                "Model MiniMaxAI/MiniMax-H3 is not supported for task image-to-video and provider fal-ai. "
                "Supported task: image-text-to-video."
            )

    monkeypatch.setattr(huggingface_hub, "InferenceClient", _Client)
    generator = InferenceProvidersGenerator(config=HumanVideoConfig(enabled=True, seed=3))
    seen = {}

    def fake_fal(image, params):
        seen.update(params)
        return b"video"

    monkeypatch.setattr(generator, "_fal_image_text_to_video", fake_fal)
    assert generator.image_to_video(PIL.Image.new("RGB", (8, 8)), "a prompt") == b"video"
    assert seen == {"prompt": "a prompt", "resolution": "480P", "duration": 5, "seed": 3}
