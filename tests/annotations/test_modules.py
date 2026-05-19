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
"""Module 1/2/3 unit tests with stubbed VLMs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.annotations.steerable_pipeline.config import (
    InterjectionsConfig,
    PlanConfig,
    VqaConfig,
)
from lerobot.annotations.steerable_pipeline.modules import (
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient

from ._helpers import make_canned_responder


@dataclass
class _StubFrameProvider:
    """Returns one sentinel object per requested timestamp."""

    sentinel: Any = field(default_factory=lambda: object())
    cameras: tuple[str, ...] = ("observation.images.top",)
    calls: list[tuple[int, tuple[float, ...], str | None]] = field(default_factory=list)
    video_calls: list[tuple[int, int, str | None]] = field(default_factory=list)

    @property
    def camera_keys(self) -> list[str]:
        return list(self.cameras)

    def frames_at(self, record, timestamps, camera_key=None):
        self.calls.append((record.episode_index, tuple(timestamps), camera_key))
        return [self.sentinel] * len(timestamps)

    def video_for_episode(self, record, max_frames, camera_key=None):
        self.video_calls.append((record.episode_index, max_frames, camera_key))
        n = min(max_frames, len(record.frame_timestamps))
        return [self.sentinel] * n


def _spy_responder(captured: list[list[dict[str, Any]]], reply: Any):
    def responder(messages):
        captured.append(list(messages))
        return reply

    return StubVlmClient(responder=responder)


def test_module1_plan_memory_subtask_smoke(fixture_dataset_root: Path, tmp_path: Path) -> None:
    vlm = make_canned_responder(
        {
            "atomic subtasks": {
                "subtasks": [
                    {"text": "grasp the handle of the sponge", "start": 0.0, "end": 0.4},
                    {"text": "wipe the counter from left to right", "start": 0.4, "end": 0.8},
                    {"text": "place the sponge into the sink", "start": 0.8, "end": 1.1},
                ]
            },
            "Update the memory": {"memory": "wiped the counter once"},
        },
    )
    module = PlanSubtasksMemoryModule(vlm=vlm, config=PlanConfig())
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("plan")

    styles = {r["style"] for r in rows}
    assert {"subtask", "plan", "memory"}.issubset(styles)
    # subtask timestamps must be exact frame timestamps
    frame_set = set(record.frame_timestamps)
    for row in rows:
        assert row["timestamp"] in frame_set
    # one plan row per subtask boundary; the first lands at t0 and each
    # plan is the deterministic numbered list of still-todo subtasks
    plan_rows = sorted((r for r in rows if r["style"] == "plan"), key=lambda r: r["timestamp"])
    subtask_rows = [r for r in rows if r["style"] == "subtask"]
    assert len(plan_rows) == len(subtask_rows)
    assert plan_rows[0]["timestamp"] == record.frame_timestamps[0]
    # the t0 plan enumerates all subtasks; later plans shrink
    assert plan_rows[0]["content"].startswith("1. ")
    assert len(plan_rows[0]["content"].splitlines()) == len(subtask_rows)
    assert len(plan_rows[-1]["content"].splitlines()) == 1


def test_module2_at_t0_emits_speech_only_no_interjection(fixture_dataset_root: Path, tmp_path: Path) -> None:
    vlm = make_canned_responder(
        {"acknowledgement the robot": {"text": "Sure, on it."}},
    )
    module = InterjectionsAndSpeechModule(
        vlm=vlm,
        config=InterjectionsConfig(max_interjections_per_episode=0),
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("interjections")
    assert len(rows) == 1
    only = rows[0]
    assert only["role"] == "assistant"
    assert only["style"] is None
    assert only["content"] is None
    assert only["timestamp"] == record.frame_timestamps[0]
    assert only["tool_calls"][0]["function"]["name"] == "say"


def test_module2_mid_episode_emits_paired_interjection_and_speech(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """Module 2 anchors interjections on Module 1's subtask boundaries.

    The executor runs Module 1 first, then Module 2 reads the subtask
    rows back from the same staging tree (see
    ``_mid_episode_interjections``). Reproduce that contract here by
    seeding the staging with two subtask rows so a single ``0 → 1``
    boundary exists for Module 2 to anchor on.
    """
    vlm = make_canned_responder(
        {
            "acknowledgement the robot": {"text": "OK."},
            # Marker matches the distinctive line of
            # ``module_2_interjection.txt``. The old marker
            # ("ONE realistic interruption") came from a previous prompt
            # version that asked for counterfactual interjections; the
            # current design anchors on subtask boundaries instead, so
            # the prompt and its marker changed.
            "Write ONE interjection": {
                "interjection": "now wipe the counter please",
                "speech": "On it.",
            },
        },
    )
    module = InterjectionsAndSpeechModule(
        vlm=vlm,
        config=InterjectionsConfig(max_interjections_per_episode=1, interjection_min_t=0.2),
        seed=7,
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    # Seed Module 1's subtask staging so Module 2 has a boundary to
    # anchor on (it bails with zero rows when no spans exist — the
    # production executor guarantees Module 1 ran first).
    boundary_ts = float(record.frame_timestamps[len(record.frame_timestamps) // 2])
    staging.write(
        "plan",
        [
            {
                "role": "assistant",
                "content": "grasp the sponge",
                "style": "subtask",
                "timestamp": float(record.frame_timestamps[0]),
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": "wipe the counter",
                "style": "subtask",
                "timestamp": boundary_ts,
                "tool_calls": None,
            },
        ],
    )
    module.run_episode(record, staging)
    rows = staging.read("interjections")

    interjections = [r for r in rows if r["style"] == "interjection"]
    speeches = [r for r in rows if r["style"] is None and r["role"] == "assistant"]
    assert len(interjections) == 1
    assert len(speeches) >= 2  # initial t=0 + one paired with the interjection
    inter_t = interjections[0]["timestamp"]
    assert any(abs(s["timestamp"] - inter_t) < 1e-9 for s in speeches)


def test_module3_vqa_unique_per_frame_and_camera(single_episode_root: Path, tmp_path: Path) -> None:
    payload = {
        "question": "How many cups?",
        "answer": {"label": "cup", "count": 2, "note": "white & blue"},
    }
    vlm = make_canned_responder({"frame-grounded visual question": payload})
    module = GeneralVqaModule(
        vlm=vlm,
        config=VqaConfig(vqa_emission_hz=1.0, K=3),
        seed=1,
        frame_provider=_StubFrameProvider(cameras=("observation.images.top", "observation.images.wrist")),
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("vqa")
    # every vqa row must carry a camera tag and one of the configured cameras
    for r in rows:
        assert r["style"] == "vqa"
        assert r.get("camera") in {"observation.images.top", "observation.images.wrist"}
    # at most one (vqa, user) and one (vqa, assistant) per (timestamp, camera)
    user_keys = [(r["timestamp"], r["camera"]) for r in rows if r["role"] == "user" and r["style"] == "vqa"]
    assistant_keys = [
        (r["timestamp"], r["camera"]) for r in rows if r["role"] == "assistant" and r["style"] == "vqa"
    ]
    assert len(user_keys) == len(set(user_keys))
    assert len(assistant_keys) == len(set(assistant_keys))
    # both cameras must be represented
    assert {c for _, c in user_keys} == {"observation.images.top", "observation.images.wrist"}
    # every emitted timestamp must be an exact source frame timestamp
    frame_set = set(record.frame_timestamps)
    for ts, _ in user_keys + assistant_keys:
        assert ts in frame_set


def test_module1_attaches_video_block_to_subtask_prompt(fixture_dataset_root: Path, tmp_path: Path) -> None:
    """Module 1 sends one ``type=video`` block covering the whole episode."""
    captured: list[list[dict[str, Any]]] = []
    payload = {
        "subtasks": [
            {"text": "grasp the handle of the sponge", "start": 0.0, "end": 0.5},
            {"text": "wipe the counter", "start": 0.5, "end": 1.1},
        ]
    }
    plan_payload = {"plan": "1. grasp\n2. wipe"}
    memory_payload = {"memory": "wiped once"}

    def responder(messages):
        captured.append(list(messages))
        text = ""
        for m in messages:
            for block in m.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
        if "concise hierarchical PLAN" in text:
            return plan_payload
        if "Update the memory" in text:
            return memory_payload
        return payload

    provider = _StubFrameProvider()
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        # Disable the rephrasings sub-prompt so the test's only video-bearing
        # call is the subtask one — keeps the assertions below focused on
        # ``_generate_subtasks`` rather than fighting the order of unrelated
        # text-only Module-1 sub-prompts.
        config=PlanConfig(max_video_frames=5, frames_per_second=10.0, n_task_rephrasings=0),
        frame_provider=provider,
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    # Find the call carrying the subtask prompt rather than blindly taking
    # captured[0] — Module 1 issues several sub-prompts and their order is
    # not part of the contract.
    assert captured, "no VLM calls made"

    def _prompt_text(messages):
        for m in messages:
            for block in m.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
        return ""

    subtask_calls = [m for m in captured if "atomic subtasks" in _prompt_text(m)]
    assert len(subtask_calls) == 1, "expected exactly one subtask-prompt VLM call"
    content = subtask_calls[0][0]["content"]
    video_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "video"]
    image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image"]
    text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
    assert len(video_blocks) == 1, f"expected exactly 1 video block, got {content}"
    assert image_blocks == [], "subtask prompt must not mix image blocks with the video block"
    assert len(text_blocks) == 1
    # video block must wrap a list of frames covering the episode
    assert isinstance(video_blocks[0]["video"], list)
    assert len(video_blocks[0]["video"]) <= 5
    # provider is called with target_count = min(duration * fps, max). With
    # fps=10 on a ~1s episode that requests >max, so max=5 wins.
    assert provider.video_calls and provider.video_calls[0][0] == record.episode_index
    assert provider.video_calls[0][1] <= 5


def test_module3_attaches_frame_image_block_to_prompt(single_episode_root: Path, tmp_path: Path) -> None:
    """Each VQA prompt must carry a single image block at the emission frame."""
    captured: list[list[dict[str, Any]]] = []
    payload = {
        "question": "How many cups?",
        "answer": {"label": "cup", "count": 1},
    }
    provider = _StubFrameProvider()
    module = GeneralVqaModule(
        vlm=_spy_responder(captured, payload),
        config=VqaConfig(vqa_emission_hz=1.0, K=1),
        seed=0,
        frame_provider=provider,
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    assert captured, "no VLM calls made"
    for messages in captured:
        content = messages[0]["content"]
        image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image"]
        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        assert len(image_blocks) == 1, f"expected 1 image block per VQA prompt, got {content}"
        assert image_blocks[0]["image"] is provider.sentinel
        assert len(text_blocks) == 1
    # provider was called once per emission per camera with the exact emission timestamp
    for ep_idx, ts_tuple, camera in provider.calls:
        assert ep_idx == record.episode_index
        assert len(ts_tuple) == 1
        assert ts_tuple[0] in record.frame_timestamps
        assert camera in provider.cameras


def test_module3_assistant_content_is_valid_json(single_episode_root: Path, tmp_path: Path) -> None:
    payload = {
        "question": "Where is the cup?",
        "answer": {"detections": [{"label": "cup", "bbox_format": "xyxy", "bbox": [10, 20, 50, 80]}]},
    }
    vlm = make_canned_responder({"frame-grounded visual question": payload})
    module = GeneralVqaModule(
        vlm=vlm,
        config=VqaConfig(vqa_emission_hz=1.0, K=2),
        seed=2,
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("vqa")
    for row in rows:
        if row["role"] == "assistant" and row["style"] == "vqa":
            decoded = json.loads(row["content"])
            assert "detections" in decoded
