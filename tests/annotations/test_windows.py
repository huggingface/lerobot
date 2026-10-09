# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Language segmentation consumes the same windows persisted by its adapter."""

from pathlib import Path

import pytest

from lerobot.annotations.steerable_pipeline.config import PlanConfig
from lerobot.annotations.steerable_pipeline.modules.plan_subtasks_memory import (
    PlanSubtasksMemoryModule,
    subtask_windows,
)
from lerobot.annotations.steerable_pipeline.reader import EpisodeRecord
from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient


def test_long_episode_reconciles_continuations_and_fails_empty_windows(monkeypatch):
    record = EpisodeRecord(
        2, "hold cup", tuple(3.0 + i / 10 for i in range(201)), tuple(range(201)), Path("unused"), 0, 201
    )
    cfg = PlanConfig(max_frames_per_prompt=8, frames_per_second=1, window_overlap_seconds=1)
    module = PlanSubtasksMemoryModule(StubVlmClient(lambda messages: {}), cfg)
    calls = []

    def predict(record, task, lo, hi):
        calls.append((lo, hi))
        return [{"text": "hold cup", "start": lo, "end": hi}]

    monkeypatch.setattr(module, "_subtasks_for_window", predict)
    assert module._generate_subtasks(record) == [{"text": "hold cup", "start": 3.0, "end": 23.0}]
    windows = subtask_windows(record, cfg)
    assert len(calls) == len(windows) > 1
    assert calls[1][0] < calls[0][1]  # Context actually overlaps.
    assert all(lo in record.frame_timestamps and hi in record.frame_timestamps for lo, hi in calls)
    monkeypatch.setattr(module, "_subtasks_for_window", lambda *args: [])
    with pytest.raises(ValueError, match="no owned spans"):
        module._generate_subtasks(record)


def test_short_episode_has_one_window_even_near_prompt_budget():
    record = EpisodeRecord(
        0, "task", tuple(float(i) for i in range(8)), tuple(range(8)), Path("unused"), 0, 8
    )
    windows = subtask_windows(
        record, PlanConfig(max_frames_per_prompt=8, frames_per_second=1, window_overlap_seconds=1)
    )
    assert len(windows) == 1 and len(windows[0].output_frames) == 8
