#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig
from lerobot.datasets import factory


def test_factory_wires_production_streaming_settings(monkeypatch):
    captured = {}

    class DummyStreamingDataset:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.meta = SimpleNamespace(camera_keys=[], depth_keys=[], stats={})

    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: object())
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *args, **kwargs: {"action": [0.0]})
    monkeypatch.setattr(factory, "StreamingLeRobotDataset", DummyStreamingDataset)
    dataset_config = DatasetConfig(
        repo_id="owner/dataset",
        streaming=True,
        video_backend="pyav",
        streaming_data_root="memory://payload",
        streaming_episode_pool_size=7,
        streaming_prefetch_episodes=3,
        streaming_byte_budget_gb=2.5,
        streaming_decode_threads=2,
        streaming_decoded_queue_size=5,
        streaming_max_open_decoders=17,
        streaming_native_http_connections=9,
        streaming_native_http_subranges=3,
    )
    cfg = SimpleNamespace(
        dataset=dataset_config,
        trainable_config=object(),
        num_workers=0,
        tolerance_s=1e-4,
    )

    dataset = factory.make_dataset(cfg)

    assert isinstance(dataset, DummyStreamingDataset)
    assert captured["args"] == ("owner/dataset",)
    assert captured["kwargs"]["data_root"] == "memory://payload"
    assert captured["kwargs"]["episode_pool_size"] == 7
    assert captured["kwargs"]["prefetch_episodes"] == 3
    assert captured["kwargs"]["byte_budget_gb"] == 2.5
    assert captured["kwargs"]["decode_threads"] == 2
    assert captured["kwargs"]["decoded_queue_size"] == 5
    assert captured["kwargs"]["max_open_decoders"] == 17
    assert captured["kwargs"]["native_http_connections"] == 9
    assert captured["kwargs"]["native_http_subranges"] == 3
    assert captured["kwargs"]["max_num_shards"] == 1
    assert captured["kwargs"]["video_backend"] == "pyav"
    assert captured["kwargs"]["return_uint8"] is True
    assert captured["kwargs"]["repeat"] is True


def test_factory_wires_local_episode_loading(monkeypatch):
    captured = {}

    class DummyDataset:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            self.meta = SimpleNamespace(camera_keys=[], depth_keys=[], stats={})

    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: object())
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *args, **kwargs: None)
    monkeypatch.setattr(factory, "LeRobotDataset", DummyDataset)
    cfg = SimpleNamespace(
        dataset=DatasetConfig(repo_id="owner/dataset", root="dataset", local_episode_loading=True),
        trainable_config=object(),
        num_workers=2,
        tolerance_s=1e-4,
    )

    factory.make_dataset(cfg)

    assert captured["kwargs"]["local_episode_loading"] is True


def test_train_eval_split_is_per_task_and_honors_selection():
    meta = SimpleNamespace(
        total_episodes=6,
        episodes=[
            {"tasks": ["a"]},
            {"tasks": ["b"]},
            {"tasks": ["a"]},
            {"tasks": ["b"]},
            {"tasks": ["a"]},
            {"tasks": ["b"]},
        ],
    )

    train, evaluation = factory.resolve_train_eval_episode_indices(meta, [0, 1, 2, 3], 0.5)

    assert train == [0, 1]
    assert evaluation == [2, 3]
