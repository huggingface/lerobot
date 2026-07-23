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
        streaming_data_root="memory://payload",
        streaming_episode_pool_size=7,
        streaming_prefetch_episodes=3,
        streaming_byte_budget_gb=2.5,
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
    assert captured["kwargs"]["max_num_shards"] == 1
    assert captured["kwargs"]["return_uint8"] is True
    assert captured["kwargs"]["repeat"] is True
