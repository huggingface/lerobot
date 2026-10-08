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
from lerobot.datasets.storage import is_bucket_root


@pytest.mark.parametrize(
    ("repo_type", "root", "expected_repo_type"),
    [
        ("dataset", None, "dataset"),
        ("bucket", None, "bucket"),
        # A bucket URI root selects bucket mode, as repo_type="bucket" does.
        ("dataset", "hf://buckets/owner/dataset", "bucket"),
    ],
)
def test_factory_wires_production_streaming_settings(
    monkeypatch: pytest.MonkeyPatch, repo_type: str, root: str | None, expected_repo_type: str
) -> None:
    captured = {}

    class DummyStreamingDataset:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.meta = SimpleNamespace(camera_keys=[], depth_keys=[], stats={})

    def load_metadata(repo_id: str, **kwargs: object) -> SimpleNamespace:
        captured["metadata_repo_id"] = repo_id
        captured["metadata_kwargs"] = kwargs
        return SimpleNamespace(storage_format="lerobot", total_episodes=1)

    monkeypatch.setattr(factory, "load_dataset_metadata", load_metadata)
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *args, **kwargs: {"action": [0.0]})
    monkeypatch.setattr(factory, "StreamingLeRobotDataset", DummyStreamingDataset)
    dataset_config = DatasetConfig(
        repo_id="owner/dataset",
        repo_type=repo_type,
        root=root,
        streaming=True,
        video_backend="pyav",
        streaming_episode_pool_size=7,
        streaming_sampling_strategy="round_robin",
        streaming_prefetch_episodes=3,
        streaming_byte_budget_gb=2.5,
        streaming_decode_threads=2,
        streaming_decoded_queue_size=5,
        video_decoder_cache_size=17,
        streaming_native_http_connections=9,
        streaming_native_http_subranges=3,
    )
    cfg = SimpleNamespace(
        dataset=dataset_config,
        trainable_config=object(),
        num_workers=0,
        tolerance_s=1e-4,
        rename_map={},
    )

    dataset = factory.make_dataset(cfg)

    assert isinstance(dataset, DummyStreamingDataset)
    assert captured["args"] == ("owner/dataset",)
    assert captured["kwargs"]["repo_type"] == expected_repo_type
    assert captured["kwargs"]["root"] is None
    assert "data_root" not in captured["kwargs"]
    assert captured["metadata_repo_id"] == "owner/dataset"
    assert captured["metadata_kwargs"]["repo_type"] == expected_repo_type
    assert captured["metadata_kwargs"]["root"] is None
    assert captured["kwargs"]["episode_pool_size"] == 7
    assert captured["kwargs"]["sampling_strategy"] == "round_robin"
    assert captured["kwargs"]["prefetch_episodes"] == 3
    assert captured["kwargs"]["byte_budget_gb"] == 2.5
    assert captured["kwargs"]["decode_threads"] == 2
    assert captured["kwargs"]["decoded_queue_size"] == 5
    assert captured["kwargs"]["video_decoder_cache_size"] == 17
    assert captured["kwargs"]["native_http_connections"] == 9
    assert captured["kwargs"]["native_http_subranges"] == 3
    assert captured["kwargs"]["max_num_shards"] == 1
    assert captured["kwargs"]["video_backend"] == "pyav"
    assert captured["kwargs"]["return_uint8"] is True
    assert captured["kwargs"]["repeat"] is True


def test_is_bucket_root() -> None:
    assert not is_bucket_root("owner/data", None)
    assert not is_bucket_root("owner/data", "local/dir")
    assert is_bucket_root("owner/data", "hf://buckets/owner/data/")
    with pytest.raises(ValueError, match="sub-directory"):
        is_bucket_root("owner/data", "hf://buckets/owner/data/sub")
    with pytest.raises(ValueError, match="does not match repo_id"):
        is_bucket_root("owner/data", "hf://buckets/owner/other")
