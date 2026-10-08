#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig
from lerobot.datasets import factory
from lerobot.datasets.storage import is_bucket_root
from lerobot.datasets.streaming_dataset import DEFAULT_STREAMING_SEED, StreamingLeRobotDataset
from tests.fixtures.constants import DUMMY_REPO_ID


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
        streaming_sidecar_lock_timeout_s=60,
    )
    cfg = SimpleNamespace(
        dataset=dataset_config,
        trainable_config=object(),
        seed=123,
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
    assert captured["kwargs"]["sidecar_lock_timeout_s"] == 60
    assert captured["kwargs"]["max_num_shards"] == 1
    assert captured["kwargs"]["video_backend"] == "pyav"
    assert captured["kwargs"]["return_uint8"] is True
    assert captured["kwargs"]["repeat"] is True
    assert captured["kwargs"]["seed"] == 123


def _streaming_cfg(root: Path | None, seed: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=DatasetConfig(
            repo_id=DUMMY_REPO_ID,
            root=str(root) if root is not None else None,
            streaming=True,
            video_backend="pyav",
            streaming_episode_pool_size=3,
        ),
        trainable_config=object(),
        seed=seed,
        num_workers=0,
        tolerance_s=1e-4,
        rename_map={},
    )


def _anchor_order(dataset: StreamingLeRobotDataset, count: int) -> list[int]:
    return [int(sample["index"]) for sample in islice(iter(dataset), count)]


@pytest.mark.parametrize(
    ("cfg_seed", "expected_seed"), [(0, 0), (7, 7), (1000, 1000), (None, DEFAULT_STREAMING_SEED)]
)
def test_factory_passes_train_seed_to_streaming_dataset(
    monkeypatch: pytest.MonkeyPatch, cfg_seed: int | None, expected_seed: int
) -> None:
    captured = {}

    class DummyStreamingDataset:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs
            self.meta = SimpleNamespace(camera_keys=[], depth_keys=[], stats={})

    monkeypatch.setattr(
        factory,
        "load_dataset_metadata",
        lambda *args, **kwargs: SimpleNamespace(storage_format="lerobot", total_episodes=1),
    )
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *args, **kwargs: None)
    monkeypatch.setattr(factory, "StreamingLeRobotDataset", DummyStreamingDataset)

    factory.make_dataset(_streaming_cfg(None, cfg_seed))

    assert captured["kwargs"]["seed"] == expected_seed


def test_make_dataset_streaming_order_follows_train_seed(
    tmp_path: Path, lerobot_dataset_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different `--seed` values give different orders; the same value repeats the order."""
    root = tmp_path / "dataset"
    lerobot_dataset_factory(
        root=root, repo_id=DUMMY_REPO_ID, total_episodes=6, total_frames=120, use_videos=False
    )
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *args, **kwargs: None)

    def order(seed: int | None) -> list[int]:
        return _anchor_order(factory.make_dataset(_streaming_cfg(root, seed)), 60)

    assert order(1) == order(1)
    assert order(1) != order(2)
    assert order(None) == order(DEFAULT_STREAMING_SEED)


def test_make_dataset_streaming_resume_continues_same_seed_order(
    tmp_path: Path, lerobot_dataset_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed run with the saved seed continues the order of the original run."""
    root = tmp_path / "dataset"
    lerobot_dataset_factory(
        root=root, repo_id=DUMMY_REPO_ID, total_episodes=6, total_frames=120, use_videos=False
    )
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *args, **kwargs: None)
    offset = 25

    full = _anchor_order(factory.make_dataset(_streaming_cfg(root, 5)), 60)
    resumed_dataset = factory.make_dataset(_streaming_cfg(root, 5))
    resumed_dataset.load_state_dict({"epoch": 0, "offset": offset, "batch_size": 1})

    assert _anchor_order(resumed_dataset, 60 - offset) == full[offset:]


def test_is_bucket_root() -> None:
    assert not is_bucket_root("owner/data", None)
    assert not is_bucket_root("owner/data", "local/dir")
    assert is_bucket_root("owner/data", "hf://buckets/owner/data/")
    with pytest.raises(ValueError, match="sub-directory"):
        is_bucket_root("owner/data", "hf://buckets/owner/data/sub")
    with pytest.raises(ValueError, match="does not match repo_id"):
        is_bucket_root("owner/data", "hf://buckets/owner/other")
