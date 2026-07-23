#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from itertools import islice
from pathlib import Path

import fsspec
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.utils.utils import cycle
from tests.fixtures.constants import DUMMY_REPO_ID


def _indices(dataset: StreamingLeRobotDataset) -> list[int]:
    return [int(item["index"]) for item in dataset]


def _assert_item_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def test_streaming_matches_map_style_with_exact_coverage(tmp_path: Path, lerobot_dataset_factory) -> None:
    root = tmp_path / "dataset"
    map_dataset = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=4,
        total_frames=40,
        use_videos=False,
    )
    streaming = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        shuffle=False,
        buffer_size=3,
    )

    samples = list(streaming)

    assert len(samples) == len(map_dataset)
    assert sorted(int(sample["index"]) for sample in samples) == list(range(len(map_dataset)))
    for sample in samples:
        _assert_item_equal(sample, map_dataset[int(sample["index"])])


def test_streaming_rgb_video_matches_map_style(tmp_path: Path, lerobot_dataset_factory) -> None:
    root = tmp_path / "dataset"
    map_dataset = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=20,
    )
    streaming = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        shuffle=False,
        buffer_size=2,
    )

    for sample in streaming:
        reference = map_dataset[int(sample["index"])]
        assert sample.keys() == reference.keys()
        for camera_key in map_dataset.meta.camera_keys:
            assert torch.equal(sample[camera_key], reference[camera_key]), (
                camera_key,
                int(sample["index"]),
                float((sample[camera_key] - reference[camera_key]).abs().max()),
            )


def test_streaming_applies_rgb_transforms_and_preserves_uint8(
    tmp_path: Path, lerobot_dataset_factory
) -> None:
    root = tmp_path / "dataset"

    def flip_width(image: torch.Tensor) -> torch.Tensor:
        return image.flip(-1)

    map_dataset = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=10,
        image_transforms=flip_width,
        return_uint8=True,
    )
    streaming = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        shuffle=False,
        buffer_size=2,
        image_transforms=flip_width,
        return_uint8=True,
    )

    sample = next(iter(streaming))
    reference = map_dataset[int(sample["index"])]
    for camera_key in map_dataset.meta.camera_keys:
        assert sample[camera_key].dtype == torch.uint8
        assert torch.equal(sample[camera_key], reference[camera_key])


def test_streaming_honors_episode_subset(tmp_path: Path, lerobot_dataset_factory) -> None:
    root = tmp_path / "dataset"
    map_dataset = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=5,
        total_frames=50,
        use_videos=False,
    )
    selected = [1, 3]
    streaming = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        episodes=selected,
        shuffle=False,
        buffer_size=2,
    )

    indices = _indices(streaming)
    expected = [
        index
        for episode in selected
        for index in range(
            map_dataset.meta.episodes[episode]["dataset_from_index"],
            map_dataset.meta.episodes[episode]["dataset_to_index"],
        )
    ]

    assert sorted(indices) == sorted(expected)


def test_streaming_reads_episode_parquet_from_configured_fsspec_root(
    tmp_path: Path, lerobot_dataset_factory
) -> None:
    root = tmp_path / "metadata"
    map_dataset = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=3,
        total_frames=30,
        use_videos=False,
    )
    remote_root = "memory://streaming-production"
    filesystem = fsspec.filesystem("memory")
    for path in (root / "data").glob("*/*.parquet"):
        relative = path.relative_to(root).as_posix()
        filesystem.put(str(path), f"streaming-production/{relative}")

    streaming = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        data_root=remote_root,
        shuffle=False,
        buffer_size=2,
    )

    assert sorted(_indices(streaming)) == list(range(len(map_dataset)))


def test_streaming_reads_video_bytes_from_configured_fsspec_root(
    tmp_path: Path, lerobot_dataset_factory
) -> None:
    root = tmp_path / "metadata"
    map_dataset = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=10,
    )
    namespace = f"streaming-video-{tmp_path.name}"
    remote_root = f"memory://{namespace}"
    filesystem = fsspec.filesystem("memory")
    for path in [*(root / "data").glob("*/*.parquet"), *(root / "videos").glob("*/*/*.mp4")]:
        relative = path.relative_to(root).as_posix()
        filesystem.put(str(path), f"{namespace}/{relative}")

    streaming = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        data_root=remote_root,
        shuffle=False,
        buffer_size=2,
    )

    sample = next(iter(streaming))
    reference = map_dataset[int(sample["index"])]
    for camera_key in map_dataset.meta.camera_keys:
        assert torch.equal(sample[camera_key], reference[camera_key])


def test_streaming_rank_shards_are_disjoint(tmp_path: Path, lerobot_dataset_factory, monkeypatch) -> None:
    root = tmp_path / "dataset"
    map_dataset = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=8,
        total_frames=80,
        use_videos=False,
    )
    per_rank = []
    for rank in range(2):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "2")
        per_rank.append(
            set(
                _indices(
                    StreamingLeRobotDataset(
                        DUMMY_REPO_ID,
                        root=root,
                        shuffle=False,
                        buffer_size=2,
                    )
                )
            )
        )

    assert per_rank[0].isdisjoint(per_rank[1])
    assert per_rank[0] | per_rank[1] == set(range(len(map_dataset)))


def test_streaming_workers_do_not_duplicate_frames(tmp_path: Path, lerobot_dataset_factory) -> None:
    root = tmp_path / "dataset"
    map_dataset = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=8,
        total_frames=80,
        use_videos=False,
    )
    streaming = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        shuffle=False,
        buffer_size=2,
    )
    loader = torch.utils.data.DataLoader(streaming, batch_size=None, num_workers=2)

    indices = [int(item["index"]) for item in loader]

    assert len(indices) == len(map_dataset)
    assert set(indices) == set(range(len(map_dataset)))


def test_streaming_persistent_workers_advance_epochs(tmp_path: Path, lerobot_dataset_factory) -> None:
    root = tmp_path / "dataset"
    map_dataset = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=8,
        total_frames=80,
        use_videos=False,
    )
    streaming = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        seed=23,
        shuffle=True,
        buffer_size=2,
    )
    loader = torch.utils.data.DataLoader(
        streaming,
        batch_size=None,
        num_workers=2,
        persistent_workers=True,
    )
    try:
        first = [int(item["index"]) for item in loader]
        second = [int(item["index"]) for item in loader]
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()

    assert sorted(first) == list(range(len(map_dataset)))
    assert sorted(second) == list(range(len(map_dataset)))
    assert first != second


def test_streaming_worker_exception_propagates_and_workers_stop(
    tmp_path: Path, lerobot_dataset_factory
) -> None:
    root = tmp_path / "dataset"
    lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=4,
        total_frames=40,
        use_videos=False,
    )
    streaming = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        shuffle=False,
        buffer_size=2,
    )
    next((root / "data").glob("*/*.parquet")).write_bytes(b"corrupt parquet")
    loader = torch.utils.data.DataLoader(
        streaming,
        batch_size=None,
        num_workers=2,
        persistent_workers=True,
    )
    try:
        with pytest.raises(Exception, match="Parquet"):
            list(loader)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
            assert not any(worker.is_alive() for worker in loader._iterator._workers)


def test_streaming_resume_reproduces_remaining_stream(tmp_path: Path, lerobot_dataset_factory) -> None:
    root = tmp_path / "dataset"
    lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=5,
        total_frames=50,
        use_videos=False,
    )
    full = _indices(
        StreamingLeRobotDataset(
            DUMMY_REPO_ID,
            root=root,
            seed=7,
            shuffle=True,
            buffer_size=3,
        )
    )
    resumed = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        seed=7,
        shuffle=True,
        buffer_size=3,
    )
    resumed.load_state_dict({"epoch": 0, "offset": 11})

    assert _indices(resumed) == full[11:]


@pytest.mark.parametrize(("batch_size", "offset"), [(None, 17), (4, 20)])
def test_streaming_worker_resume_reproduces_remaining_stream(
    tmp_path: Path,
    lerobot_dataset_factory,
    batch_size: int | None,
    offset: int,
) -> None:
    root = tmp_path / "dataset"
    lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=8,
        total_frames=80,
        use_videos=False,
    )

    def load(dataset: StreamingLeRobotDataset) -> list[int]:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)
        if batch_size is None:
            return [int(item["index"]) for item in loader]
        return [int(index) for batch in loader for index in batch["index"]]

    full = load(
        StreamingLeRobotDataset(
            DUMMY_REPO_ID,
            root=root,
            seed=31,
            shuffle=True,
            buffer_size=2,
        )
    )
    resumed = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        seed=31,
        shuffle=True,
        buffer_size=2,
    )
    resumed.load_state_dict({"epoch": 0, "offset": offset, "batch_size": batch_size or 1})

    assert load(resumed) == full[offset:]


def test_streaming_state_dict_round_trip_mid_epoch(tmp_path: Path, lerobot_dataset_factory) -> None:
    root = tmp_path / "dataset"
    lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=5,
        total_frames=50,
        use_videos=False,
    )
    source = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        seed=17,
        shuffle=True,
        buffer_size=3,
    )
    iterator = iter(source)
    consumed = [int(next(iterator)["index"]) for _ in range(13)]
    state = source.state_dict()
    remaining = [int(item["index"]) for item in iterator]

    restored = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        seed=17,
        shuffle=True,
        buffer_size=3,
    )
    restored.load_state_dict(state)

    assert len(consumed) == state["offset"]
    assert _indices(restored) == remaining


def test_streaming_worker_resume_after_epoch_boundary(tmp_path: Path, lerobot_dataset_factory) -> None:
    root = tmp_path / "dataset"
    lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=4,
        total_frames=24,
        use_videos=False,
    )

    def infinite_indices(dataset: StreamingLeRobotDataset, count: int) -> list[int]:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            num_workers=2,
            persistent_workers=True,
        )
        try:
            return [
                int(index) for batch in islice(cycle(loader), (count + 3) // 4) for index in batch["index"]
            ][:count]
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()

    full = infinite_indices(
        StreamingLeRobotDataset(
            DUMMY_REPO_ID,
            root=root,
            seed=47,
            shuffle=True,
            buffer_size=2,
            repeat=True,
        ),
        56,
    )
    offset = 32
    resumed = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        seed=47,
        shuffle=True,
        buffer_size=2,
        repeat=True,
    )
    resumed.load_state_dict({"epoch": 0, "offset": offset, "batch_size": 4})

    assert infinite_indices(resumed, 24) == full[offset : offset + 24]


def test_streaming_local_training_step_smoke(tmp_path: Path, lerobot_dataset_factory) -> None:
    root = tmp_path / "dataset"
    lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=4,
        total_frames=24,
        use_videos=False,
    )
    dataset = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        seed=53,
        buffer_size=2,
        repeat=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
    iterator = iter(loader)
    try:
        batch = next(iterator)
    finally:
        iterator._shutdown_workers()
    model = torch.nn.Linear(batch["action"].shape[-1], batch["action"].shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = torch.nn.functional.mse_loss(model(batch["action"]), batch["action"])
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert batch["index"].shape == (4,)
