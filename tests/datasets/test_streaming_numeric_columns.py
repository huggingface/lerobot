# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepared numeric columns retain the existing formatter's values and ownership."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import datasets
import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.io_utils import hf_transform_to_torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.streaming.episode_parquet import EpisodeParquetReader
from tests.fixtures.constants import DUMMY_REPO_ID


@pytest.mark.parametrize("window", [False, True])
def test_prepared_numeric_rows_are_independent(
    tmp_path: Path, lerobot_dataset_factory: Any, window: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    reference = lerobot_dataset_factory(
        root=root, repo_id=DUMMY_REPO_ID, total_episodes=1, total_frames=12, use_videos=False
    )
    deltas = {"action": [-1 / reference.fps, 0.0, 1 / reference.fps]} if window else None
    stream = StreamingLeRobotDataset(DUMMY_REPO_ID, root=root, delta_timestamps=deltas)
    data = stream._load_episode_dataset(EpisodeParquetReader(root, columns=stream._projected_columns), 0)
    assert {"action", "state", "timestamp", "index"} <= data.numeric.keys()
    expected = torch.from_numpy(data.numeric["action"][0].copy())
    getitem = datasets.Dataset.__getitem__

    def numeric_reads_are_prepared(self: datasets.Dataset, key: Any) -> Any:
        assert not set(self.column_names).intersection(data.numeric)
        return getitem(self, key)

    # Metadata reads are separate from per-frame numeric-column formatting.
    stream.meta.episodes = stream.meta.episodes.to_list()
    monkeypatch.setattr(datasets.Dataset, "__getitem__", numeric_reads_are_prepared)
    item = stream._make_episode_item(data, 0, 0, video_cache=None)
    item["action"].fill_(-100)
    assert torch.equal(torch.from_numpy(data.numeric["action"][0]), expected)
    again = stream._make_episode_item(data, 0, 0, video_cache=None)
    assert torch.equal(again["action"][0] if window else again["action"], expected)
    if window:
        assert again["action_is_pad"].tolist() == [True, False, False]


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint16", "int64", "float16", "float32", "float64"])
@pytest.mark.parametrize("vector", [False, True])
def test_numeric_dtype_matches_existing_formatter(tmp_path: Path, dtype: str, vector: bool) -> None:
    feature = datasets.Value(dtype)
    if vector:
        feature = datasets.List(feature, length=2)
    features = datasets.Features({"episode_index": datasets.Value("int64"), "value": feature})
    values = [[0, 1], [1, 0]] if vector else [0, 1]
    reference = datasets.Dataset.from_dict({"episode_index": [0, 0], "value": values}, features=features)
    pq.write_table(reference.data.table, tmp_path / "data.parquet")
    stream = object.__new__(StreamingLeRobotDataset)
    stream.meta = SimpleNamespace(
        features={},
        video_keys=[],
        get_data_file_path=lambda _: "data.parquet",
        episodes=datasets.Dataset.from_list([{"dataset_from_index": 0, "dataset_to_index": 2}]),
    )
    stream._hf_features = features
    stream.delta_indices = None
    data = stream._load_episode_dataset(EpisodeParquetReader(tmp_path, columns=tuple(features)), 0)
    reference.set_transform(hf_transform_to_torch)
    assert "value" in data.numeric
    for index in range(2):
        expected = reference[index]["value"]
        actual = data.get_item(index)["value"]
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nonnumeric_and_nullable_columns_keep_feature_path(tmp_path: Path) -> None:
    features = datasets.Features(
        {
            "episode_index": datasets.Value("int64"),
            "nullable": datasets.Value("float32"),
            "nested_null": datasets.List(datasets.Value("float32"), length=2),
            "ragged": datasets.List(datasets.Value("int64")),
            "text": datasets.Value("string"),
        }
    )
    reference = datasets.Dataset.from_dict(
        {
            "episode_index": [0, 0],
            "nullable": [None, 2.0],
            "nested_null": [[1, None], [2, 3]],
            "ragged": [[1], [2, 3]],
            "text": ["a", "b"],
        },
        features=features,
    )
    pq.write_table(reference.data.table, tmp_path / "data.parquet")
    stream = object.__new__(StreamingLeRobotDataset)
    stream.meta = SimpleNamespace(
        features={},
        video_keys=[],
        get_data_file_path=lambda _: "data.parquet",
        episodes=datasets.Dataset.from_list([{"dataset_from_index": 0, "dataset_to_index": 2}]),
    )
    stream._hf_features = features
    stream.delta_indices = None
    data = stream._load_episode_dataset(EpisodeParquetReader(tmp_path, columns=tuple(features)), 0)
    assert set(data.numeric) == {"episode_index"}
    assert set(data.other.column_names) == {"nullable", "nested_null", "ragged", "text"}
    assert data.other.with_format(None)[:] == reference.remove_columns("episode_index")[:]


@pytest.mark.parametrize("shape", [(0,), (0, 2), (2, 0), (1, 0, 2), (1, 0, 2, 3), (1, 2, 0, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
@pytest.mark.parametrize("window", [False, True])
def test_streaming_zero_width_features_preserve_dtype_and_shape(
    tmp_path: Path, shape: tuple[int, ...], dtype: str, window: bool
) -> None:
    root = tmp_path / "data"
    fps = 10
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "target": {"dtype": dtype, "shape": shape, "names": None},
    }
    recorded = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, root=root, fps=fps, features=features, use_videos=False
    )
    for episode_index in range(2):
        for frame_index in range(4):
            recorded.add_frame(
                {
                    "action": np.full((2,), 4 * episode_index + frame_index, dtype=np.float32),
                    "target": np.empty(shape, dtype=dtype),
                    "task": "Read an empty target",
                }
            )
        recorded.save_episode()
    recorded.finalize()

    deltas = {key: [-1 / fps, 0.0, 1 / fps] for key in features} if window else None
    reference = LeRobotDataset(DUMMY_REPO_ID, root=root, delta_timestamps=deltas)
    assert reference.meta.get_data_file_path(0) == reference.meta.get_data_file_path(1)
    streaming = StreamingLeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        delta_timestamps=deltas,
        shuffle=False,
        episode_pool_size=1,
        prefetch_episodes=1,
    )

    samples = list(streaming)
    assert sorted(int(sample["index"]) for sample in samples) == list(range(8))
    expected_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
    expected_shape = (3, *shape) if window else shape
    for sample in samples:
        target = sample["target"]
        assert target.dtype == expected_dtype
        assert target.shape == expected_shape
        assert target.numel() == 0
        reference_item = reference[int(sample["index"])]
        torch.testing.assert_close(target, reference_item["target"])
        torch.testing.assert_close(sample["action"], reference_item["action"])
        if window:
            assert torch.equal(sample["target_is_pad"], reference_item["target_is_pad"])
            assert torch.equal(sample["action_is_pad"], reference_item["action_is_pad"])
            assert sample["target_is_pad"].tolist() == [
                int(sample["frame_index"]) == 0,
                False,
                int(sample["frame_index"]) == 3,
            ]
