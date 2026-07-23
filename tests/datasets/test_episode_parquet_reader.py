#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from pathlib import Path

import fsspec
import pytest

pytest.importorskip("pyarrow", reason="pyarrow is required (install lerobot[dataset])")

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.episode_parquet import EpisodeParquetReader


def _table(episodes: list[int]) -> pa.Table:
    frame_counts: dict[int, int] = {}
    frame_indices = []
    values = []
    ignored = []
    for episode in episodes:
        frame_index = frame_counts.get(episode, 0)
        frame_counts[episode] = frame_index + 1
        frame_indices.append(frame_index)
        values.append(episode * 10 + frame_index)
        ignored.append(f"ignored-{episode}-{frame_index}")
    return pa.table(
        {
            "episode_index": episodes,
            "frame_index": frame_indices,
            "value": values,
            "ignored": ignored,
        }
    )


def _write_episode_row_groups(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(path, _table([0]).schema)
    try:
        writer.write_table(_table([0, 0]))
        writer.write_table(_table([1, 1, 1]))
    finally:
        writer.close()


def test_reader_projects_columns_and_reads_matching_row_group(tmp_path: Path) -> None:
    path = tmp_path / "data/chunk-000/file-000.parquet"
    _write_episode_row_groups(path)
    reader = EpisodeParquetReader(tmp_path, columns=("episode_index", "frame_index", "value"))

    table = reader.read_episode(path.relative_to(tmp_path), episode_index=1, expected_rows=3)

    assert table.column_names == ["episode_index", "frame_index", "value"]
    assert table.column("value").to_pylist() == [10, 11, 12]


def test_reader_filters_legacy_mixed_row_group(tmp_path: Path) -> None:
    path = tmp_path / "data/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(_table([0, 0, 1, 1, 1]), path)
    reader = EpisodeParquetReader(tmp_path, columns=("episode_index", "frame_index", "value"))

    table = reader.read_episode(path.relative_to(tmp_path), episode_index=1, expected_rows=3)

    assert table.column("episode_index").to_pylist() == [1, 1, 1]
    assert table.column("frame_index").to_pylist() == [0, 1, 2]


def test_reader_rejects_partial_episode(tmp_path: Path) -> None:
    path = tmp_path / "data/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(_table([2, 2]), path)
    reader = EpisodeParquetReader(tmp_path, columns=("episode_index", "frame_index"))

    with pytest.raises(ValueError, match="expected 3 rows, found 2"):
        reader.read_episode(path.relative_to(tmp_path), episode_index=2, expected_rows=3)


def test_reader_rejects_missing_episode(tmp_path: Path) -> None:
    path = tmp_path / "data/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(_table([0, 0]), path)
    reader = EpisodeParquetReader(tmp_path, columns=("episode_index", "frame_index"))

    with pytest.raises(ValueError, match="episode 4"):
        reader.read_episode(path.relative_to(tmp_path), episode_index=4, expected_rows=1)


def test_reader_supports_fsspec_remote_root() -> None:
    filesystem = fsspec.filesystem("memory")
    root = "memory://episode-reader"
    path = "episode-reader/data/chunk-000/file-000.parquet"
    with filesystem.open(path, "wb") as output:
        pq.write_table(_table([0, 0, 0]), output)
    reader = EpisodeParquetReader(root, columns=("episode_index", "value"))

    table = reader.read_episode("data/chunk-000/file-000.parquet", episode_index=0, expected_rows=3)

    assert table.column("value").to_pylist() == [0, 1, 2]
