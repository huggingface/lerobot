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
from unittest.mock import Mock

import fsspec
import pytest

pytest.importorskip("pyarrow", reason="pyarrow is required (install lerobot[dataset])")

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.streaming.episode_parquet import EpisodeParquetReader


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


def test_reader_forwards_explicit_token_to_hf_filesystem(monkeypatch) -> None:
    url_to_fs = Mock(return_value=(fsspec.filesystem("memory"), "datasets/private@revision"))
    monkeypatch.setattr(fsspec.core, "url_to_fs", url_to_fs)

    EpisodeParquetReader(
        "hf://datasets/private@revision",
        columns=("episode_index",),
        token="hf_test_token",
    )

    url_to_fs.assert_called_once_with("hf://datasets/private@revision", token="hf_test_token")


def test_reader_retries_transient_remote_file_not_found(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "episode.parquet"
    pq.write_table(_table([0, 0]), path)
    filesystem = Mock()
    attempts = 0

    def open_remote(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("partial HfFileSystem dircache")
        return path.open("rb")

    filesystem.open.side_effect = open_remote
    filesystem.invalidate_cache = Mock()
    monkeypatch.setattr(fsspec.core, "url_to_fs", Mock(return_value=(filesystem, "root")))
    reader = EpisodeParquetReader(
        "hf://datasets/private@revision",
        columns=("episode_index", "frame_index"),
        max_retries=1,
        retry_backoff_s=0,
    )

    table = reader.read_episode("data/file.parquet", episode_index=0, expected_rows=2)

    assert len(table) == 2
    assert attempts == 2
    filesystem.invalidate_cache.assert_called_once()


@pytest.mark.parametrize(
    "groups,expected_groups",
    [
        ([[0, 0], [1, 1], [2, 2]], [1]),
        ([[0, 0], [0, 1, 1], [2, 2]], [1]),
        ([[0, 0], [1, 1], [1, 1], [2, 2]], [1, 2]),
        ([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2]], [1, 2, 3]),
        ([[0, 0], [0, 1], [1, 2], [2, 2]], [1, 2]),
        # Min/max is only a candidate test: a mixed group's range can contain
        # the target episode even when no row in that group does.
        ([[0, 2], [1, 1], [3, 3]], [0, 1]),
    ],
)
def test_reader_prunes_unrelated_row_groups(
    tmp_path: Path, monkeypatch, groups: list[list[int]], expected_groups: list[int]
) -> None:
    path = tmp_path / "mixed.parquet"
    flat_episodes = [episode for group in groups for episode in group]
    source = _table(flat_episodes)
    start = 0
    with pq.ParquetWriter(path, source.schema) as writer:
        for group in groups:
            writer.write_table(source.slice(start, len(group)))
            start += len(group)

    loaded_groups: list[int] = []
    original_groups = pq.ParquetFile.read_row_groups
    original_group = pq.ParquetFile.read_row_group
    original_read = pq.ParquetFile.read

    def read_groups(self, row_groups, *args, **kwargs):
        loaded_groups.extend(row_groups)
        return original_groups(self, row_groups, *args, **kwargs)

    def read_group(self, row_group, *args, **kwargs):
        loaded_groups.append(row_group)
        return original_group(self, row_group, *args, **kwargs)

    def read(self, *args, **kwargs):
        loaded_groups.extend(range(self.metadata.num_row_groups))
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_groups", read_groups)
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read_group)
    monkeypatch.setattr(pq.ParquetFile, "read", read)
    reader = EpisodeParquetReader(tmp_path, columns=("frame_index", "value"))
    expected_rows = flat_episodes.count(1)
    table = reader.read_episode(path.name, episode_index=1, expected_rows=expected_rows)

    assert loaded_groups == expected_groups
    assert table.column_names == ["frame_index", "value"]
    assert table.to_pydict() == {
        "frame_index": list(range(expected_rows)),
        "value": list(range(10, 10 + expected_rows)),
    }


@pytest.mark.parametrize("write_statistics", [False, ["frame_index", "value"]])
def test_reader_keeps_row_groups_with_unknown_episode_statistics(
    tmp_path: Path, write_statistics: bool | list[str]
) -> None:
    path = tmp_path / "unknown-statistics.parquet"
    source = _table([0, 0, 1, 1, 1, 2])
    pq.write_table(source, path, row_group_size=2, write_statistics=write_statistics)
    reader = EpisodeParquetReader(tmp_path, columns=("episode_index", "frame_index", "value"))

    table = reader.read_episode(path.name, episode_index=1, expected_rows=3)

    assert table.to_pydict() == {"episode_index": [1, 1, 1], "frame_index": [0, 1, 2], "value": [10, 11, 12]}


def test_pruned_reader_still_rejects_out_of_order_episode(tmp_path: Path) -> None:
    path = tmp_path / "out-of-order.parquet"
    source = _table([0, 0, 1, 1, 1, 1, 2, 2]).take(pa.array([0, 1, 2, 4, 3, 5, 6, 7]))
    pq.write_table(source, path, row_group_size=2)
    reader = EpisodeParquetReader(tmp_path, columns=("episode_index", "frame_index", "value"))

    with pytest.raises(ValueError, match="non-contiguous frame indices"):
        reader.read_episode(path.name, episode_index=1, expected_rows=4)
