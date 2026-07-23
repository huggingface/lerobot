# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Episode-scoped Parquet reads for training-time dataset streaming."""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


class EpisodeParquetReader:
    """Read complete episodes with column projection from local or fsspec roots."""

    def __init__(self, data_root: str | Path, *, columns: Sequence[str]):
        if not columns:
            raise ValueError("EpisodeParquetReader requires at least one projected column")
        self.columns = tuple(dict.fromkeys(columns))
        self._read_columns = (
            self.columns if "episode_index" in self.columns else (*self.columns, "episode_index")
        )
        self._filesystem, self._root_path = fsspec.core.url_to_fs(str(data_root))

    def read_episode(
        self,
        relative_path: str | Path,
        *,
        episode_index: int,
        expected_rows: int,
    ) -> pa.Table:
        if expected_rows <= 0:
            raise ValueError(f"Episode {episode_index} must contain at least one row")

        path = posixpath.join(self._root_path.rstrip("/"), str(relative_path).lstrip("/"))
        with self._filesystem.open(path, "rb") as source:
            parquet = pq.ParquetFile(source)
            available = set(parquet.schema_arrow.names)
            missing = sorted(set(self._read_columns) - available)
            if missing:
                raise ValueError(f"Parquet file {relative_path} is missing projected columns: {missing}")
            row_group = self._matching_row_group(parquet, episode_index)
            table = (
                parquet.read_row_group(row_group, columns=list(self._read_columns))
                if row_group is not None
                else parquet.read(columns=list(self._read_columns))
            )

        if row_group is None:
            table = table.filter(pc.equal(table.column("episode_index"), episode_index))
        self._validate_complete_episode(table, episode_index, expected_rows, relative_path)
        if "episode_index" not in self.columns:
            table = table.drop_columns(["episode_index"])
        return table

    @staticmethod
    def _matching_row_group(parquet: pq.ParquetFile, episode_index: int) -> int | None:
        episode_column = next(
            index
            for index in range(parquet.metadata.num_columns)
            if parquet.metadata.schema.column(index).path == "episode_index"
        )
        matches = []
        for row_group in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(row_group).column(episode_column).statistics
            if (
                statistics is not None
                and statistics.has_min_max
                and int(statistics.min) == episode_index
                and int(statistics.max) == episode_index
            ):
                matches.append(row_group)
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _validate_complete_episode(
        table: pa.Table,
        episode_index: int,
        expected_rows: int,
        relative_path: str | Path,
    ) -> None:
        actual_rows = len(table)
        if actual_rows != expected_rows:
            raise ValueError(
                f"Parquet episode {episode_index} in {relative_path}: "
                f"expected {expected_rows} rows, found {actual_rows}"
            )
        episodes = table.column("episode_index").to_pylist()
        if any(int(value) != episode_index for value in episodes):
            raise ValueError(f"Parquet file {relative_path} returned rows outside episode {episode_index}")
        if "frame_index" in table.column_names:
            frame_indices = [int(value) for value in table.column("frame_index").to_pylist()]
            if frame_indices != list(range(expected_rows)):
                raise ValueError(
                    f"Parquet episode {episode_index} in {relative_path} has non-contiguous frame indices"
                )
