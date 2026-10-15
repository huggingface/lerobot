#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared fixtures for annotation-pipeline tests.

The on-disk dataset builder lives with the other dataset factories in
``tests/fixtures/dataset_factories.py`` (:func:`build_annotation_dataset`);
these fixtures only wire it into pytest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ``build_annotation_dataset`` pulls in ``lerobot.datasets`` (HF ``datasets``
# + ``pandas``, only in the ``dataset`` extra), so it's imported lazily inside
# each fixture — this conftest stays importable without that extra. The test
# modules ``pytest.importorskip("datasets")`` so they skip rather than error.


@pytest.fixture
def fixture_dataset_root(tmp_path: Path) -> Path:
    """A tiny dataset with two episodes, 12 frames each at 10 fps."""
    from tests.fixtures.dataset_factories import build_annotation_dataset

    return build_annotation_dataset(
        tmp_path / "ds",
        episode_specs=[
            (0, 12, "Could you tidy the kitchen please?"),
            (1, 12, "Please clean up the kitchen"),
        ],
        fps=10,
    )


@pytest.fixture
def single_episode_root(tmp_path: Path) -> Path:
    from tests.fixtures.dataset_factories import build_annotation_dataset

    return build_annotation_dataset(
        tmp_path / "ds_one",
        episode_specs=[(0, 30, "Pour water from the bottle into the cup.")],
        fps=10,
    )


@pytest.fixture
def unaligned_episode_root(tmp_path: Path) -> Path:
    """Two episodes crossing both the 1024-row hash batch and row-group cuts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.fixtures.dataset_factories import build_annotation_dataset

    root = build_annotation_dataset(
        tmp_path / "unaligned", [(0, 1200, "Pour water."), (1, 1000, "Pour water.")], fps=400
    )
    first = root / "data/chunk-000/file-000.parquet"
    second = first.with_name("file-001.parquet")
    table = pa.concat_tables([pq.read_table(first), pq.read_table(second)])
    table = table.append_column(
        "observation.state", pa.array([[float(i % 7), None] for i in range(2200)], pa.list_(pa.float32(), 2))
    )
    table = table.append_column("note", pa.array([None if i % 3 else f"frame-{i}" for i in range(2200)]))
    table = table.append_column("image_bytes", pa.array([b"rgb" if i % 2 else None for i in range(2200)]))
    pq.write_table(table, first, row_group_size=2200)
    second.unlink()
    return root


@pytest.fixture
def fragmented_parquet_reader(monkeypatch):
    """Real Arrow reads with per-row-group batches, independent of version coalescing."""
    import pyarrow.parquet as pq

    original = pq.ParquetFile.iter_batches

    def iter_batches(file, *args, **kwargs):
        groups = kwargs.pop("row_groups", None)
        for group in range(file.metadata.num_row_groups) if groups is None else groups:
            yield from original(file, *args, row_groups=[group], **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", iter_batches)
