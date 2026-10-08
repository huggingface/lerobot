# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""File-owned dense patches: missing patches never erase source rows or labels."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _pyarrow_available, require_package

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow as pa
    import pyarrow.parquet as pq


def patch_table(
    source: pa.Table,
    patch: pa.Table,
    *,
    columns: tuple[str, ...],
    keys: tuple[str, ...] = ("episode_index", "frame_index"),
) -> pa.Table:
    """Replace only explicitly owned columns at explicitly supplied identities."""
    require_package("pyarrow", "dataset")
    if not keys or not columns or len(set(keys)) != len(keys) or len(set(columns)) != len(columns):
        raise ValueError("Keys/columns must be nonempty and unique")
    if set(keys) & set(columns) or {"index", "timestamp", "episode_index", "frame_index"} & set(columns):
        raise ValueError("Patches cannot replace frame identity or timeline columns")
    if set(patch.column_names) != set(keys) | set(columns) or not set(keys) <= set(source.column_names):
        raise ValueError("Patch schema must contain exactly the keys and owned columns")
    for key in keys:
        if source.schema.field(key).type != patch.schema.field(key).type:
            raise ValueError(f"Identity type mismatch: {key}")
    source_keys = list(zip(*(source.column(key).to_pylist() for key in keys), strict=True))
    patch_keys = list(zip(*(patch.column(key).to_pylist() for key in keys), strict=True))
    if any(None in key for key in source_keys + patch_keys):
        raise ValueError("Frame identities cannot be null")
    if len(set(source_keys)) != len(source_keys) or len(set(patch_keys)) != len(patch_keys):
        raise ValueError("Frame identities must be unique")
    positions = {key: index for index, key in enumerate(source_keys)}
    if not set(patch_keys) <= positions.keys():
        raise ValueError("Patch references unknown source frames")
    result = source
    for column in columns:
        field = patch.schema.field(column)
        if column in source.column_names:
            if field.type != source.schema.field(column).type:
                raise ValueError(f"Owned column type changed: {column}")
            field = source.schema.field(column)
            values = source.column(column).to_pylist()
        else:
            values = [None] * source.num_rows
        for frame_key, value in zip(patch_keys, patch.column(column).to_pylist(), strict=True):
            values[positions[frame_key]] = value
        array = pa.array(values, type=field.type)
        if column in result.column_names:
            result = result.set_column(result.column_names.index(column), field, array)
        else:
            result = result.append_column(field, array)
    return result


def materialize_patch(source: Path, patch: pa.Table, destination: Path, *, columns: tuple[str, ...]) -> None:
    """Write a separate output file; never mutate an immutable input/cache file."""
    require_package("pyarrow", "dataset")
    if source.resolve() == destination.resolve():
        raise ValueError("Materialize into a separate version, not the input dataset")
    table = patch_table(pq.read_table(source), patch, columns=columns)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".patch-", suffix=".parquet", dir=destination.parent)
    os.close(descriptor)
    try:
        pq.write_table(table, temporary)
        if not pq.read_table(temporary).equals(table):
            raise ValueError("Parquet patch failed round-trip validation")
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
