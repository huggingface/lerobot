#!/usr/bin/env python
"""
Helpers to append/ingest rows into Lance dual tables.

- append_frames(frames_path, rows, schema): append one or more frame rows to frames table
- append_episodes(episodes_path, rows, schema): append one or more episode rows to episodes table

Note: Lance currently writes new datasets via write_dataset API; appending can be performed by opening with a write session
or by creating a new dataset and merging. For simplicity, we implement append by reading the existing table into Arrow,
concatenating with new rows, and rewriting the dataset directory. This is suitable for small-scale unit tests and demos.
For production, prefer Lance's native append APIs once available in your environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pyarrow as pa


def _append_common(path: Path, new_table: pa.Table) -> None:
    try:
        import lance  # type: ignore
    except Exception as e:
        raise ImportError("Missing 'lance' dependency; unable to append") from e

    if path.exists():
        ds = lance.dataset(str(path))
        old_tbl = ds.to_table()
        combined = pa.concat_tables([old_tbl, new_table])
        # Overwrite dataset directory with combined table
        # NOTE: This is a simple approach for demos/tests; for large datasets, use incremental append APIs.
        path.unlink(missing_ok=True)
        lance.write_dataset(combined, str(path), schema=combined.schema)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        lance.write_dataset(new_table, str(path), schema=new_table.schema)


def append_frames(frames_path: Path, rows: List[Dict[str, object]], schema: pa.Schema) -> None:
    cols: Dict[str, List[object]] = {f.name: [] for f in schema}
    for r in rows:
        for k in cols.keys():
            cols[k].append(r.get(k))
    tbl = pa.Table.from_pydict(cols, schema=schema)
    _append_common(frames_path, tbl)


def append_episodes(episodes_path: Path, rows: List[Dict[str, object]], schema: pa.Schema) -> None:
    cols: Dict[str, List[object]] = {f.name: [] for f in schema}
    for r in rows:
        for k in cols.keys():
            cols[k].append(r.get(k))
    tbl = pa.Table.from_pydict(cols, schema=schema)
    _append_common(episodes_path, tbl)
