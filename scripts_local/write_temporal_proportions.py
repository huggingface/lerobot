"""Compute + write temporal_proportions_{sparse,dense}.json from a merged ds.

For each named subtask, fraction = (sum of frames in that subtask across eps) / total frames.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.utils.constants import HF_LEROBOT_HOME


def compute_props(ds_root: Path, names_col: str, start_col: str, end_col: str) -> dict[str, float]:
    df = pq.read_table(ds_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    accum: dict[str, int] = {}
    total = 0
    for _, r in df.iterrows():
        names = r.get(names_col)
        starts = r.get(start_col)
        ends = r.get(end_col)
        if names is None or len(names) == 0:
            total += int(r["length"])
            continue
        for n, s, e in zip(list(names), list(starts), list(ends)):
            dur = int(e) - int(s) + 1
            accum[str(n)] = accum.get(str(n), 0) + dur
            total += dur
    return {k: v / total for k, v in accum.items()}


def main(repo_id: str) -> None:
    ds_root = HF_LEROBOT_HOME / repo_id
    sparse = compute_props(ds_root, "sparse_subtask_names", "sparse_subtask_start_frames", "sparse_subtask_end_frames")
    dense = compute_props(ds_root, "dense_subtask_names", "dense_subtask_start_frames", "dense_subtask_end_frames")
    if not sparse:
        sparse = {"task": 1.0}
    if not dense:
        dense = {"idle": 0.001, "task": 0.999}
    meta_dir = ds_root / "meta"
    (meta_dir / "temporal_proportions_sparse.json").write_text(json.dumps(sparse, indent=2))
    (meta_dir / "temporal_proportions_dense.json").write_text(json.dumps(dense, indent=2))
    print(f"  sparse → {sparse}")
    print(f"  dense  → {dense}")
    print(f"  written under {meta_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: write_temporal_proportions.py <repo_id>")
        raise SystemExit(2)
    main(sys.argv[1])
