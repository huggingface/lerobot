"""delete_episodes loses sparse/dense_subtask_* cols. Patch them back from src.

Maps: src kept-eps (in order, dropping deleted) → dst row positions.

Usage:
  uv run python scripts_local/patch_subtask_cols.py <src_repo_id> <dst_repo_id> <drop_eps_csv>
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from lerobot.utils.constants import HF_LEROBOT_HOME

COLS = (
    "sparse_subtask_names",
    "sparse_subtask_start_frames",
    "sparse_subtask_end_frames",
    "dense_subtask_names",
    "dense_subtask_start_frames",
    "dense_subtask_end_frames",
)


def main(src_repo: str, dst_repo: str, drop_csv: str) -> None:
    drop = set(int(x) for x in drop_csv.split(",") if x.strip())
    src_root = HF_LEROBOT_HOME / src_repo / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    dst_root = HF_LEROBOT_HOME / dst_repo / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    src = pq.read_table(src_root).to_pandas().sort_values("episode_index").reset_index(drop=True)
    dst = pq.read_table(dst_root).to_pandas().sort_values("episode_index").reset_index(drop=True)
    kept = src[~src["episode_index"].isin(drop)].reset_index(drop=True)
    if len(kept) != len(dst):
        raise RuntimeError(f"len mismatch: kept={len(kept)} dst={len(dst)}")
    for col in COLS:
        if col in kept.columns:
            dst[col] = kept[col].values
        else:
            print(f"  warn: src missing col {col}, skipping")
    dst.to_parquet(dst_root, index=False)
    print(f"patched {len(dst)} eps in {dst_root}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: patch_subtask_cols.py <src_repo_id> <dst_repo_id> <drop_eps_csv>")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
