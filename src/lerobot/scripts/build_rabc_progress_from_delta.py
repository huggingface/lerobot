"""Derive a SARM-progress parquet from a relabeled delta-reward dataset.

When `lerobot-relabel-sarm --reward-mode delta` is used, the per-frame
`next.reward` is the difference of SARM-progress between successive
frames. Cumulating per episode therefore reconstructs the (clipped)
progress curve we would have read from a fresh SARM forward — without
needing to invoke the multi-cam SARM model on every frame.

The output schema matches `lerobot.utils.rabc.RABCWeights`:
columns = index, episode_index, progress_<head_mode>.

Usage:
    uv run python -m lerobot.scripts.build_rabc_progress_from_delta \
        --src-repo-id local/sim_assemble_sarm_merged_v1_sarm_delta \
        --head-mode sparse \
        --output sarm_progress.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-repo-id", required=True, help="Local/HF id of relabeled delta ds.")
    parser.add_argument(
        "--head-mode",
        default="sparse",
        choices=["sparse", "dense", "both"],
        help="Which progress column(s) to populate. Both writes progress_sparse and progress_dense.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path. Defaults to <ds_root>/sarm_progress.parquet.",
    )
    parser.add_argument(
        "--clip-progress",
        action="store_true",
        help="Clip cumulative progress to [0, 1] (recommended for SARM-progress contract).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Loading dataset metadata: %s", args.src_repo_id)
    ds = LeRobotDataset(args.src_repo_id, download_videos=False)
    df_chunks: list[pd.DataFrame] = []
    for parquet_path in sorted(Path(ds.root).glob("data/chunk-*/file-*.parquet")):
        df_chunks.append(
            pd.read_parquet(
                parquet_path, columns=["index", "episode_index", "next.reward", "frame_index"]
            )
        )
    df = pd.concat(df_chunks, ignore_index=True).sort_values("index").reset_index(drop=True)
    logging.info("Loaded %d frames across %d episodes", len(df), df.episode_index.nunique())

    df["progress"] = df.groupby("episode_index")["next.reward"].cumsum()
    if args.clip_progress:
        df["progress"] = df["progress"].clip(lower=0.0, upper=1.0)

    out = pd.DataFrame(
        {
            "index": df["index"].astype(np.int64),
            "episode_index": df["episode_index"].astype(np.int64),
        }
    )
    if args.head_mode in ("sparse", "both"):
        out["progress_sparse"] = df["progress"].astype(np.float32)
    if args.head_mode in ("dense", "both"):
        out["progress_dense"] = df["progress"].astype(np.float32)

    output_path = Path(args.output) if args.output else Path(ds.root) / "sarm_progress.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, output_path)
    logging.info("Wrote %s (%d rows)", output_path, len(out))
    logging.info(
        "Progress stats: min=%.3f max=%.3f mean=%.3f",
        out.filter(like="progress").to_numpy().min(),
        out.filter(like="progress").to_numpy().max(),
        out.filter(like="progress").to_numpy().mean(),
    )


if __name__ == "__main__":
    main()
