"""Truncate every episode of a LeRobotDataset at end of stage K (sparse-subtask).

For each ep, keeps frames [0, sparse_subtask_end_frames[K] + 1). Updates:
- data/*.parquet — slice rows; reindex `index` globally.
- meta/episodes/*.parquet — length, dataset_from_index/to_index,
  sparse/dense_subtask_names/start/end (truncate last entry to n-1, drop extras),
  videos/<key>/to_timestamp = (n-1)/fps + 1/(2*fps).
- meta/info.json — total_frames.

Videos are left in place; the LeRobotDataset reader looks up frames by
`from_timestamp + per-frame ts` so unreferenced frames in the MP4 are dead bytes
and don't affect training.

Per-ep stats (action/state min/max etc.) in episodes meta are left as-is;
training preset uses global stats.json (recompute via `refresh_action_stats`
after this script).

Usage:
    uv run python scripts_local/truncate_episodes_at_stage.py \\
        --src-repo-id local/sim_3stage_v2_full_v2_succonly_destale_tail30 \\
        --dst-repo-id local/sim_3stage_v2_first4_destale_tail30 \\
        --stage-idx 3
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-repo-id", required=True)
    ap.add_argument("--dst-repo-id", required=True)
    ap.add_argument("--stage-idx", type=int, default=3,
                    help="0-indexed; keep stages [0, stage_idx] inclusive")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src = HF_LEROBOT_HOME / args.src_repo_id
    dst = HF_LEROBOT_HOME / args.dst_repo_id
    if not src.exists():
        raise FileNotFoundError(src)

    if not args.dry_run:
        if dst.exists():
            logging.info("removing stale dst %s", dst)
            shutil.rmtree(dst)
        logging.info("copying %s -> %s", src, dst)
        shutil.copytree(src, dst)
        work = dst
    else:
        work = src

    info = json.loads((work / "meta" / "info.json").read_text())
    fps = info["fps"]

    # Load all ep meta files into one df, preserving file mapping for later writeback.
    meta_files = sorted((work / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not meta_files:
        raise RuntimeError("no episodes meta files")
    file_to_dfs = {p: pd.read_parquet(p) for p in meta_files}
    all_meta = pd.concat(file_to_dfs.values(), ignore_index=True).sort_values("episode_index").reset_index(drop=True)

    # Compute new lengths per ep.
    new_lens: dict[int, int] = {}
    for _, row in all_meta.iterrows():
        ep_idx = int(row["episode_index"])
        ends = list(row["sparse_subtask_end_frames"])
        if args.stage_idx >= len(ends):
            raise ValueError(f"ep {ep_idx} only has {len(ends)} sparse stages")
        new_lens[ep_idx] = int(ends[args.stage_idx]) + 1

    # Compute per-ep max kept timestamp from actual data parquet (source may have
    # non-uniform sampling — can't assume ts = frame_index / fps).
    new_to_ts: dict[int, float] = {}
    src_data_files = sorted((src / "data").glob("chunk-*/file-*.parquet"))
    for p in src_data_files:
        sdf = pd.read_parquet(p, columns=["episode_index", "frame_index", "timestamp"])
        for ep_idx, sub in sdf.groupby("episode_index", sort=True):
            sub = sub.sort_values("frame_index").reset_index(drop=True)
            n = new_lens[int(ep_idx)]
            last_ts = float(sub.iloc[n - 1]["timestamp"])
            new_to_ts[int(ep_idx)] = last_ts + 0.5 / fps

    total_src = int(sum(int(r["length"]) for _, r in all_meta.iterrows()))
    total_dst = int(sum(new_lens.values()))
    logging.info("kept %d eps, %d -> %d frames (%.1f%% kept)",
                 len(new_lens), total_src, total_dst, 100 * total_dst / total_src)

    if args.dry_run:
        return

    # Slice data parquets, reindex global `index`.
    data_files = sorted((work / "data").glob("chunk-*/file-*.parquet"))
    global_idx = 0
    for p in data_files:
        df = pd.read_parquet(p)
        kept_parts = []
        for ep_idx, sub in df.groupby("episode_index", sort=True):
            sub = sub.sort_values("frame_index").reset_index(drop=True)
            n = new_lens[int(ep_idx)]
            kept_parts.append(sub.iloc[:n])
        df_new = pd.concat(kept_parts, ignore_index=True)
        df_new["index"] = range(global_idx, global_idx + len(df_new))
        global_idx += len(df_new)
        df_new.to_parquet(p, index=False)
        logging.info("wrote data %s: %d rows", p.name, len(df_new))

    assert global_idx == total_dst, f"index mismatch: {global_idx} vs {total_dst}"

    # Update episodes meta per file: length, dataset_from/to_index, subtask cols, video to_timestamp.
    cum = 0
    for p, df in file_to_dfs.items():
        df = df.sort_values("episode_index").reset_index(drop=True)
        for i, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            n = new_lens[ep_idx]
            df.at[i, "length"] = n
            df.at[i, "dataset_from_index"] = cum
            df.at[i, "dataset_to_index"] = cum + n
            cum += n
            # truncate subtask cols
            for prefix in ("sparse", "dense"):
                names_col = f"{prefix}_subtask_names"
                starts_col = f"{prefix}_subtask_start_frames"
                ends_col = f"{prefix}_subtask_end_frames"
                if names_col not in df.columns:
                    continue
                names = list(row[names_col])[: args.stage_idx + 1]
                starts = list(row[starts_col])[: args.stage_idx + 1]
                ends = list(row[ends_col])[: args.stage_idx + 1]
                ends[-1] = n - 1
                df.at[i, names_col] = np.array(names, dtype=object)
                df.at[i, starts_col] = np.array(starts, dtype=np.int64)
                df.at[i, ends_col] = np.array(ends, dtype=np.int64)
            # video to_timestamp = max kept frame's actual ts + half-frame margin
            ep_to_ts = new_to_ts[ep_idx]
            for col in df.columns:
                if col.endswith("/to_timestamp"):
                    df.at[i, col] = float(ep_to_ts)
        df.to_parquet(p, index=False)
        logging.info("wrote ep meta %s: %d eps", p.name, len(df))

    # Update info.json
    info["total_frames"] = total_dst
    (work / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    logging.info("updated info.json: total_frames=%d", total_dst)

    # Drop temporal_proportions files (will rebuild via T3).
    for f in ("temporal_proportions_sparse.json", "temporal_proportions_dense.json"):
        fp = work / "meta" / f
        if fp.exists():
            fp.unlink()
            logging.info("removed %s (rebuild via T3)", f)

    # Drop CLIP cache (rebuild if needed for RA-BC compute).
    clip_cache = work / "clip_cache.npz"
    if clip_cache.exists():
        clip_cache.unlink()
        logging.info("removed clip_cache.npz (rebuild if RA-BC needs)")

    logging.info("DONE. Next: refresh_action_stats; write_temporal_proportions; rebuild clip_cache if needed.")


if __name__ == "__main__":
    main()
