"""Convert a LeRobotDataset's discrete-class gripper action (5th element of
``action``: 0=CLOSE, 1=STAY, 2=OPEN) into a continuous value in {-1, 0, +1}.

The output dataset is a copy of the source under a new repo_id with only the
``action`` column rewritten. All other columns (observations, rewards,
metadata, videos, episode boundaries) are preserved verbatim.

Used by V6+ HIL-SERL to feed a 5-D continuous SAC actor on demos that were
recorded with the gym-hil discrete gripper convention.

Usage:
    uv run python -m lerobot.scripts.convert_gripper_to_continuous \\
        --src-repo-id local/sim_assemble_sarm_merged_v1_sarm_dense \\
        --dst-repo-id local/sim_assemble_sarm_merged_v1_sarm_dense_cont
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset


GRIPPER_MAP = {0: -1.0, 1: 0.0, 2: 1.0}


def remap_action(action_arr: np.ndarray) -> np.ndarray:
    """Map last column of an N-d action array from {0,1,2} -> {-1,0,1}."""
    out = action_arr.astype(np.float32, copy=True)
    last = out[..., -1]
    last_int = np.rint(last).astype(np.int32)
    mapped = np.full_like(last, np.nan, dtype=np.float32)
    for k, v in GRIPPER_MAP.items():
        mapped = np.where(last_int == k, v, mapped)
    if np.any(np.isnan(mapped)):
        bad = np.unique(last_int[np.isnan(mapped)])
        raise ValueError(f"Unmappable gripper class values: {bad.tolist()}")
    out[..., -1] = mapped
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-repo-id", required=True)
    ap.add_argument("--dst-repo-id", required=True)
    ap.add_argument("--src-root", default=None)
    ap.add_argument("--dst-root", default=None, help="Default: ~/.cache/huggingface/lerobot/<dst-repo-id>")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src_ds = LeRobotDataset(args.src_repo_id, root=args.src_root, download_videos=False)
    src_root = Path(src_ds.root)

    if args.dst_root:
        dst_root = Path(args.dst_root)
    else:
        # Mirror the cache layout used by HF datasets.
        dst_root = src_root.parent / args.dst_repo_id.split("/")[-1]
    if dst_root.exists():
        raise FileExistsError(f"Destination {dst_root} already exists.")

    logging.info("Copying %s -> %s (full tree, then patching action column)", src_root, dst_root)
    shutil.copytree(src_root, dst_root)

    # Rewrite each parquet shard with mapped action.
    parquets = sorted(dst_root.glob("data/chunk-*/file-*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No data parquets found under {dst_root}")
    n_total = 0
    for p in parquets:
        df = pd.read_parquet(p)
        actions = np.stack(df["action"].apply(np.asarray).tolist())
        new_actions = remap_action(actions)
        df["action"] = list(new_actions)
        df.to_parquet(p)
        n_total += len(df)
        logging.info("  patched %s (%d rows)", p, len(df))

    # Update meta/info.json's repo_id reference if present (best-effort).
    info_path = dst_root / "meta" / "info.json"
    if info_path.exists():
        import json

        with info_path.open() as f:
            info = json.load(f)
        info.setdefault("notes", {})
        info["notes"]["gripper_continuous"] = (
            "Last action element remapped from gripper class (0=CLOSE, 1=STAY, 2=OPEN) to "
            "continuous {-1, 0, +1} by convert_gripper_to_continuous.py"
        )
        with info_path.open("w") as f:
            json.dump(info, f, indent=2)

    logging.info("Done. Patched %d frames in %d parquet shards.", n_total, len(parquets))
    logging.info("New dataset at %s", dst_root)


if __name__ == "__main__":
    main()
