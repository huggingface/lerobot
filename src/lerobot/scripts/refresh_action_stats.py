"""Recompute the per-episode action stats in `meta/episodes/*.parquet` so they
match the actual action values stored in `data/*.parquet`.

`convert_gripper_to_continuous.py` rewrites the action column but leaves the
episode meta stats untouched, so the saved policy normalizer ends up with
gripper min=0/max=2/mean=0.67 even though the data is in {-1,0,+1}. Trained
policies then unnormalize to nonsense at eval.

Usage:
    uv run python -m lerobot.scripts.refresh_action_stats \\
        --repo-id local/sim_assemble_actdp_combined_cont
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


STAT_NAMES = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def _qval(arr: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(arr, q, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
    root = Path(ds.root)

    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquets under {root}")

    # Concat episode_index + action
    frames = []
    for p in data_files:
        df = pd.read_parquet(p, columns=["episode_index", "action"])
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)

    meta_files = sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    for mfp in meta_files:
        meta = pd.read_parquet(mfp)
        for i, row in meta.iterrows():
            ep_idx = int(row["episode_index"])
            ep = big[big["episode_index"] == ep_idx]
            if len(ep) == 0:
                logging.warning("ep %d empty in data shards, skipping", ep_idx)
                continue
            acts = np.stack(ep["action"].apply(np.asarray).tolist()).astype(np.float32)
            stats = {
                "min": acts.min(axis=0),
                "max": acts.max(axis=0),
                "mean": acts.mean(axis=0),
                "std": acts.std(axis=0),
                "q01": _qval(acts, 0.01),
                "q10": _qval(acts, 0.10),
                "q50": _qval(acts, 0.50),
                "q90": _qval(acts, 0.90),
                "q99": _qval(acts, 0.99),
            }
            for k in STAT_NAMES:
                meta.at[i, f"stats/action/{k}"] = list(stats[k])
        meta.to_parquet(mfp)
        logging.info("Refreshed action stats in %s (%d eps)", mfp, len(meta))

    # Sanity: print before/after action mean/std summary across eps.
    sample = pd.read_parquet(meta_files[0])
    means = np.stack(sample["stats/action/mean"].apply(np.asarray).tolist())
    stds = np.stack(sample["stats/action/std"].apply(np.asarray).tolist())
    logging.info("post: action mean (avg across eps) = %s", means.mean(axis=0))
    logging.info("post: action std  (avg across eps) = %s", stds.mean(axis=0))

    # Also rewrite the aggregated meta/stats.json so consumers that read it
    # (factory.py / dataset normalizer) see fresh action stats.
    import json
    stats_json_path = root / "meta" / "stats.json"
    if stats_json_path.exists():
        with stats_json_path.open() as f:
            agg = json.load(f)
        all_acts = np.stack(big["action"].apply(np.asarray).tolist()).astype(np.float32)
        agg["action"] = {
            "min": all_acts.min(axis=0).tolist(),
            "max": all_acts.max(axis=0).tolist(),
            "mean": all_acts.mean(axis=0).tolist(),
            "std": all_acts.std(axis=0).tolist(),
            "count": [int(len(all_acts))],
            "q01": _qval(all_acts, 0.01).tolist(),
            "q10": _qval(all_acts, 0.10).tolist(),
            "q50": _qval(all_acts, 0.50).tolist(),
            "q90": _qval(all_acts, 0.90).tolist(),
            "q99": _qval(all_acts, 0.99).tolist(),
        }
        with stats_json_path.open("w") as f:
            json.dump(agg, f, indent=2)
        logging.info("Refreshed meta/stats.json action: mean=%s std=%s", agg["action"]["mean"], agg["action"]["std"])


if __name__ == "__main__":
    main()
