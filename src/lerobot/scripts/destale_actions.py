"""Forward-fill stale (zero-motion) action targets so BC stops collapsing to zero.

Demos contain 41% frames with delta_xyz_yaw == 0 (teleop button-press idles +
slow alignment). BC + chunked policies (ACT/DP) learn a "near-goal => output 0"
attractor and stall mid-stage. Replacing those targets with the next decisive
action in the same episode breaks the attractor without dropping data or
re-encoding videos.

Operation per episode:
- stale := |action[:4]| < eps (motion dims only; gripper preserved)
- For each stale frame at t, set action[:4, t] = action[:4, t'] where t' is the
  next non-stale frame in the same ep within `lookahead` frames.
- Gripper dim (idx 4) is never modified.
- If no non-stale frame is found within lookahead: leave the row unchanged.

Then writes a copy of the source dataset (all metadata + videos copied verbatim)
into a new repo dir, with the action column rewritten in-place. Run
`refresh_action_stats` afterwards.

Usage:
    uv run python -m lerobot.scripts.destale_actions \\
        --src-repo-id local/sim_assemble_actdp_combined_cont \\
        --dst-repo-id local/sim_assemble_actdp_combined_destale \\
        --eps 1e-6 --lookahead 30
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME


def _destale_one(
    acts: np.ndarray, eps: float, lookahead: int, last_frac: float = 1.0
) -> tuple[np.ndarray, int, int]:
    """acts: (T, A) float32. Returns destaled copy + (#stale_total, #replaced).

    last_frac: only destale frames in last `last_frac` fraction of episode.
    last_frac=1.0 → all frames eligible; 0.3 → only last 30% of episode.
    """
    T, A = acts.shape
    motion = acts[:, :4]
    mag = np.abs(motion).max(axis=1)
    stale = mag < eps  # (T,)
    if last_frac < 1.0:
        cutoff = int(T * (1.0 - last_frac))
        eligible = np.zeros(T, dtype=bool)
        eligible[cutoff:] = True
        stale = stale & eligible
    n_stale = int(stale.sum())
    if n_stale == 0:
        return acts, 0, 0
    out = acts.copy()
    n_repl = 0
    for t in range(T):
        if not stale[t]:
            continue
        end = min(T, t + 1 + lookahead)
        nxt = None
        for s in range(t + 1, end):
            # need next non-stale-in-data frame (raw mag check, ignore frac mask)
            if mag[s] >= eps:
                nxt = s
                break
        if nxt is None:
            continue
        out[t, :4] = acts[nxt, :4]
        n_repl += 1
    return out, n_stale, n_repl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-repo-id", required=True)
    ap.add_argument("--dst-repo-id", required=True)
    ap.add_argument("--eps", type=float, default=1e-6, help="motion magnitude < eps => stale")
    ap.add_argument("--lookahead", type=int, default=30, help="max future frames to scan for replacement")
    ap.add_argument("--last-frac", type=float, default=1.0,
                    help="only destale stale frames in last fraction of each episode (1.0 = whole ep, 0.3 = last 30%)")
    ap.add_argument("--dry-run", action="store_true", help="report counts without writing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src_root = HF_LEROBOT_HOME / args.src_repo_id
    dst_root = HF_LEROBOT_HOME / args.dst_repo_id
    if not src_root.exists():
        raise FileNotFoundError(f"src not found: {src_root}")

    if args.dry_run:
        # Process in memory only.
        data_files = sorted(src_root.glob("data/chunk-*/file-*.parquet"))
        n_stale_tot = n_repl_tot = T_tot = 0
        for p in data_files:
            df = pd.read_parquet(p)
            for ep_idx, sub in df.groupby("episode_index", sort=True):
                acts = np.stack(sub["action"].apply(np.asarray).tolist()).astype(np.float32)
                _, n_stale, n_repl = _destale_one(acts, args.eps, args.lookahead, args.last_frac)
                n_stale_tot += n_stale
                n_repl_tot += n_repl
                T_tot += len(acts)
        logging.info(
            "DRY: total_frames=%d stale=%d (%.2f%%) replaced=%d (%.2f%%) leftover_stale=%d",
            T_tot, n_stale_tot, 100 * n_stale_tot / T_tot, n_repl_tot, 100 * n_repl_tot / T_tot,
            n_stale_tot - n_repl_tot,
        )
        return

    if dst_root.exists():
        logging.info("removing stale dst %s", dst_root)
        shutil.rmtree(dst_root)
    logging.info("copying %s -> %s", src_root, dst_root)
    shutil.copytree(src_root, dst_root)

    # Update info.json's repo_id if present (cosmetic)
    import json
    info_path = dst_root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if "repo_id" in info:
            info["repo_id"] = args.dst_repo_id
            info_path.write_text(json.dumps(info, indent=2))

    data_files = sorted(dst_root.glob("data/chunk-*/file-*.parquet"))
    n_stale_tot = n_repl_tot = T_tot = 0
    for p in data_files:
        df = pd.read_parquet(p)
        new_acts_per_idx = {}
        for ep_idx, sub in df.groupby("episode_index", sort=True):
            acts = np.stack(sub["action"].apply(np.asarray).tolist()).astype(np.float32)
            new_acts, n_stale, n_repl = _destale_one(acts, args.eps, args.lookahead, args.last_frac)
            n_stale_tot += n_stale
            n_repl_tot += n_repl
            T_tot += len(acts)
            for i, idx in enumerate(sub.index):
                new_acts_per_idx[idx] = new_acts[i]
        df["action"] = [new_acts_per_idx[i] for i in df.index]
        df.to_parquet(p)
        logging.info("wrote %s", p)

    logging.info(
        "DONE: total=%d stale=%d (%.2f%%) replaced=%d (%.2f%%) leftover_stale=%d",
        T_tot, n_stale_tot, 100 * n_stale_tot / T_tot,
        n_repl_tot, 100 * n_repl_tot / T_tot, n_stale_tot - n_repl_tot,
    )
    logging.info("Run refresh_action_stats next:\n  uv run python -m lerobot.scripts.refresh_action_stats --repo-id %s", args.dst_repo_id)


if __name__ == "__main__":
    main()
