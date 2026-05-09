"""Snap action[4] to sign({-1, +1}) — produce a discrete-gripper variant.

The 2-stage v11 ACT (40% @ thr=0.95) used discrete gripper actions {-1, +1}; v4
uses continuous gripper in [-1, 1]. ACT averages over open/close transitions
when continuous, blunting the gripper signal. Discretization at the data level
forces the policy to commit.

Usage:
    uv run python scripts_local/discretize_gripper.py \\
        --src-repo-id local/sim_3stage_v4_destale_tail30 \\
        --dst-repo-id local/sim_3stage_v4_destale_discgrip
"""
from __future__ import annotations
import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-repo-id", required=True)
    ap.add_argument("--dst-repo-id", required=True)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="action[4] >= threshold -> +1, else -1 (default 0.0)")
    args = ap.parse_args()

    src = Path(HF_LEROBOT_HOME) / args.src_repo_id
    dst = Path(HF_LEROBOT_HOME) / args.dst_repo_id
    if not src.exists():
        raise SystemExit(f"src missing: {src}")
    if dst.exists():
        logging.info(f"removing existing dst {dst}")
        shutil.rmtree(dst)
    logging.info(f"copying {src} -> {dst}")
    shutil.copytree(src, dst)

    fp = dst / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(fp)
    df = t.to_pandas()
    acts = np.stack(df["action"].values).astype(np.float32)
    cont = acts[:, 4].copy()
    new_grip = np.where(cont >= args.threshold, 1.0, -1.0).astype(np.float32)
    n_changed = int((new_grip != cont).sum())
    n_same_sign = int(np.sign(cont) == np.sign(new_grip)).sum() if False else 0  # not used
    acts[:, 4] = new_grip
    df["action"] = list(acts)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), fp)
    logging.info(f"wrote {fp} — discretized action[4] in {n_changed}/{len(cont)} rows ({n_changed/len(cont)*100:.1f}%)")
    logging.info("Run refresh_action_stats next:")
    logging.info(f"  uv run python -m lerobot.scripts.refresh_action_stats --repo-id {args.dst_repo_id}")


if __name__ == "__main__":
    main()
