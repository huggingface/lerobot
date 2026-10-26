#!/usr/bin/env python
"""Driver for `eval_grounded_grasp_v2_execute_horizon_ablation.py`: one fresh subprocess per
(execute_horizon, task) pair (same VLABench/MuJoCo state-leak workaround as every other multitask
eval on this branch). `E50` is NOT re-simulated -- it is copied verbatim from Phase 1A's own
already-completed rollout (`REUSE_E50_FROM`), which used the identical checkpoint/config/seeds.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/run_grounded_grasp_v2_execute_horizon_ablation_isolated.py --horizons 50 10 5
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402
from eval_grounded_grasp_v2_execute_horizon_ablation import OUT_DIR_ROOT, SEEDS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EVAL_SCRIPT = Path(__file__).parent / "eval_grounded_grasp_v2_execute_horizon_ablation.py"
REUSE_E50_FROM = Path("outputs/eval/safediff_vla_grounded_grasp_v2_monitor_baseline")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizons", type=int, nargs="+", default=[50, 10, 5])
    parser.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output-dir", default=str(OUT_DIR_ROOT))
    args = parser.parse_args()
    out_dir_root = Path(args.output_dir)

    for h in args.horizons:
        horizon_dir = out_dir_root / f"E{h}"
        for task in args.tasks:
            dst = horizon_dir / task / "episodes_raw.json"
            if dst.exists():
                logger.info("=== [E%d/%s] already done, skipping ===", h, task)
                continue
            if h == 50:
                src = REUSE_E50_FROM / task / "episodes_raw.json"
                src_videos = REUSE_E50_FROM / task / "videos"
                if not src.exists():
                    logger.error("=== [E50/%s] REUSE SOURCE MISSING: %s ===", task, src)
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                if src_videos.exists():
                    shutil.copytree(src_videos, horizon_dir / task / "videos", dirs_exist_ok=True)
                logger.info("=== [E50/%s] reused Phase 1A rollout from %s ===", task, src)
                continue
            logger.info("=== [E%d/%s] launching fresh subprocess ===", h, task)
            cmd = [
                sys.executable, str(EVAL_SCRIPT),
                "--execute-horizon", str(h),
                "--tasks", task,
                "--seeds", *[str(s) for s in args.seeds],
                "--output-dir", str(horizon_dir),
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error("=== [E%d/%s] subprocess FAILED (exit %d) ===", h, task, result.returncode)


if __name__ == "__main__":
    main()
