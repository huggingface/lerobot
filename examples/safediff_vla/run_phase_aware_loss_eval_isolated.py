#!/usr/bin/env python
"""Driver for `eval_grounded_grasp_v2_5_phase_aware_loss_5k.py`: gives the NEW checkpoint
(`phase_aware_loss_5k`) its own fresh subprocess per task (same VLABench/MuJoCo state-leak
workaround as every other multitask eval on this branch). `stage2_baseline` needs no subprocess:
it's read back from the existing Stage 2 eval's own results (`REUSE_FROM`), not re-simulated.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/run_phase_aware_loss_eval_isolated.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402
from eval_grounded_grasp_v2_5_phase_aware_loss_5k import CHECKPOINTS, REUSE_FROM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EVAL_SCRIPT = Path(__file__).parent / "eval_grounded_grasp_v2_5_phase_aware_loss_5k.py"
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_5k")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    for label in CHECKPOINTS:
        if label in REUSE_FROM:
            summary_paths = [out_dir / label / task / "summary.json" for task in args.tasks]
            if all(p.exists() for p in summary_paths):
                logger.info("=== [%s] already reused, skipping ===", label)
                continue
            logger.info("=== [%s] re-summarizing from Stage 2's already-isolated results ===", label)
            cmd = [
                sys.executable, str(EVAL_SCRIPT),
                "--only-label", label,
                "--tasks", *args.tasks,
                "--output-dir", str(out_dir),
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error("=== [%s] reuse subprocess FAILED (exit %d) ===", label, result.returncode)
            continue
        for task in args.tasks:
            summary_path = out_dir / label / task / "summary.json"
            if summary_path.exists():
                logger.info("=== [%s/%s] already done, skipping ===", label, task)
                continue
            logger.info("=== [%s/%s] launching fresh subprocess ===", label, task)
            cmd = [
                sys.executable, str(EVAL_SCRIPT),
                "--only-label", label,
                "--tasks", task,
                "--output-dir", str(out_dir),
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error("=== [%s/%s] subprocess FAILED (exit %d) -- leaving unavailable ===", label, task, result.returncode)

    logger.info("=== all subprocesses done, merging (including reused labels) ===")
    subprocess.run([sys.executable, str(EVAL_SCRIPT), "--merge-only", "--output-dir", str(out_dir)], check=True)


if __name__ == "__main__":
    main()
