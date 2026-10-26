#!/usr/bin/env python
"""Driver for `eval_grounded_grasp_v2_monitor_baseline.py`: one fresh subprocess per task (same
VLABench/MuJoCo state-leak workaround as every other multitask eval on this branch).

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/run_grounded_grasp_v2_monitor_baseline_isolated.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import OUT_DIR, SEEDS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EVAL_SCRIPT = Path(__file__).parent / "eval_grounded_grasp_v2_monitor_baseline.py"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    for task in args.tasks:
        raw_path = out_dir / task / "episodes_raw.json"
        if raw_path.exists():
            logger.info("=== [%s] already done, skipping ===", task)
            continue
        logger.info("=== [%s] launching fresh subprocess ===", task)
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--tasks", task,
            "--seeds", *[str(s) for s in args.seeds],
            "--output-dir", str(out_dir),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error("=== [%s] subprocess FAILED (exit %d) ===", task, result.returncode)


if __name__ == "__main__":
    main()
