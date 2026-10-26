#!/usr/bin/env python
"""Driver for `eval_grounded_grasp_v2_5_multitask_stage1_5k.py` that gives every
(checkpoint, task) pair its OWN fresh subprocess, instead of looping over all 14 combinations
inside one long-lived process.

Why: an initial combined run showed a real bug -- after ~12-13 sequential `VLABenchEnv`
constructions (different `task=` values, same process), `select_poker` under the multitask
checkpoint reported `ee_to_gt_target_dist` of 1.4-3.7m for the ENTIRE episode (the arm never
approaching the target at all), even though a byte-for-byte weight diff confirmed every module
that drives the PRE_GRASP approach (backbone, `TargetQueryExtractor`, `VisualGroundingCrossAttention`,
`TargetPointHead`, `TemporalActionDecoder`) is UNCHANGED between the single-task and multitask
checkpoints (only the 6 `local_grasp_refiner.*` tensors differ) -- so that phase's behavior cannot
legitimately differ between the two checkpoints on the same task/seed. Re-running that exact
(checkpoint, task) pair in an otherwise-empty process reproduced the SANE 0.03-0.51m distances seen
everywhere else, confirming this is VLABench/MuJoCo state leaking across many sequential env
constructions of DIFFERENT task types in one process (not a real model regression, and not
something to fix inside this branch's own scope) -- process isolation (call `_ensure_env()` fresh
per task, i.e. fresh process here) is the reliable workaround, same rationale as
`envs/vlabench.py`'s own "each worker gets its own clean rendering context rather than inheriting a
stale one" comment, just applied at the whole-eval-script level instead of `AsyncVectorEnv`'s
per-episode-batch worker level.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/run_multitask_eval_isolated.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import TASK_NAMES  # noqa: E402
from eval_grounded_grasp_v2_5_multitask_stage1_5k import (  # noqa: E402
    CHECKPOINT_MULTITASK_STAGE1,
    CHECKPOINT_SINGLE_TASK,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EVAL_SCRIPT = Path(__file__).parent / "eval_grounded_grasp_v2_5_multitask_stage1_5k.py"
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_multitask_stage1_5k_isolated")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--single-task-checkpoint", default=CHECKPOINT_SINGLE_TASK)
    parser.add_argument("--multitask-checkpoint", default=CHECKPOINT_MULTITASK_STAGE1)
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    combos = [
        ("single_task", "single_task_v2_5_10k", args.single_task_checkpoint),
        ("multitask", "multitask_stage1_v2_5_5k", args.multitask_checkpoint),
    ]
    for only_flag, label, ckpt in combos:
        for task in args.tasks:
            summary_path = out_dir / label / task / "summary.json"
            if summary_path.exists():
                logger.info("=== [%s/%s] already done, skipping ===", label, task)
                continue
            logger.info("=== [%s/%s] launching fresh subprocess ===", label, task)
            cmd = [
                sys.executable, str(EVAL_SCRIPT),
                "--only-checkpoint", only_flag,
                "--tasks", task,
                "--output-dir", str(out_dir),
                "--single-task-checkpoint", args.single_task_checkpoint,
                "--multitask-checkpoint", args.multitask_checkpoint,
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error("=== [%s/%s] subprocess FAILED (exit %d) -- leaving unavailable ===", label, task, result.returncode)

    logger.info("=== all subprocesses done, merging results ===")
    merge_cmd = [
        sys.executable, str(EVAL_SCRIPT), "--merge-only",
        "--output-dir", str(out_dir),
        "--tasks", *args.tasks,
        "--single-task-checkpoint", args.single_task_checkpoint,
        "--multitask-checkpoint", args.multitask_checkpoint,
    ]
    subprocess.run(merge_cmd, check=True)


if __name__ == "__main__":
    main()
