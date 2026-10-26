#!/usr/bin/env python
"""Driver for `final_eval_paper_multivariant.py`: one fresh OS subprocess per (model, task) pair
(30 total: 5 model variants x 6 tasks, `select_ingredient` excluded), matching the required
"process-per-task isolation" (same VLABench/MuJoCo state-leak workaround as every other multitask
eval on this branch). Skips any (model, task) pair whose `episode_results.jsonl` already has 5
records for it (from a smoke test or a prior partial run) -- never re-runs a pair that already has
real data, per the "no re-running to change results" principle for this final paper eval.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/run_final_eval_paper_isolated.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from final_eval_paper_multivariant import MODEL_VARIANTS, OUT_DIR, SEEDS, TASKS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EVAL_SCRIPT = Path(__file__).parent / "final_eval_paper_multivariant.py"


def already_done(jsonl_path: Path, model: str, task: str) -> bool:
    if not jsonl_path.exists():
        return False
    n = 0
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["model_variant"] == model and rec["task"] == task:
                n += 1
    return n >= len(SEEDS)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODEL_VARIANTS))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    jsonl_path = out_dir / "episode_results.jsonl"

    n_total = len(args.models) * len(args.tasks)
    n_run = 0
    for model in args.models:
        for task in args.tasks:
            if already_done(jsonl_path, model, task):
                logger.info("=== [%s/%s] already done (%d episodes on record), skipping ===", model, task, len(SEEDS))
                continue
            n_run += 1
            logger.info("=== [%s/%s] launching fresh subprocess (%d/%d) ===", model, task, n_run, n_total)
            cmd = [sys.executable, str(EVAL_SCRIPT), "--model", model, "--task", task, "--output-dir", str(out_dir)]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error("=== [%s/%s] subprocess FAILED (exit %d) -- leaving unavailable, continuing ===", model, task, result.returncode)

    logger.info("=== all (model, task) pairs processed ===")


if __name__ == "__main__":
    main()
