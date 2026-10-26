#!/usr/bin/env python
"""Driver for `phase_r0_reproducibility_audit.py`: one fresh subprocess per (episode, mode,
monitor, repeat) cell. For each of the 4 representative episodes:
    3x "current"       mode=actual-codebase-defaults, monitor=on
    3x "deterministic" mode=all determinism knobs on,  monitor=on
    1x "deterministic" mode=all determinism knobs on,  monitor=off  (side-effect check)
`CUBLAS_WORKSPACE_CONFIG=:4096:8` is set in the subprocess env for every "deterministic" run (must
be set before CUDA context init).

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/run_phase_r0_reproducibility_isolated.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EVAL_SCRIPT = Path(__file__).parent / "phase_r0_reproducibility_audit.py"
OUT_DIR_ROOT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_r0_reproducibility")

EPISODES = [("select_poker", 1001), ("select_fruit", 1001), ("select_book", 1002), ("select_mahjong", 1000)]


def main() -> None:
    cells = []
    for task, seed in EPISODES:
        for r in range(3):
            cells.append((task, seed, False, "on", r))
        for r in range(3):
            cells.append((task, seed, True, "on", r))
        cells.append((task, seed, True, "off", 0))

    for task, seed, deterministic, monitor, repeat in cells:
        mode = "deterministic" if deterministic else "current"
        out_dir = OUT_DIR_ROOT / mode / f"monitor_{monitor}" / task
        out_path = out_dir / f"repeat_{repeat}.json"
        if out_path.exists():
            logger.info("=== [%s/%s/monitor_%s/repeat%d] already done, skipping ===", task, mode, monitor, repeat)
            continue
        logger.info("=== [%s/%s/monitor_%s/repeat%d] launching ===", task, mode, monitor, repeat)
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        if deterministic:
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--task", task, "--seed", str(seed), "--repeat-id", str(repeat), "--monitor", monitor,
            "--output-dir", str(out_dir),
        ]
        if deterministic:
            cmd.append("--deterministic")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            logger.error("=== [%s/%s/monitor_%s/repeat%d] FAILED (exit %d) ===", task, mode, monitor, repeat, result.returncode)


if __name__ == "__main__":
    main()
