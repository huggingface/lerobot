#!/usr/bin/env python
"""Driver: fresh OS process per task x seed for `final_stageplan_v2_eval.py`. Skips finished cells."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TASKS = ["select_poker", "select_mahjong", "select_drink", "select_book", "select_fruit", "select_toy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3000, 3020)))
    ap.add_argument("--out", default="outputs/eval/safediff_stageplan_v2_final"); ap.add_argument("--head", default="outputs/train/safediff_stageplan_v2")
    a = ap.parse_args()
    for t in a.tasks:
        for s in a.seeds:
            d = Path(a.out) / t / f"seed{s}"
            if (d / "result.json").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "log.txt", "w") as f:
                rc = subprocess.run([sys.executable, str(Path(__file__).parent / "final_stageplan_v2_eval.py"), "--task", t, "--seed", str(s), "--out", a.out, "--head", a.head], stdout=f, stderr=subprocess.STDOUT).returncode
            print(t, s, "rc", rc, flush=True)


if __name__ == "__main__":
    main()
