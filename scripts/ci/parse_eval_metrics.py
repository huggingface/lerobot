#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parse lerobot-eval output into a small metrics.json artifact.

Reads eval_info.json written by lerobot-eval --output_dir and extracts the
key metrics needed by the health dashboard. Handles both single-task and
multi-task eval output formats.

Usage:
    python scripts/ci/parse_eval_metrics.py \\
        --artifacts-dir /tmp/libero-artifacts \\
        --env libero \\
        --task libero_spatial \\
        --policy pepijn223/smolvla_libero

Writes <artifacts-dir>/metrics.json. The CI workflow then uploads this file
as a GitHub Actions artifact named "<env>-metrics".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _extract_pc_success(info: dict) -> tuple[float | None, int | None]:
    """Extract (pc_success, n_episodes) from eval_info.json.

    Handles two output shapes:
      - Single-task: {"aggregated": {"pc_success": 80.0, ...}}
      - Multi-task:  {"overall": {"pc_success": 80.0, "n_episodes": 5, ...}}
    """
    # Single-task path
    if "aggregated" in info:
        agg = info["aggregated"]
        pc = agg.get("pc_success")
        n = agg.get("n_episodes")  # may be absent in older format
        if pc is not None and not math.isnan(pc):
            return float(pc), int(n) if n is not None else None

    # Multi-task path
    if "overall" in info:
        overall = info["overall"]
        pc = overall.get("pc_success")
        n = overall.get("n_episodes")
        if pc is not None and not math.isnan(pc):
            return float(pc), int(n) if n is not None else None

    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts-dir", required=True, help="Path to the mounted artifacts volume")
    parser.add_argument("--env", required=True, help="Environment name (e.g. libero)")
    parser.add_argument("--task", required=True, help="Task name (e.g. libero_spatial)")
    parser.add_argument("--policy", required=True, help="Policy hub path (e.g. pepijn223/smolvla_libero)")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    eval_info_path = artifacts_dir / "eval_info.json"

    pc_success: float | None = None
    n_episodes: int | None = None

    if eval_info_path.exists():
        try:
            info = json.loads(eval_info_path.read_text())
            pc_success, n_episodes = _extract_pc_success(info)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[parse_eval_metrics] Warning: could not parse eval_info.json: {exc}", file=sys.stderr)
    else:
        print(
            f"[parse_eval_metrics] Warning: {eval_info_path} not found — eval may have failed.",
            file=sys.stderr,
        )

    metrics = {
        "env": args.env,
        "task": args.task,
        "policy": args.policy,
        "pc_success": pc_success,
        "n_episodes": n_episodes,
    }

    out_path = artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[parse_eval_metrics] Written: {out_path}")
    print(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
