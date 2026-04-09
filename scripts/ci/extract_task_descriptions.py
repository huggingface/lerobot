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

"""Extract natural-language task descriptions for a benchmark suite.

Runs inside the benchmark Docker container (where the env library is installed)
immediately after lerobot-eval, writing a JSON file that parse_eval_metrics.py
picks up and embeds in metrics.json.

Output format: {"<suite>_<task_idx>": "<nl instruction>", ...}

Usage:
    python scripts/ci/extract_task_descriptions.py \\
        --env libero --task libero_spatial \\
        --output /tmp/eval-artifacts/task_descriptions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _libero_descriptions(task_suite: str) -> dict[str, str]:
    from libero.libero import benchmark  # type: ignore[import-untyped]

    suite_dict = benchmark.get_benchmark_dict()
    if task_suite not in suite_dict:
        print(
            f"[extract_task_descriptions] Unknown LIBERO suite '{task_suite}'. "
            f"Available: {list(suite_dict.keys())}",
            file=sys.stderr,
        )
        return {}
    suite = suite_dict[task_suite]()
    return {f"{task_suite}_{i}": suite.get_task(i).language for i in range(suite.n_tasks)}


def _metaworld_descriptions(task_name: str) -> dict[str, str]:
    # MetaWorld tasks don't expose a separate NL description attribute;
    # use a cleaned version of the task name as the description.
    label = task_name.removeprefix("metaworld-").replace("-", " ").strip()
    return {f"{task_name}_0": label}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="Environment family (libero, metaworld, ...)")
    parser.add_argument("--task", required=True, help="Task/suite name (e.g. libero_spatial)")
    parser.add_argument("--output", required=True, help="Path to write task_descriptions.json")
    args = parser.parse_args()

    descriptions: dict[str, str] = {}
    try:
        if args.env == "libero":
            descriptions = _libero_descriptions(args.task)
        elif args.env == "metaworld":
            descriptions = _metaworld_descriptions(args.task)
        else:
            print(
                f"[extract_task_descriptions] No description extractor for env '{args.env}'.",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"[extract_task_descriptions] Warning: {exc}", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(descriptions, indent=2))
    print(f"[extract_task_descriptions] {len(descriptions)} descriptions → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
