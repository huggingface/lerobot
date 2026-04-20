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


def _robotwin_descriptions(task_names: str) -> dict[str, str]:
    """Return descriptions for each requested RoboTwin task. Reads
    `description/task_instruction/<task>.json` from the RoboTwin clone
    (cwd is /opt/robotwin in CI). Falls back to the task name if missing."""
    out: dict[str, str] = {}
    root = Path("description/task_instruction")
    for name in (t.strip() for t in task_names.split(",") if t.strip()):
        desc_file = root / f"{name}.json"
        desc = name.replace("_", " ")
        if desc_file.is_file():
            data = json.loads(desc_file.read_text())
            full = data.get("full_description") or desc
            # Strip the schema placeholders ({A}, {a}) — keep the sentence readable.
            desc = full.replace("<", "").replace(">", "")
        out[f"{name}_0"] = desc
    return out


def _robocasa_descriptions(task_spec: str) -> dict[str, str]:
    """For each task in the comma-separated list, emit a cleaned-name label.

    RoboCasa episodes carry their language instruction in the env's
    `ep_meta['lang']`, populated per reset. Pulling it requires spinning
    up the full kitchen env per task (~seconds each); we use the task
    name as the key here and let the eval's episode info carry the
    actual instruction.
    """
    out: dict[str, str] = {}
    for task in (t.strip() for t in task_spec.split(",") if t.strip()):
        # Split CamelCase into words: "CloseFridge" → "close fridge".
        label = "".join(f" {c.lower()}" if c.isupper() else c for c in task).strip()
        out[f"{task}_0"] = label or task
    return out


_ROBOMME_DESCRIPTIONS = {
    "BinFill": "Fill the target bin with the correct number of cubes",
    "PickXtimes": "Pick the indicated cube the specified number of times",
    "SwingXtimes": "Swing the object the specified number of times",
    "StopCube": "Grasp and stop the moving cube",
    "VideoUnmask": "Pick the cube shown in the reference video",
    "VideoUnmaskSwap": "Pick the cube matching the reference video after a swap",
    "ButtonUnmask": "Press the button indicated by the reference",
    "ButtonUnmaskSwap": "Press the correct button after objects are swapped",
    "PickHighlight": "Pick the highlighted cube",
    "VideoRepick": "Repick the cube shown in the reference video",
    "VideoPlaceButton": "Place the cube on the button shown in the video",
    "VideoPlaceOrder": "Place cubes in the order shown in the video",
    "MoveCube": "Move the cube to the target location",
    "InsertPeg": "Insert the peg into the target hole",
    "PatternLock": "Unlock the pattern by pressing buttons in sequence",
    "RouteStick": "Route the stick through the required waypoints",
}


def _robomme_descriptions(task_names: str, task_ids: list[int] | None = None) -> dict[str, str]:
    """Return descriptions for each requested RoboMME task. Keys match the
    video filename pattern `<task>_<task_id>` used by the eval script."""
    if task_ids is None:
        task_ids = [0]
    out: dict[str, str] = {}
    for name in (t.strip() for t in task_names.split(",") if t.strip()):
        desc = _ROBOMME_DESCRIPTIONS.get(name, name)
        for tid in task_ids:
            out[f"{name}_{tid}"] = desc
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="Environment family (libero, metaworld, ...)")
    parser.add_argument("--task", required=True, help="Task/suite name (e.g. libero_spatial)")
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated task IDs (e.g. '0,1,2'). Default: [0]",
    )
    parser.add_argument("--output", required=True, help="Path to write task_descriptions.json")
    args = parser.parse_args()

    task_ids: list[int] | None = None
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]

    descriptions: dict[str, str] = {}
    try:
        if args.env == "libero":
            descriptions = _libero_descriptions(args.task)
        elif args.env == "metaworld":
            descriptions = _metaworld_descriptions(args.task)
        elif args.env == "robotwin":
            descriptions = _robotwin_descriptions(args.task)
        elif args.env == "robocasa":
            descriptions = _robocasa_descriptions(args.task)
        elif args.env == "robomme":
            descriptions = _robomme_descriptions(args.task, task_ids=task_ids)
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
