#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Report `lerobot-eval` progress for every run found under a path.

`lerobot-eval` appends one JSON line per finished task to `<output_dir>/results.jsonl` as the
evaluation proceeds (and writes `eval_manifest.json` listing every task up front). This tool searches
the given path recursively for every directory containing a `results.jsonl` and prints, for each run,
how far along it is plus the running success rate per suite and overall. It is strictly read-only and
never imports the simulator or any policy, so it is safe to run on a login node against jobs writing
to shared storage on compute nodes.

Examples:
    lerobot-watch-eval outputs/eval                 # all runs under outputs/eval
    lerobot-watch-eval outputs/eval/<single_run>    # just one run
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS_FILENAME = "results.jsonl"
MANIFEST_FILENAME = "eval_manifest.json"


def _load_manifest(output_dir: Path) -> dict:
    path = output_dir / MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _load_results(output_dir: Path) -> dict[tuple[str, int], dict]:
    """Latest record per (task_group, task_id); malformed/torn lines are skipped."""
    path = output_dir / RESULTS_FILENAME
    latest: dict[tuple[str, int], dict] = {}
    if not path.is_file():
        return latest
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[(r.get("task_group"), r.get("task_id"))] = r
    return latest


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def summarize(output_dir: Path) -> str:
    manifest = _load_manifest(output_dir)
    latest = _load_results(output_dir)

    total = manifest.get("n_tasks")
    done = len(latest)
    n_ok = sum(1 for r in latest.values() if r.get("status", "ok") == "ok")
    n_err = done - n_ok

    group_successes: dict[str, list[bool]] = defaultdict(list)
    group_task_count: dict[str, int] = defaultdict(int)
    overall_successes: list[bool] = []
    walls: list[float] = []
    for (group, _tid), r in latest.items():
        group_task_count[group] += 1
        if r.get("status", "ok") != "ok":
            continue
        successes = (r.get("metrics") or {}).get("successes") or []
        group_successes[group].extend(successes)
        overall_successes.extend(successes)
        if r.get("wall_s"):
            walls.append(r["wall_s"])

    lines = [f"eval run: {output_dir}"]
    if manifest.get("policy_path"):
        lines.append(f"policy:   {manifest['policy_path']}")

    if total:
        lines.append(f"progress: {done}/{total} tasks ({100 * done / total:.1f}%)   ok={n_ok}  err={n_err}")
    else:
        lines.append(f"progress: {done} tasks done   ok={n_ok}  err={n_err}   (no manifest; total unknown)")

    # Rough ETA from mean per-task wall time. `wall_s` measures one task's own runtime, so with
    # --env.max_parallel_tasks>1 the real ETA is shorter than this sequential-throughput estimate.
    if total and walls and done < total:
        avg = sum(walls) / len(walls)
        remaining = total - done
        lines.append(
            f"pace:     ~{avg:.1f}s/task (sequential est.); ~{_fmt_dur(avg * remaining)} left for {remaining} tasks"
        )

    if overall_successes:
        pc = 100 * sum(overall_successes) / len(overall_successes)
        lines.append(f"overall:  pc_success {pc:.1f}%  ({len(overall_successes)} episodes)")

    for group in sorted(group_successes):
        successes = group_successes[group]
        pc = 100 * sum(successes) / len(successes) if successes else float("nan")
        lines.append(f"  {group:<16} {pc:5.1f}%   ({len(successes)} ep over {group_task_count[group]} tasks)")

    errors = [(k, r) for k, r in latest.items() if r.get("status", "ok") != "ok"]
    if errors:
        lines.append(f"errors ({len(errors)}):")
        for (group, tid), r in errors[:10]:
            lines.append(f"  {group}/{tid}: {r.get('error')}")
        if len(errors) > 10:
            lines.append(f"  ... and {len(errors) - 10} more")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Directory searched recursively for eval runs (any directory containing results.jsonl).",
    )
    args = parser.parse_args()

    if not args.path.is_dir():
        raise SystemExit(f"Not a directory: {args.path}")

    run_dirs = sorted({p.parent for p in args.path.rglob(RESULTS_FILENAME)})
    if not run_dirs:
        raise SystemExit(f"No runs (directories containing {RESULTS_FILENAME}) found under {args.path}")

    print(f"Found {len(run_dirs)} run(s) under {args.path}\n")
    for i, run_dir in enumerate(run_dirs):
        if i:
            print("\n" + "-" * 72 + "\n")
        print(summarize(run_dir))


if __name__ == "__main__":
    main()
