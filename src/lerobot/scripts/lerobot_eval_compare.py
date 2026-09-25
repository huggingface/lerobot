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
"""Compare two `eval_info.json` files task by task and exit non-zero when a task regressed.

Usage:

```
lerobot-eval-compare outputs/eval/baseline/eval_info.json outputs/eval/candidate/eval_info.json
lerobot-eval-compare baseline.json candidate.json --min-drop 5 --json
```

Both files are what `lerobot-eval` writes. For every task present in both, the script reports the
baseline and candidate success rates, the difference with a 95% interval (Newcombe), a two-sided
Fisher exact p-value, and one of five calls:

- `REGRESSED`: the whole interval of the difference is at or below `-min-drop`.
- `SUSPECT`: the drop is at least `min-drop` but the interval reaches above it. At this episode count the
  drop is not separable from noise; run more episodes on this task.
- `HELD`: no drop beyond the floor. The resolution line says how large a drop could hide here.
- `IMPROVED`: the whole interval is above zero.
- `UNDERPOWERED`: fewer than `--min-episodes` episodes on either side; no call is made.

Exit codes: 0 no task regressed, 1 at least one task regressed, 2 the inputs could not be read.

The comparison treats the two evaluations as independent samples. Evaluating both policies with the same
`--seed` (same initial states) makes the true uncertainty smaller than reported here; a paired comparison
is a possible follow-up. The interval covers sampling over episodes only, not variation between training
runs of the same recipe.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from lerobot.utils.eval_stats import compare_success_counts


def load_task_successes(path: str | Path) -> dict[str, list[bool]]:
    """Read an `eval_info.json` and return `{task_label: [success, ...]}`.

    Accepts the `eval_policy_all` shape (`per_task` entries, labelled `task_group/task_id`) and the
    single-environment `eval_policy` shape (`per_episode`, labelled `all`).
    """
    with open(path) as f:
        info = json.load(f)
    if "per_task" in info:
        tasks: dict[str, list[bool]] = {}
        for entry in info["per_task"]:
            label = f"{entry.get('task_group')}/{entry.get('task_id')}"
            metrics = entry.get("metrics") or {}
            successes = metrics.get("successes")
            if successes is None:
                successes = entry.get("successes")
            if successes is None:
                raise ValueError(f"{path}: task {label} has no per-episode successes")
            tasks[label] = [bool(s) for s in successes]
        if not tasks:
            raise ValueError(f"{path}: 'per_task' is empty")
        return tasks
    if "per_episode" in info:
        return {"all": [bool(ep.get("success")) for ep in info["per_episode"]]}
    raise ValueError(f"{path}: expected an eval_info.json with 'per_task' or 'per_episode'")


def compare(
    baseline: dict[str, list[bool]],
    candidate: dict[str, list[bool]],
    min_drop_pp: float = 5.0,
    min_episodes: int = 10,
) -> dict:
    """Compare two `{task: successes}` maps and return the report as a dict."""
    shared = sorted(set(baseline) & set(candidate))
    rows = []
    for task in shared:
        b, c = baseline[task], candidate[task]
        row = compare_success_counts(
            sum(b), len(b), sum(c), len(c), min_drop_pp=min_drop_pp, min_episodes=min_episodes
        )
        row.update({"task": task, "n_baseline": len(b), "n_candidate": len(c)})
        rows.append(row)
    rows.sort(key=lambda r: r["delta_pp"] if r["delta_pp"] == r["delta_pp"] else 0.0)

    judged = [r for r in rows if r["verdict"] != "UNDERPOWERED"]
    counts = {
        v: sum(1 for r in rows if r["verdict"] == v)
        for v in ("REGRESSED", "SUSPECT", "IMPROVED", "HELD", "UNDERPOWERED")
    }
    n_b = sum(len(baseline[t]) for t in shared)
    n_c = sum(len(candidate[t]) for t in shared)
    k_b = sum(sum(baseline[t]) for t in shared)
    k_c = sum(sum(candidate[t]) for t in shared)
    summary = {
        "n_tasks": len(shared),
        "n_regressed": counts["REGRESSED"],
        "n_suspect": counts["SUSPECT"],
        "n_improved": counts["IMPROVED"],
        "n_held": counts["HELD"],
        "n_underpowered": counts["UNDERPOWERED"],
        "only_in_baseline": sorted(set(baseline) - set(candidate)),
        "only_in_candidate": sorted(set(candidate) - set(baseline)),
        "overall": {
            "baseline_pc_success": 100.0 * k_b / n_b if n_b else float("nan"),
            "candidate_pc_success": 100.0 * k_c / n_c if n_c else float("nan"),
            "n_baseline": n_b,
            "n_candidate": n_c,
        },
        "median_resolvable_drop_pp": statistics.median([r["resolvable_drop_pp"] for r in judged])
        if judged
        else float("nan"),
        "rule": {
            "min_drop_pp": min_drop_pp,
            "min_episodes": min_episodes,
            "regressed_iff": "the whole 95% interval of candidate - baseline is at or below -min_drop_pp",
        },
    }
    return {"tasks": rows, "summary": summary}


def _fmt(x: float, digits: int = 1) -> str:
    return "nan" if x != x else f"{x:.{digits}f}"


def format_report(report: dict, baseline_path: str, candidate_path: str) -> str:
    """Render the report as the text the CLI prints."""
    s = report["summary"]
    lines = [
        f"lerobot-eval-compare  baseline={baseline_path}  candidate={candidate_path}",
        "",
        f"{'task':<28} {'n':>9} {'baseline':>9} {'candidate':>9} {'change':>9} {'95% interval':>18} {'p':>8}  call",
    ]
    for r in report["tasks"]:
        n = (
            f"{r['n_baseline']}"
            if r["n_baseline"] == r["n_candidate"]
            else f"{r['n_baseline']}/{r['n_candidate']}"
        )
        ci = f"[{_fmt(r['ci95_pp'][0]):>6}, {_fmt(r['ci95_pp'][1]):>6}]"
        change = f"{r['delta_pp']:+.1f}" if r["delta_pp"] == r["delta_pp"] else "nan"
        p = f"{r['p_value']:.3g}" if r["p_value"] == r["p_value"] else "nan"
        lines.append(
            f"{r['task']:<28} {n:>9} {_fmt(r['baseline_pct']) + '%':>9} {_fmt(r['candidate_pct']) + '%':>9} "
            f"{change + ' pp':>9} {ci:>18} {p:>8}  {r['verdict']}"
        )
    o = s["overall"]
    lines += [
        "",
        f"overall (pooled over {s['n_tasks']} shared tasks): {_fmt(o['baseline_pc_success'])}% -> "
        f"{_fmt(o['candidate_pc_success'])}%",
        f"regressed {s['n_regressed']}  suspect {s['n_suspect']}  held {s['n_held']}  improved {s['n_improved']}  "
        f"underpowered {s['n_underpowered']}",
        f"smallest drop this comparison could call a regression: about {_fmt(s['median_resolvable_drop_pp'])} pp "
        "(median across tasks); smaller drops are invisible here even if real",
        f"rule: REGRESSED iff the whole 95% interval of candidate - baseline is at or below -{s['rule']['min_drop_pp']:g} pp; "
        f"tasks with fewer than {s['rule']['min_episodes']} episodes on a side are not judged",
    ]
    if s["only_in_baseline"]:
        lines.append(f"only in baseline (not compared): {', '.join(s['only_in_baseline'])}")
    if s["only_in_candidate"]:
        lines.append(f"only in candidate (not compared): {', '.join(s['only_in_candidate'])}")
    lines.append(
        "the interval covers sampling over episodes only; whether a drop comes from the model or from the "
        "training seed needs the same recipe trained more than once"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-eval-compare",
        description="Compare two eval_info.json files task by task; exit 1 if any task regressed.",
    )
    parser.add_argument("baseline", help="eval_info.json of the policy you run now")
    parser.add_argument("candidate", help="eval_info.json of the policy you are about to ship")
    parser.add_argument(
        "--min-drop",
        type=float,
        default=5.0,
        help="a task must fall at least this many percentage points, with its whole interval, to count as regressed (default 5)",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=10,
        help="below this many episodes on either side a task is reported as UNDERPOWERED and not judged (default 10)",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON instead of text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        baseline = load_task_successes(args.baseline)
        candidate = load_task_successes(args.candidate)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"lerobot-eval-compare: {e}", file=sys.stderr)
        return 2
    report = compare(baseline, candidate, min_drop_pp=args.min_drop, min_episodes=args.min_episodes)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report, args.baseline, args.candidate))
    return 1 if report["summary"]["n_regressed"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
