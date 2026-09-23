# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Plan 50 matched attempts and journal physical evidence without controlling the robot.

Begin each attempt before /start. End it even after an error. Operator labels and
model assessments are separate append-only events; missing labels remain unknown.
"""

import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

CONDITIONS = ("task_only", "semantic", "full")
METRICS = ("command_compliance", "arm_accuracy", "direction_accuracy", "point_accuracy", "language_recovery")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plan_attempts(scenarios: list[dict], seed: int) -> dict:
    if len(scenarios) != 17 or len({s["id"] for s in scenarios}) != 17:
        raise ValueError("Provide 17 distinct reset scenarios for 16 triplets and one task/full pair")
    for scenario in scenarios:
        for key in ("id", "task", "reset_instructions", "success_rubric"):
            if not isinstance(scenario.get(key), str) or not scenario[key].strip():
                raise ValueError(f"Scenario requires {key}")
    tags = {tag for scenario in scenarios for tag in scenario.get("tags", [])}
    if not {"paraphrase", "distractor", "novel_object", "multi_step"} <= tags:
        raise ValueError(
            "Scenarios must include paraphrases, distractors, novel objects, and multi-step tasks"
        )
    rng = random.Random(seed)
    orders = list(itertools.permutations(CONDITIONS))
    rng.shuffle(orders)
    attempts = []
    for index, scenario in enumerate(scenarios):
        order = orders[index % len(orders)] if index < 16 else rng.sample(["task_only", "full"], 2)
        for condition in order:
            attempts.append(
                {
                    "id": f"attempt_{len(attempts) + 1:03d}",
                    "scenario_id": scenario["id"],
                    "condition": condition,
                }
            )
    return {
        "version": 1,
        "seed": seed,
        "scenarios": scenarios,
        "attempts": attempts,
        "design": "16 complete matched triplets plus one task/full pair; counts 17/16/17. Reset before every attempt.",
    }


def read_events(stream, plan_hash: str) -> list[dict]:
    stream.seek(0)
    events = [json.loads(line) for line in stream if line.strip()]
    if any(event["plan_sha256"] != plan_hash for event in events):
        raise ValueError("Plan changed after journaling began")
    return events


def append_event(plan_path: Path, journal: Path, attempt_id: str, kind: str, data: dict) -> None:
    plan = json.loads(plan_path.read_text())
    attempts = {a["id"]: a for a in plan["attempts"]}
    if attempt_id not in attempts:
        raise ValueError("Unknown planned attempt")
    if kind not in {"begin", "end", "operator", "model", "recording"}:
        raise ValueError("Unknown evaluation event")
    json.dumps(data, allow_nan=False)
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        events = read_events(stream, digest(plan_path))
        previous = [e for e in events if e["attempt_id"] == attempt_id]
        if kind == "begin":
            if previous:
                raise ValueError("Attempt already started; never replace a failed attempt")
            started = {e["attempt_id"] for e in events if e["kind"] == "begin"}
            expected = next(a["id"] for a in plan["attempts"] if a["id"] not in started)
            if attempt_id != expected:
                raise ValueError(f"Follow the planned attempt order; next is {expected}")
            if any(
                e["kind"] == "begin"
                and not any(x["kind"] == "end" and x["attempt_id"] == e["attempt_id"] for x in events)
                for e in events
            ):
                raise ValueError("End the outstanding attempt before starting another")
            for key in ("checkpoint_sha256", "rollout_config_sha256"):
                if len(data.get(key, "")) != 64 or any(c not in "0123456789abcdef" for c in data[key]):
                    raise ValueError(f"Run provenance requires {key}")
            for key in ("reset_operator", "initial_state_evidence", "dataset_root", "planner_log"):
                if not isinstance(data.get(key), str) or not data[key].strip():
                    raise ValueError(f"Run provenance requires {key}")
            for event in events:
                if event["kind"] == "begin" and any(
                    event["data"][key] == data[key] for key in ("dataset_root", "planner_log")
                ):
                    raise ValueError("Use separate recording roots and planner logs for each attempt")
                if (
                    event["kind"] == "begin"
                    and attempts[event["attempt_id"]]["condition"] == attempts[attempt_id]["condition"]
                    and event["data"]["checkpoint_sha256"] != data["checkpoint_sha256"]
                ):
                    raise ValueError("Checkpoint changed within a condition; use a new experiment")
        elif not previous:
            raise ValueError("Begin the attempt before adding evidence")
        elif kind == "end":
            if any(e["kind"] == "end" for e in previous):
                raise ValueError("Attempt already ended")
            if data.get("reason") not in {
                "completed",
                "timeout",
                "operator_stop",
                "runtime_error",
                "interrupted",
            }:
                raise ValueError("End reason is required; it is not an outcome label")
        elif kind in {"operator", "model"}:
            if data.get("outcome") not in {"success", "failure", "unknown"}:
                raise ValueError("Outcome must be success, failure, or unknown")
            if not data.get("reviewer") or not data.get("evidence"):
                raise ValueError("Assessment requires reviewer attribution and evidence")
            for name, counts in data.get("metrics", {}).items():
                if name not in METRICS or set(counts) != {"correct", "incorrect", "unknown"}:
                    raise ValueError("Metric requires correct/incorrect/unknown counts")
                if any(type(value) is not int or value < 0 for value in counts.values()):
                    raise ValueError("Metric counts must be nonnegative integers")
        elif kind == "recording":
            if (
                not data.get("dataset_root")
                or not data.get("episode_indices")
                or not data.get("evidence_sha256")
            ):
                raise ValueError("Recording requires dataset, episode identities, and evidence hash")
        event = {
            "plan_sha256": digest(plan_path),
            "attempt_id": attempt_id,
            "kind": kind,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": data,
        }
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def planner_summary(path: Path) -> dict:
    if not path.is_file():
        return {"available": False}
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    terminal = [e for e in events if e.get("event") in {"planner_returned", "planner_hold", "planner_error"}]
    seconds = [e["elapsed_s"] for e in terminal]
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in seconds):
        raise ValueError("Invalid planner timing evidence")
    requested = {e["request_id"] for e in events if e.get("event") == "planner_request"}
    finished = {e["request_id"] for e in terminal}
    if len(finished) != len(terminal) or not finished <= requested:
        raise ValueError("Duplicate or unpaired planner terminal events")
    styles = [e["style"] for e in events if e.get("event") == "planner_proposal"]
    return {
        "available": True,
        "sha256": digest(path),
        "request_count": len(requested),
        "unfinished_requests": len(requested - finished),
        "mean_call_seconds": sum(seconds) / len(seconds) if seconds else None,
        "terminal_events": dict(Counter(e["event"] for e in terminal)),
        "proposed_styles": dict(Counter(styles)),
        "proposed_style_switches": sum(a != b for a, b in zip(styles, styles[1:], strict=False)),
        "execution_verified": False,
    }


def summarize(plan_path: Path, journal: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    events = []
    if journal.exists():
        with journal.open() as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            events = read_events(stream, digest(plan_path))
    rows = []
    for attempt in plan["attempts"]:
        evidence = [e for e in events if e["attempt_id"] == attempt["id"]]
        if not evidence:
            continue
        latest = {e["kind"]: e["data"] for e in evidence}
        rows.append(
            {
                **attempt,
                "ended": "end" in latest,
                "operator_outcome": latest.get("operator", {}).get("outcome", "unknown"),
                "model_outcome": latest.get("model", {}).get("outcome", "unknown"),
                "operator_metrics": latest.get("operator", {}).get("metrics", {}),
                "recording_reference": latest.get("recording"),
                "planner": planner_summary(Path(latest["begin"]["planner_log"])),
            }
        )
    conditions = {}
    for condition in CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        counts = Counter(r["operator_outcome"] for r in selected)
        n = len(selected)
        conditions[condition] = {
            "started": n,
            "ended": sum(r["ended"] for r in selected),
            "operator_outcomes": {key: counts[key] for key in ("success", "failure", "unknown")},
            "success_rate_bounds": [counts["success"] / n, (counts["success"] + counts["unknown"]) / n]
            if n
            else None,
            "operator_metrics": {},
        }
        for metric in METRICS:
            measurements = [
                r["operator_metrics"][metric] for r in selected if metric in r["operator_metrics"]
            ]
            totals = {k: sum(m[k] for m in measurements) for k in ("correct", "incorrect", "unknown")}
            known = totals["correct"] + totals["incorrect"]
            conditions[condition]["operator_metrics"][metric] = {
                **totals,
                "attempts_with_measurement": len(measurements),
                "known_accuracy": totals["correct"] / known if known else None,
            }
    pairs = {}
    for a, b in itertools.combinations(CONDITIONS, 2):
        paired = []
        for scenario in plan["scenarios"]:
            available = {r["condition"]: r for r in rows if r["scenario_id"] == scenario["id"]}
            if a in available and b in available:
                paired.append((available[a]["operator_outcome"], available[b]["operator_outcome"]))
        known = [(x == "success", y == "success") for x, y in paired if "unknown" not in (x, y)]
        pairs[f"{b}_minus_{a}"] = {
            "matched_started_pairs": len(paired),
            "pairs_with_unknown": len(paired) - len(known),
            "success_difference_bounds": [
                sum(int(y == "success") - int(x != "failure") for x, y in paired) / len(paired),
                sum(int(y != "failure") - int(x == "success") for x, y in paired) / len(paired),
            ]
            if paired
            else None,
            "known_pair_success_difference": sum(int(y) - int(x) for x, y in known) / len(known)
            if known
            else None,
        }
    return {
        "planned_attempts": len(plan["attempts"]),
        "started_attempts": len(rows),
        "unstarted_attempts": len(plan["attempts"]) - len(rows),
        "conditions": conditions,
        "matched_comparisons": pairs,
        "attempts": rows,
        "note": "Operator outcomes only. Unknowns retained in bounds; known-pair differences explicitly exclude unknowns. Recording references need a separate video/data audit. A journal entry is not proof of a physical rollout.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=["plan", "begin", "end", "operator", "model", "recording", "report"]
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--attempt-id")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.operation == "plan":
        if args.input is None:
            parser.error("plan requires --input with 17 scenarios")
        plan = plan_attempts(json.loads(args.input.read_text()), args.seed)
        with args.plan.open("x") as stream:
            stream.write(json.dumps(plan, indent=2) + "\n")
    elif args.journal is None:
        parser.error("This operation requires --journal")
    elif args.operation == "report":
        print(json.dumps(summarize(args.plan, args.journal), indent=2))
    else:
        if args.input is None or args.attempt_id is None:
            parser.error("Events require --input and --attempt-id")
        append_event(
            args.plan, args.journal, args.attempt_id, args.operation, json.loads(args.input.read_text())
        )


if __name__ == "__main__":
    main()
