from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lines(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    plan = json.loads(args.plan.read_text())
    root = Path(plan["evidence_root"])
    trials_root = root / "trials"
    out = root / "operator_result_v1"
    if out.exists():
        raise FileExistsError(out)
    rows = []
    terminations = Counter()
    for trial in plan["trials"]:
        stem = trial["artifact_stem"]
        evidence_path = trials_root / f"{stem}.json"
        label_path = trials_root / f"{stem}.operator_label.json"
        steps_path = trials_root / f"{stem}.steps.jsonl"
        video_path = trials_root / f"{stem}.mp4"
        ready_path = trials_root / f"{stem}.ready.jsonl"
        return_path = trials_root / f"{stem}.return.jsonl"
        for path in (evidence_path, label_path, steps_path, video_path, ready_path, return_path):
            if not path.exists():
                raise RuntimeError(f"missing {path}")
        evidence = json.loads(evidence_path.read_text())
        label = json.loads(label_path.read_text())
        if evidence.get("run_error") is not None or evidence.get("torque_disable_verified") is not True:
            raise RuntimeError(f"invalid completion: {trial['trial_id']}")
        if label["trial_id"] != trial["trial_id"] or label["model_key"] != trial["model_key"]:
            raise RuntimeError(f"label identity mismatch: {trial['trial_id']}")
        step_lines = lines(steps_path)
        if step_lines != evidence["steps_jsonl"]["lines"]:
            raise RuntimeError(f"steps mismatch: {trial['trial_id']}")
        terminations[evidence["termination"]] += 1
        rows.append({
            "trial_id": trial["trial_id"],
            "pose_order": trial["pose_order"],
            "model_key": trial["model_key"],
            "model_id": trial["model_id"],
            "cell": trial["cell"],
            "yaw": trial["nominal_yaw_degrees_modulo_90"],
            "operator_success": bool(label["success"]),
            "failure_category": label.get("failure_category"),
            "termination": evidence["termination"],
            "actual_policy_ticks": step_lines,
            "torque_disable_verified": True,
            "run_error": None,
            "evidence_sha256": sha(evidence_path),
            "video_sha256": sha(video_path),
            "steps_sha256": sha(steps_path),
            "operator_label_sha256": sha(label_path),
        })
    by_model = {}
    for key in ("S", "A"):
        group = [row for row in rows if row["model_key"] == key]
        successes = sum(row["operator_success"] for row in group)
        by_model[key] = {"success": successes, "total": len(group), "rate": successes / len(group)}
    pair_outcomes = Counter()
    for index in range(0, len(rows), 2):
        pair = {row["model_key"]: row["operator_success"] for row in rows[index:index + 2]}
        if pair == {"S": True, "A": True}:
            pair_outcomes["both_success"] += 1
        elif pair == {"S": True, "A": False}:
            pair_outcomes["source_only"] += 1
        elif pair == {"S": False, "A": True}:
            pair_outcomes["aligned_only"] += 1
        elif pair == {"S": False, "A": False}:
            pair_outcomes["both_failure"] += 1
        else:
            raise RuntimeError(f"invalid pair at rows {index + 1}-{index + 2}")
    summary = {
        "schema": "task1_csource_response_v3_simseen6_operator_result_v1",
        "status": "operator_labels_frozen_pending_canonical_video_review",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": sha(args.plan),
        "trials": len(rows),
        "by_model": by_model,
        "paired_pose_outcomes": {key: pair_outcomes[key] for key in
                                 ("source_only", "aligned_only", "both_success", "both_failure")},
        "integrity": {
            "run_error_null": sum(row["run_error"] is None for row in rows),
            "torque_disable_verified": sum(row["torque_disable_verified"] for row in rows),
            "terminations": dict(terminations),
        },
        "claim_boundary": "Operator-stage post-hoc engineering development result; canonical-video review pending; not an independent paper conclusion.",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    out.mkdir(parents=True)
    trials_path = out / "trials.jsonl"
    with trials_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    hashes_path = out / "hashes.sha256"
    hash_inputs = [trials_path, summary_path, args.plan] + sorted(trials_root.glob("*.operator_label.json"))
    hashes_path.write_text("".join(f"{sha(path)}  {path}\n" for path in hash_inputs))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
