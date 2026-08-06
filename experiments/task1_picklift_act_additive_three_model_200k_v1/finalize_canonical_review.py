from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/home/ubuntu24/Teleop/lerobot")
EXP = REPO / "experiments/task1_picklift_act_additive_three_model_200k_v1"
PLAN = EXP / "evaluation_plan.json"
ROOT = Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1")
REVIEW_ROOT = ROOT / "canonical_video_review_v1"
BLIND = REVIEW_ROOT / "blind_review_frozen_v1"
INTEGRITY = REVIEW_ROOT / "integrity_prejoin_v1.json"
ADJ = EXP / "adjudication_v1.json"
OUT = REVIEW_ROOT / "final_review_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group(rows: list[dict], field: str, value) -> dict:
    selected = [row for row in rows if row[field] == value]
    count = sum(row["final_success"] for row in selected)
    return {"success": count, "total": len(selected), "rate": count / len(selected)}


def paired(rows: list[dict], left: str, right: str) -> dict:
    by_pose: dict[int, dict[str, bool]] = {}
    for row in rows:
        by_pose.setdefault(row["pose_order"], {})[row["model_key"]] = row["final_success"]
    outcomes = Counter()
    pairs = []
    for pose_order in sorted(by_pose):
        a, b = by_pose[pose_order][left], by_pose[pose_order][right]
        category = "both_success" if a and b else "left_only" if a else "right_only" if b else "both_failure"
        outcomes[category] += 1
        pairs.append({"pose_order": pose_order, "left_success": a, "right_success": b, "outcome": category})
    left_success = sum(pair["left_success"] for pair in pairs)
    right_success = sum(pair["right_success"] for pair in pairs)
    return {
        "left_model": left,
        "right_model": right,
        "pose_count": len(pairs),
        "left_success": left_success,
        "right_success": right_success,
        "right_minus_left_success_count": right_success - left_success,
        "right_minus_left_rate": (right_success - left_success) / len(pairs),
        "outcomes": dict(outcomes),
        "pairs": pairs,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(PLAN.read_text())
    integrity = json.loads(INTEGRITY.read_text())
    if integrity["status"] != "pass" or integrity["trial_count"] != 72:
        raise RuntimeError("integrity prejoin did not pass")
    blind = [json.loads(line) for line in (BLIND / "trials.jsonl").read_text().splitlines()]
    if len(blind) != 72 or len({row["trial_id"] for row in blind}) != 72:
        raise RuntimeError("blind identity mismatch")
    adjudications = {row["trial_id"]: row for row in json.loads(ADJ.read_text())["trials"]}
    rows = []
    disagreements = []
    for blind_row in blind:
        trial_id = blind_row["trial_id"]
        operator_path = ROOT / "trials" / f"{trial_id}.operator_label.json"
        operator = json.loads(operator_path.read_text())
        raw_disagreement = bool(operator["success"]) != blind_row["success"]
        if raw_disagreement:
            disagreements.append(trial_id)
        adjudication = adjudications.get(trial_id)
        final_success = bool(adjudication["success"]) if adjudication else blind_row["success"]
        final_failure = adjudication["failure_category"] if adjudication else blind_row["failure_category"]
        rows.append(
            {
                **{key: blind_row[key] for key in ("trial_id", "model_key", "model_id", "pose_order", "within_pose_order", "cell", "coverage_tier", "yaw_degrees", "video_sha256")},
                "blind_review_success": blind_row["success"],
                "blind_review_failure_category": blind_row["failure_category"],
                "blind_review_confidence": blind_row["confidence"],
                "blind_review_evidence": blind_row["visible_evidence"],
                "operator_success": bool(operator["success"]),
                "operator_failure_category": operator.get("failure_category"),
                "operator_label_sha256": sha256(operator_path),
                "raw_operator_blind_agreement": not raw_disagreement,
                "adjudication": adjudication,
                "final_success": final_success,
                "final_failure_category": final_failure,
                "final_operator_agreement": bool(operator["success"]) == final_success,
            }
        )
    if sorted(disagreements) != sorted(adjudications):
        raise RuntimeError(f"adjudication coverage mismatch {disagreements} vs {list(adjudications)}")
    rows.sort(key=lambda row: int(row["trial_id"][1:4]))
    failures = Counter(row["final_failure_category"] for row in rows if not row["final_success"])
    operator_failures = Counter(row["operator_failure_category"] for row in rows if not row["operator_success"])
    summary = {
        "schema": "task1_additive_eval24_canonical_video_review_summary_v1",
        "status": "canonical_video_review_complete",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": sha256(PLAN),
        "overall": {"success": sum(row["final_success"] for row in rows), "total": 72, "rate": sum(row["final_success"] for row in rows) / 72},
        "by_model": {key: group(rows, "model_key", key) for key in ("A", "B", "C")},
        "by_coverage_tier": {key: group(rows, "coverage_tier", key) for key in ("seen_by_real24", "added_by_real48", "added_by_real96", "unseen_by_both")},
        "by_yaw": {str(key): group(rows, "yaw_degrees", key) for key in (0, 45)},
        "by_cell": {key: group(rows, "cell", key) for key in sorted({row["cell"] for row in rows})},
        "final_failure_categories": dict(sorted(failures.items())),
        "operator_failure_categories": dict(sorted(operator_failures.items(), key=lambda item: str(item[0]))),
        "agreement": {
            "raw_blind_operator_agree": sum(row["raw_operator_blind_agreement"] for row in rows),
            "raw_blind_operator_disagree": len(disagreements),
            "disagreement_trial_ids": disagreements,
            "adjudication_count": len(adjudications),
            "final_operator_agree": sum(row["final_operator_agreement"] for row in rows),
            "final_operator_disagree": sum(not row["final_operator_agreement"] for row in rows),
        },
        "paired_comparisons": {
            "primary_c_vs_a": paired(rows, "A", "C"),
            "key_substitution_c_vs_b": paired(rows, "B", "C"),
            "sanity_b_vs_a": paired(rows, "A", "B"),
        },
        "claim_boundary": "Single training seed and 24-pose real-robot engineering gate; descriptive evidence, not an automatic causal paper conclusion.",
    }
    OUT.mkdir(parents=True)
    trials_path = OUT / "trials.jsonl"
    with trials_path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "task1_additive_eval24_canonical_video_review_manifest_v1",
        "evaluation_id": plan["evaluation_id"],
        "status": "complete",
        "plan_sha256": sha256(PLAN),
        "blind_queue_sha256": sha256(REVIEW_ROOT / "blind_queue" / "queue.jsonl"),
        "blind_trials_sha256": sha256(BLIND / "trials.jsonl"),
        "blind_manifest_sha256": sha256(BLIND / "manifest.json"),
        "blind_frozen_before_operator_join": True,
        "operator_join_after_blind_freeze": True,
        "adjudication_sha256": sha256(ADJ),
        "integrity_prejoin_sha256": sha256(INTEGRITY),
        "trial_count": 72,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": summary["claim_boundary"],
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hashes_path = OUT / "hashes.sha256"
    files = [trials_path, summary_path, manifest_path, BLIND / "trials.jsonl", BLIND / "manifest.json", INTEGRITY, ADJ, PLAN]
    hashes_path.write_text("".join(f"{sha256(path)}  {path}\n" for path in files))
    (EXP / "canonical_review_result_index.json").write_text(json.dumps({**manifest, "summary": summary, "evidence_root": str(OUT)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
