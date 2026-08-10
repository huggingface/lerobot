from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

REPO = Path("/home/ubuntu24/Teleop/lerobot")
EXP = REPO / "experiments/task1_picklift_act_caligned_v2_response_v3_200k_v1"
PLAN = EXP / "bound_simseen6_evaluation_plan.json"
ROOT = Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_csource_vs_response_v3_simseen6_paired_eval_v1")
REVIEW = ROOT / "canonical_video_review_v1"
QUEUE = REVIEW / "blind_queue"
MAPPING = REVIEW / "private_mapping_prejoin/mapping.jsonl"
BLIND = REVIEW / "blind_review_frozen_v1"
INTEGRITY = REVIEW / "integrity_prejoin_v1.json"
OPERATOR = ROOT / "operator_result_v1"
RESEARCH = REVIEW / "research_control_operator_result_manifest.json"
OUT = REVIEW / "final_review_v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def grouped(rows: list[dict], field: str) -> dict:
    result = {}
    for value in sorted({row[field] for row in rows}, key=str):
        subset = [row for row in rows if row[field] == value]
        success = sum(row["final_success"] for row in subset)
        result[str(value)] = {"success": success, "total": len(subset), "rate": success / len(subset)}
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(RESEARCH) != "1cfb77e3ddbc7c7e36ce29d6c84de94362dfbd5f2c575220778aa3069694bc29":
        raise RuntimeError("research operator result manifest mismatch")
    plan = json.loads(PLAN.read_text())
    integrity = json.loads(INTEGRITY.read_text())
    if integrity["status"] != "pass" or integrity["trial_count"] != 12:
        raise RuntimeError("prejoin integrity did not pass")
    blind_rows = json_lines(BLIND / "trials.jsonl")
    if len(blind_rows) != 12 or len({row["neutral_id"] for row in blind_rows}) != 12:
        raise RuntimeError("blind row identity mismatch")
    mapping = {row["neutral_id"]: row for row in json_lines(MAPPING)}
    queue = {row["neutral_id"]: row for row in json_lines(QUEUE / "queue.jsonl")}
    trial_by_id = {row["trial_id"]: row for row in plan["trials"]}
    rows = []
    disagreements = []
    for blind in blind_rows:
        mapped = mapping[blind["neutral_id"]]
        trial = trial_by_id[mapped["trial_id"]]
        video_sha256 = queue[blind["neutral_id"]]["video_sha256"]
        if video_sha256 != mapped["video_sha256"]:
            raise RuntimeError(f"blind mapping video mismatch: {blind['neutral_id']}")
        operator_path = ROOT / "trials" / f"{trial['artifact_stem']}.operator_label.json"
        operator = json.loads(operator_path.read_text())
        disagreement = bool(operator["success"]) != bool(blind["success"])
        if disagreement:
            disagreements.append(trial["trial_id"])
        rows.append({
            "trial_id": trial["trial_id"],
            "pose_order": trial["pose_order"],
            "model_key": trial["model_key"],
            "model_id": trial["model_id"],
            "cell": trial["cell"],
            "yaw_degrees": trial["nominal_yaw_degrees_modulo_90"],
            "neutral_id": blind["neutral_id"],
            "video_sha256": video_sha256,
            "blind_review_success": bool(blind["success"]),
            "blind_review_failure_category": blind["failure_category"],
            "blind_review_confidence": blind["confidence"],
            "blind_reviewed_time_range_s": blind["reviewed_time_range_s"],
            "blind_candidate_intervals_s": blind["candidate_intervals_s"],
            "blind_evidence_intervals_s": blind["evidence_intervals_s"],
            "blind_visible_evidence": blind["visible_evidence"],
            "operator_success": bool(operator["success"]),
            "operator_failure_category": operator.get("failure_category"),
            "operator_label_sha256": sha(operator_path),
            "raw_operator_blind_agreement": not disagreement,
            "adjudication": None,
            "final_success": bool(blind["success"]),
            "final_failure_category": blind["failure_category"],
        })
    rows.sort(key=lambda row: int(row["trial_id"][1:4]))
    if disagreements:
        raise RuntimeError(f"disagreements require separate blind adjudication: {disagreements}")
    pairs = []
    outcomes = Counter()
    for pose in range(1, 7):
        pair_rows = [row for row in rows if row["pose_order"] == pose]
        result = {row["model_key"]: row["final_success"] for row in pair_rows}
        category = "both_success" if result == {"S": True, "A": True} else "source_only" if result == {"S": True, "A": False} else "aligned_only" if result == {"S": False, "A": True} else "both_failure"
        outcomes[category] += 1
        pairs.append({"pose_order": pose, "source_success": result["S"], "aligned_success": result["A"], "outcome": category})
    by_model = grouped(rows, "model_key")
    discordant = outcomes["source_only"] + outcomes["aligned_only"]
    mcnemar_p = min(1.0, 2.0 * (0.5 ** discordant)) if discordant else 1.0
    summary = {
        "schema": "task1_response_v3_simseen6_canonical_video_review_summary_v1",
        "status": "canonical_video_review_complete",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": sha(PLAN),
        "statistical_unit": "paired_pose",
        "paired_pose_count": 6,
        "by_model": by_model,
        "aligned_minus_source_success_count": by_model["A"]["success"] - by_model["S"]["success"],
        "aligned_minus_source_rate": by_model["A"]["rate"] - by_model["S"]["rate"],
        "paired_outcomes": dict(outcomes),
        "paired_rows": pairs,
        "exact_mcnemar_two_sided_p": mcnemar_p,
        "by_cell": grouped(rows, "cell"),
        "by_yaw": grouped(rows, "yaw_degrees"),
        "final_failure_categories": dict(sorted(Counter(row["final_failure_category"] for row in rows if not row["final_success"]).items())),
        "agreement": {"agree": 12, "disagree": 0, "disagreement_trial_ids": [], "adjudication_count": 0},
        "execution_observation": "All 12 trials ran maximum duration; success early-stop was not triggered symmetrically. No rerun is warranted.",
        "provenance_caveat": "No independent research-control hardware-GO manifest exists for this execution; the owning conversation contains explicit user hardware authorization.",
        "claim_boundary": "Six paired poses are a post-hoc engineering development gate. p=0.125 is not significant; do not claim response alignment is significantly worse, a response-only causal effect, Sim net benefit, or a paper conclusion.",
    }
    OUT.mkdir(parents=True)
    trials_path = OUT / "trials.jsonl"
    trials_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "task1_response_v3_simseen6_canonical_video_review_manifest_v1",
        "status": "complete_pending_independent_validation",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": sha(PLAN),
        "research_commit": "e2677f9d6da818c631ed8f7a4924bd9982d806c3",
        "research_operator_result_manifest_sha256": sha(RESEARCH),
        "operator_result_v1": {name: sha(OPERATOR / name) for name in ("summary.json", "trials.jsonl", "hashes.sha256")},
        "operator_result_failed_attempts_excluded": ["operator_result_failed_attempt1", "operator_result_failed_attempt2"],
        "neutral_queue_sha256": sha(QUEUE / "queue.jsonl"),
        "neutral_queue_manifest_sha256": sha(QUEUE / "manifest.json"),
        "private_mapping_sha256": sha(MAPPING),
        "blind_trials_sha256": sha(BLIND / "trials.jsonl"),
        "blind_manifest_sha256": sha(BLIND / "manifest.json"),
        "blind_labels_frozen_before_operator_join": True,
        "operator_join_after_blind_freeze": True,
        "disagreement_count": 0,
        "adjudication_required": False,
        "integrity_prejoin_sha256": sha(INTEGRITY),
        "trial_count": 12,
        "provenance_caveat": summary["provenance_caveat"],
        "claim_boundary": summary["claim_boundary"],
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hashes_path = OUT / "hashes.sha256"
    files = [trials_path, summary_path, manifest_path, PLAN, RESEARCH, OPERATOR / "summary.json", OPERATOR / "trials.jsonl", OPERATOR / "hashes.sha256", QUEUE / "queue.jsonl", QUEUE / "manifest.json", MAPPING, BLIND / "trials.jsonl", BLIND / "manifest.json", INTEGRITY]
    hashes_path.write_text("".join(f"{sha(path)}  {path}\n" for path in files))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
