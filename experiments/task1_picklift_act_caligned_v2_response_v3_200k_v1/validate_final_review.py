from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path("/home/ubuntu24/Teleop/lerobot")
EXP = REPO / "experiments/task1_picklift_act_caligned_v2_response_v3_200k_v1"
PLAN = EXP / "bound_simseen6_evaluation_plan.json"
ROOT = Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_csource_vs_response_v3_simseen6_paired_eval_v1")
REVIEW = ROOT / "canonical_video_review_v1"
FINAL = REVIEW / "final_review_v1"
OUT = FINAL / "independent_validation.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def group(rows: list[dict], field: str) -> dict:
    result = {}
    for value in sorted({row[field] for row in rows}, key=str):
        subset = [row for row in rows if row[field] == value]
        success = sum(row["final_success"] for row in subset)
        result[str(value)] = {"success": success, "total": len(subset), "rate": success / len(subset)}
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(PLAN.read_text())
    plan_by_id = {row["trial_id"]: row for row in plan["trials"]}
    final = lines(FINAL / "trials.jsonl")
    summary = json.loads((FINAL / "summary.json").read_text())
    manifest = json.loads((FINAL / "manifest.json").read_text())
    blind = {row["neutral_id"]: row for row in lines(REVIEW / "blind_review_frozen_v1/trials.jsonl")}
    queue = {row["neutral_id"]: row for row in lines(REVIEW / "blind_queue/queue.jsonl")}
    mapping = {row["neutral_id"]: row for row in lines(REVIEW / "private_mapping_prejoin/mapping.jsonl")}
    errors = []
    if len(final) != 12 or len({row["trial_id"] for row in final}) != 12 or set(plan_by_id) != {row["trial_id"] for row in final}:
        errors.append("trial_identity")
    video_pass = 0
    for row in final:
        trial = plan_by_id[row["trial_id"]]
        stem = trial["artifact_stem"]
        operator_path = ROOT / "trials" / f"{stem}.operator_label.json"
        operator = json.loads(operator_path.read_text())
        evidence = json.loads((ROOT / "trials" / f"{stem}.json").read_text())
        steps = lines(ROOT / "trials" / f"{stem}.steps.jsonl")
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(ROOT / "trials" / f"{stem}.mp4")], text=True))["streams"][0]
        neutral = blind[row["neutral_id"]]
        mapped = mapping[row["neutral_id"]]
        checks = [
            mapped["trial_id"] == row["trial_id"], row["video_sha256"] == mapped["video_sha256"] == queue[row["neutral_id"]]["video_sha256"],
            row["blind_review_success"] == bool(neutral["success"]), row["operator_success"] == bool(operator["success"]),
            row["raw_operator_blind_agreement"] == (bool(neutral["success"]) == bool(operator["success"])),
            row["final_success"] == row["blind_review_success"], row["adjudication"] is None,
            int(probe["width"]) == 640, int(probe["height"]) == 480, probe["r_frame_rate"] == "20/1",
            int(probe["nb_read_frames"]) == len(steps), steps[-1]["tick_started_elapsed_seconds"] >= 29.9,
            evidence.get("run_error") is None, evidence.get("torque_disable_verified") is True,
            evidence.get("upstream_action_modified_events") == 0, evidence["termination"] == "maximum_duration",
        ]
        if not all(checks):
            errors.append(f"trial_integrity:{row['trial_id']}")
        else:
            video_pass += 1
    by_model = group(final, "model_key")
    if by_model != summary["by_model"] or group(final, "cell") != summary["by_cell"] or group(final, "yaw_degrees") != summary["by_yaw"]:
        errors.append("group_summary")
    failures = dict(sorted(Counter(row["final_failure_category"] for row in final if not row["final_success"]).items()))
    if failures != summary["final_failure_categories"]:
        errors.append("failure_categories")
    disagreements = [row["trial_id"] for row in final if not row["raw_operator_blind_agreement"]]
    if disagreements or summary["agreement"]["disagree"] != 0 or summary["agreement"]["adjudication_count"] != 0:
        errors.append("agreement")
    if not manifest["blind_labels_frozen_before_operator_join"] or not manifest["operator_join_after_blind_freeze"]:
        errors.append("provenance_order")
    if manifest["operator_result_failed_attempts_excluded"] != ["operator_result_failed_attempt1", "operator_result_failed_attempt2"]:
        errors.append("failed_attempt_exclusion")
    validation = {
        "schema": "task1_response_v3_simseen6_canonical_review_independent_validation_v1",
        "status": "pass" if not errors else "fail",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": sha(PLAN),
        "trial_count": len(final),
        "unique_trial_count": len({row["trial_id"] for row in final}),
        "video_integrity_checks_passed": video_pass,
        "recomputed_by_model": by_model,
        "recomputed_failure_categories": failures,
        "recomputed_disagreement_trial_ids": disagreements,
        "final_trials_sha256": sha(FINAL / "trials.jsonl"),
        "summary_sha256": sha(FINAL / "summary.json"),
        "manifest_sha256": sha(FINAL / "manifest.json"),
        "errors": errors,
        "validated_at_utc": datetime.now(UTC).isoformat(),
    }
    OUT.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    hashes = FINAL / "hashes_v2.sha256"
    files = [FINAL / "trials.jsonl", FINAL / "summary.json", FINAL / "manifest.json", FINAL / "hashes.sha256", OUT, REVIEW / "blind_review_frozen_v1/trials.jsonl", REVIEW / "blind_review_frozen_v1/manifest.json", REVIEW / "integrity_prejoin_v1.json", PLAN]
    hashes.write_text("".join(f"{sha(path)}  {path}\n" for path in files))
    print(json.dumps(validation, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
