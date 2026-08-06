from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/home/ubuntu24/Teleop/lerobot")
EXP = REPO / "experiments/task1_picklift_act_additive_three_model_200k_v1"
PLAN = EXP / "evaluation_plan.json"
ROOT = Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1")
REVIEW = ROOT / "canonical_video_review_v1"
BLIND = REVIEW / "blind_review_frozen_v1"
FINAL = REVIEW / "final_review_v1"
OUT = FINAL / "independent_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
    plan_by_id = {trial["trial_id"]: trial for trial in plan["trials"]}
    blind = lines(BLIND / "trials.jsonl")
    final = lines(FINAL / "trials.jsonl")
    summary = json.loads((FINAL / "summary.json").read_text())
    manifest = json.loads((FINAL / "manifest.json").read_text())
    errors = []
    if sha256(PLAN) != "ad711d39df8f2e5334665add1c2fdfee00e4a57e4550fa9960cffaf32728c8c3": errors.append("plan_sha")
    if len(blind) != 72 or len(final) != 72: errors.append("trial_count")
    if set(plan_by_id) != {row["trial_id"] for row in blind} or set(plan_by_id) != {row["trial_id"] for row in final}: errors.append("identity_set")
    blind_by_id = {row["trial_id"]: row for row in blind}
    video_checks = 0
    for row in final:
        trial_id = row["trial_id"]
        trial = plan_by_id[trial_id]
        if row["model_key"] != trial["model_key"] or row["pose_order"] != trial["pose_order"]: errors.append(f"plan_binding:{trial_id}")
        if row["blind_review_success"] != blind_by_id[trial_id]["success"]: errors.append(f"blind_changed:{trial_id}")
        operator_path = ROOT / "trials" / f"{trial_id}.operator_label.json"
        operator = json.loads(operator_path.read_text())
        if row["operator_success"] != bool(operator["success"]) or row["operator_label_sha256"] != sha256(operator_path): errors.append(f"operator_join:{trial_id}")
        evidence = json.loads((ROOT / "trials" / f"{trial_id}.json").read_text())
        steps = lines(ROOT / "trials" / f"{trial_id}.steps.jsonl")
        raw = subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json",
            str(ROOT / "trials" / f"{trial_id}.mp4"),
        ], text=True)
        stream = json.loads(raw)["streams"][0]
        checks = [
            int(stream["width"]) == 640,
            int(stream["height"]) == 480,
            stream["r_frame_rate"] == "20/1",
            int(stream["nb_read_frames"]) == len(steps),
            steps[-1]["tick_started_elapsed_seconds"] >= 29.9,
            evidence.get("run_error") is None,
            evidence.get("torque_disable_verified") is True,
            evidence.get("upstream_action_modified_events") == 0,
            evidence["model_sha256"] == plan["models"][trial["model_key"]]["model_sha256"],
        ]
        if not all(checks): errors.append(f"trial_integrity:{trial_id}")
        video_checks += 1
    calc_overall = {"success": sum(row["final_success"] for row in final), "total": 72, "rate": sum(row["final_success"] for row in final) / 72}
    if calc_overall != summary["overall"]: errors.append("overall")
    for field, key in [("model_key", "by_model"), ("coverage_tier", "by_coverage_tier"), ("yaw_degrees", "by_yaw"), ("cell", "by_cell")]:
        if group(final, field) != summary[key]: errors.append(key)
    final_failures = dict(sorted(Counter(row["final_failure_category"] for row in final if not row["final_success"]).items()))
    if final_failures != summary["final_failure_categories"]: errors.append("failure_categories")
    raw_disagreements = sorted(row["trial_id"] for row in final if row["operator_success"] != row["blind_review_success"])
    if raw_disagreements != sorted(summary["agreement"]["disagreement_trial_ids"]): errors.append("disagreements")
    if sum(row["final_operator_agreement"] for row in final) != summary["agreement"]["final_operator_agree"]: errors.append("final_agreement")
    if not manifest["blind_frozen_before_operator_join"] or not manifest["operator_join_after_blind_freeze"]: errors.append("provenance_order")
    validation = {
        "schema": "task1_additive_eval24_canonical_review_independent_validation_v1",
        "status": "pass" if not errors else "fail",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": sha256(PLAN),
        "blind_trials_sha256": sha256(BLIND / "trials.jsonl"),
        "final_trials_sha256": sha256(FINAL / "trials.jsonl"),
        "summary_sha256": sha256(FINAL / "summary.json"),
        "manifest_sha256": sha256(FINAL / "manifest.json"),
        "trial_count": len(final),
        "unique_trial_count": len({row["trial_id"] for row in final}),
        "video_integrity_checks_passed": video_checks if not any(error.startswith("trial_integrity") for error in errors) else None,
        "recomputed": {
            "overall": calc_overall,
            "by_model": group(final, "model_key"),
            "by_coverage_tier": group(final, "coverage_tier"),
            "by_yaw": group(final, "yaw_degrees"),
            "by_cell": group(final, "cell"),
            "failure_categories": final_failures,
            "raw_disagreement_trial_ids": raw_disagreements,
        },
        "errors": errors,
        "validated_at_utc": datetime.now(UTC).isoformat(),
    }
    OUT.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    hashes = FINAL / "hashes_v2.sha256"
    files = [FINAL / "trials.jsonl", FINAL / "summary.json", FINAL / "manifest.json", OUT, BLIND / "trials.jsonl", BLIND / "manifest.json", REVIEW / "integrity_prejoin_v1.json", EXP / "adjudication_v1.json", PLAN]
    hashes.write_text("".join(f"{sha256(path)}  {path}\n" for path in files))
    print(json.dumps(validation, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
