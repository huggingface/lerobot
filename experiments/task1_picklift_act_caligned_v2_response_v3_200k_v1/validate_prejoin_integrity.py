from __future__ import annotations

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
OUT = REVIEW / "integrity_prejoin_v1.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(PLAN.read_text())
    errors = []
    checks = []
    for trial in plan["trials"]:
        stem = trial["artifact_stem"]
        evidence_path = ROOT / "trials" / f"{stem}.json"
        steps_path = ROOT / "trials" / f"{stem}.steps.jsonl"
        ready_path = ROOT / "trials" / f"{stem}.ready.jsonl"
        return_path = ROOT / "trials" / f"{stem}.return.jsonl"
        video_path = ROOT / "trials" / f"{stem}.mp4"
        evidence = json.loads(evidence_path.read_text())
        steps = json_lines(steps_path)
        ready = json_lines(ready_path)
        returned = json_lines(return_path)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(video_path)
        ], text=True))["streams"][0]
        masks_modified = sum(any(row["upstream_action_modified_mask"]) for row in steps)
        local = {
            "trial_id": trial["trial_id"],
            "video_sha256": sha(video_path),
            "steps": len(steps),
            "video_frames": int(probe["nb_read_frames"]),
            "width": int(probe["width"]),
            "height": int(probe["height"]),
            "fps": probe["r_frame_rate"],
            "last_tick_elapsed_seconds": steps[-1]["tick_started_elapsed_seconds"],
            "ready_rows": len(ready),
            "return_rows": len(returned),
            "termination": evidence["termination"],
            "run_error": evidence.get("run_error"),
            "torque_disable_verified": evidence.get("torque_disable_verified"),
            "upstream_action_modified_events": evidence.get("upstream_action_modified_events"),
            "recomputed_modified_tick_count": masks_modified,
        }
        valid = all([
            evidence["trial"]["trial_id"] == trial["trial_id"],
            evidence["evaluation_plan_sha256"] == sha(PLAN),
            evidence["model_sha256"] == plan["models"][trial["model_key"]]["model_sha256"],
            evidence.get("run_error") is None,
            evidence.get("torque_disable_verified") is True,
            evidence.get("upstream_action_modified_events") == 0,
            masks_modified == 0,
            evidence["termination"] == "maximum_duration",
            sha(video_path) == evidence["video"]["sha256"],
            sha(steps_path) == evidence["steps_jsonl"]["sha256"],
            sha(ready_path) == evidence["ready_pose_alignment"]["trajectory"]["sha256"],
            sha(return_path) == evidence["automatic_return"]["trajectory"]["sha256"],
            len(steps) == evidence["steps_jsonl"]["lines"] == evidence["video"]["frames"] == int(probe["nb_read_frames"]),
            len(ready) == evidence["ready_pose_alignment"]["trajectory"]["lines"],
            len(returned) == evidence["automatic_return"]["trajectory"]["lines"],
            int(probe["width"]) == 640 and int(probe["height"]) == 480 and probe["r_frame_rate"] == "20/1",
            steps[-1]["tick_started_elapsed_seconds"] >= 29.9,
        ])
        local["status"] = "pass" if valid else "fail"
        checks.append(local)
        if not valid:
            errors.append(trial["trial_id"])
    result = {
        "schema": "task1_response_v3_simseen6_integrity_prejoin_v1",
        "status": "pass" if not errors else "fail",
        "plan_sha256": sha(PLAN),
        "trial_count": len(checks),
        "video_integrity_passed": sum(row["status"] == "pass" for row in checks),
        "all_maximum_duration": all(row["termination"] == "maximum_duration" for row in checks),
        "failed_attempt_directories_excluded": True,
        "operator_labels_read": False,
        "checks": checks,
        "errors": errors,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
