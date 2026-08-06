from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PLAN = Path("/home/ubuntu24/Teleop/lerobot/experiments/task1_picklift_act_additive_three_model_200k_v1/evaluation_plan.json")
ROOT = Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1")
TRIALS = ROOT / "trials"
OUT = ROOT / "canonical_video_review_v1" / "integrity_prejoin_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def probe(path: Path) -> dict:
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path),
        ],
        text=True,
    )
    return json.loads(raw)["streams"][0]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(PLAN.read_text())
    models = plan["models"]
    rows = []
    errors = []
    for expected_order, trial in enumerate(plan["trials"], 1):
        stem = trial["artifact_stem"]
        main_path = TRIALS / f"{stem}.json"
        steps_path = TRIALS / f"{stem}.steps.jsonl"
        ready_path = TRIALS / f"{stem}.ready.jsonl"
        return_path = TRIALS / f"{stem}.return.jsonl"
        video_path = TRIALS / f"{stem}.mp4"
        missing = []
        for path in (main_path, steps_path, ready_path, return_path, video_path):
            if not path.is_file():
                missing.append(f"missing:{path}")
        if missing:
            errors.extend(missing)
            continue
        evidence = json.loads(main_path.read_text())
        steps = jsonl(steps_path)
        ready = jsonl(ready_path)
        returned = jsonl(return_path)
        video = probe(video_path)
        checks = {
            "order": trial["order"] == expected_order,
            "trial_identity": evidence["trial"]["trial_id"] == trial["trial_id"],
            "plan_sha": evidence["evaluation_plan_sha256"] == sha256(PLAN),
            "model_sha": evidence["model_sha256"] == models[trial["model_key"]]["model_sha256"],
            "run_error_null": evidence.get("run_error") is None,
            "torque_disable": evidence.get("torque_disable_verified") is True,
            "ready_evidence_valid": bool(ready) or (
                evidence["ready_pose_alignment"]["trajectory"]["lines"] == 0
                and evidence["ready_pose_alignment"]["result"]["status"] == "ready_pose_observed"
            ),
            "return_nonempty": bool(returned),
            "policy_reset_after_ready": evidence["policy_start"]["policy_reset_after_ready_pose"] is True,
            "upstream_modification_zero": evidence.get("upstream_action_modified_events") == 0,
            "video_width": int(video["width"]) == 640,
            "video_height": int(video["height"]) == 480,
            "video_fps": video["r_frame_rate"] == "20/1",
            "video_frames_equal_steps": int(video["nb_read_frames"]) == len(steps),
            "main_frames_equal_steps": evidence["video"]["frames"] == len(steps),
            "last_tick_at_least_29_9": float(steps[-1]["tick_started_elapsed_seconds"]) >= 29.9,
            "step_indices_contiguous": all(row["step"] == i for i, row in enumerate(steps)),
            "step_upstream_unmodified": all(not any(row["upstream_action_modified_mask"]) for row in steps),
            "video_hash": evidence["video"]["sha256"] == sha256(video_path),
            "steps_hash": evidence["steps_jsonl"]["sha256"] == sha256(steps_path),
        }
        bad = [name for name, passed in checks.items() if not passed]
        if bad:
            errors.append(f"{stem}:{','.join(bad)}")
        rows.append(
            {
                "trial_id": trial["trial_id"],
                "model_key": trial["model_key"],
                "pose_order": trial["pose_order"],
                "within_pose_order": trial["within_pose_order"],
                "video_frames": int(video["nb_read_frames"]),
                "steps_rows": len(steps),
                "last_tick_seconds": steps[-1]["tick_started_elapsed_seconds"],
                "termination": evidence["termination"],
                "checks": checks,
            }
        )
    counts = Counter(row["termination"] for row in rows)
    result = {
        "schema": "task1_additive_eval24_integrity_prejoin_v1",
        "status": "pass" if len(rows) == 72 and not errors else "fail",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": sha256(PLAN),
        "trial_count": len(rows),
        "unique_trial_count": len({row["trial_id"] for row in rows}),
        "termination_counts": dict(counts),
        "errors": errors,
        "operator_labels_read": False,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "trials": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "trial_count", "unique_trial_count", "termination_counts", "errors")}, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
