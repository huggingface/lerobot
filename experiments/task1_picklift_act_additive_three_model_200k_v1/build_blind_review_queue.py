from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

PLAN = Path("/home/ubuntu24/Teleop/lerobot/experiments/task1_picklift_act_additive_three_model_200k_v1/evaluation_plan.json")
EVAL_ROOT = Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1")
OUT = EVAL_ROOT / "canonical_video_review_v1" / "blind_queue"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(PLAN.read_text())
    rows = []
    for trial in plan["trials"]:
        stem = trial["artifact_stem"]
        video = EVAL_ROOT / "trials" / f"{stem}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        rows.append(
            {
                "trial_id": trial["trial_id"],
                "pose_order": trial["pose_order"],
                "within_pose_order": trial["within_pose_order"],
                "model_key": trial["model_key"],
                "model_id": trial["model_id"],
                "cell": trial["cell"],
                "coverage_tier": trial["coverage_tier"],
                "yaw_degrees": trial["nominal_yaw_degrees_modulo_90"],
                "video_path": str(video),
                "video_sha256": sha256(video),
            }
        )
    if len(rows) != 72 or len({row["trial_id"] for row in rows}) != 72:
        raise RuntimeError("blind queue identity mismatch")
    OUT.mkdir(parents=True)
    queue = OUT / "queue.jsonl"
    with queue.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema": "task1_additive_eval24_blind_review_queue_v1",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": sha256(PLAN),
        "trial_count": len(rows),
        "operator_labels_included": False,
        "operator_results_included": False,
        "failure_categories_included": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "queue_sha256": sha256(queue),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
