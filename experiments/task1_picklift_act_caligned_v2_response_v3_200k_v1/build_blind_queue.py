from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

REPO = Path("/home/ubuntu24/Teleop/lerobot")
EXP = REPO / "experiments/task1_picklift_act_caligned_v2_response_v3_200k_v1"
PLAN = EXP / "bound_simseen6_evaluation_plan.json"
ROOT = Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_csource_vs_response_v3_simseen6_paired_eval_v1")
OPERATOR = ROOT / "operator_result_v1"
OUT = ROOT / "canonical_video_review_v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(PLAN.read_text())
    expected = {
        "summary.json": "6c2c9b015c99d131ed61bc39908a9d5a3d764b120666cf6537060233e6d296cc",
        "trials.jsonl": "52cecf7677e2a3a7e7e888abb56fd97045dce6c337dc3eb0aa2189b6088e6e02",
        "hashes.sha256": "d7a2693974c6d3615810a777fd742c4b7db024c0649cbe9c1560fb665ed57f42",
    }
    for name, digest in expected.items():
        if sha(OPERATOR / name) != digest:
            raise RuntimeError(f"operator_result_v1 mismatch: {name}")
    if (ROOT / "operator_result_failed_attempt1").exists() is False or (ROOT / "operator_result_failed_attempt2").exists() is False:
        raise RuntimeError("failed attempts must remain preserved and excluded")
    seed = 20260810
    rng = random.Random(seed)
    neutral_ids = [f"n{value:02d}" for value in range(1, 13)]
    rng.shuffle(neutral_ids)
    temp = ROOT / ".canonical_video_review_v1.tmp"
    if temp.exists():
        raise FileExistsError(temp)
    videos = temp / "blind_videos"
    queue_dir = temp / "blind_queue"
    private = temp / "private_mapping_prejoin"
    videos.mkdir(parents=True)
    queue_dir.mkdir()
    private.mkdir()
    queue_rows = []
    mapping_rows = []
    for trial, neutral_id in zip(plan["trials"], neutral_ids, strict=True):
        source = ROOT / "trials" / f"{trial['artifact_stem']}.mp4"
        target = videos / f"{neutral_id}.mp4"
        shutil.copyfile(source, target)
        digest = sha(source)
        if sha(target) != digest:
            raise RuntimeError(f"neutral copy mismatch: {neutral_id}")
        queue_rows.append({"neutral_id": neutral_id, "video_path": str(target), "video_sha256": digest})
        mapping_rows.append({"neutral_id": neutral_id, "trial_id": trial["trial_id"], "artifact_stem": trial["artifact_stem"], "video_sha256": digest})
    queue_rows.sort(key=lambda row: row["neutral_id"])
    mapping_rows.sort(key=lambda row: row["neutral_id"])
    queue = queue_dir / "queue.jsonl"
    mapping = private / "mapping.jsonl"
    queue.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in queue_rows))
    mapping.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in mapping_rows))
    manifest = {
        "schema": "task1_response_v3_simseen6_neutral_blind_queue_v1",
        "status": "frozen_before_blind_review",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": sha(PLAN),
        "operator_result_v1_bound": expected,
        "operator_result_failed_attempts_excluded": ["operator_result_failed_attempt1", "operator_result_failed_attempt2"],
        "neutral_id_seed": seed,
        "trial_count": 12,
        "queue_contains_model_identity": False,
        "queue_contains_trial_or_pose_identity": False,
        "queue_contains_operator_labels": False,
        "queue_sha256": sha(queue),
        "private_mapping_sha256": sha(mapping),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (queue_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temp, OUT)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
