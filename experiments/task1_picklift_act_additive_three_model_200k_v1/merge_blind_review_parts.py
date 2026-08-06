from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1/canonical_video_review_v1")
QUEUE = ROOT / "blind_queue" / "queue.jsonl"
PARTS = ["part_001_012", "part_013_024", "part_025_048", "part_049_060", "part_061_072"]
OUT = ROOT / "blind_review_frozen_v1"
EXPECTED_QUEUE_SHA = "9d1be6eaca335a62b71bf2f471c19a568abe6a996731d9cf00eb6eec0edfaa09"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first(row: dict, *keys: str):
    for key in keys:
        if key in row:
            return row[key]
    return None


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha256(QUEUE) != EXPECTED_QUEUE_SHA:
        raise RuntimeError("queue hash mismatch")
    queue = [json.loads(line) for line in QUEUE.read_text().splitlines()]
    queue_by_id = {row["trial_id"]: row for row in queue}
    normalized = []
    part_provenance = []
    for part_name in PARTS:
        part = ROOT / "blind_review_parts_v1" / part_name
        trials_path = part / "trials.jsonl"
        manifest_path = part / "manifest.json"
        if not trials_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(part)
        part_provenance.append(
            {
                "part": part_name,
                "trials_sha256": sha256(trials_path),
                "manifest_sha256": sha256(manifest_path),
            }
        )
        for raw in [json.loads(line) for line in trials_path.read_text().splitlines() if line]:
            trial_id = raw["trial_id"]
            source = queue_by_id[trial_id]
            category = first(raw, "failure_category", "visible_failure_category")
            raw_category = category
            if category == "no_grasp":
                category = "missed_grasp"
            success = bool(raw["success"])
            if success:
                category = None
            elif category not in {"missed_grasp", "spatial_offset", "top_contact", "grasp_then_drop", "insufficient_lift", "unknown"}:
                raise RuntimeError(f"unsupported category {trial_id}:{category}")
            normalized.append(
                {
                    "trial_id": trial_id,
                    "queue_row_1_indexed": queue.index(source) + 1,
                    "model_key": source["model_key"],
                    "model_id": source["model_id"],
                    "pose_order": source["pose_order"],
                    "within_pose_order": source["within_pose_order"],
                    "cell": source["cell"],
                    "coverage_tier": source["coverage_tier"],
                    "yaw_degrees": source["yaw_degrees"],
                    "video_sha256": source["video_sha256"],
                    "success": success,
                    "failure_category": category,
                    "raw_failure_category": raw_category,
                    "confidence": raw.get("confidence"),
                    "review_coverage": first(raw, "coverage", "coverage_seconds"),
                    "visible_two_finger_grasp": bool(first(raw, "visible_two_finger_grasp", "grasp_visible")),
                    "visible_unsupported_lift_gt_5cm": bool(first(raw, "visible_unsupported_lift_over_5cm", "visible_unsupported_lift_gt_5cm", "lift_over_5cm_visible", "lift_gt5cm_visible", "lift_gt_5cm_visible")),
                    "hold_interval": first(raw, "hold_interval_seconds", "hold_interval_s", "hold_interval_sec"),
                    "visible_evidence": first(raw, "visible_evidence", "evidence"),
                    "source_part": part_name,
                    "source_raw_row": raw,
                }
            )
    normalized.sort(key=lambda row: row["queue_row_1_indexed"])
    if len(normalized) != 72 or [row["queue_row_1_indexed"] for row in normalized] != list(range(1, 73)):
        raise RuntimeError("blind coverage mismatch")
    if len({row["trial_id"] for row in normalized}) != 72:
        raise RuntimeError("duplicate trial")
    OUT.mkdir(parents=True)
    trials_out = OUT / "trials.jsonl"
    with trials_out.open("x") as handle:
        for row in normalized:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    failures = Counter(row["failure_category"] for row in normalized if not row["success"])
    manifest = {
        "schema": "task1_additive_eval24_blind_review_frozen_v1",
        "evaluation_id": "task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1",
        "status": "blind_labels_frozen_before_operator_join",
        "queue_sha256": EXPECTED_QUEUE_SHA,
        "trial_count": 72,
        "unique_trial_count": 72,
        "operator_labels_read_by_reviewers": False,
        "operator_results_read_by_reviewers": False,
        "part_provenance": part_provenance,
        "normalization": {"no_grasp": "missed_grasp", "raw_fields_preserved": True},
        "aggregate_computed_after_all_labels_frozen": {
            "success": sum(row["success"] for row in normalized),
            "failure": sum(not row["success"] for row in normalized),
            "failure_categories": dict(sorted(failures.items())),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_out = OUT / "manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hashes_out = OUT / "hashes.sha256"
    hashes_out.write_text(f"{sha256(trials_out)}  {trials_out.name}\n{sha256(manifest_out)}  {manifest_out.name}\n")
    for path in (trials_out, manifest_out, hashes_out):
        os.chmod(path, 0o444)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
