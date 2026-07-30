from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from evalv2_pilot import (
    find_trial,
    infrastructure_invalid_marker_path,
    load_frozen_plan,
    original_evidence_path,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an immutable operator-confirmed placement-mismatch marker "
            "without rewriting completed trial evidence."
        )
    )
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--observed-x-m", type=float, required=True)
    parser.add_argument("--observed-y-m", type=float, required=True)
    parser.add_argument("--observed-yaw-degrees", type=float, required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--operator-confirmed-placement-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.operator_confirmed_placement_mismatch:
        raise RuntimeError("Placement-invalid marking requires explicit operator confirmation.")
    plan = load_frozen_plan()
    trial = find_trial(plan, args.trial_id)
    original_path = original_evidence_path(plan, trial)
    if not original_path.exists():
        raise RuntimeError(f"Original trial evidence does not exist: {original_path}")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original.get("status") != "completed_pending_operator_annotation":
        raise RuntimeError("Placement mismatch marker requires completed, unannotated trial evidence.")
    label_path = (
        Path(plan["evidence_root"])
        / "labels"
        / f"{trial['spawn_region']}.operator.json"
    )
    if label_path.exists():
        raise RuntimeError("Refusing placement-invalid marking after an operator success/failure label exists.")
    expected_pose = {
        "x_forward_m": trial["nominal_x_forward_m"],
        "y_lateral_m": trial["nominal_y_lateral_m"],
        "yaw_degrees_modulo_90": trial["nominal_yaw_degrees_modulo_90"],
    }
    observed_pose = {
        "x_forward_m": args.observed_x_m,
        "y_lateral_m": args.observed_y_m,
        "yaw_degrees_modulo_90": args.observed_yaw_degrees,
    }
    if observed_pose == expected_pose:
        raise RuntimeError("Observed placement equals the frozen pose; mismatch marker is invalid.")
    marker_path = infrastructure_invalid_marker_path(plan, trial)
    if marker_path.exists():
        raise RuntimeError(f"Refusing to overwrite infrastructure-invalid marker: {marker_path}")
    marker = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "status": "infrastructure_invalid",
        "reason": "operator_placement_mismatch",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "trial_id": trial["trial_id"],
        "artifact_stem": trial["spawn_region"],
        "scored_trial": False,
        "policy_failure": False,
        "replacement_allowed": True,
        "replacement_limit": 1,
        "expected_nominal_pose": expected_pose,
        "operator_reported_pose": observed_pose,
        "operator_notes": args.notes,
        "original_evidence": {
            "path": str(original_path),
            "sha256": sha256_file(original_path),
        },
        "canonical_video": original["video"],
        "steps_jsonl": original["steps_jsonl"],
        "preservation": {
            "original_evidence_rewritten": False,
            "video_rewritten": False,
            "steps_rewritten": False,
            "replacement_must_use_same_frozen_pose_model_and_order": True,
        },
    }
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved immutable placement-invalid marker to {marker_path}")


if __name__ == "__main__":
    main()
