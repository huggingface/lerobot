from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from paired_evaluator import (
    find_trial,
    infrastructure_invalid_marker_path,
    load_frozen_plan,
    original_evidence_path,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preserve and mark one operator-placement mismatch as infrastructure-invalid."
    )
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()

    plan = load_frozen_plan()
    trial = find_trial(plan, args.trial_id)
    evidence_path = original_evidence_path(plan, trial)
    if not evidence_path.exists():
        raise RuntimeError(f"Original trial evidence does not exist: {evidence_path}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    marker_path = infrastructure_invalid_marker_path(plan, trial)
    if marker_path.exists():
        raise RuntimeError(f"Refusing to overwrite infrastructure-invalid marker: {marker_path}")
    marker = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "trial_id": trial["trial_id"],
        "artifact_stem": trial["spawn_region"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "infrastructure_invalid",
        "reason": "operator_placement_mismatch",
        "operator_notes": args.notes,
        "operator_reported_pose": {"status": "not_specified"},
        "expected_nominal_pose": {
            "x_forward_m": trial["nominal_x_forward_m"],
            "y_lateral_m": trial["nominal_y_lateral_m"],
            "yaw_degrees_modulo_90": trial["nominal_yaw_degrees_modulo_90"],
        },
        "policy_failure": False,
        "scored_trial": False,
        "replacement_allowed": True,
        "replacement_limit": 1,
        "original_evidence": {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
        },
        "canonical_video": evidence["video"],
        "steps_jsonl": evidence["steps_jsonl"],
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
    print(f"Saved immutable infrastructure-invalid marker to {marker_path}")


if __name__ == "__main__":
    main()
