from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from paired_evaluator import find_trial, load_frozen_plan, sha256_file

FAILURE_CATEGORIES = {
    "missed_grasp",
    "insufficient_lift",
    "dropped_before_0p5s",
    "timeout",
    "unknown",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write one immutable operator label without modifying trial evidence."
    )
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--artifact-stem")
    parser.add_argument("--success", choices=("true", "false"), required=True)
    parser.add_argument("--failure-category")
    parser.add_argument("--time-to-first-valid-success-seconds", type=float)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    plan = load_frozen_plan()
    trial = find_trial(plan, args.trial_id)
    artifact_stem = args.artifact_stem or trial["spawn_region"]
    if artifact_stem not in {
        trial["spawn_region"],
        f"{trial['spawn_region']}__replacement1",
    }:
        raise RuntimeError("Artifact stem is not the original or linked replacement for the trial.")
    trials_root = Path(plan["evidence_root"]) / "trials"
    evidence_path = trials_root / f"{artifact_stem}.json"
    if not evidence_path.exists():
        raise RuntimeError(f"Trial evidence does not exist: {evidence_path}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    success = args.success == "true"
    if success and args.failure_category is not None:
        raise RuntimeError("Successful labels must not have a failure category.")
    if not success and args.failure_category not in FAILURE_CATEGORIES:
        raise RuntimeError(
            f"Failed labels require one frozen failure category: {sorted(FAILURE_CATEGORIES)}"
        )
    if (
        args.time_to_first_valid_success_seconds is not None
        and not 0.0 <= args.time_to_first_valid_success_seconds <= 30.0
    ):
        raise RuntimeError("Success time must lie inside the frozen 30-second window.")

    labels_root = Path(plan["evidence_root"]) / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    output_path = labels_root / f"{artifact_stem}.operator.json"
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite immutable label: {output_path}")
    label = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "trial_id": trial["trial_id"],
        "artifact_stem": artifact_stem,
        "cell_id": trial["cell_id"],
        "model_id": trial["model_id"],
        "source": "operator",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "success": success,
        "failure_category": args.failure_category,
        "time_to_first_valid_success_seconds": args.time_to_first_valid_success_seconds,
        "notes": args.notes,
        "notes_are_operator_estimate_not_measurement_truth": True,
        "success_contract": plan["success_contract"],
        "held_at_end_required": False,
        "trial_evidence": {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
        },
        "canonical_video": {"status": "not_reviewed"},
    }
    output_path.write_text(
        json.dumps(label, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved immutable operator label to {output_path}")


if __name__ == "__main__":
    main()
