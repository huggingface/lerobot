"""Create one immutable operator annotation sidecar for a real trial."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

FAILURE_CATEGORIES = {
    "perception_failure",
    "missed_grasp",
    "unstable_grasp",
    "wrong_trajectory",
    "collision",
    "premature_release",
    "timeout",
    "out_of_workspace",
    "unknown",
}
SUCCESS_CONTRACT = (
    "Visible bilateral grasp, at least 5 cm lift, stable hold for at least "
    "0.5 s, and no drop before trial end."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--result",
        choices=("success", "failure", "aborted"),
        required=True,
    )
    parser.add_argument("--failure-category", choices=sorted(FAILURE_CATEGORIES))
    parser.add_argument("--notes")
    args = parser.parse_args()

    machine = json.loads(args.evidence.read_text(encoding="utf-8"))
    if machine["operator_annotation_status"] != "pending":
        raise RuntimeError("Machine evidence is not pending annotation.")
    if args.result == "failure" and args.failure_category is None:
        raise RuntimeError("Failure annotation requires a failure category.")
    if args.result != "failure" and args.failure_category is not None:
        raise RuntimeError("Failure category is only valid for failure.")

    output = args.evidence.with_suffix(".annotation.json")
    if output.exists():
        raise RuntimeError(f"Annotation already exists: {output}")
    payload = {
        "schema_version": 1,
        "spawn_region": machine["trial"]["spawn_region"],
        "machine_evidence": str(args.evidence),
        "machine_evidence_status": machine["status"],
        "success_contract": SUCCESS_CONTRACT,
        "operator_result": args.result,
        "failure_category": args.failure_category,
        "notes": args.notes,
        "annotated_at_utc": datetime.now(UTC).isoformat(),
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
