from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--success", action="store_true")
    parser.add_argument("--report-zh", required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    trial = next(row for row in plan["trials"] if row["trial_id"] == args.trial_id)
    root = Path(plan["evidence_root"]) / "trials"
    stem = trial["artifact_stem"]
    evidence = root / f"{stem}.json"
    if not evidence.exists():
        raise RuntimeError(f"missing evidence: {evidence}")
    output = root / f"{stem}.operator_label.json"
    if output.exists():
        raise FileExistsError(output)
    value = {
        "schema": "task1_csource_response_v3_simseen6_operator_label_v1",
        "evaluation_id": plan["evaluation_id"],
        "trial_id": args.trial_id,
        "artifact_stem": stem,
        "model_key": trial["model_key"],
        "model_id": trial["model_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "labeled",
        "success": args.success,
        "failure_category": None if args.success else "operator_unspecified_failure",
        "operator_report_zh": args.report_zh,
    }
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
