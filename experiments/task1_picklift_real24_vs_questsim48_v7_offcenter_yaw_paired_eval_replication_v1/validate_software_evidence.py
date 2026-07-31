from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_EVIDENCE = Path(
    "/home/ubuntu24/Teleop/artifacts/evaluation/"
    "task1_picklift_real24_vs_questsim48_v7_offcenter_yaw_paired_eval24_replication_v1/"
    "software_preparation_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> dict:
    hashes_path = root / "hashes.sha256"
    expected_rows = {}
    for row in hashes_path.read_text(encoding="utf-8").splitlines():
        digest, name = row.split("  ", 1)
        expected_rows[name] = digest
    required = {
        "evaluation_plan.json",
        "research_identity_verification.json",
        "dry_run.json",
        "manifest.json",
    }
    if set(expected_rows) != required:
        raise RuntimeError(f"Unexpected frozen hash inventory: {sorted(expected_rows)}")
    actual_rows = {name: sha256_file(root / name) for name in sorted(required)}
    if actual_rows != {name: expected_rows[name] for name in sorted(required)}:
        raise RuntimeError("Frozen evidence content does not match hashes.sha256.")

    plan = json.loads((root / "evaluation_plan.json").read_text(encoding="utf-8"))
    research = json.loads(
        (root / "research_identity_verification.json").read_text(encoding="utf-8")
    )
    dry_run = json.loads((root / "dry_run.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if len(plan["trials"]) != 24:
        raise RuntimeError("Frozen plan does not contain 24 paired trials.")
    if dry_run["fake_protocol"]["trials_exercised"] != 24:
        raise RuntimeError("Dry run did not exercise all 24 paired trials.")
    if dry_run["fake_protocol"]["policy_reset_calls"] != {
        "real24_only": 12,
        "questsim48_v7": 12,
    }:
        raise RuntimeError("Dry run policy reset counts differ from the frozen pairing.")
    if any(dry_run["hardware_access"].values()):
        raise RuntimeError("Dry-run evidence claims hardware access.")
    if research["working_tree_status"] != "clean":
        raise RuntimeError("Research identity was not frozen from a clean checkout.")
    if manifest["plan"]["sha256"] != actual_rows["evaluation_plan.json"]:
        raise RuntimeError("Manifest plan hash does not bind the frozen plan.")
    if (
        manifest["research_identity_verification"]["sha256"]
        != actual_rows["research_identity_verification.json"]
    ):
        raise RuntimeError("Manifest research hash does not bind the frozen verification.")
    if manifest["dry_run"]["sha256"] != actual_rows["dry_run.json"]:
        raise RuntimeError("Manifest dry-run hash does not bind the frozen dry run.")
    runner_path = Path(manifest["runner"]["path"])
    if not runner_path.exists():
        raise RuntimeError("Manifest runner path is unavailable.")
    if sha256_file(runner_path) != manifest["runner"]["sha256"]:
        raise RuntimeError("Current runner does not match the software-evidence manifest.")
    return {
        "schema_version": 1,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "status": "independent_validation_passed",
        "evaluation_id": plan["evaluation_id"],
        "evidence_root": str(root),
        "paired_trials": 24,
        "model_trial_counts": {"real24_only": 12, "questsim48_v7": 12},
        "hardware_access": dry_run["hardware_access"],
        "plan_sha256": actual_rows["evaluation_plan.json"],
        "dry_run_sha256": actual_rows["dry_run.json"],
        "manifest_sha256": actual_rows["manifest.json"],
        "hashes_sha256": sha256_file(hashes_path),
        "runner_sha256": manifest["runner"]["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.evidence_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise RuntimeError(f"Refusing to overwrite validation evidence: {args.output}")
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
