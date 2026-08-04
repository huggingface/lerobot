from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_PLAN_SHA256 = "7de77eed859898a6397265244ab2f4c189d91dbb565202b99a0a5bdd208214f1"
EXPECTED_MODELS = {
    "real48": "73f61996c0ebba444c1bce070ec36735425f3307e420eab47cf29a3ab7ffa14c",
    "real96": "2d80bbddff5c3e5862a6e9f0b639619628fb637b9beea4826a6469e95f851e44",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate frozen Eval48 software-gate evidence")
    parser.add_argument("--software-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.software_root
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    dry_run = json.loads((root / "dry_run.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "evaluation_plan.json").read_text(encoding="utf-8"))
    if sha256_file(root / "evaluation_plan.json") != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Frozen evidence plan hash mismatch")
    hash_rows = {}
    for line in (root / "hashes.sha256").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        hash_rows[filename] = digest
    for filename, digest in hash_rows.items():
        if sha256_file(root / filename) != digest:
            raise RuntimeError(f"Evidence hash mismatch: {filename}")
    if dry_run["status"] != "software_dry_run_passed_hardware_not_accessed":
        raise RuntimeError("Dry-run status mismatch")
    if dry_run["hardware_access"] != {
        "serial": False, "camera": False, "robot": False, "torque": False, "rollout": False
    }:
        raise RuntimeError("Dry-run hardware boundary mismatch")
    fake = dry_run["fake_protocol"]
    if fake["trials_exercised"] != 96 or fake["policy_reset_calls"] != {"real48": 48, "real96": 48}:
        raise RuntimeError("Fake protocol count mismatch")
    if fake["pose_trial_ids_in_frozen_order"] != [trial["trial_id"] for trial in plan["trials"]]:
        raise RuntimeError("Fake protocol order mismatch")
    if not all((fake["all_ready_before_policy"], fake["all_ready_after_trial"], fake["all_canonical_rgb_640x480"], fake["all_pre_action_frames_before_policy_send"], fake["all_official_sent_equals_requested"], fake["all_torque_disabled"])):
        raise RuntimeError("Fake protocol invariant failed")
    smoke_hashes = {}
    for model_id, model_hash in EXPECTED_MODELS.items():
        path = root / f"offline_inference_{model_id}.json"
        smoke = json.loads(path.read_text(encoding="utf-8"))
        if smoke["status"] != "pass" or smoke["model_sha256"] != model_hash:
            raise RuntimeError(f"Offline smoke identity mismatch: {model_id}")
        if smoke["output_shape"] != [1, 6] or smoke["output_finite"] is not True:
            raise RuntimeError(f"Offline smoke output mismatch: {model_id}")
        smoke_hashes[model_id] = sha256_file(path)
    result = {
        "schema": "task1_picklift_real48_vs_real96_eval48_software_gate_independent_validation_v1",
        "status": "pass",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "trials": 96,
        "poses": 48,
        "model_trials": {"real48": 48, "real96": 48},
        "coverage_tiers": {"seen_by_real48": 24, "added_by_real96": 18, "unseen_by_both": 6},
        "offline_inference_smoke_sha256": smoke_hashes,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "hashes_sha256": sha256_file(root / "hashes.sha256"),
        "hardware_accessed": False,
        "rollout_executed": False,
    }
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite independent validation: {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
