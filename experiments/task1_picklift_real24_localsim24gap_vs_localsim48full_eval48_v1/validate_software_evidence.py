from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_PLAN_SHA256 = "61efcb5e4298c86a103d41a963643962801375a4a78f0919b50c037c783ce176"
EXPECTED_MODELS = {
    "real24_localsim24_gap": "e9bbbc96e3104435d670e450090ab143610e2cdba8d38485beec339d5230577c",
    "real24_localsim48_full": "735af1dc914c1ea5b82fada65a3c72439cb5603ac8731425f43819f849972c0e",
}
EXPECTED_SOURCE_HASHES = {
    "source_research_contract.json": "696980b1a78d5f2d2ee71c96a72e7ede23a34b51fddde93caabfa04767394342",
    "source_training_result.json": "a2a320bcfbc3ff6bfcbf2000a17ca15804e1a7b00e9477e9f16a1577b82b2477",
    "source_pose_manifest.json": "f6bc79e9b99818f12f0e6a374688850374ea6f5cb971ba5da7ef3f32ae8322e7",
    "source_eval48_plan.json": "7de77eed859898a6397265244ab2f4c189d91dbb565202b99a0a5bdd208214f1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently validate frozen Eval48 software-gate evidence"
    )
    parser.add_argument("--software-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.software_root
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
    for filename, digest in EXPECTED_SOURCE_HASHES.items():
        if sha256_file(root / filename) != digest:
            raise RuntimeError(f"Frozen source identity mismatch: {filename}")
    if dry_run["status"] != "software_dry_run_passed_hardware_not_accessed":
        raise RuntimeError("Dry-run status mismatch")
    if dry_run["hardware_access"] != {
        "serial": False,
        "camera": False,
        "robot": False,
        "torque": False,
        "rollout": False,
    }:
        raise RuntimeError("Dry-run hardware boundary mismatch")
    fake = dry_run["fake_protocol"]
    if fake["trials_exercised"] != 96 or fake["policy_reset_calls"] != {
        "real24_localsim24_gap": 48,
        "real24_localsim48_full": 48,
    }:
        raise RuntimeError("Fake protocol count mismatch")
    if fake["pose_trial_ids_in_frozen_order"] != [trial["trial_id"] for trial in plan["trials"]]:
        raise RuntimeError("Fake protocol order mismatch")
    if not all(
        (
            fake["all_ready_before_policy"],
            fake["all_ready_after_trial"],
            fake["all_canonical_rgb_640x480"],
            fake["all_pre_action_frames_before_policy_send"],
            fake["all_official_sent_equals_requested"],
            fake["all_torque_disabled"],
        )
    ):
        raise RuntimeError("Fake protocol invariant failed")
    trials = plan["trials"]
    if len(trials) != 96 or [trial["order"] for trial in trials] != list(range(1, 97)):
        raise RuntimeError("Frozen paired trial order mismatch")
    for pose_order, (left, right) in enumerate(zip(trials[::2], trials[1::2], strict=True), start=1):
        expected = tuple(EXPECTED_MODELS) if pose_order % 2 == 1 else tuple(reversed(EXPECTED_MODELS))
        if (left["model_key"], right["model_key"]) != expected:
            raise RuntimeError(f"First-model alternation mismatch at pose {pose_order}")
        if left["eval_pose_id"] != right["eval_pose_id"] or left["pose_order"] != pose_order:
            raise RuntimeError(f"Paired pose identity mismatch at pose {pose_order}")
    source_trials = trials[::2]
    tier_counts = {
        tier: sum(trial["coverage_tier"] == tier for trial in source_trials)
        for tier in ("seen_by_real48", "added_by_real96", "unseen_by_both")
    }
    if tier_counts != {"seen_by_real48": 24, "added_by_real96": 18, "unseen_by_both": 6}:
        raise RuntimeError("Coverage-tier counts changed")
    if {
        yaw: sum(trial["nominal_yaw_degrees_modulo_90"] == yaw for trial in source_trials) for yaw in (0, 45)
    } != {0: 24, 45: 24}:
        raise RuntimeError("Yaw balance changed")
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
        "schema": "task1_picklift_real24_localsim_gap_full_eval48_software_gate_independent_validation_v1",
        "status": "pass",
        "evaluation_id": plan["evaluation_id"],
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "trials": 96,
        "poses": 48,
        "model_trials": dict.fromkeys(EXPECTED_MODELS, 48),
        "first_model_counts": dict.fromkeys(EXPECTED_MODELS, 24),
        "coverage_tiers": tier_counts,
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
