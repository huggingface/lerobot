from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPERIMENT_ID = "task1_picklift_real24_localsim48_gap_recovery_act_v1"
RESULT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/evidence") / EXPERIMENT_ID / "training_result_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    hashes = json.loads((RESULT_ROOT / "hashes.json").read_text(encoding="utf-8"))
    checked = 0
    for entry in hashes["entries"].values():
        path = Path(entry["path"])
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Frozen evidence mismatch: {path}")
        checked += 1
    manifest = json.loads((RESULT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULT_ROOT / "run_summary.json").read_text(encoding="utf-8"))
    if manifest["status"] != "frozen_both_models_offline_complete":
        raise RuntimeError("Pair result manifest is not complete")
    if summary["status"] != "both_models_offline_training_and_validation_complete":
        raise RuntimeError("Pair run summary is not complete")
    for condition_id in ("C", "D"):
        condition = summary["conditions"][condition_id]
        if condition["status"] != "offline_training_and_validation_complete_no_rollout_started":
            raise RuntimeError(f"Condition {condition_id} is incomplete")
        if condition["full_training"]["domain_counts"]["actual_samples_seen_by_main_process"] != {
            "real": 400000,
            "simulation": 400000,
        }:
            raise RuntimeError(f"Condition {condition_id} full sampling count mismatch")
        checkpoint = Path(condition["full_training"]["selected_checkpoint"])
        if (
            sha256_file(checkpoint / "model.safetensors")
            != condition["full_training"]["selected_model_sha256"]
        ):
            raise RuntimeError(f"Condition {condition_id} selected model SHA mismatch")
    result = {
        "status": "pass",
        "checked_hash_entries": checked,
        "conditions": ["C", "D"],
        "both_selected_checkpoints_match": True,
        "both_domain_sampling_counts_match": True,
        "hardware_accessed": False,
        "rollout_started": False,
    }
    output = RESULT_ROOT / "independent_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
