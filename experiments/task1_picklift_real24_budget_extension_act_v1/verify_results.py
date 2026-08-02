from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPERIMENT_ID = "task1_picklift_real24_budget_extension_act_v1"
RESULT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/evidence") / EXPERIMENT_ID / "training_result_v1"
INDEX = Path("/home/ubuntu24/Teleop/lerobot/experiments") / EXPERIMENT_ID / "run_result.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_path = RESULT_ROOT / "manifest.json"
    hashes_path = RESULT_ROOT / "hashes.json"
    summary_path = RESULT_ROOT / "run_summary.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = json.loads(hashes_path.read_text())
    summary = json.loads(summary_path.read_text())
    index = json.loads(INDEX.read_text())
    for name, item in hashes["entries"].items():
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Hash mismatch: {name}")
    if sha256_file(summary_path) != manifest["run_summary_sha256"]:
        raise RuntimeError("Summary hash mismatch")
    if sha256_file(hashes_path) != manifest["hashes_sha256"]:
        raise RuntimeError("Hashes index mismatch")
    if index["full_training"]["selected_model_sha256"] != summary["full_training"]["selected_model_sha256"]:
        raise RuntimeError("Selected model identity mismatch")
    if summary["status"] != "offline_training_and_validation_complete_no_rollout_started":
        raise RuntimeError("Summary status mismatch")
    print(json.dumps({
        "schema": "task1_picklift_real24_budget_extension_act_independent_verification_v1",
        "status": "pass",
        "verified_entries": len(hashes["entries"]),
        "selected_model_sha256": summary["full_training"]["selected_model_sha256"],
        "hardware_accessed": False,
        "rollout_executed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
