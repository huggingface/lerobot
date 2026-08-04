from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


EXPERIMENT_ID = "task1_picklift_real24_budget_extension_act_v1"
REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / EXPERIMENT_ID
ARTIFACT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/training") / EXPERIMENT_ID
CONTRACT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/evidence") / EXPERIMENT_ID / "data_and_contract_v1"
RESULT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/evidence") / EXPERIMENT_ID / "training_result_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if RESULT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {RESULT_ROOT}")
    index_path = EXPERIMENT_ROOT / "run_result.json"
    if index_path.exists():
        raise FileExistsError(f"Refusing to overwrite {index_path}")

    checkpoint_root = ARTIFACT_ROOT / "full_100k/checkpoints"
    validation = json.loads((ARTIFACT_ROOT / "offline_validation_100k.json").read_text())
    smoke_validation = json.loads((ARTIFACT_ROOT / "smoke_500/offline_validation.json").read_text())
    audit = json.loads((CONTRACT_ROOT / "dataset_audit.json").read_text())
    contract = json.loads((CONTRACT_ROOT / "training_contract_verification.json").read_text())
    paths = {
        "experiment_manifest": EXPERIMENT_ROOT / "experiment_manifest.json",
        "train_config_smoke": EXPERIMENT_ROOT / "train_config_smoke.json",
        "train_config_full": EXPERIMENT_ROOT / "train_config_full.json",
        "dataset_audit": CONTRACT_ROOT / "dataset_audit.json",
        "training_contract": CONTRACT_ROOT / "training_contract_verification.json",
        "runtime_snapshot": CONTRACT_ROOT / "runtime_snapshot.json",
        "smoke_log": ARTIFACT_ROOT / "smoke_500.log",
        "smoke_validation": ARTIFACT_ROOT / "smoke_500/offline_validation.json",
        "full_log": ARTIFACT_ROOT / "full_100k.log",
        "full_validation": ARTIFACT_ROOT / "offline_validation_100k.json",
        "selected_train_config": checkpoint_root / "100000/pretrained_model/train_config.json",
        "selected_processor_stats": checkpoint_root / "100000/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors",
    }
    for step in ("020000", "040000", "060000", "080000", "100000"):
        paths[f"model_{step}"] = checkpoint_root / step / "pretrained_model/model.safetensors"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if validation["status"] != "pass" or validation["model_sha256"] != sha256_file(paths["model_100000"]):
        raise RuntimeError("Selected checkpoint validation/hash mismatch")
    if smoke_validation["status"] != "pass" or audit["status"] != "pass" or contract["status"] != "pass":
        raise RuntimeError("Prerequisite offline evidence is incomplete")

    checkpoint_hashes = {
        step: sha256_file(checkpoint_root / step / "pretrained_model/model.safetensors")
        for step in ("020000", "040000", "060000", "080000", "100000")
    }
    summary = {
        "schema": "task1_picklift_real24_budget_extension_act_run_summary_v1",
        "status": "offline_training_and_validation_complete_no_rollout_started",
        "experiment_id": EXPERIMENT_ID,
        "research_contract_commit": "428bdcda4506ce8024925cb27e8f151d967da0f4",
        "dataset": {
            "root": audit["datasets"]["real24"]["root"],
            "repo_id": audit["datasets"]["real24"]["repo_id"],
            "episodes": 24,
            "frames": 4263,
            "tree_sha256": audit["datasets"]["real24"]["tree"]["tree_sha256"],
            "stats_sha256": audit["datasets"]["real24"]["stats_sha256"],
        },
        "runtime": contract["runtime"],
        "smoke": {
            "steps": 500,
            "duration_wall_seconds": 34.73,
            "final_loss": 2.504,
            "model_sha256": smoke_validation["model_sha256"],
            "offline_validation": "pass",
        },
        "full_training": {
            "steps": 100000,
            "duration_wall_seconds": 5043,
            "final_loss": 0.029,
            "final_l1_loss": 0.029,
            "final_kld_loss": 0.0,
            "checkpoint_hashes": checkpoint_hashes,
            "selected_step": 100000,
            "selected_checkpoint": str(checkpoint_root / "100000/pretrained_model"),
            "selected_model_sha256": validation["model_sha256"],
            "processor_stats_sha256": validation["processor_stats_sha256"],
            "full_log_sha256": sha256_file(paths["full_log"]),
        },
        "offline_validation": validation,
        "boundaries": {
            "hardware_accessed": False,
            "rollout_executed": False,
            "paused_eval_next_cursor": "t050_p25_real96",
            "paper_effect_conclusion": False,
        },
    }

    RESULT_ROOT.mkdir(parents=True)
    summary_path = RESULT_ROOT / "run_summary.json"
    write_json(summary_path, summary)
    hashes = {
        "schema": "task1_picklift_real24_budget_extension_act_hashes_v1",
        "entries": {name: entry(path) for name, path in sorted(paths.items())},
        "run_summary": entry(summary_path),
    }
    hashes_path = RESULT_ROOT / "hashes.json"
    write_json(hashes_path, hashes)
    manifest = {
        "schema": "task1_picklift_real24_budget_extension_act_training_result_freeze_v1",
        "freeze_id": f"{EXPERIMENT_ID}_training_result_v1",
        "status": "frozen_offline_training_and_validation_complete",
        "selected_checkpoint_step": 100000,
        "selected_model_sha256": validation["model_sha256"],
        "run_summary_sha256": sha256_file(summary_path),
        "hashes_sha256": sha256_file(hashes_path),
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "hardware_accessed": False,
        "rollout_executed": False,
    }
    manifest_path = RESULT_ROOT / "manifest.json"
    write_json(manifest_path, manifest)
    index = {
        "schema": "task1_picklift_real24_budget_extension_act_result_index_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": summary["status"],
        "dataset": summary["dataset"],
        "smoke": summary["smoke"],
        "full_training": summary["full_training"],
        "offline_validation": {
            "status": validation["status"],
            "input_shapes": validation["input_shapes"],
            "output_shape": validation["output_shape"],
            "output_finite": validation["output_finite"],
        },
        "evidence": {
            "root": str(RESULT_ROOT),
            "manifest_sha256": sha256_file(manifest_path),
            "hashes_sha256": sha256_file(hashes_path),
            "run_summary_sha256": sha256_file(summary_path),
        },
        "hardware_accessed": False,
        "rollout_executed": False,
    }
    write_json(index_path, index)
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
