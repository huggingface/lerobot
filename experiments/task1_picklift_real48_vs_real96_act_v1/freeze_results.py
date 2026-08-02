from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


EXPERIMENT_ID = "task1_picklift_real48_vs_real96_act_v1"
REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / EXPERIMENT_ID
ARTIFACT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/training") / EXPERIMENT_ID
CONTRACT_EVIDENCE = Path("/home/ubuntu24/Teleop/artifacts/evidence") / EXPERIMENT_ID / "data_and_contract_v1"
RESULT_EVIDENCE = Path("/home/ubuntu24/Teleop/artifacts/evidence") / EXPERIMENT_ID / "training_result_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def hash_entry(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze matched Task1 Real48/Real96 ACT results")
    parser.add_argument("--run-summary", type=Path, default=ARTIFACT_ROOT / "run_summary.json")
    parser.add_argument("--evidence-root", type=Path, default=RESULT_EVIDENCE)
    args = parser.parse_args()
    if args.evidence_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.evidence_root}")
    result_indices = [EXPERIMENT_ROOT / "run_result.json", EXPERIMENT_ROOT / "offline_validation_result.json"]
    if any(path.exists() for path in result_indices):
        raise FileExistsError("Refusing to overwrite checked-in result indices")
    summary = json.loads(args.run_summary.read_text())
    if summary["status"] != "offline_training_and_validation_complete_no_rollout_started":
        raise RuntimeError("Run summary is incomplete")

    paths = {
        "experiment_manifest": EXPERIMENT_ROOT / "experiment_manifest.json",
        "dataset_audit": CONTRACT_EVIDENCE / "dataset_audit.json",
        "training_contract": CONTRACT_EVIDENCE / "training_contract_verification.json",
        "runtime_snapshot": CONTRACT_EVIDENCE / "runtime_snapshot.json",
        "result_independent_verification": CONTRACT_EVIDENCE / "result_independent_verification.json",
    }
    for condition in ("real48", "real96"):
        paths[f"{condition}_config_smoke"] = EXPERIMENT_ROOT / f"{condition}_train_config_smoke.json"
        paths[f"{condition}_config_full"] = EXPERIMENT_ROOT / f"{condition}_train_config_full.json"
        paths[f"{condition}_smoke_log"] = ARTIFACT_ROOT / condition / "smoke_500.log"
        paths[f"{condition}_smoke_validation"] = ARTIFACT_ROOT / condition / "smoke_500/offline_validation.json"
        paths[f"{condition}_full_log"] = ARTIFACT_ROOT / condition / "full_100k.log"
        paths[f"{condition}_full_validation"] = ARTIFACT_ROOT / condition / "offline_validation_100k.json"
        checkpoint_root = ARTIFACT_ROOT / condition / "full_100k/checkpoints"
        for step in ("020000", "040000", "060000", "080000", "100000"):
            paths[f"{condition}_model_{step}"] = checkpoint_root / step / "pretrained_model/model.safetensors"
        selected = checkpoint_root / "100000/pretrained_model"
        paths[f"{condition}_selected_train_config"] = selected / "train_config.json"
        paths[f"{condition}_selected_preprocessor"] = selected / "policy_preprocessor.json"
        paths[f"{condition}_selected_processor_stats"] = selected / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing result evidence: {missing}")

    args.evidence_root.mkdir(parents=True)
    frozen_summary = args.evidence_root / "run_summary.json"
    shutil.copy2(args.run_summary, frozen_summary)
    hashes = {
        "schema": "task1_picklift_real48_vs_real96_act_hashes_v1",
        "entries": {name: hash_entry(path) for name, path in sorted(paths.items())},
        "run_summary": hash_entry(frozen_summary),
    }
    hashes_path = args.evidence_root / "hashes.json"
    write_json(hashes_path, hashes)
    freeze_manifest = {
        "schema": "task1_picklift_real48_vs_real96_act_training_result_freeze_v1",
        "freeze_id": "task1_picklift_real48_vs_real96_act_v1_training_result_v1",
        "status": "frozen_offline_training_and_validation_complete",
        "experiment_id": EXPERIMENT_ID,
        "research_contract_commit": "73908355df1add52cd04753216c13f8b1c0b400a",
        "selected_checkpoint_step": 100000,
        "selected_models": {
            condition: summary["conditions"][condition]["full_training"]["selected_model_sha256"]
            for condition in ("real48", "real96")
        },
        "hashes_path": str(hashes_path),
        "hashes_sha256": sha256_file(hashes_path),
        "run_summary_path": str(frozen_summary),
        "run_summary_sha256": sha256_file(frozen_summary),
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "hardware_accessed": False,
        "rollout_executed": False,
    }
    freeze_path = args.evidence_root / "manifest.json"
    write_json(freeze_path, freeze_manifest)

    run_result = {
        "schema": "task1_picklift_real48_vs_real96_act_run_result_index_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": summary["status"],
        "research_contract_commit": summary["research_contract_commit"],
        "datasets": summary["dataset_audit"],
        "training_contract": summary["training_contract"],
        "conditions": summary["conditions"],
        "comparison_boundary": summary["comparison_boundary"],
        "result_evidence": {
            "root": str(args.evidence_root),
            "manifest_sha256": sha256_file(freeze_path),
            "hashes_sha256": sha256_file(hashes_path),
            "run_summary_sha256": sha256_file(frozen_summary),
        },
    }
    offline_result = {
        "schema": "task1_picklift_real48_vs_real96_act_offline_validation_index_v1",
        "real48": summary["conditions"]["real48"]["offline_validation"],
        "real96": summary["conditions"]["real96"]["offline_validation"],
        "hardware_accessed": False,
    }
    write_json(EXPERIMENT_ROOT / "run_result.json", run_result)
    write_json(EXPERIMENT_ROOT / "offline_validation_result.json", offline_result)
    print(json.dumps(run_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
