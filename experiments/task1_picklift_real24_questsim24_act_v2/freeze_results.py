from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENT_ID = "task1_picklift_real24_questsim24_act_v2"
REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / EXPERIMENT_ID
ARTIFACT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/training") / EXPERIMENT_ID
DATASET_EVIDENCE = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_real24_questsim24_act_v2/combined48_freeze_v3"
)
RESULT_EVIDENCE = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_real24_questsim24_act_v2/training_result_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path)],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def hash_entry(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze controlled Mixed v2 results")
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=ARTIFACT_ROOT / "run_summary.json",
    )
    parser.add_argument("--evidence-root", type=Path, default=RESULT_EVIDENCE)
    args = parser.parse_args()

    if args.evidence_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.evidence_root}")
    result_files = [
        EXPERIMENT_ROOT / "combined48_result.json",
        EXPERIMENT_ROOT / "offline_validation_result.json",
        EXPERIMENT_ROOT / "run_result.json",
    ]
    if any(path.exists() for path in result_files):
        raise FileExistsError("Refusing to overwrite an existing checked-in result index")

    summary = json.loads(args.run_summary.read_text())
    if summary["status"] != "offline_training_and_validation_complete_no_rollout_started":
        raise RuntimeError("Run summary is not complete")
    model_sha = summary["full_training"]["checkpoints"]["100000"]
    if model_sha != summary["offline_validation"]["model_sha256"]:
        raise RuntimeError("Selected checkpoint and validation model hashes differ")

    key_paths = {
        "dataset_source_audit": DATASET_EVIDENCE / "source_compatibility_audit.json",
        "dataset_derived_manifest": DATASET_EVIDENCE / "derived_manifest.json",
        "dataset_freeze": DATASET_EVIDENCE / "freeze_manifest.json",
        "dataset_verification": DATASET_EVIDENCE / "verification.json",
        "training_contract_verification": (DATASET_EVIDENCE / "training_contract_verification.json"),
        "train_config_full": EXPERIMENT_ROOT / "train_config_full.json",
        "train_config_smoke": EXPERIMENT_ROOT / "train_config_smoke.json",
        "experiment_manifest": EXPERIMENT_ROOT / "experiment_manifest.json",
        "smoke_log": ARTIFACT_ROOT / "smoke_500.log",
        "smoke_sampling_counts": (ARTIFACT_ROOT / "smoke_500/domain_sampling_counts.json"),
        "smoke_offline_validation": (ARTIFACT_ROOT / "smoke_500/offline_validation.json"),
        "smoke_model": (ARTIFACT_ROOT / "smoke_500/checkpoints/000500/pretrained_model/model.safetensors"),
        "full_log": ARTIFACT_ROOT / "full_100k.log",
        "full_sampling_counts": (ARTIFACT_ROOT / "full_100k/domain_sampling_counts.json"),
        "full_offline_validation": ARTIFACT_ROOT / "offline_validation_100k.json",
        "full_saved_train_config": (
            ARTIFACT_ROOT / "full_100k/checkpoints/100000/pretrained_model/train_config.json"
        ),
        "full_saved_preprocessor": (
            ARTIFACT_ROOT / "full_100k/checkpoints/100000/pretrained_model/"
            "policy_preprocessor_step_3_normalizer_processor.safetensors"
        ),
    }
    for step in ("020000", "040000", "060000", "080000", "100000"):
        key_paths[f"model_{step}"] = (
            ARTIFACT_ROOT / f"full_100k/checkpoints/{step}/pretrained_model/model.safetensors"
        )
    missing = [str(path) for path in key_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing result evidence: {missing}")

    args.evidence_root.mkdir(parents=True)
    frozen_summary = args.evidence_root / "run_summary.json"
    shutil.copy2(args.run_summary, frozen_summary)
    hashes = {
        "schema": "task1_picklift_mixed_v2_training_hashes_v1",
        "entries": {name: hash_entry(path) for name, path in sorted(key_paths.items())},
        "run_summary": hash_entry(frozen_summary),
    }
    hashes_path = args.evidence_root / "hashes.json"
    write_json(hashes_path, hashes)
    freeze_manifest = {
        "schema": "task1_picklift_mixed_v2_training_result_freeze_v1",
        "freeze_id": "task1_picklift_real24_questsim24_act_v2_training_result_v1",
        "status": "frozen_offline_training_and_validation_complete",
        "experiment_id": EXPERIMENT_ID,
        "prepared_training_code_commit": git_last_commit(EXPERIMENT_ROOT / "train_config_full.json"),
        "dataset_tree_sha256": summary["dataset"]["tree_sha256"],
        "selected_checkpoint_step": 100000,
        "selected_model_sha256": model_sha,
        "hashes_path": str(hashes_path),
        "hashes_sha256": sha256_file(hashes_path),
        "run_summary_path": str(frozen_summary),
        "run_summary_sha256": sha256_file(frozen_summary),
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "boundaries": summary["boundaries"],
    }
    freeze_path = args.evidence_root / "manifest.json"
    write_json(freeze_path, freeze_manifest)

    dataset_result = {
        "schema": "task1_picklift_mixed_v2_dataset_result_index_v1",
        "status": "frozen_official_loader_pass",
        "dataset": summary["dataset"],
        "config_contract": summary["config_contract"],
        "evidence_root": str(DATASET_EVIDENCE),
        "freeze_manifest_sha256": sha256_file(DATASET_EVIDENCE / "freeze_manifest.json"),
    }
    offline_result = {
        "schema": "task1_picklift_mixed_v2_offline_validation_index_v1",
        **summary["offline_validation"],
    }
    run_result = {
        "schema": "task1_picklift_mixed_v2_run_result_index_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": summary["status"],
        "training_config_commit": summary["training_config_commit"],
        "dataset": {
            "repo_id": summary["dataset"]["repo_id"],
            "root": summary["dataset"]["root"],
            "tree_sha256": summary["dataset"]["tree_sha256"],
            "episodes": summary["dataset"]["episodes"],
            "frames": summary["dataset"]["frames"],
            "domain_episode_counts": summary["dataset"]["domain_episode_counts"],
            "domain_frame_counts": summary["dataset"]["domain_frame_counts"],
        },
        "controlled_corrections": summary["config_contract"]["verification"]["controlled_corrections"],
        "smoke": summary["smoke"],
        "full_training": summary["full_training"],
        "offline_validation": summary["offline_validation"],
        "control_models": summary["control_models"],
        "boundaries": summary["boundaries"],
        "interpretation": summary["interpretation"],
        "artifact_summary": hash_entry(frozen_summary),
        "result_evidence": {
            "root": str(args.evidence_root),
            "manifest_path": str(freeze_path),
            "manifest_sha256": sha256_file(freeze_path),
            "hashes_path": str(hashes_path),
            "hashes_sha256": sha256_file(hashes_path),
        },
    }
    write_json(EXPERIMENT_ROOT / "combined48_result.json", dataset_result)
    write_json(EXPERIMENT_ROOT / "offline_validation_result.json", offline_result)
    write_json(EXPERIMENT_ROOT / "run_result.json", run_result)
    print(json.dumps(run_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
