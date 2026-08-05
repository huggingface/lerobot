from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENT_ID = "task1_picklift_real24_localsim48_gap_recovery_act_v1"
REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / EXPERIMENT_ID
BINDINGS_PATH = EXPERIMENT_ROOT / "source_bindings.json"
RESULT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/evidence") / EXPERIMENT_ID / "training_result_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hash_entry(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def last_metric(log_path: Path) -> dict:
    lines = [
        line
        for line in log_path.read_text(errors="replace").replace("\r", "\n").splitlines()
        if re.search(r"ot_train\.py:\d+ step:", line)
    ]
    if not lines:
        raise RuntimeError(f"No metric line in {log_path}")
    match = re.search(r"ot_train\.py:\d+\s+(.*)$", lines[-1])
    fields = dict(re.findall(r"([A-Za-z0-9_/]+):(\S+)", match.group(1)))
    return {
        "step": fields["step"],
        "loss": float(fields["loss"]),
        "l1_loss": float(fields["l1_loss"]),
        "kld_loss": float(fields["kld_loss"]),
        "gradient_norm": float(fields["grdn"]),
        "samples_per_second": float(fields["smp/s"]),
        "gpu_memory_gb": float(fields["mem_gb"]),
    }


def verify_saved_training_contract(
    condition_id: str,
    expected_config_path: Path,
    checkpoint: Path,
) -> dict:
    expected = json.loads(expected_config_path.read_text(encoding="utf-8"))
    saved_path = checkpoint / "train_config.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    checks = {
        "dataset.repo_id": (saved["dataset"]["repo_id"], expected["dataset"]["repo_id"]),
        "dataset.root": (saved["dataset"]["root"], expected["dataset"]["root"]),
        "dataset.domain_balanced_episode_groups": (
            saved["dataset"]["domain_balanced_episode_groups"],
            expected["dataset"]["domain_balanced_episode_groups"],
        ),
        "dataset.use_imagenet_stats": (
            saved["dataset"]["use_imagenet_stats"],
            expected["dataset"]["use_imagenet_stats"],
        ),
        "seed": (saved["seed"], 1000),
        "steps": (saved["steps"], 100000),
        "batch_size": (saved["batch_size"], 8),
        "resume": (saved["resume"], False),
        "output_dir": (saved["output_dir"], expected["output_dir"]),
        "policy.type": (saved["policy"]["type"], "act"),
        "policy.chunk_size": (saved["policy"]["chunk_size"], 67),
        "policy.n_action_steps": (saved["policy"]["n_action_steps"], 67),
        "policy.pretrained_path": (saved["policy"]["pretrained_path"], None),
    }
    mismatches = {
        key: {"actual": actual, "expected": wanted}
        for key, (actual, wanted) in checks.items()
        if actual != wanted
    }
    if mismatches:
        raise RuntimeError(f"Saved training contract mismatch for {condition_id}: {mismatches}")
    return hash_entry(saved_path)


def condition_summary(condition_id: str, condition: dict) -> dict:
    training_root = Path(condition["training_root"])
    smoke_root = training_root / "smoke_500"
    full_root = training_root / "full_100k"
    smoke_log = training_root / "smoke_500.log"
    full_log = training_root / "full_100k.log"
    if "End of training" not in smoke_log.read_text(errors="replace"):
        raise RuntimeError(f"Smoke did not finish for {condition_id}")
    if "End of training" not in full_log.read_text(errors="replace"):
        raise RuntimeError(f"100k did not finish for {condition_id}")
    smoke_validation_path = smoke_root / "offline_validation.json"
    full_validation_path = training_root / "offline_validation_100k.json"
    smoke_validation = json.loads(smoke_validation_path.read_text(encoding="utf-8"))
    full_validation = json.loads(full_validation_path.read_text(encoding="utf-8"))
    if smoke_validation["status"] != "pass" or full_validation["status"] != "pass":
        raise RuntimeError(f"Offline validation failed for {condition_id}")
    for label, validation in (("smoke", smoke_validation), ("full", full_validation)):
        if not validation["all_outputs_shape_1x6_and_finite"]:
            raise RuntimeError(f"{label} output validation failed for {condition_id}")
        if [sample["domain"] for sample in validation["samples"]] != [
            "real",
            "simulation",
        ] or not all(sample["output_finite"] for sample in validation["samples"]):
            raise RuntimeError(f"{label} domain validation mismatch for {condition_id}")
    smoke_counts_path = smoke_root / "domain_sampling_counts.json"
    full_counts_path = full_root / "domain_sampling_counts.json"
    smoke_counts = json.loads(smoke_counts_path.read_text(encoding="utf-8"))
    full_counts = json.loads(full_counts_path.read_text(encoding="utf-8"))
    if smoke_counts["actual_samples_seen_by_main_process"] != {
        "real": 2000,
        "simulation": 2000,
    }:
        raise RuntimeError(f"Smoke domain counts mismatch for {condition_id}")
    if full_counts["actual_samples_seen_by_main_process"] != {
        "real": 400000,
        "simulation": 400000,
    }:
        raise RuntimeError(f"100k domain counts mismatch for {condition_id}")
    checkpoints = {}
    for step in (20000, 40000, 60000, 80000, 100000):
        model = full_root / f"checkpoints/{step:06d}/pretrained_model/model.safetensors"
        checkpoints[f"{step:06d}"] = sha256_file(model)
    selected = checkpoints["100000"]
    if full_validation["model_sha256"] != selected:
        raise RuntimeError(f"100k model/validation SHA mismatch for {condition_id}")
    dataset_freeze = Path(condition["evidence_root"]) / "freeze_manifest.json"
    contract_verification = Path(condition["evidence_root"]) / "training_contract_verification.json"
    config_smoke = EXPERIMENT_ROOT / "configs" / f"{condition_id}_smoke.json"
    config_full = EXPERIMENT_ROOT / "configs" / f"{condition_id}_full.json"
    selected_checkpoint = full_root / "checkpoints/100000/pretrained_model"
    saved_train_config = verify_saved_training_contract(condition_id, config_full, selected_checkpoint)
    return {
        "condition": condition_id,
        "status": "offline_training_and_validation_complete_no_rollout_started",
        "dataset": json.loads(dataset_freeze.read_text(encoding="utf-8")),
        "dataset_freeze_sha256": sha256_file(dataset_freeze),
        "training_contract_sha256": sha256_file(contract_verification),
        "configs": {
            "smoke": hash_entry(config_smoke),
            "full": hash_entry(config_full),
        },
        "smoke": {
            "steps": 500,
            "status": "pass",
            "metrics": last_metric(smoke_log),
            "model_sha256": smoke_validation["model_sha256"],
            "checkpoint": str(smoke_root / "checkpoints/000500/pretrained_model"),
            "log": hash_entry(smoke_log),
            "domain_counts": smoke_counts,
            "domain_counts_file": hash_entry(smoke_counts_path),
            "offline_validation": hash_entry(smoke_validation_path),
        },
        "full_training": {
            "steps": 100000,
            "status": "pass",
            "metrics": last_metric(full_log),
            "checkpoints": checkpoints,
            "selected_checkpoint_step": 100000,
            "selected_checkpoint": str(selected_checkpoint),
            "selected_model_sha256": selected,
            "saved_train_config": saved_train_config,
            "log": hash_entry(full_log),
            "domain_counts": full_counts,
            "domain_counts_file": hash_entry(full_counts_path),
            "offline_validation": hash_entry(full_validation_path),
            "processor_sha256": full_validation["processor_sha256"],
            "cuda_reload_finite_real_and_sim": True,
        },
    }


def main() -> None:
    if RESULT_ROOT.exists():
        raise FileExistsError(f"Refusing existing result root: {RESULT_ROOT}")
    if (EXPERIMENT_ROOT / "result_index.json").exists():
        raise FileExistsError("Refusing existing result index")
    bindings = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))
    summaries = {
        condition_id: condition_summary(condition_id, condition)
        for condition_id, condition in bindings["conditions"].items()
    }
    RESULT_ROOT.mkdir(parents=True)
    for condition_id, summary in summaries.items():
        write_json(RESULT_ROOT / f"condition_{condition_id}.json", summary)
    pair_summary = {
        "schema": "task1_picklift_real24_localsim_gap_full_act_pair_result_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "both_models_offline_training_and_validation_complete",
        "execution_order": ["C", "D"],
        "source_bindings_sha256": sha256_file(BINDINGS_PATH),
        "postcollection_result_sha256": bindings["postcollection_finalization"]["result_manifest_sha256"],
        "conditions": summaries,
        "boundaries": {
            "hardware_accessed": False,
            "serial_accessed": False,
            "camera_hardware_accessed": False,
            "robot_actions_sent": False,
            "simulation_rollout_started": False,
            "real_evaluation_started": False,
            "push_performed": False,
            "paper_result": False,
        },
        "next_gate": "paired real Eval48 software preparation and explicit hardware GO",
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    pair_summary_path = RESULT_ROOT / "run_summary.json"
    write_json(pair_summary_path, pair_summary)
    entries = {
        "source_bindings": hash_entry(BINDINGS_PATH),
        "postcollection_result": hash_entry(EXPERIMENT_ROOT / "localsim_finalization_result.json"),
        "materializer": hash_entry(EXPERIMENT_ROOT / "materialize_pair.py"),
        "config_builder": hash_entry(EXPERIMENT_ROOT / "build_and_verify_configs.py"),
        "checkpoint_validator": hash_entry(EXPERIMENT_ROOT / "validate_checkpoint.py"),
        "finalizer": hash_entry(EXPERIMENT_ROOT / "finalize_pair.py"),
        "result_verifier": hash_entry(EXPERIMENT_ROOT / "verify_final_result.py"),
        "condition_C": hash_entry(RESULT_ROOT / "condition_C.json"),
        "condition_D": hash_entry(RESULT_ROOT / "condition_D.json"),
        "run_summary": hash_entry(pair_summary_path),
    }
    hashes_path = RESULT_ROOT / "hashes.json"
    write_json(hashes_path, {"schema": "task1_act_pair_result_hashes_v1", "entries": entries})
    manifest = {
        "schema": "task1_picklift_real24_localsim_gap_full_act_pair_freeze_v1",
        "status": "frozen_both_models_offline_complete",
        "experiment_id": EXPERIMENT_ID,
        "run_summary_sha256": sha256_file(pair_summary_path),
        "hashes_sha256": sha256_file(hashes_path),
        "condition_C_model_sha256": summaries["C"]["full_training"]["selected_model_sha256"],
        "condition_D_model_sha256": summaries["D"]["full_training"]["selected_model_sha256"],
        "hardware_accessed": False,
        "rollout_started": False,
    }
    manifest_path = RESULT_ROOT / "manifest.json"
    write_json(manifest_path, manifest)
    result_index = {
        **manifest,
        "result_root": str(RESULT_ROOT),
        "manifest_sha256": sha256_file(manifest_path),
        "run_summary_path": str(pair_summary_path),
        "next_gate": pair_summary["next_gate"],
    }
    write_json(EXPERIMENT_ROOT / "result_index.json", result_index)
    print(json.dumps(result_index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
