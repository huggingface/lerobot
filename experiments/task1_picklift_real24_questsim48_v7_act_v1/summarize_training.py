from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

import lerobot

EXPERIMENT_ID = "task1_picklift_real24_questsim48_v7_act_v1"
REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / EXPERIMENT_ID
ARTIFACT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/training") / EXPERIMENT_ID
DATASET_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/datasets/"
    "task1_picklift_real24_questsim48_v7_act_v1/combined72_v1"
)
DATASET_EVIDENCE = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/"
    "task1_picklift_real24_questsim48_v7_act_v1/combined72_freeze_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_last_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path)],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def metric_lines(path: Path) -> list[str]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    return [line for line in text.splitlines() if re.search(r"ot_train\.py:\d+ step:", line)]


def parse_metric(line: str) -> dict:
    match = re.search(r"ot_train\.py:\d+\s+(.*)$", line)
    if not match:
        raise RuntimeError(f"Cannot parse metric line: {line}")
    fields = dict(re.findall(r"([A-Za-z0-9_/]+):(\S+)", match.group(1)))
    return {
        "reported_step": fields["step"],
        "samples": fields["smpl"],
        "episodes_sampled": fields["ep"],
        "epochs": float(fields["epch"]),
        "loss": float(fields["loss"]),
        "gradient_norm": float(fields["grdn"]),
        "learning_rate": float(fields["lr"]),
        "update_seconds": float(fields["updt_s"]),
        "data_seconds": float(fields["data_s"]),
        "samples_per_second": float(fields["smp/s"]),
        "gpu_memory_gb": float(fields["mem_gb"]),
        "l1_loss": float(fields["l1_loss"]),
        "kld_loss": float(fields["kld_loss"]),
    }


def log_times(path: Path) -> dict:
    text = path.read_text(errors="replace").replace("\r", "\n")
    timestamps = [
        datetime.strptime(match, "%Y-%m-%d %H:%M:%S")
        for match in re.findall(
            r"INFO (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ot_train\.py:",
            text,
        )
    ]
    if not timestamps:
        raise RuntimeError(f"No training timestamps found in {path}")
    return {
        "started_at_local": timestamps[0].isoformat(),
        "ended_at_local": timestamps[-1].isoformat(),
        "duration_seconds": int((timestamps[-1] - timestamps[0]).total_seconds()),
    }


def checkpoint_hashes(checkpoints_root: Path) -> dict[str, str]:
    result = {}
    for step in ("020000", "040000", "060000", "080000", "100000"):
        model = checkpoints_root / step / "pretrained_model/model.safetensors"
        if not model.exists():
            raise FileNotFoundError(model)
        result[step] = sha256_file(model)
    return result


def checked_sampling_counts(path: Path, steps: int) -> dict:
    result = json.loads(path.read_text())
    expected = steps * 4
    counts = result["actual_samples_seen_by_main_process"]
    if counts != {"real": expected, "simulation": expected}:
        raise RuntimeError(f"Unexpected domain counts in {path}: {counts}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize controlled Real24 + Sim48-v7 ACT training")
    parser.add_argument(
        "--offline-validation",
        type=Path,
        default=ARTIFACT_ROOT / "offline_validation_100k.json",
    )
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "run_summary.json")
    args = parser.parse_args()

    smoke_log = ARTIFACT_ROOT / "smoke_500.log"
    full_log = ARTIFACT_ROOT / "full_100k.log"
    full_output = ARTIFACT_ROOT / "full_100k"
    full_checkpoint = full_output / "checkpoints/100000/pretrained_model"
    if "End of training" not in smoke_log.read_text(errors="replace"):
        raise RuntimeError("Smoke log has not reached End of training")
    if "End of training" not in full_log.read_text(errors="replace"):
        raise RuntimeError("Full training log has not reached End of training")
    validation = json.loads(args.offline_validation.read_text())
    if validation["status"] != "pass":
        raise RuntimeError("Offline checkpoint validation did not pass")
    if set(validation["domains_validated"]) != {"real", "simulation"}:
        raise RuntimeError("Offline validation did not cover both domains")

    dataset_freeze = json.loads((DATASET_EVIDENCE / "freeze_manifest.json").read_text())
    dataset_verification = json.loads((DATASET_EVIDENCE / "verification.json").read_text())
    source_audit = json.loads((DATASET_EVIDENCE / "source_compatibility_audit.json").read_text())
    training_contract = json.loads((DATASET_EVIDENCE / "training_contract_verification.json").read_text())
    smoke_counts_path = ARTIFACT_ROOT / "smoke_500/domain_sampling_counts.json"
    full_counts_path = ARTIFACT_ROOT / "full_100k/domain_sampling_counts.json"
    smoke_counts = checked_sampling_counts(smoke_counts_path, 500)
    full_counts = checked_sampling_counts(full_counts_path, 100000)

    result = {
        "schema": "task1_picklift_real24_questsim48_v7_act_run_result_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "offline_training_and_validation_complete_no_rollout_started",
        "training_config_commit": git_last_commit(EXPERIMENT_ROOT / "train_config_full.json"),
        "dataset": {
            "repo_id": dataset_freeze["repo_id"],
            "root": str(DATASET_ROOT),
            "freeze_id": dataset_freeze["freeze_id"],
            "tree_sha256": dataset_freeze["tree_sha256"],
            "episodes": 72,
            "frames": 10147,
            "domain_episode_counts": {
                "real": 24,
                "quest_remote_mujoco_sim48_v7": 48,
            },
            "domain_frame_counts": {
                "real": 3790,
                "quest_remote_mujoco_sim48_v7": 6357,
            },
            "official_loader_and_provenance_verification": (dataset_verification["status"]),
            "state_action_identity_against_sources": dataset_verification["derived"][
                "state_action_identity_against_sources"
            ],
            "gripper_semantics": source_audit["gripper_semantics"],
            "verification_sha256": sha256_file(DATASET_EVIDENCE / "verification.json"),
        },
        "config_contract": {
            "verification": training_contract,
            "verification_sha256": sha256_file(DATASET_EVIDENCE / "training_contract_verification.json"),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "lerobot": lerobot.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "smoke": {
            "steps": 500,
            "status": "pass",
            **log_times(smoke_log),
            "final_metrics": parse_metric(metric_lines(smoke_log)[-1]),
            "checkpoint_path": str(ARTIFACT_ROOT / "smoke_500/checkpoints/000500/pretrained_model"),
            "model_sha256": sha256_file(
                ARTIFACT_ROOT / "smoke_500/checkpoints/000500/pretrained_model/model.safetensors"
            ),
            "log_path": str(smoke_log),
            "log_sha256": sha256_file(smoke_log),
            "domain_sampling": smoke_counts,
            "domain_sampling_sha256": sha256_file(smoke_counts_path),
            "offline_validation_path": str(ARTIFACT_ROOT / "smoke_500/offline_validation.json"),
            "offline_validation_sha256": sha256_file(ARTIFACT_ROOT / "smoke_500/offline_validation.json"),
        },
        "full_training": {
            "steps": 100000,
            "status": "pass",
            **log_times(full_log),
            "final_metrics": parse_metric(metric_lines(full_log)[-1]),
            "log_path": str(full_log),
            "log_sha256": sha256_file(full_log),
            "domain_sampling": full_counts,
            "domain_sampling_sha256": sha256_file(full_counts_path),
            "checkpoints": checkpoint_hashes(full_output / "checkpoints"),
            "checkpoint_hash_field": "SHA-256 of pretrained_model/model.safetensors",
            "selected_checkpoint_step": 100000,
            "selected_checkpoint_path": str(full_checkpoint),
            "saved_train_config_sha256": sha256_file(full_checkpoint / "train_config.json"),
            "initialized_from_policy_checkpoint": None,
        },
        "offline_validation": {
            **validation,
            "evidence_path": str(args.offline_validation),
            "evidence_sha256": sha256_file(args.offline_validation),
        },
        "control_models": {
            "real24_only_sha256": ("ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"),
            "mixed_v1_sha256": ("e054e682057f09a4653af00a4580da173d3d1658ef5c34244bdbf3ca1a125de5"),
            "mixed_v2_sha256": ("b7faae880393bdbf5e44ebeaab1f399f732d6ee325be698f999c90eb865cee68"),
        },
        "boundaries": {
            "hardware_accessed": False,
            "serial_accessed": False,
            "camera_hardware_accessed": False,
            "robot_actions_sent": False,
            "simulation_rollout_started": False,
            "real_evaluation_started": False,
            "trained_or_finetuned_from_old_checkpoint": False,
            "push_performed": False,
        },
        "interpretation": (
            "Offline engineering baseline only. Sim48-v7 replaces prior simulation "
            "data under the fixed Real24-only ACT recipe and the controlled Mixed-v2 "
            "sampler contract. No simulation or real-robot performance conclusion "
            "is made by this run."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
