from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


EXPERIMENT_ID = "task1_picklift_real48_vs_real96_act_v1"
REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / EXPERIMENT_ID
ARTIFACT_ROOT = Path("/home/ubuntu24/Teleop/artifacts/training") / EXPERIMENT_ID
CONTRACT_EVIDENCE = Path("/home/ubuntu24/Teleop/artifacts/evidence") / EXPERIMENT_ID / "data_and_contract_v1"
CONDITIONS = ("real48", "real96")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        for match in re.findall(r"INFO (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ot_train\.py:", text)
    ]
    if not timestamps:
        raise RuntimeError(f"No LeRobot timestamps found in {path}")
    return {
        "started_at_local": timestamps[0].isoformat(),
        "ended_at_local": timestamps[-1].isoformat(),
        "duration_seconds": int((timestamps[-1] - timestamps[0]).total_seconds()),
    }


def checkpoint_hashes(root: Path) -> dict[str, dict]:
    result = {}
    for step in ("020000", "040000", "060000", "080000", "100000"):
        checkpoint = root / step / "pretrained_model"
        model = checkpoint / "model.safetensors"
        if not model.is_file():
            raise FileNotFoundError(model)
        result[step] = {
            "checkpoint": str(checkpoint),
            "model_sha256": sha256_file(model),
            "saved_train_config_sha256": sha256_file(checkpoint / "train_config.json"),
        }
    return result


def summarize_condition(condition: str) -> dict:
    root = ARTIFACT_ROOT / condition
    smoke_log = root / "smoke_500.log"
    full_log = root / "full_100k.log"
    if "End of training" not in smoke_log.read_text(errors="replace"):
        raise RuntimeError(f"{condition} smoke did not complete")
    if "End of training" not in full_log.read_text(errors="replace"):
        raise RuntimeError(f"{condition} full training did not complete")
    smoke_validation_path = root / "smoke_500/offline_validation.json"
    full_validation_path = root / "offline_validation_100k.json"
    smoke_validation = json.loads(smoke_validation_path.read_text())
    full_validation = json.loads(full_validation_path.read_text())
    if smoke_validation["status"] != "pass" or full_validation["status"] != "pass":
        raise RuntimeError(f"{condition} offline validation failed")
    checkpoints = checkpoint_hashes(root / "full_100k/checkpoints")
    if checkpoints["100000"]["model_sha256"] != full_validation["model_sha256"]:
        raise RuntimeError(f"{condition} selected checkpoint hash differs from validation")
    smoke_model = root / "smoke_500/checkpoints/000500/pretrained_model/model.safetensors"
    if sha256_file(smoke_model) != smoke_validation["model_sha256"]:
        raise RuntimeError(f"{condition} smoke checkpoint hash differs from validation")
    return {
        "smoke": {
            "status": "pass",
            "steps": 500,
            **log_times(smoke_log),
            "final_metrics": parse_metric(metric_lines(smoke_log)[-1]),
            "log_path": str(smoke_log),
            "log_sha256": sha256_file(smoke_log),
            "checkpoint": str(smoke_model.parent),
            "model_sha256": sha256_file(smoke_model),
            "offline_validation_path": str(smoke_validation_path),
            "offline_validation_sha256": sha256_file(smoke_validation_path),
        },
        "full_training": {
            "status": "pass",
            "steps": 100000,
            **log_times(full_log),
            "final_metrics": parse_metric(metric_lines(full_log)[-1]),
            "log_path": str(full_log),
            "log_sha256": sha256_file(full_log),
            "checkpoints": checkpoints,
            "selected_checkpoint_step": 100000,
            "selected_checkpoint": checkpoints["100000"]["checkpoint"],
            "selected_model_sha256": checkpoints["100000"]["model_sha256"],
            "processor_sha256": full_validation["files"]["policy_preprocessor"]["sha256"],
            "processor_stats_sha256": full_validation["processor_stats_sha256"],
            "initialized_from_checkpoint": None,
        },
        "offline_validation": {
            **full_validation,
            "evidence_path": str(full_validation_path),
            "evidence_sha256": sha256_file(full_validation_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched Task1 Real48/Real96 ACT training")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "run_summary.json")
    args = parser.parse_args()
    dataset_audit = json.loads((CONTRACT_EVIDENCE / "dataset_audit.json").read_text())
    contract = json.loads((CONTRACT_EVIDENCE / "training_contract_verification.json").read_text())
    runtime = json.loads((CONTRACT_EVIDENCE / "runtime_snapshot.json").read_text())
    if dataset_audit["status"] != "pass" or contract["status"] != "pass":
        raise RuntimeError("Input dataset or training contract evidence is not passing")
    conditions = {condition: summarize_condition(condition) for condition in CONDITIONS}
    result = {
        "schema": "task1_picklift_real48_vs_real96_act_run_summary_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "offline_training_and_validation_complete_no_rollout_started",
        "research_contract_commit": "73908355df1add52cd04753216c13f8b1c0b400a",
        "dataset_audit": {
            "path": str(CONTRACT_EVIDENCE / "dataset_audit.json"),
            "sha256": sha256_file(CONTRACT_EVIDENCE / "dataset_audit.json"),
            "datasets": {
                name: {
                    key: dataset_audit["datasets"][name][key]
                    for key in ("repo_id", "root", "episodes", "frames", "tree", "stats_sha256")
                }
                for name in CONDITIONS
            },
            "discard_boundary": dataset_audit["discard_boundary"],
        },
        "training_contract": {
            "path": str(CONTRACT_EVIDENCE / "training_contract_verification.json"),
            "sha256": sha256_file(CONTRACT_EVIDENCE / "training_contract_verification.json"),
            "matching_verified": True,
            "full_config_differences": contract["full_config_differences"],
        },
        "runtime": runtime,
        "conditions": conditions,
        "comparison_boundary": {
            "single_seed": 1000,
            "fixed_step": 100000,
            "checkpoint_selection_by_loss_or_result": False,
            "training_loss_is_not_policy_effect": True,
            "real_or_sim_rollout_executed": False,
            "paper_effect_conclusion": False,
        },
        "hardware_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
