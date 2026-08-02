from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from copy import deepcopy
from pathlib import Path

import torch
import torchvision
from torchvision.models import ResNet18_Weights

import lerobot
from lerobot.optim import AdamWConfig


REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments/task1_picklift_real48_vs_real96_act_v1"
EVIDENCE_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/"
    "task1_picklift_real48_vs_real96_act_v1/data_and_contract_v1"
)
REAL24_REFERENCE = REPO_ROOT / "experiments/task1_picklift_real24_act_v1/train_config.json"
CONFIGS = {
    "real48_full": EXPERIMENT_ROOT / "real48_train_config_full.json",
    "real96_full": EXPERIMENT_ROOT / "real96_train_config_full.json",
    "real48_smoke": EXPERIMENT_ROOT / "real48_train_config_smoke.json",
    "real96_smoke": EXPERIMENT_ROOT / "real96_train_config_smoke.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def training_semantics(config: dict, *, smoke: bool = False) -> dict:
    value = deepcopy(config)
    for key in ("repo_id", "root", "episodes"):
        value["dataset"].pop(key)
    value.pop("output_dir")
    value.pop("job_name")
    if smoke:
        value.pop("steps")
        value.pop("save_freq")
        value.pop("log_freq")
    return value


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, object] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        result.update(flatten(item, path))
    return result


def differences(left: dict, right: dict) -> list[dict]:
    a, b = flatten(left), flatten(right)
    return [
        {"field": key, "left": a.get(key), "right": b.get(key)}
        for key in sorted(set(a) | set(b))
        if a.get(key) != b.get(key)
    ]


def verify_recipe(config: dict, condition: str, steps: int) -> None:
    dataset = config["dataset"]
    policy = config["policy"]
    expected_episodes = 48 if condition == "real48" else 96
    expected_repo_id = (
        "local/task1_picklift_real96_accepted_v1_accepted"
        if condition == "real48"
        else "local/task1_picklift_real96_accepted_v1"
    )
    if dataset["repo_id"] != expected_repo_id or dataset["episodes"] != list(range(expected_episodes)):
        raise RuntimeError(f"{condition} dataset identity/episode list mismatch")
    if dataset["use_imagenet_stats"] is not True or dataset["image_transforms"]["enable"] is not False:
        raise RuntimeError("ImageNet preprocessing or no-transform contract changed")
    if config["seed"] != 1000 or config["batch_size"] != 8 or config["num_workers"] != 4:
        raise RuntimeError("Seed/batch/workers mismatch")
    if config["steps"] != steps or config["resume"] is not False:
        raise RuntimeError("Step/from-scratch contract mismatch")
    if config["sample_weighting"] is not None:
        raise RuntimeError("Pure-Real standard sampling must not use sample weighting")
    if policy["pretrained_path"] is not None or policy["device"] != "cuda" or policy["use_amp"] is not False:
        raise RuntimeError("Checkpoint/device/AMP contract mismatch")
    expected_policy = {
        "n_obs_steps": 1,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "chunk_size": 67,
        "n_action_steps": 67,
        "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "use_vae": True,
        "latent_dim": 32,
        "n_vae_encoder_layers": 4,
        "dropout": 0.1,
        "kl_weight": 10.0,
        "optimizer_lr": 1e-5,
        "optimizer_weight_decay": 1e-4,
        "optimizer_lr_backbone": 1e-5,
    }
    for key, expected in expected_policy.items():
        if policy[key] != expected:
            raise RuntimeError(f"Policy field {key} mismatch: {policy[key]} != {expected}")
    if not config["use_policy_training_preset"] or AdamWConfig().grad_clip_norm != 10.0:
        raise RuntimeError("AdamW preset gradient clipping contract mismatch")


def runtime_snapshot() -> dict:
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("CUDA is unavailable")
    transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
    if list(transforms.mean) != [0.485, 0.456, 0.406] or list(transforms.std) != [0.229, 0.224, 0.225]:
        raise RuntimeError("Torchvision ImageNet1K V1 statistics drifted")
    return {
        "python": platform.python_version(),
        "lerobot": lerobot.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "imagenet_mean": list(transforms.mean),
        "imagenet_std": list(transforms.std),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "hardware_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify matched Task1 Real48/Real96 ACT configs")
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT / "training_contract_verification.json")
    parser.add_argument("--runtime-output", type=Path, default=EVIDENCE_ROOT / "runtime_snapshot.json")
    args = parser.parse_args()
    configs = {name: load(path) for name, path in CONFIGS.items()}
    reference = load(REAL24_REFERENCE)
    verify_recipe(configs["real48_full"], "real48", 100000)
    verify_recipe(configs["real96_full"], "real96", 100000)
    verify_recipe(configs["real48_smoke"], "real48", 500)
    verify_recipe(configs["real96_smoke"], "real96", 500)

    if training_semantics(configs["real48_full"]) != training_semantics(configs["real96_full"]):
        raise RuntimeError("Real48 and Real96 full training semantics differ beyond dataset/output identity")
    if training_semantics(configs["real48_full"]) != training_semantics(reference):
        raise RuntimeError("Full recipe differs from frozen Real24 ACT baseline recipe")
    for condition in ("real48", "real96"):
        if training_semantics(configs[f"{condition}_smoke"], smoke=True) != training_semantics(configs[f"{condition}_full"], smoke=True):
            raise RuntimeError(f"{condition} smoke differs from full beyond smoke run controls")

    full_diff = differences(configs["real48_full"], configs["real96_full"])
    allowed_fields = {
        "dataset.repo_id",
        "dataset.root",
        "dataset.episodes",
        "output_dir",
        "job_name",
    }
    if {row["field"] for row in full_diff} != allowed_fields:
        raise RuntimeError(f"Unexpected Real48/Real96 full-config differences: {full_diff}")

    runtime = runtime_snapshot()
    result = {
        "schema": "task1_picklift_real48_real96_act_training_contract_verification_v1",
        "status": "pass",
        "experiment_id": "task1_picklift_real48_vs_real96_act_v1",
        "config_hashes": {name: sha256_file(path) for name, path in CONFIGS.items()},
        "real24_reference_config_sha256": sha256_file(REAL24_REFERENCE),
        "full_config_differences": full_diff,
        "allowed_full_config_difference_fields": sorted(allowed_fields),
        "recipe_equal_after_allowed_identity_fields": True,
        "smoke_full_semantics_equal_after_smoke_controls": True,
        "condition_local_stats_source": {
            "real48": "/home/ubuntu24/Teleop/artifacts/derived/task1_picklift_real48_accepted_v1/accepted/meta/stats.json",
            "real96": "/home/ubuntu24/Teleop/artifacts/derived/task1_picklift_real96_accepted_v1/meta/stats.json",
        },
        "sampling": {
            "type": "standard LeRobot pure-Real frame sampling",
            "domain_balancing": False,
            "episode_duplication": False,
            "result_weighting": False,
        },
        "runtime": runtime,
        "hardware_accessed": False,
    }
    write_json(args.runtime_output, runtime)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
