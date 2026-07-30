from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lerobot.datasets import DomainBalancedSampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments/task1_picklift_real24_questsim48_v7_act_v1"
REAL_CONFIG = REPO_ROOT / "experiments/task1_picklift_real24_act_v1/train_config.json"
CONTROLLED_MIXED_V2_CONFIG = (
    REPO_ROOT / "experiments/task1_picklift_real24_questsim24_act_v2/train_config_full.json"
)
REAL_MODEL = Path(
    "/home/ubuntu24/Teleop/artifacts/training/task1_picklift_real24_act_v1/"
    "full_100k/checkpoints/100000/pretrained_model/model.safetensors"
)
REAL_MODEL_SHA256 = "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
DERIVED_DATASET_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/datasets/"
    "task1_picklift_real24_questsim48_v7_act_v1/combined72_v1"
)
DERIVED_REPO_ID = "local/task1_picklift_real24_questsim48_v7_combined72_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def recipe(config: dict) -> dict:
    policy = config["policy"]
    return {
        "seed": config["seed"],
        "batch_size": config["batch_size"],
        "steps": config["steps"],
        "save_freq": config["save_freq"],
        "env_eval_freq": config["env_eval_freq"],
        "image_transforms": config["dataset"]["image_transforms"],
        "use_imagenet_stats": config["dataset"]["use_imagenet_stats"],
        "policy": {
            key: policy[key]
            for key in (
                "type",
                "n_obs_steps",
                "input_features",
                "output_features",
                "device",
                "use_amp",
                "pretrained_path",
                "chunk_size",
                "n_action_steps",
                "normalization_mapping",
                "vision_backbone",
                "pretrained_backbone_weights",
                "replace_final_stride_with_dilation",
                "pre_norm",
                "dim_model",
                "n_heads",
                "dim_feedforward",
                "feedforward_activation",
                "n_encoder_layers",
                "n_decoder_layers",
                "use_vae",
                "latent_dim",
                "n_vae_encoder_layers",
                "temporal_ensemble_coeff",
                "dropout",
                "kl_weight",
                "optimizer_lr",
                "optimizer_weight_decay",
                "optimizer_lr_backbone",
            )
        },
    }


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        flattened.update(flatten(item, path))
    return flattened


def differences(left: dict, right: dict) -> list[dict]:
    left_flat = flatten(left)
    right_flat = flatten(right)
    rows = []
    for key in sorted(set(left_flat) | set(right_flat)):
        if left_flat.get(key) != right_flat.get(key):
            rows.append(
                {
                    "field": key,
                    "left": left_flat.get(key),
                    "right": right_flat.get(key),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify fixed Real24 + Sim48-v7 training contract")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "train_config_full.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    real = load(REAL_CONFIG)
    controlled_mixed_v2 = load(CONTROLLED_MIXED_V2_CONFIG)
    candidate = load(args.config)
    real_recipe = recipe(real)
    controlled_v2_recipe = recipe(controlled_mixed_v2)
    candidate_recipe = recipe(candidate)

    if real_recipe != candidate_recipe:
        raise RuntimeError("Candidate recipe differs from Real24-only beyond dataset/sampling identity")
    if candidate_recipe != controlled_v2_recipe:
        raise RuntimeError("Candidate recipe differs from controlled Mixed v2")
    if candidate_recipe["use_imagenet_stats"] is not True:
        raise RuntimeError("Candidate must use ImageNet visual statistics")

    groups = candidate["dataset"]["domain_balanced_episode_groups"]
    if groups != {"real": list(range(24)), "simulation": list(range(24, 72))}:
        raise RuntimeError("Domain episode groups do not match the frozen 24+48 partition")
    if candidate["batch_size"] != 8:
        raise RuntimeError("Batch size must remain 8")
    if candidate["policy"]["pretrained_path"] is not None:
        raise RuntimeError("Training must initialize from scratch")

    dataset = LeRobotDataset(DERIVED_REPO_ID, root=DERIVED_DATASET_ROOT, video_backend="pyav")
    sampler_kwargs = {
        "dataset_from_indices": dataset.meta.episodes["dataset_from_index"],
        "dataset_to_indices": dataset.meta.episodes["dataset_to_index"],
        "episode_indices": dataset.meta.episodes["episode_index"],
        "domain_episode_groups": groups,
        "batch_size": candidate["batch_size"],
        "episode_indices_to_use": dataset.episodes,
        "seed": candidate["seed"],
        "absolute_to_relative_idx": dataset.absolute_to_relative_idx,
    }
    sampler = DomainBalancedSampler(**sampler_kwargs)
    epoch0 = list(sampler)
    epoch0_repeat = list(DomainBalancedSampler(**sampler_kwargs))
    if epoch0 != epoch0_repeat:
        raise RuntimeError("Balanced sampler is not deterministic for the frozen seed")
    if len(epoch0) % candidate["batch_size"] != 0:
        raise RuntimeError("Balanced sampler emitted a partial batch")
    for offset in range(0, len(epoch0), candidate["batch_size"]):
        batch = epoch0[offset : offset + candidate["batch_size"]]
        if sum(index < 3790 for index in batch) != 4:
            raise RuntimeError(f"Batch {offset // candidate['batch_size']} is not 4 Real + 4 Sim")
    if len(set(epoch0)) != len(epoch0):
        raise RuntimeError("Balanced sampler duplicated a frame inside one epoch")

    real_sha = sha256_file(REAL_MODEL)
    if real_sha != REAL_MODEL_SHA256:
        raise RuntimeError("Real24-only control checkpoint identity mismatch")

    result = {
        "schema": "task1_picklift_real24_questsim48_v7_training_contract_verification_v1",
        "status": "pass",
        "candidate_config": str(args.config),
        "candidate_config_sha256": sha256_file(args.config),
        "real24_only_config": str(REAL_CONFIG),
        "real24_only_config_sha256": sha256_file(REAL_CONFIG),
        "controlled_mixed_v2_config": str(CONTROLLED_MIXED_V2_CONFIG),
        "controlled_mixed_v2_config_sha256": sha256_file(CONTROLLED_MIXED_V2_CONFIG),
        "recipe_equal_to_real24_only": True,
        "recipe_equal_to_controlled_mixed_v2": True,
        "config_diff_against_real24_only": differences(real, candidate),
        "config_diff_against_controlled_mixed_v2": differences(controlled_mixed_v2, candidate),
        "controlled_contract": {
            "use_imagenet_stats": True,
            "gripper_canonicalization": "identity, verified by dataset audit",
            "domain_balancing": {
                "batch_size": 8,
                "per_batch_real": 4,
                "per_batch_simulation": 4,
                "episode_partition": groups,
            },
        },
        "sampler_verification": {
            "status": "pass",
            "seed": candidate["seed"],
            "source_domain_frame_counts": sampler.domain_frame_counts,
            "complete_batches_per_epoch": sampler.num_batches,
            "samples_per_epoch": len(sampler),
            "samples_per_domain_per_batch": sampler.samples_per_domain_per_batch,
            "selected_samples_per_domain_per_epoch": (
                sampler.num_batches * sampler.samples_per_domain_per_batch
            ),
            "per_batch_domain_counts": {"real": 4, "simulation": 4},
            "deterministic_repeat_equal": True,
            "duplicates_within_epoch": 0,
            "oversampling": False,
        },
        "control_models": {"real24_only_sha256": real_sha},
        "initialized_from_checkpoint": None,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
