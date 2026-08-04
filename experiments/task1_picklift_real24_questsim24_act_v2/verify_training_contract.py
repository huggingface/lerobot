from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lerobot.datasets import DomainBalancedSampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments/task1_picklift_real24_questsim24_act_v2"
REAL_CONFIG = REPO_ROOT / "experiments/task1_picklift_real24_act_v1/train_config.json"
MIXED_V1_CONFIG = REPO_ROOT / "experiments/task1_picklift_real24_questsim24_act_v1/train_config_full.json"
REAL_MODEL = Path(
    "/home/ubuntu24/Teleop/artifacts/training/task1_picklift_real24_act_v1/"
    "full_100k/checkpoints/100000/pretrained_model/model.safetensors"
)
MIXED_V1_MODEL = Path(
    "/home/ubuntu24/Teleop/artifacts/training/"
    "task1_picklift_real24_questsim24_act_v1/full_100k/"
    "checkpoints/100000/pretrained_model/model.safetensors"
)
REAL_MODEL_SHA256 = "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
MIXED_V1_MODEL_SHA256 = "e054e682057f09a4653af00a4580da173d3d1658ef5c34244bdbf3ca1a125de5"
DERIVED_DATASET_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/datasets/task1_picklift_real24_questsim24_act_v2/combined48_v3"
)
DERIVED_REPO_ID = "local/task1_picklift_real24_questsim24_combined48_v3"


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
    parser = argparse.ArgumentParser(description="Verify fixed Mixed v2 training contract")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "train_config_full.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    real = load(REAL_CONFIG)
    mixed_v1 = load(MIXED_V1_CONFIG)
    mixed_v2 = load(args.config)
    real_recipe = recipe(real)
    v1_recipe = recipe(mixed_v1)
    v2_recipe = recipe(mixed_v2)

    if real_recipe != v2_recipe:
        raise RuntimeError("Mixed v2 recipe differs from Real24-only beyond the dataset/sampling contract")
    if v2_recipe["use_imagenet_stats"] is not True:
        raise RuntimeError("Mixed v2 must use ImageNet visual statistics")
    if v1_recipe["use_imagenet_stats"] is not False:
        raise RuntimeError("Frozen Mixed v1 visual preprocessing identity changed")

    groups = mixed_v2["dataset"]["domain_balanced_episode_groups"]
    if groups != {"real": list(range(24)), "simulation": list(range(24, 48))}:
        raise RuntimeError("Mixed v2 domain episode groups do not match the frozen 24+24 partition")
    if mixed_v2["batch_size"] != 8:
        raise RuntimeError("Mixed v2 batch size must remain 8")
    if mixed_v2["policy"]["pretrained_path"] is not None:
        raise RuntimeError("Mixed v2 must initialize from scratch")

    dataset = LeRobotDataset(DERIVED_REPO_ID, root=DERIVED_DATASET_ROOT, video_backend="pyav")
    sampler_kwargs = {
        "dataset_from_indices": dataset.meta.episodes["dataset_from_index"],
        "dataset_to_indices": dataset.meta.episodes["dataset_to_index"],
        "episode_indices": dataset.meta.episodes["episode_index"],
        "domain_episode_groups": groups,
        "batch_size": mixed_v2["batch_size"],
        "episode_indices_to_use": dataset.episodes,
        "seed": mixed_v2["seed"],
        "absolute_to_relative_idx": dataset.absolute_to_relative_idx,
    }
    sampler = DomainBalancedSampler(**sampler_kwargs)
    epoch0 = list(sampler)
    epoch0_repeat = list(DomainBalancedSampler(**sampler_kwargs))
    if epoch0 != epoch0_repeat:
        raise RuntimeError("Balanced sampler is not deterministic for the frozen seed")
    if len(epoch0) % mixed_v2["batch_size"] != 0:
        raise RuntimeError("Balanced sampler emitted a partial batch")
    for offset in range(0, len(epoch0), mixed_v2["batch_size"]):
        batch = epoch0[offset : offset + mixed_v2["batch_size"]]
        if sum(index < 3790 for index in batch) != 4:
            raise RuntimeError(f"Batch {offset // mixed_v2['batch_size']} is not 4 Real + 4 Sim")
    if len(set(epoch0)) != len(epoch0):
        raise RuntimeError("Balanced sampler duplicated a frame inside one epoch")

    real_sha = sha256_file(REAL_MODEL)
    mixed_v1_sha = sha256_file(MIXED_V1_MODEL)
    if real_sha != REAL_MODEL_SHA256 or mixed_v1_sha != MIXED_V1_MODEL_SHA256:
        raise RuntimeError("Control checkpoint identity mismatch")

    result = {
        "schema": "task1_picklift_mixed_v2_training_contract_verification_v1",
        "status": "pass",
        "mixed_v2_config": str(args.config),
        "mixed_v2_config_sha256": sha256_file(args.config),
        "real24_only_config": str(REAL_CONFIG),
        "real24_only_config_sha256": sha256_file(REAL_CONFIG),
        "mixed_v1_config": str(MIXED_V1_CONFIG),
        "mixed_v1_config_sha256": sha256_file(MIXED_V1_CONFIG),
        "recipe_equal_to_real24_only": True,
        "config_diff_against_real24_only": differences(real, mixed_v2),
        "config_diff_against_mixed_v1": differences(mixed_v1, mixed_v2),
        "controlled_corrections": {
            "use_imagenet_stats": {"mixed_v1": False, "mixed_v2": True},
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
            "seed": mixed_v2["seed"],
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
        "control_models": {
            "real24_only_sha256": real_sha,
            "mixed_v1_sha256": mixed_v1_sha,
        },
        "initialized_from_checkpoint": None,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
