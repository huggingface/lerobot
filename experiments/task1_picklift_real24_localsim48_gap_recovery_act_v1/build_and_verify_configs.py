from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lerobot.datasets import DomainBalancedSampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset

EXPERIMENT_ID = "task1_picklift_real24_localsim48_gap_recovery_act_v1"
REPO_ROOT = Path("/home/ubuntu24/Teleop/lerobot")
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / EXPERIMENT_ID
BINDINGS_PATH = EXPERIMENT_ROOT / "source_bindings.json"
TEMPLATE_PATH = REPO_ROOT / "experiments/task1_picklift_real24_questsim48_v7_act_v1/train_config_full.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def recipe(config: dict) -> dict:
    policy = config["policy"]
    return {
        "seed": config["seed"],
        "batch_size": config["batch_size"],
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


def config_for(template: dict, condition_id: str, condition: dict, episodes: int, smoke: bool) -> dict:
    config = json.loads(json.dumps(template))
    config["dataset"]["repo_id"] = condition["repo_id"]
    config["dataset"]["root"] = condition["dataset_root"]
    config["dataset"]["episodes"] = list(range(episodes))
    config["dataset"]["domain_balanced_episode_groups"] = {
        "real": list(range(24)),
        "simulation": list(range(24, episodes)),
    }
    suffix = "smoke_500" if smoke else "full_100k"
    config["output_dir"] = f"{condition['training_root']}/{suffix}"
    config["job_name"] = f"{EXPERIMENT_ID}_{condition_id.lower()}_{suffix}"
    config["steps"] = 500 if smoke else 100000
    config["save_freq"] = 500 if smoke else 20000
    return config


def validate_sampler(condition: dict, config: dict) -> dict:
    dataset = LeRobotDataset(condition["repo_id"], root=Path(condition["dataset_root"]), video_backend="pyav")
    sampler_kwargs = {
        "dataset_from_indices": dataset.meta.episodes["dataset_from_index"],
        "dataset_to_indices": dataset.meta.episodes["dataset_to_index"],
        "episode_indices": dataset.meta.episodes["episode_index"],
        "domain_episode_groups": config["dataset"]["domain_balanced_episode_groups"],
        "batch_size": config["batch_size"],
        "episode_indices_to_use": dataset.episodes,
        "seed": config["seed"],
        "absolute_to_relative_idx": dataset.absolute_to_relative_idx,
    }
    sampler = DomainBalancedSampler(**sampler_kwargs)
    indices = list(sampler)
    repeat = list(DomainBalancedSampler(**sampler_kwargs))
    if indices != repeat or len(indices) % 8:
        raise RuntimeError("DomainBalancedSampler determinism or complete-batch invariant failed")
    real_end = 4263
    for offset in range(0, len(indices), 8):
        batch = indices[offset : offset + 8]
        if sum(index < real_end for index in batch) != 4:
            raise RuntimeError(f"Batch {offset // 8} is not 4 Real + 4 Sim")
    if len(set(indices)) != len(indices):
        raise RuntimeError("Balanced sampler duplicated frames within one epoch")
    return {
        "complete_batches": sampler.num_batches,
        "samples_per_epoch": len(indices),
        "samples_per_domain_per_batch": sampler.samples_per_domain_per_batch,
        "source_domain_frame_counts": sampler.domain_frame_counts,
        "deterministic_repeat_equal": True,
        "duplicates_within_epoch": 0,
    }


def main() -> None:
    bindings = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))
    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    template_recipe = recipe(template)
    results = {}
    for condition_id, condition in bindings["conditions"].items():
        sim = bindings["simulation"][condition["simulation_key"]]
        episodes = bindings["real"]["episodes"] + sim["episodes"]
        freeze_path = Path(condition["evidence_root"]) / "freeze_manifest.json"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze["episodes"] != episodes or freeze["frames"] != bindings["real"]["frames"] + sim["frames"]:
            raise RuntimeError(f"Dataset freeze mismatch for {condition_id}")
        configs = {}
        for smoke in (True, False):
            config = config_for(template, condition_id, condition, episodes, smoke)
            path = EXPERIMENT_ROOT / "configs" / f"{condition_id}_{'smoke' if smoke else 'full'}.json"
            write_json(path, config)
            if recipe(config) != {
                **template_recipe,
                "save_freq": 500 if smoke else template_recipe["save_freq"],
            }:
                raise RuntimeError(f"Frozen ACT recipe drift for {condition_id} smoke={smoke}")
            if config["policy"]["pretrained_path"] is not None:
                raise RuntimeError("Formal training must start from scratch")
            configs["smoke" if smoke else "full"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        full = json.loads(Path(configs["full"]["path"]).read_text(encoding="utf-8"))
        sampler = validate_sampler(condition, full)
        verification = {
            "status": "pass",
            "condition": condition_id,
            "dataset_freeze_sha256": sha256_file(freeze_path),
            "dataset_tree_sha256": freeze["tree_sha256"],
            "configs": configs,
            "fixed_recipe": {
                "seed": 1000,
                "steps": 100000,
                "smoke_steps": 500,
                "batch_size": 8,
                "real_per_batch": 4,
                "simulation_per_batch": 4,
                "chunk_size": 67,
                "n_action_steps": 67,
                "use_imagenet_stats": True,
                "initialized_from_checkpoint": None,
            },
            "sampler": sampler,
            "hardware_accessed": False,
            "rollout_started": False,
        }
        verification_path = Path(condition["evidence_root"]) / "training_contract_verification.json"
        write_json(verification_path, verification)
        results[condition_id] = verification
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
