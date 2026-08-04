from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy


CONDITIONS = {
    "real24": {
        "root": Path(
            "/home/ubuntu24/Teleop/artifacts/derived/"
            "task1_picklift_real24_budget_extension_v1/accepted"
        ),
        "repo_id": "local/task1_picklift_real24_budget_extension_v1_accepted",
        "sample_index": 0,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_values_for_feature(state: dict[str, torch.Tensor], feature: str) -> dict[str, list]:
    return {
        key: value.detach().cpu().numpy().astype(float).reshape(-1).tolist()
        for key, value in state.items()
        if feature in key
    }


def contains_vector(values: dict[str, list], expected: list[float], atol: float = 1e-6) -> bool:
    return any(
        len(vector) == len(expected) and np.allclose(vector, expected, atol=atol, rtol=0)
        for vector in values.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline validate fixed Task1 Real24 budget ACT checkpoint")
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    spec = CONDITIONS[args.condition]
    dataset = LeRobotDataset(spec["repo_id"], root=spec["root"], video_backend="pyav")
    sample = dataset[spec["sample_index"]]
    inputs = {
        "observation.state": sample["observation.state"].unsqueeze(0),
        "observation.images.front": sample["observation.images.front"].unsqueeze(0),
    }

    model = ACTPolicy.from_pretrained(args.checkpoint)
    model.to(args.device)
    model.eval()
    model.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    with torch.inference_mode():
        action = postprocessor(model.select_action(preprocessor(inputs)))
    raw_action = action.detach().cpu().numpy()
    if raw_action.shape != (1, 6) or not np.isfinite(raw_action).all():
        raise RuntimeError(f"Invalid offline inference output: shape={raw_action.shape}")

    train_config_path = args.checkpoint / "train_config.json"
    train_config = json.loads(train_config_path.read_text())
    if train_config["dataset"]["use_imagenet_stats"] is not True:
        raise RuntimeError("Saved checkpoint train config does not enable ImageNet statistics")
    if train_config["policy"]["chunk_size"] != 67 or train_config["policy"]["n_action_steps"] != 67:
        raise RuntimeError("Saved checkpoint ACT horizon mismatch")
    if train_config["policy"]["pretrained_path"] is not None or train_config.get("checkpoint_path") is not None:
        raise RuntimeError("Saved checkpoint reports non-scratch initialization")
    if train_config["seed"] != 1000 or train_config["steps"] not in (500, 100000):
        raise RuntimeError("Saved checkpoint seed/step contract mismatch")
    expected_repo_id = spec["repo_id"]
    if train_config["dataset"]["repo_id"] != expected_repo_id:
        raise RuntimeError("Saved checkpoint dataset identity mismatch")
    optimizer = train_config["optimizer"]
    if optimizer["type"] != "adamw" or optimizer["lr"] != 1e-5 or optimizer["weight_decay"] != 1e-4 or optimizer["grad_clip_norm"] != 10.0:
        raise RuntimeError("Saved optimizer contract mismatch")

    normalizer_path = args.checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    normalizer_state = load_file(normalizer_path)
    visual_values = tensor_values_for_feature(normalizer_state, "observation.images.front")
    if not contains_vector(visual_values, [0.485, 0.456, 0.406]) or not contains_vector(
        visual_values, [0.229, 0.224, 0.225]
    ):
        raise RuntimeError(f"Saved processor lacks frozen ImageNet mean/std: {visual_values}")
    state_values = tensor_values_for_feature(normalizer_state, "observation.state")
    action_values = tensor_values_for_feature(normalizer_state, "action")
    dataset_stats = json.loads((spec["root"] / "meta/stats.json").read_text())
    for feature, values in (("observation.state", state_values), ("action", action_values)):
        if not contains_vector(values, dataset_stats[feature]["mean"], atol=1e-4) or not contains_vector(
            values, dataset_stats[feature]["std"], atol=1e-4
        ):
            raise RuntimeError(f"Saved {feature} processor stats differ from condition dataset stats")

    hashed_files = {
        name: {
            "path": str(args.checkpoint / filename),
            "sha256": sha256_file(args.checkpoint / filename),
        }
        for name, filename in {
            "model": "model.safetensors",
            "policy_config": "config.json",
            "saved_train_config": "train_config.json",
            "policy_preprocessor": "policy_preprocessor.json",
            "policy_preprocessor_stats": "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor": "policy_postprocessor.json",
            "policy_postprocessor_stats": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        }.items()
    }
    result = {
        "schema": "task1_picklift_real24_budget_extension_act_offline_checkpoint_validation_v1",
        "status": "pass",
        "condition": args.condition,
        "checkpoint": str(args.checkpoint),
        "sample_index": spec["sample_index"],
        "dataset_repo_id": spec["repo_id"],
        "input_shapes": {
            "observation.state": list(inputs["observation.state"].shape),
            "observation.images.front": list(inputs["observation.images.front"].shape),
        },
        "output_shape": list(raw_action.shape),
        "output_finite": True,
        "raw_action": raw_action[0].tolist(),
        "use_imagenet_stats": True,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "condition_local_state_action_stats_match": True,
        "normalizer_tensor_keys": sorted(normalizer_state),
        "files": hashed_files,
        "model_sha256": hashed_files["model"]["sha256"],
        "processor_stats_sha256": hashed_files["policy_preprocessor_stats"]["sha256"],
        "hardware_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
