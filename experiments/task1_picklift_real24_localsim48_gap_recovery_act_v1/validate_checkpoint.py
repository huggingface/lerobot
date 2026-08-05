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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ACT checkpoint validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, video_backend="pyav")
    indices = [0, 4263]
    if len(dataset) <= indices[-1]:
        raise RuntimeError("Dataset does not include both Real and Simulation frames")
    model = ACTPolicy.from_pretrained(args.checkpoint)
    model.to(args.device)
    model.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    samples = []
    for index, domain in zip(indices, ("real", "simulation"), strict=True):
        sample = dataset[index]
        inputs = {
            "observation.state": sample["observation.state"].unsqueeze(0),
            "observation.images.front": sample["observation.images.front"].unsqueeze(0),
        }
        model.reset()
        with torch.inference_mode():
            action = postprocessor(model.select_action(preprocessor(inputs)))
        array = action.detach().cpu().numpy()
        if array.shape != (1, 6) or not np.isfinite(array).all():
            raise RuntimeError(f"Invalid output for {domain}: shape={array.shape}")
        samples.append(
            {
                "domain": domain,
                "index": index,
                "state_shape": list(inputs["observation.state"].shape),
                "image_shape": list(inputs["observation.images.front"].shape),
                "output_shape": list(array.shape),
                "output_finite": True,
                "action": array[0].tolist(),
            }
        )

    processor_path = args.checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    processor = load_file(processor_path)
    mean = processor["observation.images.front.mean"].cpu().flatten().tolist()
    std = processor["observation.images.front.std"].cpu().flatten().tolist()
    if not np.allclose(mean, [0.485, 0.456, 0.406], atol=1e-7) or not np.allclose(
        std, [0.229, 0.224, 0.225], atol=1e-7
    ):
        raise RuntimeError(f"Saved visual normalization is not ImageNet: {mean=} {std=}")
    result = {
        "status": "pass",
        "checkpoint": str(args.checkpoint),
        "model_sha256": sha256_file(args.checkpoint / "model.safetensors"),
        "processor_sha256": sha256_file(processor_path),
        "visual_mean": mean,
        "visual_std": std,
        "samples": samples,
        "all_outputs_shape_1x6_and_finite": True,
        "hardware_accessed": False,
        "rollout_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
