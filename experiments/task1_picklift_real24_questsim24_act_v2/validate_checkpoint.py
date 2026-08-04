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

DEFAULT_CHECKPOINT = Path(
    "/home/ubuntu24/Teleop/artifacts/training/"
    "task1_picklift_real24_questsim24_act_v2/full_100k/"
    "checkpoints/100000/pretrained_model"
)
DEFAULT_DATASET = Path(
    "/home/ubuntu24/Teleop/artifacts/datasets/task1_picklift_real24_questsim24_act_v2/combined48_v3"
)
REPO_ID = "local/task1_picklift_real24_questsim24_combined48_v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline mixed Task1 ACT validation")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=[0, 3790],
        help="At least one Real and one Sim frame; defaults to each domain's first frame.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=REPO_ID, root=args.dataset_root)
    model = ACTPolicy.from_pretrained(args.checkpoint)
    model.to(args.device)
    model.eval()
    model.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    samples = []
    for sample_index in args.sample_indices:
        sample = dataset[sample_index]
        inputs = {
            "observation.state": sample["observation.state"].unsqueeze(0),
            "observation.images.front": sample["observation.images.front"].unsqueeze(0),
        }
        model.reset()
        with torch.inference_mode():
            action = postprocessor(model.select_action(preprocessor(inputs)))
        raw_action = action.detach().cpu().numpy()
        if raw_action.shape != (1, 6):
            raise RuntimeError(f"Expected output shape (1, 6), got {raw_action.shape}")
        if not np.isfinite(raw_action).all():
            raise RuntimeError("Checkpoint output contains NaN or infinity")
        episode_index = int(sample["episode_index"])
        samples.append(
            {
                "sample_index": sample_index,
                "episode_index": episode_index,
                "source_domain": "real" if episode_index < 24 else "simulation",
                "input_shapes": {
                    "observation.state": list(inputs["observation.state"].shape),
                    "observation.images.front": list(inputs["observation.images.front"].shape),
                },
                "output_shape": list(raw_action.shape),
                "output_finite": True,
                "raw_action": raw_action[0].tolist(),
            }
        )
    if {sample["source_domain"] for sample in samples} != {"real", "simulation"}:
        raise RuntimeError("Offline validation must include both Real and Sim samples")

    processor_stats_path = args.checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    processor_stats = load_file(processor_stats_path)
    image_mean = processor_stats["observation.images.front.mean"].cpu().flatten().tolist()
    image_std = processor_stats["observation.images.front.std"].cpu().flatten().tolist()
    expected_mean = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    if not np.allclose(image_mean, expected_mean, atol=1e-7) or not np.allclose(
        image_std, expected_std, atol=1e-7
    ):
        raise RuntimeError(
            f"Saved visual processor is not ImageNet normalization: mean={image_mean}, std={image_std}"
        )

    result = {
        "status": "pass",
        "checkpoint": str(args.checkpoint),
        "model_sha256": sha256_file(args.checkpoint / "model.safetensors"),
        "samples": samples,
        "domains_validated": ["real", "simulation"],
        "all_outputs_shape_1x6_and_finite": True,
        "saved_visual_processor": {
            "status": "pass_imagenet_stats",
            "path": str(processor_stats_path),
            "sha256": sha256_file(processor_stats_path),
            "mean": image_mean,
            "std": image_std,
        },
        "hardware_accessed": False,
        "simulation_rollout_started": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
