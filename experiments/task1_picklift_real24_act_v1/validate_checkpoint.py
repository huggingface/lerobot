from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deployment_safety import clamp_action_fail_closed, sha256_file
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Task1 ACT checkpoint validation.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/ubuntu24/Teleop/artifacts/training/task1_picklift_real24_act_v1/"
            "full_100k/checkpoints/100000/pretrained_model"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/ubuntu24/Teleop/artifacts/task1_picklift_formal24_s03_20260728"),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LeRobotDataset(
        repo_id="local/task1_picklift_formal24_s03_20260728",
        root=args.dataset_root,
    )
    sample = dataset[args.sample_index]
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
        processed = preprocessor(inputs)
        action = model.select_action(processed)
        action = postprocessor(action)

    raw_action = action.detach().cpu().numpy()
    if raw_action.shape != (1, 6):
        raise RuntimeError(f"Expected checkpoint output shape (1, 6), got {raw_action.shape}.")
    if not np.isfinite(raw_action).all():
        raise RuntimeError("Checkpoint output contains NaN or infinity.")
    clipped_action, clip_mask = clamp_action_fail_closed(raw_action[0])

    result = {
        "status": "pass",
        "checkpoint": str(args.checkpoint),
        "model_sha256": sha256_file(args.checkpoint / "model.safetensors"),
        "sample_index": args.sample_index,
        "input_shapes": {
            "observation.state": list(inputs["observation.state"].shape),
            "observation.images.front": list(inputs["observation.images.front"].shape),
        },
        "output_shape": list(raw_action.shape),
        "output_finite": bool(np.isfinite(raw_action).all()),
        "raw_action": raw_action[0].tolist(),
        "calibration_clipped_action": clipped_action.tolist(),
        "calibration_clip_mask": clip_mask.tolist(),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
