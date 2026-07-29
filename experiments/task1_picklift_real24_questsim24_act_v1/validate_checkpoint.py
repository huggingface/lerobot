from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy


DEFAULT_CHECKPOINT = Path(
    "/home/ubuntu24/Teleop/artifacts/training/"
    "task1_picklift_real24_questsim24_act_v1/full_100k/"
    "checkpoints/100000/pretrained_model"
)
DEFAULT_DATASET = Path(
    "/home/ubuntu24/Teleop/artifacts/datasets/"
    "task1_picklift_real24_questsim24_act_v1/combined48_v1"
)
REPO_ID = "local/task1_picklift_real24_questsim24_combined48_v1"


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
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=REPO_ID, root=args.dataset_root)
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
        action = postprocessor(model.select_action(preprocessor(inputs)))
    raw_action = action.detach().cpu().numpy()
    if raw_action.shape != (1, 6):
        raise RuntimeError(f"Expected output shape (1, 6), got {raw_action.shape}")
    if not np.isfinite(raw_action).all():
        raise RuntimeError("Checkpoint output contains NaN or infinity")

    result = {
        "status": "pass",
        "checkpoint": str(args.checkpoint),
        "model_sha256": sha256_file(args.checkpoint / "model.safetensors"),
        "sample_index": args.sample_index,
        "input_shapes": {
            "observation.state": list(inputs["observation.state"].shape),
            "observation.images.front": list(
                inputs["observation.images.front"].shape
            ),
        },
        "output_shape": list(raw_action.shape),
        "output_finite": True,
        "raw_action": raw_action[0].tolist(),
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
