#!/usr/bin/env python
"""Reload a converted checkpoint and predict a chunk without connecting to a robot."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.bi_yam_follower.bi_yam_follower import validate_target

# Published example state paired with the checkpoint's three sample images.
SAMPLE_STATE = [
    -0.06656748,
    0.01468681,
    0.01659419,
    -0.08602273,
    -0.01468681,
    0.13904783,
    0.99223638,
    0.19512475,
    0.01087205,
    0.01087205,
    -0.06771191,
    -0.07305257,
    -0.08945601,
    0.98885375,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--observation", type=Path, help="JSON containing state and top/left/right image paths"
    )
    parser.add_argument("--task", default="pick up the red block")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.checkpoint / "yam_source.json").read_text())
    if args.observation:
        observed = json.loads(args.observation.read_text())
    else:
        observed = {
            "state": SAMPLE_STATE,
            **{
                name: hf_hub_download(
                    source["checkpoint"], f"assets/sample_{name}_rgb.png", revision=source["revision"]
                )
                for name in ("top", "left", "right")
            },
        }
    observation = {
        "observation.state": np.asarray(observed["state"], dtype=np.float32),
        **{
            f"observation.images.{name}": np.array(Image.open(observed[name]).convert("RGB"))
            for name in ("top", "left", "right")
        },
    }
    if observation["observation.state"].shape != (14,):
        raise ValueError("Expected 14 physical joint/gripper values")
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = args.checkpoint
    config.device = "cuda"
    policy = MolmoAct2Policy.from_pretrained(args.checkpoint, config=config).to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=args.checkpoint)
    processed = preprocessor(
        prepare_observation_for_inference(observation, torch.device("cuda"), args.task, "bi_yam_follower")
    )
    durations = []
    actions = None
    for _ in range(2):
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            # Like SyncInferenceEngine, postprocess each selected action separately.
            chunk = policy.predict_action_chunk(processed)
            actions = torch.stack([postprocessor(action) for action in chunk.transpose(0, 1)], dim=1)
        torch.cuda.synchronize()
        durations.append(time.monotonic() - start)
    assert actions is not None
    values = actions.detach().float().cpu().numpy()
    if values.shape != (1, 30, 14) or not np.isfinite(values).all():
        raise ValueError(f"Invalid prediction: shape={values.shape}")
    violations = []
    for step, action in enumerate(values[0]):
        for side, target in (("left", action[:7]), ("right", action[7:])):
            try:
                validate_target(target)
            except ValueError as exc:
                violations.append({"step": step, "side": side, "reason": str(exc)})
    report = {
        "task": args.task,
        "source": "live snapshot" if args.observation else "published sample",
        "shape": list(values.shape),
        "chunk_seconds": durations,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "actions": values[0].tolist(),
        "limit_violations": violations,
        "robot_connected": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "actions"}, indent=2))


if __name__ == "__main__":
    main()
