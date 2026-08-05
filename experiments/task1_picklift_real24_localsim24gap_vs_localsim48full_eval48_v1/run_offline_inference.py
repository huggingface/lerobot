from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy

READY_STATE = [
    7.4285712242126465,
    -98.32967376708984,
    45.010990142822266,
    92.21977996826172,
    1.8461538553237915,
    19.765840530395508,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded offline ACT CUDA inference")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weights = args.checkpoint / "model.safetensors"
    actual_model_sha256 = sha256_file(weights)
    if actual_model_sha256 != args.expected_model_sha256:
        raise RuntimeError("Checkpoint hash mismatch before offline inference")

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to("cuda")
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    inputs = {
        "observation.state": torch.tensor(READY_STATE, dtype=torch.float32).unsqueeze(0),
        "observation.images.front": torch.zeros((1, 3, 480, 640), dtype=torch.float32),
    }
    policy.reset()
    with torch.inference_mode():
        action = postprocessor(policy.select_action(preprocessor(inputs)))
    output = action.detach().cpu().numpy()
    if output.shape != (1, 6) or not np.isfinite(output).all():
        raise RuntimeError(f"Invalid ACT output: shape={output.shape}")

    result = {
        "schema": "task1_act_eval48_offline_inference_v1",
        "status": "pass",
        "model_key": args.model_key,
        "checkpoint": str(args.checkpoint),
        "model_sha256": actual_model_sha256,
        "input_state_shape": [1, 6],
        "input_front_shape": [1, 3, 480, 640],
        "output_shape": list(output.shape),
        "output_finite": True,
        "output_action": output[0].tolist(),
        "device": "cuda",
        "hardware_accessed": False,
        "rollout_started": False,
    }
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite offline inference: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
