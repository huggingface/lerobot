#!/usr/bin/env python
"""Export an ACT policy's network to ONNX and verify numerical parity.

Only the inference network is exported (ResNet backbone + transformer enc/dec +
action head). The VAE encoder is training-only and the inference latent is zeros,
so the exported graph is a pure function of (state, images) -> action_chunk.
Normalization stays in the LeRobot processor pipeline (outside ONNX).

Usage:
    python examples/onnx/export_act.py \
        --policy-path=outputs/converted/act_aloha_sim_transfer_cube_human \
        --output=outputs/onnx/act_transfer_cube.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTExportWrapper(nn.Module):
    """Tensor-in/tensor-out wrapper around ACT's inference network."""

    def __init__(self, model: nn.Module, image_keys: list[str], has_state: bool, has_env_state: bool):
        super().__init__()
        self.model = model
        self.image_keys = image_keys
        self.has_state = has_state
        self.has_env_state = has_env_state

    def forward(self, state: torch.Tensor, *images: torch.Tensor) -> torch.Tensor:
        batch: dict = {}
        if self.has_state:
            batch[OBS_STATE] = state
        if self.has_env_state:
            # Convention: when env_state is used it is passed as `state`.
            batch[OBS_ENV_STATE] = state
        batch[OBS_IMAGES] = list(images)
        actions, _ = self.model(batch)
        return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="Converted ACT checkpoint dir or repo id")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading ACT policy from '{args.policy_path}'...")
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.eval()
    policy.to(args.device)
    cfg = policy.config

    image_keys = list(cfg.image_features)
    has_state = cfg.robot_state_feature is not None
    has_env_state = cfg.env_state_feature is not None
    state_dim = (cfg.robot_state_feature or cfg.env_state_feature).shape[0]

    print(f"      image_keys={image_keys} state_dim={state_dim} "
          f"chunk_size={cfg.chunk_size} action_dim={cfg.action_feature.shape[0]}")

    wrapper = ACTExportWrapper(policy.model, image_keys, has_state, has_env_state).eval().to(args.device)

    # Build example inputs (batch size 1) from the config feature shapes.
    state_example = torch.randn(1, state_dim, device=args.device)
    image_examples = [
        torch.rand(1, *cfg.image_features[k].shape, device=args.device) for k in image_keys
    ]
    example_inputs = (state_example, *image_examples)

    input_names = ["state"] + [f"image_{i}" for i in range(len(image_keys))]
    output_names = ["action_chunk"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    print(f"[2/4] Exporting to ONNX (opset {args.opset}) -> {out}")
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(out),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    print("[3/4] Running parity check (torch vs onnxruntime)...")
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(out), sess_options=so, providers=providers)

    # Fresh random inputs for the check.
    state_check = torch.randn(2, state_dim, device=args.device)
    image_check = [torch.rand(2, *cfg.image_features[k].shape, device=args.device) for k in image_keys]

    with torch.no_grad():
        torch_out = wrapper(state_check, *image_check).cpu().numpy()

    ort_inputs = {"state": state_check.cpu().numpy()}
    for i, img in enumerate(image_check):
        ort_inputs[f"image_{i}"] = img.cpu().numpy()
    ort_out = sess.run(None, ort_inputs)[0]

    max_abs = float(np.max(np.abs(torch_out - ort_out)))
    mean_abs = float(np.mean(np.abs(torch_out - ort_out)))
    print(f"      shapes: torch={torch_out.shape} onnx={ort_out.shape}")
    print(f"      max_abs_diff={max_abs:.3e} mean_abs_diff={mean_abs:.3e} (atol={args.atol:.0e})")

    ok = max_abs <= args.atol
    print(f"[4/4] Parity: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"Parity check failed: max_abs_diff {max_abs:.3e} > atol {args.atol:.0e}")
    print(f"\nDone. ONNX model at: {out}")


if __name__ == "__main__":
    main()
