"""Step 1: PyTorch reference output for the real pi0 checkpoint.

Loads the checkpoint in float32/CPU, builds a deterministic input, runs the real
`sample_actions` (fixed noise) to get the reference action chunk, and also captures
the prefix K/V cache. Everything is saved to an .npz that the ONNX steps compare
against and reuse for export tracing shapes.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from pi0_onnx.common import (
    ACTION,
    PrefixEncoder,
    camera_keys,
    load_policy_fp32,
    make_fixed_inputs,
    patch_sincos_float32,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="alexzai/pi0_your_task")
    ap.add_argument("--out", default="pi0_onnx/artifacts")
    ap.add_argument("--steps", type=int, default=None, help="Override num_inference_steps.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    patch_sincos_float32()
    print(f"Loading {args.model_id} (float32/CPU)...")
    policy = load_policy_fp32(args.model_id)
    if args.steps is not None:
        policy.config.num_inference_steps = args.steps
    steps = policy.config.num_inference_steps
    adim = policy.config.output_features[ACTION].shape[0]
    cams = camera_keys(policy)

    inp = make_fixed_inputs(policy)

    print("Running reference sample_actions...")
    with torch.no_grad():
        ref = policy.model.sample_actions(
            inp["images"], inp["img_masks"], inp["lang_tokens"], inp["lang_masks"],
            inp["state"], noise=inp["noise"].clone(),
        )
    ref = ref[:, :, :adim].numpy()

    print("Capturing prefix K/V cache...")
    prefix = PrefixEncoder(policy)
    with torch.no_grad():
        keys, values, ppm = prefix(*inp["images"], *inp["img_masks"], inp["lang_tokens"], inp["lang_masks"])

    np.savez(
        out / "reference.npz",
        images=np.stack([x.numpy() for x in inp["images"]], axis=0),
        img_masks=np.stack([x.numpy() for x in inp["img_masks"]], axis=0),
        lang_tokens=inp["lang_tokens"].numpy(),
        lang_masks=inp["lang_masks"].numpy(),
        state=inp["state"].numpy(),
        noise=inp["noise"].numpy(),
        ref_actions=ref,
        keys=keys.numpy(),
        values=values.numpy(),
        prefix_pad_masks=ppm.numpy(),
        num_inference_steps=np.int64(steps),
        action_dim=np.int64(adim),
        cam_keys=np.array(cams),
    )
    print(f"Saved {out / 'reference.npz'}")
    print(f"  action chunk shape {ref.shape}, prefix_len {keys.shape[3]}, layers {keys.shape[0]}, steps {steps}")
    print(f"  ref action[0,0] = {ref[0, 0]}")


if __name__ == "__main__":
    main()
