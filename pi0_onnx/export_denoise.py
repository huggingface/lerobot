"""Step 3: export the single denoise-step graph to ONNX.

Inputs : state, x_t (noisy actions), timestep, keys, values, prefix_pad_masks.
Output : v_t (velocity), consumed by the host Euler loop.

Only the gemma_300m action expert + small projections are needed here, so we free
the ~9GB paligemma tower before export to keep peak memory low. Tracing shapes for
keys/values come from reference.npz (run reference.py first).
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from pi0_onnx.common import (
    DenoiseStep,
    export_to_onnx,
    load_policy_fp32_lowmem,
    patch_sincos_float32,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="alexzai/pi0_your_task")
    ap.add_argument("--out", default="pi0_onnx/artifacts")
    args = ap.parse_args()

    out = Path(args.out)
    ref = np.load(out / "reference.npz", allow_pickle=True)

    patch_sincos_float32()
    print(f"Loading {args.model_id} (float32/CPU, low-mem)...")
    policy = load_policy_fp32_lowmem(args.model_id)

    # The denoise branch only touches gemma_expert + projections; free paligemma (~9GB).
    policy.model.paligemma_with_expert.paligemma = None
    gc.collect()

    denoise = DenoiseStep(policy)
    state = torch.from_numpy(ref["state"])
    x_t = torch.from_numpy(ref["noise"])
    timestep = torch.ones(state.shape[0], dtype=torch.float32)
    keys = torch.from_numpy(ref["keys"])
    values = torch.from_numpy(ref["values"])
    ppm = torch.from_numpy(ref["prefix_pad_masks"])

    path = str(out / "denoise.onnx")
    print("Exporting denoise.onnx...")
    used = export_to_onnx(
        denoise,
        (state, x_t, timestep, keys, values, ppm),
        ["state", "x_t", "timestep", "keys", "values", "prefix_pad_masks"],
        ["v_t"],
        path,
    )
    print(f"Exported {path} via {used} exporter")


if __name__ == "__main__":
    main()
