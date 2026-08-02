"""Step 4: run the ONNX graphs end-to-end and verify against the PyTorch reference.

Runs prefix.onnx once, then the 10-step Euler flow-matching loop calling denoise.onnx
each step, and compares the resulting action chunk (and the prefix K/V) to reference.npz.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pi0_onnx.common import euler_loop_numpy, make_ort_denoise_callable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pi0_onnx/artifacts")
    ap.add_argument("--atol", type=float, default=2e-3, help="Max abs diff tolerance on actions.")
    args = ap.parse_args()

    out = Path(args.out)
    ref = np.load(out / "reference.npz", allow_pickle=True)
    steps = int(ref["num_inference_steps"])
    adim = int(ref["action_dim"])
    n_cam = ref["images"].shape[0]

    print("Loading ONNX sessions...")
    psess = ort.InferenceSession(str(out / "prefix.onnx"), providers=["CPUExecutionProvider"])
    dsess = ort.InferenceSession(str(out / "denoise.onnx"), providers=["CPUExecutionProvider"])

    pin = [i.name for i in psess.get_inputs()]
    feeds = {}
    for i in range(n_cam):
        feeds[pin[i]] = ref["images"][i]
        feeds[pin[n_cam + i]] = ref["img_masks"][i]
    feeds[pin[-2]] = ref["lang_tokens"]
    feeds[pin[-1]] = ref["lang_masks"]

    print("Running prefix.onnx...")
    keys_o, values_o, ppm_o = psess.run(None, feeds)
    dk = np.abs(keys_o - ref["keys"]).max()
    dv = np.abs(values_o - ref["values"]).max()
    print(f"  prefix K/V max abs diff: keys={dk:.3e} values={dv:.3e}")

    print(f"Running {steps}-step Euler denoise loop (denoise.onnx)...")
    ort_call = make_ort_denoise_callable(dsess)
    onnx_actions = euler_loop_numpy(
        (keys_o, values_o, ppm_o, ref["state"]), ort_call, ref["noise"], steps
    )
    onnx_actions = onnx_actions[:, :, :adim]

    ref_actions = ref["ref_actions"]
    diff = np.abs(onnx_actions - ref_actions)
    print("\n===== VERIFICATION =====")
    print(f"action chunk shape : {onnx_actions.shape}")
    print(f"max  abs diff      : {diff.max():.3e}")
    print(f"mean abs diff      : {diff.mean():.3e}")
    print(f"ref  action[0,0]   : {ref_actions[0, 0]}")
    print(f"onnx action[0,0]   : {onnx_actions[0, 0]}")
    ok = diff.max() <= args.atol
    print(f"\n{'PASS' if ok else 'FAIL'}: max abs diff {diff.max():.3e} {'<=' if ok else '>'} atol {args.atol}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
