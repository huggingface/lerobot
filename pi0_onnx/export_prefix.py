"""Step 2: export the prefix / VLM encoder graph to ONNX.

Inputs : preprocessed camera images + image masks + language tokens/mask.
Outputs: stacked per-layer K/V cache tensors (keys, values) + prefix_pad_masks.

This graph contains the SigLIP vision tower + gemma_2b prefix and is large (~9GB
fp32), so it is written with ONNX external-data (weights beside the .onnx). The
dynamo exporter handles >2GB; classic tracing (used for the small denoise graph)
cannot serialize it.
"""

import argparse
import os
from pathlib import Path

from pi0_onnx.common import (
    PrefixEncoder,
    camera_keys,
    export_to_onnx,
    load_policy_fp32_lowmem,
    make_fixed_inputs,
    patch_sincos_float32,
    prefix_arg_names,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="alexzai/pi0_your_task")
    ap.add_argument("--out", default="pi0_onnx/artifacts")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = str(out / "prefix.onnx")

    # >2GB graph: force the dynamo exporter with external-data serialization.
    os.environ.setdefault("PI0_ONNX_EXPORTER", "dynamo")

    patch_sincos_float32()
    print(f"Loading {args.model_id} (float32/CPU, low-mem)...")
    policy = load_policy_fp32_lowmem(args.model_id)
    cams = camera_keys(policy)

    inp = make_fixed_inputs(policy)
    prefix = PrefixEncoder(policy)
    args_t = (*inp["images"], *inp["img_masks"], inp["lang_tokens"], inp["lang_masks"])

    print("Exporting prefix.onnx (this is the heavy one; external data used)...")
    used = export_to_onnx(
        prefix,
        args_t,
        prefix_arg_names(len(cams)),
        ["keys", "values", "prefix_pad_masks"],
        path,
        external_data=True,
    )
    print(f"Exported {path} via {used} exporter")


if __name__ == "__main__":
    main()
