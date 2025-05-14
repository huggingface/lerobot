#!/usr/bin/env python3
"""
Peek into a safetensors file:
  - list keys & shapes
  - dump metadata (if present)
  - count any NaN / Inf entries

Usage:
    python peek_safetensors.py /path/to/model.safetensors
"""
import argparse
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser(description="Inspect a .safetensors file")
    parser.add_argument("file", help="Path to the .safetensors file")
    args = parser.parse_args()
    path = args.file

    print(f"[1] Scanning keys & metadata in {path!r}…")
    try:
        with safe_open(path, framework="pt") as f:
            # metadata
            meta = {}
            if hasattr(f, "metadata"):
                try:
                    meta = f.metadata() or {}
                except TypeError:
                    # some versions expose metadata as a property
                    meta = getattr(f, "metadata", {}) or {}
            print("  Metadata:", meta if meta else "{}")

            # keys & shapes
            print("  Keys / Shapes:")
            for key in f.keys():
                # try the faster API first...
                if hasattr(f, "get_shape"):
                    shape = f.get_shape(key)
                else:
                    # fallback: load only this tensor to get its shape
                    tensor = f.get_tensor(key)
                    # could be np.ndarray or torch.Tensor
                    shape = tuple(tensor.shape)
                print(f"    {key:60s} → {shape}")
    except Exception as e:
        print(f"  ✗ Failed to list keys: {e}", file=sys.stderr)
        return 1

    print(f"\n[2] Loading weights and checking for NaN/Inf…")
    try:
        sd = load_file(path, device="cpu")
    except Exception as e:
        print(f"  ✗ Failed to load file: {e}", file=sys.stderr)
        return 1

    total_inf = total_nan = 0
    inf_map = {}
    nan_map = {}

    for name, tensor in sd.items():
        inf_c = int(torch.isinf(tensor).sum())
        nan_c = int(torch.isnan(tensor).sum())
        total_inf += inf_c
        total_nan += nan_c
        if inf_c:
            inf_map[name] = inf_c
        if nan_c:
            nan_map[name] = nan_c

    print(f"  Total Inf: {total_inf}, Total NaN: {total_nan}")
    if inf_map:
        print("  Inf breakdown:")
        for n, c in inf_map.items():
            print(f"    {n:60s} : {c}")
    if nan_map:
        print("  NaN breakdown:")
        for n, c in nan_map.items():
            print(f"    {n:60s} : {c}")

    return 0

if __name__ == "__main__":
    sys.exit(main())