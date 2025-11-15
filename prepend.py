#!/usr/bin/env python3
import safetensors.torch as st
import torch
import argparse
import os

def prefix_state_dict(input_path, output_path, prefix="model."):
    # Load original checkpoint
    state_dict = st.load_file(input_path)

    print(f"Loaded {len(state_dict)} tensors from {input_path}")

    # Add prefix to every key
    new_state_dict = {f"{prefix}{k}": v for k, v in state_dict.items()}

    print(f"Writing prefixed checkpoint with {len(new_state_dict)} keys...")
    st.save_file(new_state_dict, output_path)

    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to model.safetensors")
    parser.add_argument("--output", type=str, required=True, help="Output prefixed model.safetensors")
    parser.add_argument("--prefix", type=str, default="model.", help="Prefix to add to each key")
    args = parser.parse_args()

    prefix_state_dict(args.input, args.output, args.prefix)
