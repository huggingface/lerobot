# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a local released LIBERO bundle on CPU; never uploads artifacts."""

import argparse
import logging
from pathlib import Path

import torch

from lerobot.policies.longwam.conversion_longwam import convert_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(8)
    convert_release(args.bundle, args.output, backbone=args.backbone, tokenizer_model=args.tokenizer_model)
    print(f"Native checkpoint saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
