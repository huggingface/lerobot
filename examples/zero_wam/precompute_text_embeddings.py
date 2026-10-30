#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precompute UMT5 embeddings for every task of a LeRobot dataset (and the empty prompt).

Training then skips the 5.7B UMT5 encoder: pass the output with
``--policy.text_embedding_cache_path``. Task strings are preserved as supplied by the dataset.

    python examples/zero_wam/precompute_text_embeddings.py \
        --dataset-root /path/to/dataset --output text_embeddings.pt
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from lerobot.policies.lingbot_va.utils import clean_prompt, load_text_encoder, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-components-path", default="robbyant-research/zero-wam-pretrain")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tasks = pd.read_parquet(args.dataset_root / "meta" / "tasks.parquet")
    names = tasks.index if "task" not in tasks.columns else tasks["task"]
    prompts = {""}
    for task in names:
        prompts.add(str(task))
    ordered_prompts = sorted(prompts)
    print(f"{len(prompts)} unique prompts")

    tokenizer = load_tokenizer(args.frozen_components_path, subfolder="tokenizer")
    encoder = load_text_encoder(
        args.frozen_components_path,
        torch_dtype=torch.bfloat16,
        torch_device=args.device,
        subfolder="text_encoder",
    ).eval()
    cache = {}
    with torch.no_grad():
        for prompt in tqdm(ordered_prompts):
            tokens = tokenizer(
                [clean_prompt(prompt)],
                padding="max_length",
                max_length=args.max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            hidden = encoder(
                tokens.input_ids.to(args.device), tokens.attention_mask.to(args.device)
            ).last_hidden_state
            length = int(tokens.attention_mask.gt(0).sum())
            cache[prompt] = hidden[0, :length].to("cpu", torch.bfloat16).clone()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)
    print(f"Saved {len(cache)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
