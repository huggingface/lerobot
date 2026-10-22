#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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


"""Offline conversion of historical Wall-OSS-0.5 snapshots to native LeRobot files.

uv run python -m lerobot.policies.wall_oss_05.convert_checkpoint \
    --source /path/to/old/snapshot --output-dir /path/to/native/checkpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file

from lerobot.processor import NormalizerProcessorStep
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_wall_oss_05 import WallOSS05Config
from .modeling_wall_oss_05 import WallOSS05Policy
from .processor_wall_oss_05 import (
    WALL_OSS_ACTION_TOKEN,
    WALL_OSS_PROPRIO_TOKEN,
    make_wall_oss_05_pre_post_processors,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import Qwen2_5_VLProcessor


def convert_checkpoint(
    source: Path,
    output_dir: Path,
    processor_repo: str = "lerobot/wall-oss-0.5",
) -> dict:
    """Convert once, preserving every neural tensor and the saved quantile statistics.

    This utility is deliberately outside runtime loading. The policy uses the
    ordinary PreTrainedPolicy loader and distributed saver after conversion.
    """
    require_package("transformers", extra="wall_oss_05")
    source = source.resolve()
    output_dir = output_dir.resolve()
    if source == output_dir:
        raise ValueError("Convert into a different directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.safetensors"
    if weights_path.exists() and weights_path.stat().st_size:
        raise FileExistsError(weights_path)

    config = WallOSS05Config.from_pretrained(source)
    processor = Qwen2_5_VLProcessor.from_pretrained(source, fix_mistral_regex=True)
    processor.tokenizer.add_tokens([WALL_OSS_PROPRIO_TOKEN, WALL_OSS_ACTION_TOKEN])
    processor.tokenizer.padding_side = "left"
    state = load_file(source / "model.safetensors")
    if "model.embed_tokens.weight" not in state:
        raise ValueError("Expected the historical checkpoint's model.embed_tokens.weight.")
    ignored = [key for key in state if ".normalizer_" in key]
    for key in ignored:
        state.pop(key)

    # Historical snapshots used the author model as the root. LeRobot saves
    # the policy root, whose model child adds exactly one namespace prefix.
    state.pop("lm_head.weight", None)
    native = {f"model.{key}": value for key, value in state.items()}
    backbone = config.vlm_backbone_config
    backbone.text_config.vocab_size = state["model.embed_tokens.weight"].shape[0]
    backbone.text_config.pad_token_id = processor.tokenizer.pad_token_id
    backbone.text_config.eos_token_id = processor.tokenizer.eos_token_id
    backbone.action_token_id = processor.tokenizer.convert_tokens_to_ids(WALL_OSS_ACTION_TOKEN)
    config.vlm_config = backbone.to_dict()
    config.dtype = state["model.embed_tokens.weight"].dtype
    config.pretrained_name_or_path = processor_repo
    config.pretrained_path = None
    config.pretrained_revision = None
    config.device = "cpu"

    # Check the complete native key contract without allocating a second 4B model.
    with torch.device("meta"):
        expected = WallOSS05Policy(config).state_dict()
    missing = set(expected) - native.keys() - {"model.lm_head.weight"}
    unexpected = native.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(
            f"Native tensor mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for key, tensor in native.items():
        if tensor.shape != expected[key].shape:
            raise ValueError(f"Wrong shape for {key}: {tensor.shape}, expected {expected[key].shape}")
    del expected

    saved = json.loads((source / "policy_preprocessor.json").read_text())
    entry = next(step for step in saved["steps"] if step["registry_name"] == "normalizer_processor")
    normalizer = NormalizerProcessorStep(**entry["config"])
    normalizer.load_state_dict(load_file(source / entry["state_file"]))
    pre, post = make_wall_oss_05_pre_post_processors(config, dataset_stats=normalizer.stats)

    save_file(native, weights_path, metadata={"format": "pt"})
    restored = load_file(weights_path)
    if not all(torch.equal(tensor, restored[key]) for key, tensor in native.items()):
        raise RuntimeError("Converted tensor values changed.")
    del restored
    config._save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    pre.save_pretrained(output_dir)
    post.save_pretrained(output_dir)

    digest = hashlib.sha256()
    with weights_path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    report = {
        "format": "native LeRobot policy state_dict",
        "neural_tensors": len(native),
        "parameters": sum(value.numel() for value in native.values()),
        "dropped_model_owned_normalizer_tensors": len(ignored),
        "all_neural_tensors_preserved_exactly": True,
        "weight_sha256": digest.hexdigest(),
        "weight_bytes": weights_path.stat().st_size,
        "processor_repo": processor_repo,
    }
    (output_dir / "native_conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processor-repo", default="lerobot/wall-oss-0.5")
    args = parser.parse_args()
    print(json.dumps(convert_checkpoint(args.source, args.output_dir, args.processor_repo), indent=2))


if __name__ == "__main__":
    main()
