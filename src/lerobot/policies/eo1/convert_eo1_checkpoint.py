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

"""Convert the released EO-1 checkpoint into a native LeRobot policy checkpoint."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLProcessor

from .configuration_eo1 import EO1Config

DEFAULT_SOURCE_REPO_ID = "IPEC-COMMUNITY/EO-1-3B"
DEFAULT_QWEN_BASE_REPO_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DESTINATION_REPO_ID = "lerobot/eo1-base"


def convert_upstream_state_key(key: str) -> str:
    """Map an upstream Transformers 4.x EO-1 key to the LeRobot/Transformers 5.x layout."""
    if key.startswith("vlm_backbone.model."):
        key = key.replace(
            "vlm_backbone.model.",
            "vlm_backbone.model.language_model.",
            1,
        )
    elif key.startswith("vlm_backbone.visual."):
        key = key.replace("vlm_backbone.visual.", "vlm_backbone.model.visual.", 1)
    return f"model.{key}"


def build_vlm_config(upstream_config: dict[str, Any], qwen_base_config: dict[str, Any]) -> dict[str, Any]:
    """Build a current-Transformers Qwen config while preserving EO-1 token settings."""
    config = deepcopy(qwen_base_config)
    for key in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "vision_token_id",
        "vocab_size",
        "eos_token_id",
        "pad_token_id",
        "state_token_id",
        "action_token_id",
        "action_pass_id",
    ):
        if key in upstream_config:
            config[key] = upstream_config[key]
    config["tie_word_embeddings"] = True
    config["use_cache"] = upstream_config.get("use_cache", False)
    return config


def build_lerobot_config(
    upstream_config: dict[str, Any],
    vlm_config: dict[str, Any],
    destination_repo_id: str,
) -> EO1Config:
    """Translate released EO-1 architectural defaults into a LeRobot config."""
    action_chunk_size = int(upstream_config["action_chunk_size"])
    max_action_dim = int(upstream_config["max_action_dim"])
    return EO1Config(
        vlm_base=destination_repo_id,
        vlm_config=vlm_config,
        chunk_size=action_chunk_size,
        n_action_steps=action_chunk_size,
        max_state_dim=max_action_dim,
        max_action_dim=max_action_dim,
        num_denoise_steps=int(upstream_config["num_denoise_steps"]),
        num_action_layers=int(upstream_config["num_action_layers"]),
        action_act=str(upstream_config["action_act"]),
        dtype="bfloat16",
        device="cpu",
        pretrained_path=Path(destination_repo_id),
        repo_id=destination_repo_id,
        license="mit",
        tags=["eo1", "robotics", "vision-language-action", "lerobot"],
    )


def _strip_remote_code_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_remote_code_metadata(item) for key, item in value.items() if key != "auto_map"}
    if isinstance(value, list):
        return [_strip_remote_code_metadata(item) for item in value]
    return value


def save_processor(source_repo_id: str, output_dir: Path, revision: str) -> None:
    """Save a self-contained native Qwen processor with EO-1's trained special tokens."""
    processor = Qwen2_5_VLProcessor.from_pretrained(
        source_repo_id,
        revision=revision,
        use_fast=False,
        fix_mistral_regex=True,
        trust_remote_code=False,
    )
    processor.save_pretrained(output_dir)
    for path in output_dir.glob("*.json"):
        data = json.loads(path.read_text())
        cleaned = _strip_remote_code_metadata(data)
        path.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n")


def convert_weights(snapshot_dir: Path, output_file: Path, source_repo_id: str) -> int:
    """Stream the two released shards into one LeRobot safetensors file."""
    index = json.loads((snapshot_dir / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    converted_state = {}
    for shard_name in shard_names:
        for key, tensor in load_file(snapshot_dir / shard_name, device="cpu").items():
            converted_key = convert_upstream_state_key(key)
            if converted_key in converted_state:
                raise ValueError(f"Converted EO-1 key collision: {converted_key}")
            converted_state[converted_key] = tensor

    expected_keys = set(index["weight_map"])
    if len(converted_state) != len(expected_keys):
        raise ValueError(f"Expected {len(expected_keys)} converted EO-1 tensors, got {len(converted_state)}.")
    save_file(
        converted_state,
        output_file,
        metadata={"format": "pt", "source": source_repo_id},
    )
    return len(converted_state)


def write_model_card(
    output_dir: Path,
    source_repo_id: str,
    destination_repo_id: str,
    source_revision: str,
) -> None:
    card = f"""---
license: mit
library_name: lerobot
pipeline_tag: robotics
tags:
- lerobot
- eo1
- vision-language-action
base_model: {source_repo_id}
---

# EO-1 base for LeRobot

This is a lossless key-layout conversion of
[`{source_repo_id}`](https://huggingface.co/{source_repo_id}) at revision
`{source_revision}` for LeRobot's native `eo1` policy implementation.

The checkpoint preserves the released EO-1 Qwen vision-language weights,
language-model head, state/action projectors, and flow-matching head. Its defaults
match the release: action chunk size 16, maximum state/action dimension 32, two
action-projector layers, and ten denoising steps.

Use `--policy.path={destination_repo_id}` for training or rollout. LeRobot then
applies the dataset or environment's camera, state, and action feature shapes
before instantiating the policy.
"""
    (output_dir / "README.md").write_text(card)


def convert_checkpoint(
    output_dir: Path,
    *,
    source_repo_id: str = DEFAULT_SOURCE_REPO_ID,
    qwen_base_repo_id: str = DEFAULT_QWEN_BASE_REPO_ID,
    destination_repo_id: str = DEFAULT_DESTINATION_REPO_ID,
    revision: str = "main",
    push_to_hub: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "model.safetensors"
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output_file}")

    snapshot_dir = Path(
        snapshot_download(
            repo_id=source_repo_id,
            revision=revision,
            allow_patterns=["config.json", "model*.safetensors*"],
        )
    )
    upstream_config = json.loads((snapshot_dir / "config.json").read_text())
    qwen_base_config = Qwen2_5_VLConfig.from_pretrained(qwen_base_repo_id).to_dict()
    vlm_config = build_vlm_config(upstream_config, qwen_base_config)
    config = build_lerobot_config(upstream_config, vlm_config, destination_repo_id)

    config._save_pretrained(output_dir)
    save_processor(source_repo_id, output_dir, revision)
    source_revision = snapshot_dir.name
    write_model_card(output_dir, source_repo_id, destination_repo_id, source_revision)
    tensor_count = convert_weights(snapshot_dir, output_file, source_repo_id)
    print(f"Converted {tensor_count} tensors to {output_file}")

    if push_to_hub:
        api = HfApi()
        api.create_repo(destination_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=destination_repo_id,
            repo_type="model",
            folder_path=output_dir,
            commit_message=f"Convert {source_repo_id} to LeRobot EO-1 format",
        )
        print(f"Uploaded checkpoint to https://huggingface.co/{destination_repo_id}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    parser.add_argument("--qwen-base-repo-id", default=DEFAULT_QWEN_BASE_REPO_ID)
    parser.add_argument("--destination-repo-id", default=DEFAULT_DESTINATION_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()
    convert_checkpoint(
        args.output_dir,
        source_repo_id=args.source_repo_id,
        qwen_base_repo_id=args.qwen_base_repo_id,
        destination_repo_id=args.destination_repo_id,
        revision=args.revision,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
