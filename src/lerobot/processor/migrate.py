#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
Script to migrate pretrained pi0 model from lerobot/pi0 HuggingFace repository
with updated input/output features configuration and option to push to hub.

Usage:
    # Basic migration (saves locally)
    python migrate.py

    # Migrate and push to hub
    python migrate.py --push-to-hub --repo-id username/my-pi0-model

    # Push to specific branch
    python migrate.py --push-to-hub --repo-id username/my-pi0-model --branch dev

    # Push to same repo with different branch
    python migrate.py --push-to-hub --repo-id lerobot/pi0 --branch migrated-v2
"""

import argparse
import json
import os

import safetensors
import torch
from huggingface_hub import HfApi, create_repo, snapshot_download

from lerobot.configs.types import FeatureType
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import SmolVLANewLineProcessor
from lerobot.processor import (
    DeviceProcessor,
    NormalizerProcessor,
    ProcessorStep,
    RenameProcessor,
    RobotProcessor,
    ToBatchProcessor,
    UnnormalizerProcessor,
)
from lerobot.processor.tokenizer_processor import TokenizerProcessor


def update_config_with_features(config_dict: dict) -> dict:
    """Update the configuration dictionary with the new input/output features."""

    # Define input features
    config_dict["input_features"] = {
        "observation.state": {"type": FeatureType.STATE.value, "shape": (6,)},
        "observation.images.camera0": {
            "type": FeatureType.VISUAL.value,
            "shape": (3, 480, 640),  # C, H, W format for RGB image
        },
        "observation.images.camera1": {
            "type": FeatureType.VISUAL.value,
            "shape": (3, 480, 640),  # C, H, W format for RGB image
        },
        "observation.images.camera2": {
            "type": FeatureType.VISUAL.value,
            "shape": (3, 480, 640),  # C, H, W format for RGB image
        },
    }

    # Define output features
    config_dict["output_features"] = {"action": {"type": FeatureType.ACTION.value, "shape": (6,)}}

    return config_dict


def update_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Update the state dictionary with the new input/output features."""
    updated_state_dict = {}
    removed_keys = []

    for key, value in state_dict.items():
        # Remove keys containing normalization terms
        if any(term in key for term in ["unnormalize_outputs", "normalize_targets", "normalize_inputs"]):
            removed_keys.append(key)
            continue

        # Rename keys: remove "_orig_mod." from anywhere in the key
        new_key = key.replace("_orig_mod.", "")
        updated_state_dict[new_key] = value

    if removed_keys:
        print(f"Removed {len(removed_keys)} normalization-related keys from state dict:")
        for key in removed_keys:
            print(f"  - {key}")

    return updated_state_dict


def make_smolvla_processor(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[RobotProcessor, RobotProcessor]:
    """Create preprocessing and postprocessing pipelines for PI0 policy."""

    input_steps: list[ProcessorStep] = [
        # Add tokenizer processor first (always included for pi0)
        # Add normalizers
        RenameProcessor(rename_map={}),
        NormalizerProcessor(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        ToBatchProcessor(),
        SmolVLANewLineProcessor(),  # Add newlines before tokenization for PaliGemma
        TokenizerProcessor(
            tokenizer_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessor(device=config.device),
    ]

    output_steps = [
        DeviceProcessor(device="cpu"),
        UnnormalizerProcessor(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    # Create and return the processors
    preprocessor = RobotProcessor(steps=input_steps, name="robot_preprocessor")

    postprocessor = RobotProcessor(steps=output_steps, name="robot_postprocessor")

    return preprocessor, postprocessor


def migrate_smolvla_model(
    source_repo_id: str = "lerobot/smolvla_base",
    save_directory: str = "./migrated_smolvla",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    push_to_hub: bool = False,
    hub_repo_id: str = None,
    hub_branch: str = "main",
    create_pr: bool = False,
) -> tuple[SmolVLAPolicy, SmolVLAConfig, RobotProcessor, RobotProcessor]:
    """
    Migrate pretrained smolvla model with updated configuration.

    Args:
        source_repo_id: HuggingFace repository ID where the pretrained model is stored
        save_directory: Local directory to save the migrated model
        device: Device to load the model on
        push_to_hub: Whether to push the migrated model to HuggingFace Hub
        hub_repo_id: Repository ID to push to (if push_to_hub is True)
        hub_branch: Branch to push to (default: "main")
        create_pr: Whether to create a pull request instead of pushing directly

    Returns:
        Tuple of (model, config, preprocessor, postprocessor)
    """

    print(f"Starting migration of smolvla model from {source_repo_id}")

    # Create save directory
    os.makedirs(save_directory, exist_ok=True)

    # Step 1: Download the entire repository
    print("Downloading entire model repository...")
    snapshot_dir = snapshot_download(repo_id=source_repo_id, cache_dir=".cache")
    print(f"Repository downloaded to: {snapshot_dir}")

    # Step 2: Load and update configuration
    config_path = os.path.join(snapshot_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    # Update with new input/output features
    config_dict = update_config_with_features(config_dict)

    state_dict = safetensors.torch.load_file(filename=f"{snapshot_dir}/model.safetensors", device="cpu")

    # Update state dict with key renaming and normalization removal
    state_dict = update_state_dict(state_dict)

    # Create SmolVLAConfig instance from updated dictionary
    # Remove 'type' field as it's not needed for instantiation
    if "type" in config_dict:
        del config_dict["type"]
    config = SmolVLAConfig(**config_dict)

    # Step 3: Create model with updated config and load updated state dict
    print("Loading model with pretrained weights...")
    model = SmolVLAPolicy(config)
    model.load_state_dict(state_dict, strict=True)
    model.to("cpu")
    model.to(torch.float32)
    # Step 4: Create processors (without stats for now)
    print("Creating processors...")
    preprocessor, postprocessor = make_smolvla_processor(config, dataset_stats=None)

    # Step 5: Save the migrated model locally
    print(f"Saving migrated model to {save_directory}")

    # Save configuration
    config_save_path = os.path.join(save_directory, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save the model
    model.save_pretrained(save_directory)

    # Save processors
    preprocessor.save_pretrained(save_directory, config_filename="robot_preprocessor.json")
    postprocessor.save_pretrained(save_directory, config_filename="robot_postprocessor.json")

    # Also copy tokenizer files if they exist
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]

    for tokenizer_file in tokenizer_files:
        src_path = os.path.join(snapshot_dir, tokenizer_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(save_directory, tokenizer_file)
            import shutil

            shutil.copy2(src_path, dst_path)
            print(f"Copied {tokenizer_file}")

    print("Migration completed successfully!")
    print(f"Migrated model saved to: {save_directory}")
    print("\nUpdated configuration:")
    print(f"  Input features: {list(config.input_features.keys())}")
    print(f"  Output features: {list(config.output_features.keys())}")

    # Step 6: Push to hub if requested
    if push_to_hub:
        if hub_repo_id is None:
            raise ValueError("--repo-id must be specified when using --push-to-hub")

        print(f"\nPushing to HuggingFace Hub: {hub_repo_id}")

        # Create repository if it doesn't exist
        api = HfApi()
        try:
            create_repo(repo_id=hub_repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"Note: Could not create repo (may already exist): {e}")

        # Push the model
        model.push_to_hub(repo_id=hub_repo_id, branch=hub_branch, create_pr=create_pr)

        # Push processors
        api.upload_file(
            path_or_fileobj=os.path.join(save_directory, "preprocessor.json"),
            path_in_repo="preprocessor.json",
            repo_id=hub_repo_id,
            repo_type="model",
            revision=hub_branch,
            create_pr=create_pr,
        )

        # Upload processor state files if they exist
        for file in os.listdir(save_directory):
            if file.endswith("_preprocessor.safetensors"):
                api.upload_file(
                    path_or_fileobj=os.path.join(save_directory, file),
                    path_in_repo=file,
                    repo_id=hub_repo_id,
                    repo_type="model",
                    revision=hub_branch,
                    create_pr=create_pr,
                )

        api.upload_file(
            path_or_fileobj=os.path.join(save_directory, "postprocessor.json"),
            path_in_repo="postprocessor.json",
            repo_id=hub_repo_id,
            repo_type="model",
            revision=hub_branch,
            create_pr=create_pr,
        )

        # Upload postprocessor state files if they exist
        for file in os.listdir(save_directory):
            if file.endswith("_postprocessor.safetensors"):
                api.upload_file(
                    path_or_fileobj=os.path.join(save_directory, file),
                    path_in_repo=file,
                    repo_id=hub_repo_id,
                    repo_type="model",
                    revision=hub_branch,
                    create_pr=create_pr,
                )

        # Upload tokenizer files
        for tokenizer_file in tokenizer_files:
            file_path = os.path.join(save_directory, tokenizer_file)
            if os.path.exists(file_path):
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=tokenizer_file,
                    repo_id=hub_repo_id,
                    repo_type="model",
                    revision=hub_branch,
                    create_pr=create_pr,
                )

        print(f"Successfully pushed to {hub_repo_id} on branch '{hub_branch}'")
        if create_pr:
            print("A pull request has been created for review.")

    return model, config, preprocessor, postprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Migrate pretrained smolvla model with updated configuration"
    )
    parser.add_argument(
        "--source-repo-id",
        type=str,
        default="lerobot/smolvla",
        help="Source HuggingFace repository ID (default: lerobot/smolvla)",
    )
    parser.add_argument(
        "--save-directory",
        type=str,
        default="./migrated_smolvla",
        help="Local directory to save migrated model (default: ./migrated_smolvla)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load model on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push the migrated model to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-id", type=str, help="HuggingFace Hub repository ID to push to (required if --push-to-hub)"
    )
    parser.add_argument("--branch", type=str, default="main", help="Branch to push to (default: main)")
    parser.add_argument(
        "--create-pr", action="store_true", help="Create a pull request instead of pushing directly"
    )

    args = parser.parse_args()

    # Run migration
    model, config, preprocessor, postprocessor = migrate_smolvla_model(
        source_repo_id=args.source_repo_id,
        save_directory=args.save_directory,
        device=args.device,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.repo_id,
        hub_branch=args.branch,
        create_pr=args.create_pr,
    )

    print("\nMigration complete! ✓")

    # Print example usage
    print("\nExample usage:")
    print("```python")
    print("from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy")
    print("from lerobot.processor import RobotProcessor")
    print()
    print("# Load migrated model")
    print(f"model = SmolVLAPolicy.from_pretrained('{args.save_directory}')")
    print(
        f"preprocessor = RobotProcessor.from_pretrained('{args.save_directory}', config_filename='preprocessor.json')"
    )
    print(
        f"postprocessor = RobotProcessor.from_pretrained('{args.save_directory}', config_filename='postprocessor.json')"
    )
    print()
    print("# Use for inference")
    print("observation = {...}  # Your observation dict")
    print("processed_obs = preprocessor(observation)")
    print("action = model.predict_action_chunk(processed_obs)")
    print("action = postprocessor({'action': action})['action']")
    print("```")


if __name__ == "__main__":
    main()
