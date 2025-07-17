#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Generic script to migrate any policy model with normalization layers to the new pipeline-based system.

This script:
1. Loads an existing pretrained policy model
2. Extracts normalization statistics from the model
3. Creates a NormalizerProcessor with these statistics
4. Removes normalization layers from the model state_dict
5. Saves the new model and processor

Usage:
    python scripts/migration/migrate_policy_normalization.py \
        --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
        --policy-type act \
        --push-to-hub
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor.normalize_processor import NormalizerProcessor
from lerobot.processor.pipeline import RobotProcessor

# Policy type to class mapping
POLICY_CLASSES = {
    "act": "lerobot.policies.act.modeling_act.ACTPolicy",
    "diffusion": "lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy",
    "pi0": "lerobot.policies.pi0.modeling_pi0.PI0Policy",
    "pi0fast": "lerobot.policies.pi0fast.modeling_pi0fast.PI0FASTPolicy",
    "smolvla": "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
    "tdmpc": "lerobot.policies.tdmpc.modeling_tdmpc.TDMPCPolicy",
    "vqbet": "lerobot.policies.vqbet.modeling_vqbet.VQBeTPolicy",
    "sac": "lerobot.policies.sac.modeling_sac.SACPolicy",
    "classifier": "lerobot.policies.classifier.modeling_classifier.ClassifierPolicy",
}


def extract_normalization_stats(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Extract normalization statistics from model state_dict."""
    stats = {}

    # Common patterns for normalization layers
    normalization_patterns = [
        ("normalize_inputs.buffer_", "unnormalize_outputs.buffer_"),  # Common pattern
        ("normalize.", "unnormalize."),  # Alternative pattern
        ("input_normalizer.", "output_normalizer."),  # SAC pattern
    ]

    # Extract all normalization buffers
    for key, tensor in state_dict.items():
        for norm_prefix, _ in normalization_patterns:
            if norm_prefix in key:
                # Extract the feature name and stat type
                parts = key.replace(norm_prefix, "").split(".")
                if len(parts) >= 2:
                    # Handle keys like "buffer_observation_state.mean"
                    feature_parts = parts[:-1]
                    stat_type = parts[-1]

                    # Reconstruct feature name (e.g., "observation.state")
                    feature_name = ".".join(feature_parts).replace("_", ".")

                    if feature_name not in stats:
                        stats[feature_name] = {}

                    stats[feature_name][stat_type] = tensor.clone()

    return stats


def detect_features_and_norm_modes(
    config: Dict[str, Any], stats: Dict[str, Dict[str, torch.Tensor]]
) -> tuple[Dict[str, PolicyFeature], Dict[FeatureType, NormalizationMode]]:
    """Detect features and normalization modes from config and stats."""
    features = {}
    norm_modes = {}

    # First, check if there's a normalization_mapping in the config
    if "normalization_mapping" in config:
        print(f"Found normalization_mapping in config: {config['normalization_mapping']}")
        # Extract normalization modes from config
        for feature_name, mode_str in config["normalization_mapping"].items():
            # Convert string to NormalizationMode enum
            if mode_str == "mean_std":
                mode = NormalizationMode.MEAN_STD
            elif mode_str == "min_max":
                mode = NormalizationMode.MIN_MAX
            else:
                print(f"Warning: Unknown normalization mode '{mode_str}' for feature '{feature_name}'")
                continue

            # Determine feature type from feature name
            if "image" in feature_name or "visual" in feature_name:
                feature_type = FeatureType.VISUAL
            elif "state" in feature_name:
                feature_type = FeatureType.STATE
            elif "action" in feature_name:
                feature_type = FeatureType.ACTION
            else:
                feature_type = FeatureType.STATE

            norm_modes[feature_type] = mode

    # Try to extract from config
    if "features" in config:
        for key, feature_config in config["features"].items():
            shape = feature_config.get("shape", feature_config.get("dim"))
            shape = (shape,) if isinstance(shape, int) else tuple(shape)

            # Determine feature type
            if "image" in key or "visual" in key:
                feature_type = FeatureType.VISUAL
            elif "state" in key:
                feature_type = FeatureType.STATE
            elif "action" in key:
                feature_type = FeatureType.ACTION
            else:
                feature_type = FeatureType.STATE  # Default

            features[key] = PolicyFeature(feature_type, shape)

    # If no features in config, infer from stats
    if not features:
        for key, stat_dict in stats.items():
            # Get shape from any stat tensor
            tensor = next(iter(stat_dict.values()))
            shape = tuple(tensor.shape)

            # Determine feature type based on key
            if "image" in key or "visual" in key or "pixels" in key:
                feature_type = FeatureType.VISUAL
            elif "state" in key or "joint" in key or "position" in key:
                feature_type = FeatureType.STATE
            elif "action" in key:
                feature_type = FeatureType.ACTION
            else:
                feature_type = FeatureType.STATE

            features[key] = PolicyFeature(feature_type, shape)

    # If normalization modes weren't in config, determine based on available stats
    if not norm_modes:
        for key, stat_dict in stats.items():
            if key in features:
                if "mean" in stat_dict and "std" in stat_dict:
                    feature_type = features[key].type
                    if feature_type not in norm_modes:
                        norm_modes[feature_type] = NormalizationMode.MEAN_STD
                elif "min" in stat_dict and "max" in stat_dict:
                    feature_type = features[key].type
                    if feature_type not in norm_modes:
                        norm_modes[feature_type] = NormalizationMode.MIN_MAX

    # Default normalization modes if not detected
    if FeatureType.VISUAL not in norm_modes:
        norm_modes[FeatureType.VISUAL] = NormalizationMode.MEAN_STD
    if FeatureType.STATE not in norm_modes:
        norm_modes[FeatureType.STATE] = NormalizationMode.MIN_MAX
    if FeatureType.ACTION not in norm_modes:
        norm_modes[FeatureType.ACTION] = NormalizationMode.MEAN_STD

    return features, norm_modes


def remove_normalization_layers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove normalization layers from state_dict."""
    new_state_dict = {}

    # Patterns to remove
    remove_patterns = [
        "normalize_inputs.",
        "unnormalize_outputs.",
        "normalize.",
        "unnormalize.",
        "input_normalizer.",
        "output_normalizer.",
        "normalizer.",
    ]

    for key, tensor in state_dict.items():
        should_remove = any(pattern in key for pattern in remove_patterns)
        if not should_remove:
            new_state_dict[key] = tensor

    return new_state_dict


def load_model_from_hub(repo_id: str, revision: str = None) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load model state_dict and config from hub."""
    # Download files
    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)

    # Load state_dict
    state_dict = load_safetensors(safetensors_path)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    return state_dict, config


def main():
    parser = argparse.ArgumentParser(
        description="Migrate policy models with normalization layers to new pipeline system"
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        required=True,
        help="Path to pretrained model (hub repo or local directory)",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        required=True,
        choices=list(POLICY_CLASSES.keys()),
        help="Type of policy model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for migrated model (default: same as pretrained-path)",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push migrated model to hub")
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="Hub repository ID for pushing (default: same as pretrained-path)",
    )
    parser.add_argument("--revision", type=str, default=None, help="Revision of the model to load")
    parser.add_argument("--private", action="store_true", help="Make the hub repository private")

    args = parser.parse_args()

    # Load model and config
    print(f"Loading model from {args.pretrained_path}...")
    if os.path.isdir(args.pretrained_path):
        # Local directory
        state_dict = load_safetensors(os.path.join(args.pretrained_path, "model.safetensors"))
        with open(os.path.join(args.pretrained_path, "config.json")) as f:
            config = json.load(f)
    else:
        # Hub repository
        state_dict, config = load_model_from_hub(args.pretrained_path, args.revision)

    # Extract normalization statistics
    print("Extracting normalization statistics...")
    stats = extract_normalization_stats(state_dict)

    if not stats:
        print("Warning: No normalization statistics found in model. The model might already be migrated.")
    else:
        print(f"Found normalization statistics for: {list(stats.keys())}")

    # Detect features and normalization modes
    print("Detecting features and normalization modes...")
    features, norm_map = detect_features_and_norm_modes(config, stats)

    print(f"Detected features: {list(features.keys())}")
    print(f"Normalization modes: {norm_map}")

    # Create NormalizerProcessor
    print("Creating NormalizerProcessor...")
    if stats:
        processor = RobotProcessor(
            [NormalizerProcessor(features, norm_map, stats)], name=f"{args.policy_type}_normalizer"
        )
    else:
        # No normalization needed
        processor = RobotProcessor([], name=f"{args.policy_type}_normalizer")

    # Remove normalization layers from state_dict
    print("Removing normalization layers from model...")
    new_state_dict = remove_normalization_layers(state_dict)

    removed_keys = set(state_dict.keys()) - set(new_state_dict.keys())
    if removed_keys:
        print(f"Removed {len(removed_keys)} normalization layer keys")

    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if os.path.isdir(args.pretrained_path):
            output_dir = Path(args.pretrained_path).parent / f"{Path(args.pretrained_path).name}_migrated"
        else:
            output_dir = Path(f"./{args.pretrained_path.replace('/', '_')}_migrated")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save migrated model
    print(f"Saving migrated model to {output_dir}...")
    save_safetensors(new_state_dict, output_dir / "model.safetensors")

    # Clean up config - remove normalization_mapping field
    cleaned_config = dict(config)
    if "normalization_mapping" in cleaned_config:
        print("Removing 'normalization_mapping' field from config")
        del cleaned_config["normalization_mapping"]

    # Save cleaned config
    with open(output_dir / "config.json", "w") as f:
        json.dump(cleaned_config, f, indent=2)

    # Determine hub repo ID if pushing to hub
    if args.push_to_hub:
        if args.hub_repo_id:
            hub_repo_id = args.hub_repo_id
        else:
            if not os.path.isdir(args.pretrained_path):
                # Use same repo with "_migrated" suffix
                hub_repo_id = f"{args.pretrained_path}_migrated"
            else:
                raise ValueError("--hub-repo-id must be specified when pushing local model to hub")
    else:
        hub_repo_id = None

    # Save processor (and optionally push to hub)
    processor_dir = output_dir / "processor"
    processor.save_pretrained(
        processor_dir,
        repo_id=hub_repo_id,
        push_to_hub=args.push_to_hub,
        private=args.private,
        commit_message=f"Upload {args.policy_type} normalization processor",
    )
    print(f"Saved processor to {processor_dir}")

    # If pushing to hub, also upload the model files
    if args.push_to_hub:
        print(f"Pushing to hub repository: {hub_repo_id}")

        # For the model, we still need to use API since it's just safetensors + config
        api = HfApi()
        api.create_repo(repo_id=hub_repo_id, repo_type="model", private=args.private, exist_ok=True)

        # Upload model files
        api.upload_file(
            path_or_fileobj=str(output_dir / "model.safetensors"),
            path_in_repo="model.safetensors",
            repo_id=hub_repo_id,
            repo_type="model",
            commit_message=f"Upload {args.policy_type} model weights without normalization",
        )

        api.upload_file(
            path_or_fileobj=str(output_dir / "config.json"),
            path_in_repo="config.json",
            repo_id=hub_repo_id,
            repo_type="model",
            commit_message=f"Upload {args.policy_type} model config",
        )

        print(f"Successfully pushed to https://huggingface.co/{hub_repo_id}")

    print("\nMigration complete!")
    print(f"Migrated model saved to: {output_dir}")
    if processor.steps:
        print("Normalization processor created with statistics")
    else:
        print("No normalization processor needed (model had no normalization layers)")


if __name__ == "__main__":
    main()
