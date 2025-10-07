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
A generic script to migrate LeRobot policies with built-in normalization layers to the new
pipeline-based processor system.

This script performs the following steps:
1.  Loads a pretrained policy model and its configuration from a local path or the
    Hugging Face Hub.
2.  Scans the model's state dictionary to extract normalization statistics (e.g., mean,
    std, min, max) for all features.
3.  Creates two new processor pipelines:
    - A preprocessor that normalizes inputs (observations) and outputs (actions).
    - A postprocessor that unnormalizes outputs (actions) for inference.
4.  Removes the original normalization layers from the model's state dictionary,
    creating a "clean" model.
5.  Saves the new clean model, the preprocessor, the postprocessor, and a generated
    model card to a new directory.
6.  Optionally pushes all the new artifacts to the Hugging Face Hub.

Usage:
    python src/lerobot/processor/migrate_policy_normalization.py \
        --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
        --push-to-hub \
        --branch main

Note: This script now uses the modern `make_pre_post_processors` and `make_policy_config`
factory functions from `lerobot.policies.factory` to create processors and configurations,
ensuring consistency with the current codebase.

The script extracts normalization statistics from the old model's state_dict, creates clean
processor pipelines using the factory functions, and saves a migrated model that is compatible
with the new PolicyProcessorPipeline architecture.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file as load_safetensors

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION


def extract_normalization_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """
    Scans a model's state_dict to find and extract normalization statistics.

    This function identifies keys corresponding to normalization layers (e.g., those
    for mean, std, min, max) based on a set of predefined patterns and organizes
    them into a nested dictionary.

    Args:
        state_dict: The state dictionary of a pretrained policy model.

    Returns:
        A nested dictionary where outer keys are feature names (e.g.,
        'observation.state') and inner keys are statistic types ('mean', 'std'),
        mapping to their corresponding tensor values.
    """
    stats = {}

    # Define patterns to match and their prefixes to remove
    normalization_patterns = [
        "normalize_inputs.buffer_",
        "unnormalize_outputs.buffer_",
        "normalize_targets.buffer_",
        "normalize.",  # Must come after normalize_* patterns
        "unnormalize.",  # Must come after unnormalize_* patterns
        "input_normalizer.",
        "output_normalizer.",
        "normalalize_inputs.",
        "unnormalize_outputs.",
        "normalize_targets.",
        "unnormalize_targets.",
    ]

    # Process each key in state_dict
    for key, tensor in state_dict.items():
        # Try each pattern
        for pattern in normalization_patterns:
            if key.startswith(pattern):
                # Extract the remaining part after the pattern
                remaining = key[len(pattern) :]
                parts = remaining.split(".")

                # Need at least feature name and stat type
                if len(parts) >= 2:
                    # Last part is the stat type (mean, std, min, max, etc.)
                    stat_type = parts[-1]
                    # Everything else is the feature name
                    feature_name = ".".join(parts[:-1]).replace("_", ".")

                    # Add to stats
                    if feature_name not in stats:
                        stats[feature_name] = {}
                    stats[feature_name][stat_type] = tensor.clone()

                # Only process the first matching pattern
                break

    return stats


def detect_features_and_norm_modes(
    config: dict[str, Any], stats: dict[str, dict[str, torch.Tensor]]
) -> tuple[dict[str, PolicyFeature], dict[FeatureType, NormalizationMode]]:
    """
    Infers policy features and normalization modes from the model config and stats.

    This function first attempts to find feature definitions and normalization
    mappings directly from the policy's configuration file. If this information is
    not present, it infers it from the extracted normalization statistics, using
    tensor shapes to determine feature shapes and the presence of specific stat
    keys (e.g., 'mean'/'std' vs 'min'/'max') to determine the normalization mode.
    It applies sensible defaults if inference is not possible.

    Args:
        config: The policy's configuration dictionary from `config.json`.
        stats: The normalization statistics extracted from the model's state_dict.

    Returns:
        A tuple containing:
        - A dictionary mapping feature names to `PolicyFeature` objects.
        - A dictionary mapping `FeatureType` enums to `NormalizationMode` enums.
    """
    features = {}
    norm_modes = {}

    # First, check if there's a normalization_mapping in the config
    if "normalization_mapping" in config:
        print(f"Found normalization_mapping in config: {config['normalization_mapping']}")
        # Extract normalization modes from config
        for feature_type_str, mode_str in config["normalization_mapping"].items():
            # Convert string to FeatureType enum
            try:
                if feature_type_str == "VISUAL":
                    feature_type = FeatureType.VISUAL
                elif feature_type_str == "STATE":
                    feature_type = FeatureType.STATE
                elif feature_type_str == "ACTION":
                    feature_type = FeatureType.ACTION
                else:
                    print(f"Warning: Unknown feature type '{feature_type_str}', skipping")
                    continue
            except (AttributeError, ValueError):
                print(f"Warning: Could not parse feature type '{feature_type_str}', skipping")
                continue

            # Convert string to NormalizationMode enum
            try:
                if mode_str == "MEAN_STD":
                    mode = NormalizationMode.MEAN_STD
                elif mode_str == "MIN_MAX":
                    mode = NormalizationMode.MIN_MAX
                elif mode_str == "IDENTITY":
                    mode = NormalizationMode.IDENTITY
                else:
                    print(
                        f"Warning: Unknown normalization mode '{mode_str}' for feature type '{feature_type_str}'"
                    )
                    continue
            except (AttributeError, ValueError):
                print(f"Warning: Could not parse normalization mode '{mode_str}', skipping")
                continue

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
            elif ACTION in key:
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
            elif ACTION in key:
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


def remove_normalization_layers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Creates a new state_dict with all normalization-related layers removed.

    This function filters the original state dictionary, excluding any keys that
    match a set of predefined patterns associated with normalization modules.

    Args:
        state_dict: The original model state dictionary.

    Returns:
        A new state dictionary containing only the core model weights, without
        any normalization parameters.
    """
    new_state_dict = {}

    # Patterns to remove
    remove_patterns = [
        "normalize_inputs.",
        "unnormalize_outputs.",
        "normalize_targets.",  # Added pattern for target normalization
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


def clean_state_dict(
    state_dict: dict[str, torch.Tensor], remove_str: str = "._orig_mod"
) -> dict[str, torch.Tensor]:
    """
    Remove a substring (e.g. '._orig_mod') from all keys in a state dict.

    Args:
        state_dict (dict): The original state dict.
        remove_str (str): The substring to remove from the keys.

    Returns:
        dict: A new state dict with cleaned keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace(remove_str, "")
        new_state_dict[new_k] = v
    return new_state_dict


def load_state_dict_with_missing_key_handling(
    policy: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    policy_type: str,
    known_missing_keys_whitelist: dict[str, list[str]],
) -> list[str]:
    """
    Load state dict into policy with graceful handling of missing keys.

    This function loads the state dict with strict=False, filters out whitelisted
    missing keys, and provides detailed reporting about any issues found.

    Args:
        policy: The policy model to load the state dict into.
        state_dict: The cleaned state dictionary to load.
        policy_type: The type of policy (used for whitelist lookup).
        known_missing_keys_whitelist: Dictionary mapping policy types to lists of
                                     known acceptable missing keys.

    Returns:
        List of problematic missing keys that weren't in the whitelist.
    """
    # Load the cleaned state dict with strict=False to capture missing/unexpected keys
    load_result = policy.load_state_dict(state_dict, strict=False)

    # Check for missing keys
    missing_keys = load_result.missing_keys
    unexpected_keys = load_result.unexpected_keys

    # Filter out whitelisted missing keys
    policy_type_lower = policy_type.lower()
    whitelisted_keys = known_missing_keys_whitelist.get(policy_type_lower, [])
    problematic_missing_keys = [key for key in missing_keys if key not in whitelisted_keys]

    if missing_keys:
        if problematic_missing_keys:
            print(f"WARNING: Found {len(problematic_missing_keys)} unexpected missing keys:")
            for key in problematic_missing_keys:
                print(f"   - {key}")

        if len(missing_keys) > len(problematic_missing_keys):
            whitelisted_missing = [key for key in missing_keys if key in whitelisted_keys]
            print(f"INFO: Found {len(whitelisted_missing)} expected missing keys (whitelisted):")
            for key in whitelisted_missing:
                print(f"   - {key}")

    if unexpected_keys:
        print(f"WARNING: Found {len(unexpected_keys)} unexpected keys:")
        for key in unexpected_keys:
            print(f"   - {key}")

    if not missing_keys and not unexpected_keys:
        print("Successfully loaded cleaned state dict into policy model (all keys matched)")
    else:
        print("State dict loaded with some missing/unexpected keys (see details above)")

    return problematic_missing_keys


def convert_features_to_policy_features(features_dict: dict[str, dict]) -> dict[str, PolicyFeature]:
    """
    Converts a feature dictionary from the old config format to the new `PolicyFeature` format.

    Args:
        features_dict: The feature dictionary in the old format, where values are
                       simple dictionaries (e.g., `{"shape": [7]}`).

    Returns:
        A dictionary mapping feature names to `PolicyFeature` dataclass objects.
    """
    converted_features = {}

    for key, feature_dict in features_dict.items():
        # Determine feature type based on key
        if "image" in key or "visual" in key:
            feature_type = FeatureType.VISUAL
        elif "state" in key:
            feature_type = FeatureType.STATE
        elif ACTION in key:
            feature_type = FeatureType.ACTION
        else:
            feature_type = FeatureType.STATE

        # Get shape from feature dict
        shape = feature_dict.get("shape", feature_dict.get("dim"))
        shape = (shape,) if isinstance(shape, int) else tuple(shape) if shape is not None else ()

        converted_features[key] = PolicyFeature(feature_type, shape)

    return converted_features


def display_migration_summary_with_warnings(problematic_missing_keys: list[str]) -> None:
    """
    Display final migration summary with warnings about problematic missing keys.

    Args:
        problematic_missing_keys: List of missing keys that weren't in the whitelist.
    """
    if not problematic_missing_keys:
        return

    print("\n" + "=" * 60)
    print("IMPORTANT: MIGRATION COMPLETED WITH WARNINGS")
    print("=" * 60)
    print(
        f"The migration was successful, but {len(problematic_missing_keys)} unexpected missing keys were found:"
    )
    print()
    for key in problematic_missing_keys:
        print(f"   - {key}")
    print()
    print("These missing keys may indicate:")
    print("  • The model architecture has changed")
    print("  • Some components were not properly saved in the original model")
    print("  • The migration script needs to be updated for this policy type")
    print()
    print("What to do next:")
    print("  1. Test your migrated model carefully to ensure it works as expected")
    print("  2. If you encounter issues, please open an issue at:")
    print("     https://github.com/huggingface/lerobot/issues")
    print("  3. Include this migration log and the missing keys listed above")
    print()
    print("If the model works correctly despite these warnings, the missing keys")
    print("might be expected for your policy type and can be added to the whitelist.")
    print("=" * 60)


def load_model_from_hub(
    repo_id: str, revision: str | None = None
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any] | None]:
    """
    Downloads and loads a model's state_dict and configs from the Hugging Face Hub.

    Args:
        repo_id: The repository ID on the Hub (e.g., 'lerobot/aloha').
        revision: The specific git revision (branch, tag, or commit hash) to use.

    Returns:
        A tuple containing the model's state dictionary, the policy configuration,
        and the training configuration (None if train_config.json is not found).
    """
    # Download files.
    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)

    # Load state_dict
    state_dict = load_safetensors(safetensors_path)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Try to load train_config (optional)
    train_config = None
    try:
        train_config_path = hf_hub_download(repo_id=repo_id, filename="train_config.json", revision=revision)
        with open(train_config_path) as f:
            train_config = json.load(f)
    except FileNotFoundError:
        print("train_config.json not found - continuing without training configuration")

    return state_dict, config, train_config


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
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Git branch to use when pushing to hub. If specified, a PR will be created automatically (default: push directly to main)",
    )

    args = parser.parse_args()

    # Load model and config
    print(f"Loading model from {args.pretrained_path}...")
    if os.path.isdir(args.pretrained_path):
        # Local directory
        state_dict = load_safetensors(os.path.join(args.pretrained_path, "model.safetensors"))
        with open(os.path.join(args.pretrained_path, "config.json")) as f:
            config = json.load(f)

        # Try to load train_config (optional)
        train_config = None
        train_config_path = os.path.join(args.pretrained_path, "train_config.json")
        if os.path.exists(train_config_path):
            with open(train_config_path) as f:
                train_config = json.load(f)
        else:
            print("train_config.json not found - continuing without training configuration")
    else:
        # Hub repository
        state_dict, config, train_config = load_model_from_hub(args.pretrained_path, args.revision)

    # Extract normalization statistics
    print("Extracting normalization statistics...")
    stats = extract_normalization_stats(state_dict)

    print(f"Found normalization statistics for: {list(stats.keys())}")

    # Detect input features and normalization modes
    print("Detecting features and normalization modes...")
    features, norm_map = detect_features_and_norm_modes(config, stats)

    print(f"Detected features: {list(features.keys())}")
    print(f"Normalization modes: {norm_map}")

    # Remove normalization layers from state_dict
    print("Removing normalization layers from model...")
    new_state_dict = remove_normalization_layers(state_dict)
    new_state_dict = clean_state_dict(new_state_dict, remove_str="._orig_mod")

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

    # Extract policy type from config
    if "type" not in config:
        raise ValueError("Policy type not found in config.json. The config must contain a 'type' field.")

    policy_type = config["type"]
    print(f"Detected policy type: {policy_type}")

    # Clean up config - remove fields that shouldn't be passed to config constructor
    cleaned_config = dict(config)

    # Remove fields that are not part of the config class constructors
    fields_to_remove = ["normalization_mapping", "type"]
    for field in fields_to_remove:
        if field in cleaned_config:
            print(f"Removing '{field}' field from config")
            del cleaned_config[field]

    # Convert input_features and output_features to PolicyFeature objects if they exist
    if "input_features" in cleaned_config:
        cleaned_config["input_features"] = convert_features_to_policy_features(
            cleaned_config["input_features"]
        )
    if "output_features" in cleaned_config:
        cleaned_config["output_features"] = convert_features_to_policy_features(
            cleaned_config["output_features"]
        )

    # Add normalization mapping to config
    cleaned_config["normalization_mapping"] = norm_map

    # Create policy configuration using the factory
    print(f"Creating {policy_type} policy configuration...")
    policy_config = make_policy_config(policy_type, **cleaned_config)

    # Create policy instance using the factory
    print(f"Instantiating {policy_type} policy...")
    policy_class = get_policy_class(policy_type)
    policy = policy_class(policy_config)

    # Define whitelist of known missing keys that are acceptable (for example weight tie) for certain policy types
    known_missing_keys_whitelist = {
        "pi0": ["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"],
        # Add other policy types and their known missing keys here as needed
    }

    # Load state dict with graceful missing key handling
    problematic_missing_keys = load_state_dict_with_missing_key_handling(
        policy=policy,
        state_dict=new_state_dict,
        policy_type=policy_type,
        known_missing_keys_whitelist=known_missing_keys_whitelist,
    )
    policy.to(torch.float32)
    # Create preprocessor and postprocessor using the factory
    print("Creating preprocessor and postprocessor using make_pre_post_processors...")
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_config, dataset_stats=stats)

    # Determine hub repo ID if pushing to hub
    hub_repo_id = None
    if args.push_to_hub:
        if args.hub_repo_id:
            hub_repo_id = args.hub_repo_id
        else:
            if not os.path.isdir(args.pretrained_path):
                # Use same repo with "_migrated" suffix
                hub_repo_id = f"{args.pretrained_path}_migrated"
            else:
                raise ValueError("--hub-repo-id must be specified when pushing local model to hub")

    # Save all components to local directory first
    print(f"Saving preprocessor to {output_dir}...")
    preprocessor.save_pretrained(output_dir)

    print(f"Saving postprocessor to {output_dir}...")
    postprocessor.save_pretrained(output_dir)

    print(f"Saving model to {output_dir}...")
    policy.save_pretrained(output_dir)

    # Generate and save model card
    print("Generating model card...")
    # Get metadata from original config
    dataset_repo_id = "unknown"
    if train_config is not None:
        dataset_repo_id = train_config.get("repo_id", "unknown")
    license = config.get("license", "apache-2.0")

    tags = config.get("tags", ["robotics", "lerobot", policy_type]) or ["robotics", "lerobot", policy_type]
    tags = set(tags).union({"robotics", "lerobot", policy_type})
    tags = list(tags)

    # Generate model card
    card = policy.generate_model_card(
        dataset_repo_id=dataset_repo_id, model_type=policy_type, license=license, tags=tags
    )

    # Save model card locally
    card.save(str(output_dir / "README.md"))
    print(f"Model card saved to {output_dir / 'README.md'}")
    # Push all files to hub in a single operation if requested
    if args.push_to_hub and hub_repo_id:
        api = HfApi()

        # Determine if we should create a PR (automatically if branch is specified)
        create_pr = args.branch is not None
        target_location = f"branch '{args.branch}'" if args.branch else "main branch"

        print(f"Pushing all migrated files to {hub_repo_id} on {target_location}...")

        # Upload all files in a single commit with automatic PR creation if branch specified
        commit_message = "Migrate policy to PolicyProcessorPipeline system"
        commit_description = None

        if create_pr:
            # Separate commit description for PR body
            commit_description = """**Automated Policy Migration to PolicyProcessorPipeline**

This PR migrates your model to the new LeRobot policy format using the modern PolicyProcessorPipeline architecture.

## What Changed

### **New Architecture - PolicyProcessorPipeline**
Your model now uses external PolicyProcessorPipeline components for data processing instead of built-in normalization layers. This provides:
- **Modularity**: Separate preprocessing and postprocessing pipelines
- **Flexibility**: Easy to swap, configure, and debug processing steps
- **Compatibility**: Works with the latest LeRobot ecosystem

### **Normalization Extraction**
We've extracted normalization statistics from your model's state_dict and removed the built-in normalization layers:
- **Extracted patterns**: `normalize_inputs.*`, `unnormalize_outputs.*`, `normalize.*`, `unnormalize.*`, `input_normalizer.*`, `output_normalizer.*`
- **Statistics preserved**: Mean, std, min, max values for all features
- **Clean model**: State dict now contains only core model weights

### **Files Added**
- **preprocessor_config.json**: Configuration for input preprocessing pipeline
- **postprocessor_config.json**: Configuration for output postprocessing pipeline
- **model.safetensors**: Clean model weights without normalization layers
- **config.json**: Updated model configuration
- **train_config.json**: Training configuration
- **README.md**: Updated model card with migration information

### **Benefits**
- **Backward Compatible**: Your model behavior remains identical
- **Future Ready**: Compatible with latest LeRobot features and updates
- **Debuggable**: Easy to inspect and modify processing steps
- **Portable**: Processors can be shared and reused across models

### **Usage**
```python
# Load your migrated model
from lerobot.policies import get_policy_class
from lerobot.processor import PolicyProcessorPipeline

# The preprocessor and postprocessor are now external
preprocessor = PolicyProcessorPipeline.from_pretrained("your-model-repo", config_filename="preprocessor_config.json")
postprocessor = PolicyProcessorPipeline.from_pretrained("your-model-repo", config_filename="postprocessor_config.json")
policy = get_policy_class("your-policy-type").from_pretrained("your-model-repo")

# Process data through the pipeline
processed_batch = preprocessor(raw_batch)
action = policy(processed_batch)
final_action = postprocessor(action)
```

*Generated automatically by the LeRobot policy migration script*"""

        upload_kwargs = {
            "repo_id": hub_repo_id,
            "folder_path": output_dir,
            "repo_type": "model",
            "commit_message": commit_message,
            "revision": args.branch,
            "create_pr": create_pr,
            "allow_patterns": ["*.json", "*.safetensors", "*.md"],
            "ignore_patterns": ["*.tmp", "*.log"],
        }

        # Add commit_description for PR body if creating PR
        if create_pr and commit_description:
            upload_kwargs["commit_description"] = commit_description

        api.upload_folder(**upload_kwargs)

        if create_pr:
            print("All files pushed and pull request created successfully!")
        else:
            print("All files pushed to main branch successfully!")

    print("\nMigration complete!")
    print(f"Migrated model saved to: {output_dir}")
    if args.push_to_hub and hub_repo_id:
        if args.branch:
            print(
                f"Successfully pushed all files to branch '{args.branch}' and created PR on https://huggingface.co/{hub_repo_id}"
            )
        else:
            print(f"Successfully pushed to https://huggingface.co/{hub_repo_id}")
        if args.branch:
            print(f"\nView the branch at: https://huggingface.co/{hub_repo_id}/tree/{args.branch}")
            print(
                f"View the PR at: https://huggingface.co/{hub_repo_id}/discussions (look for the most recent PR)"
            )
        else:
            print(f"\nView the changes at: https://huggingface.co/{hub_repo_id}")

    # Display final summary about any problematic missing keys
    display_migration_summary_with_warnings(problematic_missing_keys)


if __name__ == "__main__":
    main()
