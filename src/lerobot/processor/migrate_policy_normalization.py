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
        --policy-type act \
        --push-to-hub
"""

import argparse
import importlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

from .batch_processor import AddBatchDimensionProcessorStep
from .device_processor import DeviceProcessorStep
from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from .pipeline import PolicyProcessorPipeline
from .rename_processor import RenameProcessorStep

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
        elif "action" in key:
            feature_type = FeatureType.ACTION
        else:
            feature_type = FeatureType.STATE

        # Get shape from feature dict
        shape = feature_dict.get("shape", feature_dict.get("dim"))
        shape = (shape,) if isinstance(shape, int) else tuple(shape)

        converted_features[key] = PolicyFeature(feature_type, shape)

    return converted_features


def load_model_from_hub(
    repo_id: str, revision: str = None
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """
    Downloads and loads a model's state_dict and configs from the Hugging Face Hub.

    Args:
        repo_id: The repository ID on the Hub (e.g., 'lerobot/aloha').
        revision: The specific git revision (branch, tag, or commit hash) to use.

    Returns:
        A tuple containing the model's state dictionary, the policy configuration,
        and the training configuration.
    """
    # Download files.
    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
    train_config_path = hf_hub_download(repo_id=repo_id, filename="train_config.json", revision=revision)

    # Load state_dict
    state_dict = load_safetensors(safetensors_path)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    with open(train_config_path) as f:
        train_config = json.load(f)

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

    args = parser.parse_args()

    # Load model and config
    print(f"Loading model from {args.pretrained_path}...")
    if os.path.isdir(args.pretrained_path):
        # Local directory
        state_dict = load_safetensors(os.path.join(args.pretrained_path, "model.safetensors"))
        with open(os.path.join(args.pretrained_path, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(args.pretrained_path, "train_config.json")) as f:
            train_config = json.load(f)
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

    # Clean up config - remove normalization_mapping field
    cleaned_config = dict(config)
    if "normalization_mapping" in cleaned_config:
        print("Removing 'normalization_mapping' field from config")
        del cleaned_config["normalization_mapping"]
    policy_type = deepcopy(cleaned_config["type"])

    del cleaned_config["type"]

    # Instantiate the policy model with cleaned config and load the cleaned state dict
    print(f"Instantiating {policy_type} policy model...")
    policy_class_path = POLICY_CLASSES[policy_type]
    module_path, class_name = policy_class_path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    policy_class = getattr(module, class_name)

    # Create config class instance
    config_module_path = module_path.replace("modeling", "configuration")
    config_module = importlib.import_module(config_module_path)
    # Handle special cases for config class names
    config_class_names = {
        "act": "ACTConfig",
        "diffusion": "DiffusionConfig",
        "pi0": "PI0Config",
        "pi0fast": "PI0FASTConfig",
        "smolvla": "SmolVLAConfig",
        "tdmpc": "TDMPCConfig",
        "vqbet": "VQBeTConfig",
        "sac": "SACConfig",
        "classifier": "ClassifierConfig",
    }
    config_class_name = config_class_names.get(policy_type, f"{policy_type.upper()}Config")
    config_class = getattr(config_module, config_class_name)

    # Convert input_features and output_features to PolicyFeature objects - these are mandatory
    if "input_features" not in cleaned_config:
        raise ValueError("Missing mandatory 'input_features' in config")
    if "output_features" not in cleaned_config:
        raise ValueError("Missing mandatory 'output_features' in config")

    cleaned_config["input_features"] = convert_features_to_policy_features(cleaned_config["input_features"])
    cleaned_config["output_features"] = convert_features_to_policy_features(cleaned_config["output_features"])

    # Create config instance from cleaned config dict
    policy_config = config_class(**cleaned_config)

    # Create policy instance - some policies expect dataset_stats
    policy = policy_class(policy_config)

    # Load the cleaned state dict
    policy.load_state_dict(new_state_dict, strict=True)
    print("Successfully loaded cleaned state dict into policy model")

    # Now create preprocessor and postprocessor with cleaned_config available
    print("Creating preprocessor and postprocessor...")
    # The pattern from existing processor factories:
    # - Preprocessor has two NormalizerProcessorSteps: one for input_features, one for output_features
    # - Postprocessor has one UnnormalizerProcessorStep for output_features only

    # Get features from cleaned_config (now they're PolicyFeature objects)
    input_features = cleaned_config.get("input_features", {})
    output_features = cleaned_config.get("output_features", {})

    # Create preprocessor with two normalizers (following the pattern from processor factories)
    preprocessor_steps = [
        RenameProcessorStep(rename_map={}),
        NormalizerProcessorStep(
            features={**input_features, **output_features},
            norm_map=norm_map,
            stats=stats,
        ),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=policy_config.device),
    ]
    preprocessor = PolicyProcessorPipeline(steps=preprocessor_steps, name="robot_preprocessor")

    # Create postprocessor with unnormalizer for outputs only
    postprocessor_steps = [
        DeviceProcessorStep(device="cpu"),
        UnnormalizerProcessorStep(features=output_features, norm_map=norm_map, stats=stats),
    ]
    postprocessor = PolicyProcessorPipeline(steps=postprocessor_steps, name="robot_postprocessor")

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

    # Save preprocessor and postprocessor to root directory
    print(f"Saving preprocessor to {output_dir}...")
    preprocessor.save_pretrained(output_dir)
    if args.push_to_hub:
        preprocessor.push_to_hub(repo_id=hub_repo_id, private=args.private)

    print(f"Saving postprocessor to {output_dir}...")
    postprocessor.save_pretrained(output_dir)
    if args.push_to_hub:
        postprocessor.push_to_hub(repo_id=hub_repo_id, private=args.private)

    # Save model using the policy's save_pretrained method
    print(f"Saving model to {output_dir}...")
    policy.save_pretrained(
        output_dir, push_to_hub=args.push_to_hub, repo_id=hub_repo_id, private=args.private
    )

    # Generate and save model card
    print("Generating model card...")
    # Get metadata from original config
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
    # Push model card to hub if requested
    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(output_dir / "README.md"),
            path_in_repo="README.md",
            repo_id=hub_repo_id,
            repo_type="model",
            commit_message="Add model card for migrated model",
        )
        print("Model card pushed to hub")

    print("\nMigration complete!")
    print(f"Migrated model saved to: {output_dir}")
    if args.push_to_hub:
        print(f"Successfully pushed to https://huggingface.co/{hub_repo_id}")


if __name__ == "__main__":
    main()
