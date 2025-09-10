#!/usr/bin/env python
"""Script for Pi0 pretrained policy inference and Hub upload."""

import argparse
from datetime import datetime

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

# Set seed
torch.manual_seed(42)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pi0 policy inference and Hub upload")
    parser.add_argument(
        "--source-model-id",
        type=str,
        default="pepijn223/pi0_libero_lerobot",
        help="Source model repository ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--dataset-id", type=str, default="pepijn223/libero", help="Dataset repository ID on Hugging Face Hub"
    )
    parser.add_argument(
        "--output-model-id",
        type=str,
        required=True,
        help="Output model repository ID to upload to (e.g., 'your-username/pi0-libero-fixed')",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to run inference on"
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to load from dataset")
    parser.add_argument(
        "--sample-idx", type=int, default=10, help="Sample index within episode to use for inference"
    )
    parser.add_argument("--private", action="store_true", help="Make the uploaded model private")
    parser.add_argument(
        "--commit-message", type=str, default=None, help="Custom commit message for the upload"
    )
    return parser.parse_args()


def _inject_normalization_stats(policy: PI0Policy, dataset_meta: LeRobotDatasetMetadata, key_mapping: dict):
    """Recreate normalization layers with proper stats from the dataset."""
    from lerobot.policies.normalize import Normalize, Unnormalize

    # Convert numpy stats to the format expected by normalization layers and remap keys
    stats = {}
    for dataset_key, stat_dict in dataset_meta.stats.items():
        # Use mapped key if available, otherwise use original key
        policy_key = key_mapping.get(dataset_key, dataset_key)

        stats[policy_key] = {
            stat_type: torch.from_numpy(stat_array) if isinstance(stat_array, np.ndarray) else stat_array
            for stat_type, stat_array in stat_dict.items()
        }

    print(f"Available stats keys: {list(stats.keys())}")
    print(
        f"Policy expects keys: input={list(policy.config.input_features.keys())}, output={list(policy.config.output_features.keys())}"
    )

    # Recreate normalization layers with proper stats
    normalize_inputs = Normalize(policy.config.input_features, policy.config.normalization_mapping, stats)

    normalize_targets = Normalize(policy.config.output_features, policy.config.normalization_mapping, stats)

    unnormalize_outputs = Unnormalize(
        policy.config.output_features, policy.config.normalization_mapping, stats
    )

    # Replace the normalization layers on the policy
    policy.normalize_inputs = normalize_inputs
    policy.normalize_targets = normalize_targets
    policy.unnormalize_outputs = unnormalize_outputs

    print("Normalization layers recreated with dataset stats.")


def configure_policy_features(policy: PI0Policy, dataset: LeRobotDataset):
    """Configure policy input and output features based on dataset metadata."""
    print(f"Dataset features: {list(dataset.meta.features.keys())}")

    # Create a proper mapping from dataset keys to policy keys
    dataset_to_policy_mapping = {}

    # Handle images
    if "image" in dataset.meta.features:
        dataset_to_policy_mapping["image"] = "observation.images.image"
    if "wrist_image" in dataset.meta.features:
        dataset_to_policy_mapping["wrist_image"] = "observation.images.image2"

    # Handle state
    if "state" in dataset.meta.features:
        dataset_to_policy_mapping["state"] = "observation.state"

    # Handle actions
    if "actions" in dataset.meta.features:
        dataset_to_policy_mapping["actions"] = "action"

    print(f"Key mapping: {dataset_to_policy_mapping}")

    # Clear existing input features and reconfigure with proper mapping
    policy.config.input_features = {}
    policy.config.output_features = {}

    # Map visual features
    for dataset_key, policy_key in dataset_to_policy_mapping.items():
        if dataset_key in ["image", "wrist_image"]:
            feature_info = dataset.meta.features[dataset_key]
            # Convert HWC to CHW format and resize
            shape = (3, 224, 224)  # Pi0 expects CHW format
            policy.config.input_features[policy_key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)

    # Map state features
    for dataset_key, policy_key in dataset_to_policy_mapping.items():
        if dataset_key == "state":
            feature_info = dataset.meta.features[dataset_key]
            shape = tuple(feature_info["shape"])
            policy.config.input_features[policy_key] = PolicyFeature(type=FeatureType.STATE, shape=shape)

    # Map action features
    for dataset_key, policy_key in dataset_to_policy_mapping.items():
        if dataset_key == "actions":
            feature_info = dataset.meta.features[dataset_key]
            shape = tuple(feature_info["shape"])
            policy.config.output_features[policy_key] = PolicyFeature(type=FeatureType.ACTION, shape=shape)

    print(f"Policy input_features: {list(policy.config.input_features.keys())}")
    print(f"Policy output_features: {list(policy.config.output_features.keys())}")
    print(f"Policy image_features: {list(policy.config.image_features.keys())}")
    print(f"Policy action_feature: {policy.config.action_feature}")

    return dataset_to_policy_mapping


def fix_buffer_naming(policy: PI0Policy):
    """Fix buffer naming issues in the loaded policy state dict."""
    print("Fixing normalization buffer naming issues...")

    state_dict = policy.state_dict()
    corrected_state_dict = {}
    fixes_applied = 0

    for key, value in state_dict.items():
        new_key = key

        # Fix buffer naming: buffer_observation_state_mean -> buffer_observation_state.mean
        if "buffer_observation_state_mean" in key:
            new_key = key.replace("buffer_observation_state_mean", "buffer_observation_state.mean")
            fixes_applied += 1
            print(f"  Fixed: {key} -> {new_key}")
        elif "buffer_observation_state_std" in key:
            new_key = key.replace("buffer_observation_state_std", "buffer_observation_state.std")
            fixes_applied += 1
            print(f"  Fixed: {key} -> {new_key}")
        # Remove image buffers that aren't expected (they cause conflicts)
        elif "buffer_observation_image_mean" in key or "buffer_observation_image_std" in key:
            print(f"  Removed unexpected buffer: {key}")
            continue  # Skip this buffer

        corrected_state_dict[new_key] = value

    # Add missing action buffers with dummy values (will be replaced by dataset stats)
    missing_buffers = [
        "normalize_targets.buffer_action.mean",
        "normalize_targets.buffer_action.std",
        "unnormalize_outputs.buffer_action.mean",
        "unnormalize_outputs.buffer_action.std",
    ]

    for buffer_key in missing_buffers:
        if buffer_key not in corrected_state_dict:
            # Use dummy values - these will be overwritten by proper dataset stats later
            if "mean" in buffer_key:
                corrected_state_dict[buffer_key] = torch.zeros(8)  # Assume 8-dim action
            else:  # std
                corrected_state_dict[buffer_key] = torch.ones(8)  # Assume 8-dim action
            fixes_applied += 1
            print(f"  Added missing buffer: {buffer_key}")

    print(f"Applied {fixes_applied} buffer fixes")

    # Load the corrected state dict back into the policy
    policy.load_state_dict(corrected_state_dict)
    return policy


def main():
    """Main function to run the Pi0 inference and upload."""
    args = parse_args()

    # Load pretrained Pi0 model directly from Hugging Face Hub
    print(f"Loading pretrained Pi0 model from {args.source_model_id}...")

    # Load with strict=False to allow missing/unexpected keys, then fix them manually
    policy = PI0Policy.from_pretrained(args.source_model_id, strict=False)
    policy = fix_buffer_naming(policy)
    policy.eval()
    policy.to(args.device)

    # Load dataset and get a sample
    print(f"Loading dataset: {args.dataset_id}")
    dataset = LeRobotDataset(args.dataset_id, episodes=[args.episode])
    meta: LeRobotDatasetMetadata = dataset.meta
    sample = dataset[args.sample_idx]

    # Configure policy features
    key_mapping = configure_policy_features(policy, dataset)

    # Inject normalization stats with proper key mapping
    _inject_normalization_stats(policy, meta, key_mapping)

    # Prepare batch for PI0 (handle temporal dimensions)
    batch = {}

    # Map dataset sample keys to policy keys
    reverse_mapping = {v: k for k, v in key_mapping.items()}

    for policy_key in policy.config.input_features:
        # Find the corresponding dataset key
        dataset_key = reverse_mapping.get(policy_key, policy_key)

        if dataset_key in sample:
            data = sample[dataset_key]

            # Handle image data: convert from HWC to CHW and normalize
            if policy_key.startswith("observation.images."):
                if data.dim() == 3 and data.shape[-1] == 3:  # HWC format
                    data = data.permute(2, 0, 1)  # Convert to CHW
                # Normalize to [0, 1] range if needed
                if data.dtype == torch.uint8:
                    data = data.float() / 255.0
                # Resize to expected size if needed
                if data.shape[-2:] != (224, 224):
                    import torch.nn.functional as F  # noqa: N812

                    data = F.interpolate(
                        data.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                    )[0]

            # Remove temporal dimension if present
            if data.dim() > len(policy.config.input_features[policy_key].shape):
                data = data[0]

            batch[policy_key] = data.unsqueeze(0)  # Add batch dimension

    # Debug: print what's in the sample
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Batch keys prepared: {list(batch.keys())}")

    # Pi0 requires task description - add a default if not available
    if "task" in sample:
        batch["task"] = [sample["task"]]  # Keep as list of strings
    else:
        print("No task in sample, using default task description")
        batch["task"] = ["Complete the manipulation task"]

    print(f"Task: {batch['task'][0]}")
    print(f"Final batch keys: {list(batch.keys())}")

    # Run inference
    with torch.no_grad():
        action = policy.select_action(batch)
        print(f"Predicted action shape: {action.shape}")
        print(f"Predicted action: {action.tolist()}")

    print("✅ Pi0 pretrained inference completed successfully!")

    # Upload to Hugging Face Hub
    print(f"\n📤 Uploading model to Hugging Face Hub: {args.output_model_id}")

    # Create commit message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = (
        args.commit_message
        or f"Pi0 model with injected normalization stats from {args.dataset_id} - {timestamp}"
    )

    # Update model configuration with dataset info
    policy.config.push_to_hub = True
    policy.config.repo_id = args.output_model_id
    policy.config.private = args.private

    # Add metadata about the adaptation
    adaptation_info = {
        "source_model": args.source_model_id,
        "dataset_used": args.dataset_id,
        "adaptation_date": timestamp,
        "stats_injected": True,
        "key_mapping": key_mapping,
        "inference_test_passed": True,
        "sample_action_shape": list(action.shape),
    }

    try:
        # Push to hub
        policy.push_to_hub(
            repo_id=args.output_model_id,
            private=args.private,
            commit_message=commit_message,
            create_pr=False,
        )

        # Also save the adaptation info as a separate file
        import json
        import os
        import tempfile

        from huggingface_hub import HfApi

        api = HfApi()

        # Create a temporary file with adaptation info
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(adaptation_info, f, indent=2)
            temp_path = f.name

        try:
            api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="adaptation_info.json",
                repo_id=args.output_model_id,
                commit_message=f"Add adaptation metadata - {timestamp}",
            )
        finally:
            os.unlink(temp_path)

        print(f"✅ Model successfully uploaded to: https://huggingface.co/{args.output_model_id}")
        print("📋 Adaptation info:")
        for key, value in adaptation_info.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Error uploading to Hub: {e}")
        raise


if __name__ == "__main__":
    main()
