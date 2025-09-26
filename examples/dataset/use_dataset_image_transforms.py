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
This example demonstrates how to use image transforms with LeRobot datasets for data augmentation during training.

Image transforms are applied to camera frames to improve model robustness and generalization. They are applied
at training time only, not during dataset recording, allowing you to experiment with different augmentations
without re-recording data.
"""

import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig


def save_image(tensor, filename):
    """Helper function to save a tensor as an image file."""
    if tensor.dim() == 3:  # [C, H, W]
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = torch.clamp(tensor, 0.0, 1.0)
        pil_image = to_pil_image(tensor)
        pil_image.save(filename)
        print(f"Saved: {filename}")
    else:
        print(f"Skipped {filename}: unexpected tensor shape {tensor.shape}")


def example_1_default_transforms():
    """Example 1: Use default transform configuration and save original vs transformed images"""
    print("\n Example 1: Default Transform Configuration with Image Saving")

    repo_id = "pepijn223/record_main_0"  # Example dataset

    try:
        # Load dataset without transforms (original)
        dataset_original = LeRobotDataset(repo_id=repo_id)

        # Load dataset with transforms enabled
        transforms_config = ImageTransformsConfig(
            enable=True,  # Enable transforms (disabled by default)
            max_num_transforms=2,  # Apply up to 2 transforms per frame
            random_order=False,  # Apply in standard order
        )
        dataset_with_transforms = LeRobotDataset(
            repo_id=repo_id, image_transforms=ImageTransforms(transforms_config)
        )

        # Save original and transformed images for comparison
        if len(dataset_original) > 0:
            frame_idx = 0  # Use first frame
            original_sample = dataset_original[frame_idx]
            transformed_sample = dataset_with_transforms[frame_idx]

            print(f"Saving comparison images (frame {frame_idx}):")

            for cam_key in dataset_original.meta.camera_keys:
                if cam_key in original_sample and cam_key in transformed_sample:
                    cam_name = cam_key.replace(".", "_").replace("/", "_")

                    # Save original and transformed images
                    save_image(original_sample[cam_key], f"{cam_name}_original.png")
                    save_image(transformed_sample[cam_key], f"{cam_name}_transformed.png")

    except Exception as e:
        print(f"Could not load dataset '{repo_id}': {e}")


def example_2_custom_transforms():
    """Example 2: Create custom transform configuration and save examples"""
    print("\n Example 2: Custom Transform Configuration")

    repo_id = "pepijn223/record_main_0"  # Example dataset

    try:
        # Create custom transform configuration with strong effects
        custom_transforms_config = ImageTransformsConfig(
            enable=True,
            max_num_transforms=2,  # Apply up to 2 transforms per frame
            random_order=True,  # Apply transforms in random order
            tfs={
                "brightness": ImageTransformConfig(
                    weight=1.0,
                    type="ColorJitter",
                    kwargs={"brightness": (0.5, 1.5)},  # Strong brightness range
                ),
                "contrast": ImageTransformConfig(
                    weight=1.0,  # Higher weight = more likely to be selected
                    type="ColorJitter",
                    kwargs={"contrast": (0.6, 1.4)},  # Strong contrast
                ),
                "sharpness": ImageTransformConfig(
                    weight=0.5,  # Lower weight = less likely to be selected
                    type="SharpnessJitter",
                    kwargs={"sharpness": (0.2, 2.0)},  # Strong sharpness variation
                ),
            },
        )

        dataset_with_custom_transforms = LeRobotDataset(
            repo_id=repo_id, image_transforms=ImageTransforms(custom_transforms_config)
        )

        # Save examples with strong transforms
        if len(dataset_with_custom_transforms) > 0:
            sample = dataset_with_custom_transforms[0]
            print("Saving custom transform examples:")

            for cam_key in dataset_with_custom_transforms.meta.camera_keys:
                if cam_key in sample:
                    cam_name = cam_key.replace(".", "_").replace("/", "_")
                    save_image(sample[cam_key], f"{cam_name}_custom_transforms.png")

    except Exception as e:
        print(f"Could not load dataset '{repo_id}': {e}")


def example_3_torchvision_transforms():
    """Example 3: Use pure torchvision transforms and save examples"""
    print("\n Example 3: Pure Torchvision Transforms")

    repo_id = "pepijn223/record_main_0"  # Example dataset

    try:
        # Create torchvision transform pipeline
        torchvision_transforms = v2.Compose(
            [
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                v2.RandomRotation(degrees=10),  # Small rotation
            ]
        )

        dataset_with_torchvision = LeRobotDataset(repo_id=repo_id, image_transforms=torchvision_transforms)

        # Save examples with torchvision transforms
        if len(dataset_with_torchvision) > 0:
            sample = dataset_with_torchvision[0]
            print("Saving torchvision transform examples:")

            for cam_key in dataset_with_torchvision.meta.camera_keys:
                if cam_key in sample:
                    cam_name = cam_key.replace(".", "_").replace("/", "_")
                    save_image(sample[cam_key], f"{cam_name}_torchvision.png")

    except Exception as e:
        print(f"Could not load dataset '{repo_id}': {e}")


def main():
    """Run all examples"""
    print("LeRobot Dataset Image Transforms Examples")

    example_1_default_transforms()
    example_2_custom_transforms()
    example_3_torchvision_transforms()


if __name__ == "__main__":
    main()
