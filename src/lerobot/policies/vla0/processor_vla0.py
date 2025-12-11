#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
VLA-0 Processor

Handles pre-processing and post-processing for VLA-0 policy.
"""

from typing import Any

import torch
import torch.nn.functional as F

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.vla0.configuration_vla0 import VLA0Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_vla0_pre_post_processors(
    config: VLA0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the VLA-0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match expected configurations.
    2. Adding a batch dimension for single samples.
    3. Resizing images to the target size.
    4. Normalizing inputs based on dataset statistics.
    5. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing actions to their original scale (if needed).

    Args:
        config: The configuration object for the VLA-0 policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        VLA0ImagePreprocessor(target_size=config.image_size),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@ProcessorStepRegistry.register(name="vla0_image_preprocessor")
class VLA0ImagePreprocessor(ComplementaryDataProcessorStep):
    """
    Preprocessor step that resizes images to the target size for VLA-0.

    VLA-0 uses Qwen2.5-VL which expects specific image dimensions.
    This preprocessor ensures images are properly resized.
    """

    def __init__(self, target_size: int = 224):
        """
        Args:
            target_size: Target size for image resizing (both height and width)
        """
        self.target_size = target_size

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process the data by resizing images."""
        result = dict(data)

        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 3:
                # Check if this looks like an image (has 3 or 4 channels in position 0 or 1)
                if value.shape[-3] in [1, 3, 4]:  # Assumes (B, C, H, W) or (C, H, W)
                    # Resize if not already target size
                    h, w = value.shape[-2:]
                    if h != self.target_size or w != self.target_size:
                        result[key] = self._resize_image(value)

        return result

    def _resize_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Resize image to target size while maintaining aspect ratio with padding.

        Args:
            img: Image tensor of shape (..., C, H, W)

        Returns:
            Resized image tensor of shape (..., C, target_size, target_size)
        """
        # Store original shape for reshaping later
        orig_shape = img.shape[:-3]
        c, h, w = img.shape[-3:]

        # Flatten batch dimensions if needed
        if img.ndim > 4:
            img = img.reshape(-1, c, h, w)
        elif img.ndim == 3:
            img = img.unsqueeze(0)

        # Calculate scaling factor to fit in target size
        scale = min(self.target_size / h, self.target_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize
        resized = F.interpolate(
            img.float(),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

        # Pad to target size (pad on right and bottom)
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        padded = F.pad(resized, (0, pad_w, 0, pad_h), mode="constant", value=0)

        # Restore original batch dimensions
        if orig_shape:
            padded = padded.reshape(*orig_shape, c, self.target_size, self.target_size)
        else:
            padded = padded.squeeze(0)

        return padded.to(img.dtype)

    def complementary_data(self, complementary_data: dict) -> dict:
        """Pass through complementary data unchanged."""
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Update feature shapes to reflect resized images."""
        new_features = {}
        for ft_type, ft_dict in features.items():
            new_ft_dict = {}
            for name, feature in ft_dict.items():
                # Check if this is a visual feature
                if hasattr(feature, "type") and str(feature.type).upper() == "VISUAL":
                    # Update shape to target size
                    if len(feature.shape) >= 2:
                        new_shape = (*feature.shape[:-2], self.target_size, self.target_size)
                        new_feature = PolicyFeature(
                            type=feature.type,
                            shape=new_shape,
                        )
                        new_ft_dict[name] = new_feature
                    else:
                        new_ft_dict[name] = feature
                else:
                    new_ft_dict[name] = feature
            new_features[ft_type] = new_ft_dict
        return new_features

    def to_json_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            "name": "vla0_image_preprocessor",
            "target_size": self.target_size,
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "VLA0ImagePreprocessor":
        """Deserialize from JSON-compatible dictionary."""
        return cls(target_size=json_dict.get("target_size", 224))


@ProcessorStepRegistry.register(name="vla0_action_bounds_processor")
class VLA0ActionBoundsProcessor(ComplementaryDataProcessorStep):
    """
    Processor step that normalizes actions to the bounds expected by VLA-0.

    VLA-0 discretizes actions to integers in [0, num_bins], so this processor
    ensures actions are scaled appropriately.
    """

    def __init__(
        self,
        action_min: list[float],
        action_max: list[float],
        num_bins: int = 1000,
    ):
        """
        Args:
            action_min: Minimum values for each action dimension
            action_max: Maximum values for each action dimension
            num_bins: Number of bins for discretization
        """
        self.action_min = torch.tensor(action_min, dtype=torch.float32)
        self.action_max = torch.tensor(action_max, dtype=torch.float32)
        self.num_bins = num_bins

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pass through - action bounds are handled in the model."""
        return data

    def complementary_data(self, complementary_data: dict) -> dict:
        """Pass through complementary data unchanged."""
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass through features unchanged."""
        return features

    def to_json_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            "name": "vla0_action_bounds_processor",
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
            "num_bins": self.num_bins,
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "VLA0ActionBoundsProcessor":
        """Deserialize from JSON-compatible dictionary."""
        return cls(
            action_min=json_dict["action_min"],
            action_max=json_dict["action_max"],
            num_bins=json_dict.get("num_bins", 1000),
        )
