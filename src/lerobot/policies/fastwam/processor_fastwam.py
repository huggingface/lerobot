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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ActionProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    ImageCropResizeProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_fastwam import FastWAMConfig


@dataclass
@ProcessorStepRegistry.register(name="fastwam_image_crop_resize_processor")
class FastWAMImageCropResizeProcessorStep(ImageCropResizeProcessorStep):
    """`ImageCropResizeProcessorStep` that tolerates a leading temporal/batch stack.

    FastWAM loads a per-camera video stack, so image observations arrive as
    ``[B, T, C, H, W]``. torchvision's crop/resize only accept ``[..., C, H, W]`` with a
    single leading batch dim (resize raises on 5-D input), so we flatten any leading
    dims into the batch, apply the base 4-D crop/resize, then restore the leading shape.
    Crop/resize params and feature-shape bookkeeping are inherited unchanged.
    """

    def observation(self, observation: dict) -> dict:
        leads: dict[str, tuple] = {}
        flat_input = dict(observation)
        for key, img in observation.items():
            if "image" in key and torch.is_tensor(img) and img.ndim > 4:
                leads[key] = tuple(img.shape[:-3])
                flat_input[key] = img.reshape(-1, *img.shape[-3:])
        processed = super().observation(flat_input)
        if not leads:
            return processed
        out = dict(processed)
        for key, lead in leads.items():
            im = processed[key]
            out[key] = im.reshape(*lead, *im.shape[-3:])
        return out


@dataclass
@ProcessorStepRegistry.register(name="fastwam_action_toggle_processor")
class FastWAMActionToggleProcessorStep(ActionProcessorStep):
    """Apply FastWAM LIBERO toggle semantics to configured action dimensions."""

    toggle_dimensions: list[int]

    def action(self, action: PolicyAction) -> PolicyAction:
        if not self.toggle_dimensions:
            return action
        processed_action = action.clone()
        action_dim = int(processed_action.shape[-1])
        for dim in self.toggle_dimensions:
            resolved_dim = dim if dim >= 0 else action_dim + dim
            if resolved_dim < 0 or resolved_dim >= action_dim:
                raise ValueError(
                    f"FastWAM action toggle dimension {dim} is out of bounds for action dim {action_dim}."
                )
            value = processed_action[..., resolved_dim]
            value = value * 2.0 - 1.0
            processed_action[..., resolved_dim] = torch.sign(-value)
        return processed_action

    def get_config(self) -> dict[str, Any]:
        return {"toggle_dimensions": self.toggle_dimensions}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_fastwam_pre_post_processors(
    config: FastWAMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Create LeRobot pre- and post-processing pipelines for FastWAM.

    Args:
        config (FastWAMConfig): Policy configuration controlling device and
            normalization feature metadata.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): Optional
            LeRobot dataset statistics used by normalization processors.

    Returns:
        tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]: Input and
        output processor pipelines discoverable by LeRobot.
    """

    # force visual stats to be mean 0.5 and std 0.5 to map [0, 1] data to [-1, 1]
    normalization_stats: dict[str, dict[str, Any]] = dict(dataset_stats or {})
    for key, feature in config.input_features.items():
        if feature.type != FeatureType.VISUAL:
            continue
        channels = int(feature.shape[0])
        normalization_stats[key] = {
            "mean": torch.full((channels, 1, 1), 0.5, dtype=torch.float32),
            "std": torch.full((channels, 1, 1), 0.5, dtype=torch.float32),
        }

    # resize visual inputs to match model expected input size, if necessary
    visual_shapes = [
        feature.shape
        for feature in config.input_features.values()
        if feature.type == FeatureType.VISUAL
    ]
    resize_steps = []
    if visual_shapes:
        target_hw = (int(visual_shapes[0][1]), int(visual_shapes[0][2]))
        # FastWAM-aware resize: tolerates the leading temporal dim of the video stack.
        resize_steps.append(FastWAMImageCropResizeProcessorStep(resize_size=target_hw))

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        *resize_steps,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=normalization_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=normalization_stats,
        ),
    ]
    if config.toggle_action_dimensions:
        output_steps.append(
            FastWAMActionToggleProcessorStep(toggle_dimensions=config.toggle_action_dimensions)
        )
    output_steps.append(DeviceProcessorStep(device="cpu"))
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
