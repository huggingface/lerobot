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

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ActionProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_fastwam import FastWAMConfig


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

    # NOTE: no visual normalization here. VISUAL is IDENTITY (see configuration_fastwam.normalization_mapping)
    # — images pass through in [0, 1] and the model maps them to the Wan VAE's [-1, 1] at the encode
    # boundary. This is deliberate: `lerobot_train.py` overrides the normalizer stats with
    # `dataset.meta.stats` when fine-tuning, and a real dataset's per-channel image std is the tiny
    # frame-to-frame brightness variance, which would blow images far outside [-1,1] and saturate them.
    # STATE/ACTION still normalize with dataset stats below.
    normalization_stats: dict[str, dict[str, Any]] = dict(dataset_stats or {})

    # NOTE: no resize step here. The model is the single authority on input resolution: it resizes
    # each camera to the per-camera target (image_size split across cameras) in
    # `_stack_video_from_images` / `_prepare_infer_image`, on every path (train forward, rollout and
    # eval select_action). A preprocessor resize step would be both redundant (the model re-resizes
    # anyway) and unsafe across fine-tuning: its `resize_size` would be inherited from the base
    # checkpoint's camera geometry, not this dataset's, making the concatenation N_cameras x too wide.

    steps = make_default_policy_processor_steps(config, normalization_stats, normalizer_device=config.device)

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
    ]
    if config.toggle_action_dimensions:
        output_steps.append(
            FastWAMActionToggleProcessorStep(toggle_dimensions=config.toggle_action_dimensions)
        )
    output_steps.append(steps.to_cpu)
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
