#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenderGenerationPromptStep,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.processor.render_messages_processor import RenderMessagesStep
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_wall_oss_05 import WallOSS05Config, _load_recipe


@ProcessorStepRegistry.register(name="wall_oss_05_task_passthrough")
class WallOSS05TaskPassthrough(ComplementaryDataProcessorStep):
    """Validate language input without selecting, punctuating, or rewriting it."""

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        if "task" not in complementary_data:
            raise KeyError("Wall-OSS-0.5 requires the already-selected LeRobot task string.")
        task = complementary_data["task"]
        if not isinstance(task, str) and not (
            isinstance(task, list) and all(isinstance(value, str) for value in task)
        ):
            raise TypeError("task must be a string or a list of strings.")
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="wall_oss_05_pad_state")
class WallOSS05PadStateProcessorStep(ObservationProcessorStep):
    """Zero-pad state to the fixed 26D Wall-OSS contract before normalization."""

    max_state_dim: int = 26
    # Shared with the crop step so OMX 6D→26D auto-crops actions back to 6D.
    native_dim_holder: dict[str, int] | None = None

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if OBS_STATE not in observation:
            return observation
        state = observation[OBS_STATE]
        state_dim = state.shape[-1]
        if state_dim > self.max_state_dim:
            raise ValueError(
                f"Wall-OSS-0.5 state has {state_dim} dims, which exceeds max_state_dim={self.max_state_dim}."
            )
        if self.native_dim_holder is not None:
            self.native_dim_holder["dim"] = state_dim
        if state_dim < self.max_state_dim:
            observation = dict(observation)
            observation[OBS_STATE] = torch.nn.functional.pad(state, (0, self.max_state_dim - state_dim))
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = {ft: feats.copy() for ft, feats in features.items()}
        obs_feats = new_features.setdefault(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in obs_feats:
            obs_feats[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,))
        return new_features

    def get_config(self) -> dict[str, Any]:
        return {"max_state_dim": self.max_state_dim}


@dataclass
@ProcessorStepRegistry.register(name="wall_oss_05_pad_action")
class WallOSS05PadActionProcessorStep(ProcessorStep):
    """Zero-pad training actions to the fixed 26D Wall-OSS contract before normalization."""

    max_action_dim: int = 26

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Wall-OSS-0.5 action should be a PolicyAction tensor, got {type(action)}.")
        action_dim = action.shape[-1]
        if action_dim > self.max_action_dim:
            raise ValueError(
                f"Wall-OSS-0.5 action has {action_dim} dims, which exceeds max_action_dim={self.max_action_dim}."
            )
        if action_dim == self.max_action_dim:
            return transition
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = torch.nn.functional.pad(
            action, (0, self.max_action_dim - action_dim)
        )
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = {ft: feats.copy() for ft, feats in features.items()}
        action_feats = new_features.setdefault(PipelineFeatureType.ACTION, {})
        action_feats[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.max_action_dim,))
        return new_features

    def get_config(self) -> dict[str, Any]:
        return {"max_action_dim": self.max_action_dim}


@dataclass
@ProcessorStepRegistry.register(name="wall_oss_05_crop_action")
class WallOSS05CropActionProcessorStep(PolicyActionProcessorStep):
    """Crop padded actions back to the robot/env action width after unnormalization.

    ``action_dim`` overrides the auto width tracked by the paired pad step.
    """

    action_dim: int | None = None
    native_dim_holder: dict[str, int] | None = None

    def _resolved_dim(self, action: PolicyAction) -> int | None:
        if self.action_dim is not None:
            return self.action_dim
        if self.native_dim_holder is not None:
            return self.native_dim_holder.get("dim")
        return None

    def action(self, action: PolicyAction) -> PolicyAction:
        dim = self._resolved_dim(action)
        if dim is None or action.shape[-1] <= dim:
            return action
        return action[..., :dim]

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        dim = self.action_dim
        if dim is None:
            return features
        new_features = {ft: feats.copy() for ft, feats in features.items()}
        action_feats = new_features.setdefault(PipelineFeatureType.ACTION, {})
        action_feats[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(dim,))
        return new_features

    def get_config(self) -> dict[str, Any]:
        return {"action_dim": self.action_dim}


@ProcessorStepRegistry.register(name="wall_oss_05_clamp_normalized")
class WallOSS05ClampNormalizedProcessorStep(ProcessorStep):
    """Match Wall's clipped q01/q99 normalization after the standard LeRobot normalizer."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        output = transition.copy()
        observation = output.get(TransitionKey.OBSERVATION)
        if observation is not None and OBS_STATE in observation:
            observation = dict(observation)
            observation[OBS_STATE] = observation[OBS_STATE].clamp(-1, 1)
            output[TransitionKey.OBSERVATION] = observation
        action = output.get(TransitionKey.ACTION)
        if action is not None:
            output[TransitionKey.ACTION] = action.clamp(-1, 1)
        return output

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def reconcile_wall_oss_05_processors(
    config: WallOSS05Config,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Insert pad/crop steps into Hub pipelines that predate auto-padding.

    Pad/crop share a holder so a 6D robot state is zero-padded to 26D for the
    model, then actions are cropped back to 6D automatically.
    """
    native_dim_holder: dict[str, int] = {}
    pre_steps = list(preprocessor.steps)
    if not any(isinstance(step, WallOSS05PadStateProcessorStep) for step in pre_steps):
        insert_idx = next(
            (idx for idx, step in enumerate(pre_steps) if isinstance(step, NormalizerProcessorStep)),
            len(pre_steps),
        )
        pre_steps[insert_idx:insert_idx] = [
            WallOSS05PadStateProcessorStep(
                max_state_dim=config.max_state_dim, native_dim_holder=native_dim_holder
            ),
            WallOSS05PadActionProcessorStep(max_action_dim=config.max_action_dim),
        ]
        preprocessor.steps = pre_steps
    else:
        for step in pre_steps:
            if isinstance(step, WallOSS05PadStateProcessorStep):
                step.native_dim_holder = native_dim_holder
                break

    post_steps = list(postprocessor.steps)
    crop_step = WallOSS05CropActionProcessorStep(
        action_dim=config.postprocess_action_dim,
        native_dim_holder=native_dim_holder,
    )
    crop_idx = next(
        (idx for idx, step in enumerate(post_steps) if isinstance(step, WallOSS05CropActionProcessorStep)),
        None,
    )
    if crop_idx is None:
        insert_idx = next(
            (idx + 1 for idx, step in enumerate(post_steps) if isinstance(step, UnnormalizerProcessorStep)),
            0,
        )
        post_steps.insert(insert_idx, crop_step)
    else:
        post_steps[crop_idx] = crop_step
    postprocessor.steps = post_steps

    return preprocessor, postprocessor


def make_wall_oss_05_pre_post_processors(
    config: WallOSS05Config,
    dataset_stats: dict | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the serializable Wall input/output pipelines."""

    steps = make_default_policy_processor_steps(config, dataset_stats)
    language_steps = []
    if config.recipe_path:
        # Re-resolve the path here, not only in `WallOSS05Config.__post_init__`: a
        # caller may set `recipe_path` after construction, and training must render
        # the same recipe the checkpoint prompts itself with (`config.recipe`).
        config.recipe = _load_recipe(config.recipe_path)

    if config.use_language_recipe or config.recipe_path:
        if config.recipe is None:
            raise ValueError("Wall-OSS-0.5 language training requires a recipe in policy config.")
        language_steps.append(RenderMessagesStep(recipe=config.recipe))
    native_dim_holder: dict[str, int] = {}
    return make_policy_processor_pipelines(
        input_steps=[
            RenderGenerationPromptStep(config.recipe),
            steps.rename_observations,
            steps.add_batch_dim,
            WallOSS05TaskPassthrough(),
            WallOSS05PadStateProcessorStep(
                max_state_dim=config.max_state_dim, native_dim_holder=native_dim_holder
            ),
            WallOSS05PadActionProcessorStep(max_action_dim=config.max_action_dim),
            steps.normalize,
            WallOSS05ClampNormalizedProcessorStep(),
            *language_steps,
            steps.to_device,
        ],
        output_steps=[
            steps.unnormalize,
            WallOSS05CropActionProcessorStep(
                action_dim=config.postprocess_action_dim,
                native_dim_holder=native_dim_holder,
            ),
            steps.to_cpu,
        ],
    )
