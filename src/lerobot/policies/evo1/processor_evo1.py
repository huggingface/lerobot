# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    create_transition,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    DONE,
    INFO,
    OBS_PREFIX,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    REWARD,
    TRUNCATED,
)

from .configuration_evo1 import Evo1Config


def evo1_batch_to_transition(batch: dict[str, Any]):
    transition = batch_to_transition(batch)
    complementary_data = dict(transition.get("complementary_data") or {})
    reserved = {ACTION, REWARD, DONE, TRUNCATED, INFO}
    for key, value in batch.items():
        if key in reserved or key.startswith(OBS_PREFIX):
            continue
        complementary_data.setdefault(key, value)
    return create_transition(
        observation=transition.get("observation"),
        action=transition.get("action"),
        reward=transition.get("reward", 0.0),
        done=transition.get("done", False),
        truncated=transition.get("truncated", False),
        info=transition.get("info", {}),
        complementary_data=complementary_data,
    )


@dataclass
@ProcessorStepRegistry.register(name="evo1_pad_state_processor")
class Evo1PadStateProcessorStep(ObservationProcessorStep):
    """Pad policy observations to EVO1's fixed state width before normalization."""

    max_state_dim: int = 24

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if OBS_STATE not in observation:
            return observation

        state = observation[OBS_STATE]
        state_dim = state.shape[-1]
        if state_dim > self.max_state_dim:
            raise ValueError(
                f"EVO1 state has {state_dim} dims, which exceeds max_state_dim={self.max_state_dim}."
            )
        if state_dim < self.max_state_dim:
            observation = observation.copy()
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
@ProcessorStepRegistry.register(name="evo1_pad_action_processor")
class Evo1PadActionProcessorStep(ProcessorStep):
    """Pad training actions and preserve the active action dimensions with action_mask."""

    max_action_dim: int = 24

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"EVO1 action should be a PolicyAction tensor, but got {type(action)}.")

        action_dim = action.shape[-1]
        if action_dim > self.max_action_dim:
            raise ValueError(
                f"EVO1 action has {action_dim} dims, which exceeds max_action_dim={self.max_action_dim}."
            )

        new_transition = transition.copy()
        new_action = action
        if action_dim < self.max_action_dim:
            new_action = torch.nn.functional.pad(action, (0, self.max_action_dim - action_dim))

        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        action_mask = complementary_data.get("action_mask")
        if action_mask is None:
            action_mask = torch.ones(action.shape, dtype=torch.bool, device=action.device)
        else:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=action.device)
            if action_mask.shape != action.shape:
                raise ValueError(
                    f"action_mask shape {tuple(action_mask.shape)} does not match action shape {tuple(action.shape)}."
                )
        if action_dim < self.max_action_dim:
            action_mask = torch.nn.functional.pad(action_mask, (0, self.max_action_dim - action_dim))

        complementary_data["action_mask"] = action_mask
        new_transition[TransitionKey.ACTION] = new_action
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
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
@ProcessorStepRegistry.register(name="evo1_action_processor")
class Evo1ActionProcessorStep(PolicyActionProcessorStep):
    """Crop padded EVO1 actions and optionally binarize the LIBERO gripper channel."""

    action_dim: int
    binarize_gripper: bool = False
    gripper_index: int = 6
    gripper_threshold: float = 0.5
    gripper_below_threshold_value: float = 1.0
    gripper_above_threshold_value: float = -1.0

    def action(self, action: PolicyAction) -> PolicyAction:
        if action.shape[-1] < self.action_dim:
            raise ValueError(
                f"EVO1 action has {action.shape[-1]} dims, which is smaller than action_dim={self.action_dim}."
            )

        action = action[..., : self.action_dim]
        if not self.binarize_gripper:
            return action

        if not 0 <= self.gripper_index < self.action_dim:
            raise ValueError(
                f"gripper_index={self.gripper_index} must be within action_dim={self.action_dim}."
            )

        action = action.clone()
        below = torch.as_tensor(
            self.gripper_below_threshold_value,
            dtype=action.dtype,
            device=action.device,
        )
        above = torch.as_tensor(
            self.gripper_above_threshold_value,
            dtype=action.dtype,
            device=action.device,
        )
        action[..., self.gripper_index] = torch.where(
            action[..., self.gripper_index] > self.gripper_threshold,
            above,
            below,
        )
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = {ft: feats.copy() for ft, feats in features.items()}
        action_feats = new_features.setdefault(PipelineFeatureType.ACTION, {})
        action_feats[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        return new_features

    def get_config(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "binarize_gripper": self.binarize_gripper,
            "gripper_index": self.gripper_index,
            "gripper_threshold": self.gripper_threshold,
            "gripper_below_threshold_value": self.gripper_below_threshold_value,
            "gripper_above_threshold_value": self.gripper_above_threshold_value,
        }


def _evo1_action_dim(config: Evo1Config) -> int:
    if config.postprocess_action_dim is not None:
        return config.postprocess_action_dim
    action_feature = config.action_feature
    if action_feature is None:
        return config.max_action_dim
    return int(action_feature.shape[0])


def _evo1_normalization_features(config: Evo1Config) -> dict[str, PolicyFeature]:
    features = {**config.input_features, **config.output_features}
    features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(config.max_state_dim,))
    features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(config.max_action_dim,))
    return features


def _evo1_action_features(config: Evo1Config) -> dict[str, PolicyFeature]:
    return {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(config.max_action_dim,))}


_STAT_PAD_VALUES = {
    "mean": 0.0,
    "std": 1.0,
    "min": -1.0,
    "max": 1.0,
    "q01": -1.0,
    "q99": 1.0,
    "q10": -1.0,
    "q90": 1.0,
}


def _pad_stat_value(value: Any, target_dim: int, stat_name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 0 or tensor.shape[-1] >= target_dim:
        return tensor

    pad_shape = (*tensor.shape[:-1], target_dim - tensor.shape[-1])
    pad_value = _STAT_PAD_VALUES.get(stat_name, 0.0)
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=-1)


def _pad_feature_stats(
    stats: dict[str, dict[str, Any]],
    feature_key: str,
    target_dim: int,
) -> None:
    if feature_key not in stats:
        return
    stats[feature_key] = {
        stat_name: _pad_stat_value(stat_value, target_dim, stat_name)
        for stat_name, stat_value in stats[feature_key].items()
    }


def _pad_evo1_stats(
    config: Evo1Config,
    stats: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    if stats is None:
        return None

    padded_stats = deepcopy(stats)
    # Added dimensions represent zero-padding inside EVO1. These neutral stats keep
    # padded observations at normalized zero and only provide shape compatibility.
    _pad_feature_stats(padded_stats, OBS_STATE, config.max_state_dim)
    _pad_feature_stats(padded_stats, ACTION, config.max_action_dim)
    return padded_stats


def reconcile_evo1_processors(
    config: Evo1Config,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Reconcile checkpoint-loaded pipelines with the current EVO1 config.

    Two things cannot be restored from a serialized pipeline alone: the EVO1 batch converter
    (converters are plain functions and are never serialized), and eval-time CLI overrides of the
    action postprocessing flags (`postprocess_action_dim`, `binarize_gripper`, `gripper_*`). This
    restores the converter and rebuilds the action step from the current config so those overrides
    take effect.
    """
    # Pipelines reloaded from a checkpoint come back with the default batch converter, which drops
    # non-observation extras (embodiment_id, state_mask, custom task fields) needed by EVO1.
    preprocessor.to_transition = evo1_batch_to_transition

    action_step = Evo1ActionProcessorStep(
        action_dim=_evo1_action_dim(config),
        binarize_gripper=config.binarize_gripper,
        gripper_index=config.gripper_index,
        gripper_threshold=config.gripper_threshold,
        gripper_below_threshold_value=config.gripper_below_threshold_value,
        gripper_above_threshold_value=config.gripper_above_threshold_value,
    )
    steps = list(postprocessor.steps)
    action_step_idx = next(
        (idx for idx, step in enumerate(steps) if isinstance(step, Evo1ActionProcessorStep)), None
    )
    if action_step_idx is None:
        insert_idx = next(
            (idx + 1 for idx, step in enumerate(steps) if isinstance(step, UnnormalizerProcessorStep)),
            0,
        )
        steps.insert(insert_idx, action_step)
    else:
        steps[action_step_idx] = action_step
    postprocessor.steps = steps

    return preprocessor, postprocessor


def make_evo1_pre_post_processors(
    config: Evo1Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    normalization_features = _evo1_normalization_features(config)
    action_features = _evo1_action_features(config)
    normalization_stats = _pad_evo1_stats(config, dataset_stats)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        Evo1PadStateProcessorStep(max_state_dim=config.max_state_dim),
        Evo1PadActionProcessorStep(max_action_dim=config.max_action_dim),
        NormalizerProcessorStep(
            features=normalization_features,
            norm_map=config.normalization_mapping,
            stats=normalization_stats,
        ),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=action_features,
            norm_map=config.normalization_mapping,
            stats=normalization_stats,
        ),
        Evo1ActionProcessorStep(
            action_dim=_evo1_action_dim(config),
            binarize_gripper=config.binarize_gripper,
            gripper_index=config.gripper_index,
            gripper_threshold=config.gripper_threshold,
            gripper_below_threshold_value=config.gripper_below_threshold_value,
            gripper_above_threshold_value=config.gripper_above_threshold_value,
        ),
        # float32 so downstream numpy conversion works even when the policy computes in bf16.
        DeviceProcessorStep(device="cpu", float_dtype="float32"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=evo1_batch_to_transition,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
