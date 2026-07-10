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

import json
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_lawam import LaWAMConfig


@ProcessorStepRegistry.register(name="lawam_clip_actions")
class LaWAMClipActionsProcessorStep(ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        transition = dict(transition)
        transition[TransitionKey.ACTION] = action.clamp(-1.0, 1.0)
        return transition

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="lawam_pre_snap_gripper")
class LaWAMPreSnapGripperProcessorStep(ProcessorStep):
    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None or action.shape[-1] <= self.gripper_dim:
            return transition
        transition = dict(transition)
        snapped = action.clone()
        snapped[..., self.gripper_dim] = (snapped[..., self.gripper_dim] >= self.threshold).float()
        transition[TransitionKey.ACTION] = snapped
        return transition

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="lawam_binarize_gripper")
class LaWAMBinarizeGripperProcessorStep(ProcessorStep):
    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None or action.shape[-1] <= self.gripper_dim:
            return transition
        transition = dict(transition)
        binarized = action.clone()
        binarized[..., self.gripper_dim] = (
            2.0 * (binarized[..., self.gripper_dim] > self.threshold).float() - 1.0
        )
        transition[TransitionKey.ACTION] = binarized
        return transition

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="lawam_libero_state")
class LaWAMLiberoStateProcessorStep(ProcessorStep):
    def __init__(self, target_state_dim: int):
        self.target_state_dim = int(target_state_dim)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or OBS_STATE not in observation:
            return transition

        state = torch.as_tensor(observation[OBS_STATE])
        if state.shape[-1] == self.target_state_dim:
            return transition
        if state.shape[-1] != 8 or self.target_state_dim != 7:
            return transition

        transition = dict(transition)
        observation = dict(observation)
        observation[OBS_STATE] = torch.cat((state[..., :6], state[..., -1:]), dim=-1)
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"target_state_dim": self.target_state_dim}

    def transform_features(self, features):
        return features


def _resolve_native_lawam_stats_path(config: LaWAMConfig) -> Path | None:
    if config.lawam_dataset_stats_path is not None:
        stats_path = Path(config.lawam_dataset_stats_path).expanduser()
        if not stats_path.exists():
            raise FileNotFoundError(f"`policy.lawam_dataset_stats_path` does not exist: {stats_path}")
        return stats_path

    if config.lawam_checkpoint_path is None:
        return None

    checkpoint_path = Path(config.lawam_checkpoint_path).expanduser()
    if checkpoint_path.suffix != ".pt" or len(checkpoint_path.parents) < 2:
        return None

    stats_path = checkpoint_path.parents[1] / "dataset_statistics.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            "Original LaWAM checkpoints require dataset statistics for evaluation. "
            f"Expected `{stats_path}` next to the checkpoint run directory, or pass "
            "`policy.lawam_dataset_stats_path` explicitly."
        )
    return stats_path


def _pick_native_lawam_dataset_stats(native_stats: dict[str, Any], unnorm_key: str | None) -> dict[str, Any]:
    if unnorm_key is None:
        if len(native_stats) != 1:
            raise ValueError(
                "LaWAM dataset statistics contain multiple keys; please pass "
                f"`policy.lawam_unnorm_key`. Available: {list(native_stats.keys())}"
            )
        unnorm_key = next(iter(native_stats.keys()))

    if unnorm_key not in native_stats:
        raise ValueError(
            f"Invalid `policy.lawam_unnorm_key`={unnorm_key}. Available: {list(native_stats.keys())}"
        )
    return native_stats[unnorm_key]


def _convert_native_lawam_stats(
    config: LaWAMConfig, native_dataset_stats: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    dataset_stats: dict[str, dict[str, Any]] = {}
    if "state" in native_dataset_stats:
        dataset_stats[OBS_STATE] = dict(native_dataset_stats["state"])
    if config.action_feature is not None and "action" in native_dataset_stats:
        dataset_stats[ACTION] = dict(native_dataset_stats["action"])
    return dataset_stats


def _load_native_lawam_dataset_stats(config: LaWAMConfig) -> dict[str, dict[str, Any]] | None:
    stats_path = _resolve_native_lawam_stats_path(config)
    if stats_path is None:
        return None

    with open(stats_path) as f:
        native_stats = json.load(f)

    native_dataset_stats = _pick_native_lawam_dataset_stats(native_stats, config.lawam_unnorm_key)
    return _convert_native_lawam_stats(config, native_dataset_stats)


def _stats_feature_dim(stats: dict[str, Any]) -> int | None:
    for key in ("min", "max", "mean", "std", "q01", "q99"):
        values = stats.get(key)
        if values is not None:
            return int(torch.as_tensor(values).numel())
    return None


def make_lawam_pre_post_processors(
    config: LaWAMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if dataset_stats is None:
        dataset_stats = _load_native_lawam_dataset_stats(config)

    features = {**config.input_features, **config.output_features}
    state_stats = dataset_stats.get(OBS_STATE) if dataset_stats is not None else None
    state_stats_dim = _stats_feature_dim(state_stats) if state_stats is not None else None
    if state_stats_dim is not None and OBS_STATE not in features:
        features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_stats_dim,))

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]
    if state_stats_dim is not None:
        input_steps.append(LaWAMLiberoStateProcessorStep(target_state_dim=state_stats_dim))
    input_steps.append(
        NormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    )

    output_steps: list[ProcessorStep] = []
    if config.clip_normalized_actions:
        output_steps.append(LaWAMClipActionsProcessorStep())
    if config.pre_snap_gripper_action:
        output_steps.append(
            LaWAMPreSnapGripperProcessorStep(
                gripper_dim=config.gripper_dim,
                threshold=config.gripper_threshold,
            )
        )
    output_steps.append(
        UnnormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    )
    if config.binarize_gripper_action:
        output_steps.append(
            LaWAMBinarizeGripperProcessorStep(
                gripper_dim=config.gripper_dim,
                threshold=config.gripper_threshold,
            )
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
