# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Serializable preprocessing and inverse projection for G0.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_g05 import G05_EMBODIMENT_MAPPINGS, G05Config


def _copy_feature_tree(
    features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
    return {kind: values.copy() for kind, values in features.items()}


@dataclass
@ProcessorStepRegistry.register(name="g05_image_transform")
class G05ImageTransformStep(ProcessorStep):
    """Apply the checkpoint's per-camera resize and ``[0,1]`` to ``[-1,1]`` transform."""

    camera_order: tuple[str, ...]
    camera_sizes: dict[str, tuple[int, int]]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __post_init__(self) -> None:
        self.camera_order = tuple(self.camera_order)
        self.camera_sizes = {key: tuple(value) for key, value in self.camera_sizes.items()}
        self.mean = tuple(self.mean)
        self.std = tuple(self.std)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition
        missing = [key for key in self.camera_order if key not in observation]
        if missing:
            raise ValueError(f"G0.5 is missing camera(s) {missing}; required order is {self.camera_order}.")
        transition = transition.copy()
        observation = dict(observation)
        for key in self.camera_order:
            image = torch.as_tensor(observation[key])
            if image.ndim < 3 or image.shape[-3] != 3:
                raise ValueError(f"G0.5 camera {key!r} must end in [3,H,W], got {image.shape}.")
            was_floating_point = torch.is_floating_point(image)
            image = image.float()
            if not was_floating_point:
                image = image / 255.0
            flat = image.reshape(-1, *image.shape[-3:])
            target_size = self.camera_sizes[key]
            if tuple(flat.shape[-2:]) != target_size:
                flat = functional.interpolate(flat, size=target_size, mode="bilinear", align_corners=False)
            mean = flat.new_tensor(self.mean).view(1, 3, 1, 1)
            std = flat.new_tensor(self.std).view(1, 3, 1, 1)
            observation[key] = ((flat - mean) / std).reshape(*image.shape[:-3], 3, *target_size)
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = _copy_feature_tree(features)
        observations = result.setdefault(PipelineFeatureType.OBSERVATION, {})
        for key in self.camera_order:
            if key in observations:
                height, width = self.camera_sizes[key]
                observations[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_order": list(self.camera_order),
            "camera_sizes": {key: list(value) for key, value in self.camera_sizes.items()},
            "mean": list(self.mean),
            "std": list(self.std),
        }


@dataclass
@ProcessorStepRegistry.register(name="g05_embodiment_projection")
class G05EmbodimentProjectionStep(ProcessorStep):
    """Map raw embodiment coordinates into the checkpoint's padded policy layout."""

    embodiment: str
    policy_state_dim: int
    policy_action_dim: int
    camera_order: tuple[str, ...]

    def __post_init__(self) -> None:
        self.camera_order = tuple(self.camera_order)
        if self.embodiment not in G05_EMBODIMENT_MAPPINGS:
            raise ValueError(f"No projection is defined for G0.5 embodiment {self.embodiment!r}.")

    @property
    def mapping(self) -> dict[str, tuple[int, ...]]:
        return G05_EMBODIMENT_MAPPINGS[self.embodiment]

    @staticmethod
    def _project(value: torch.Tensor, indices: tuple[int, ...], width: int) -> torch.Tensor:
        if value.shape[-1] != len(indices):
            raise ValueError(f"Raw G0.5 tensor has {value.shape[-1]} dimensions, expected {len(indices)}.")
        projected = value.new_zeros(*value.shape[:-1], width)
        projected[..., list(indices)] = value
        return projected

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        missing_cameras = [key for key in self.camera_order if key not in observation]
        if missing_cameras and any(key.startswith("observation.images.") for key in observation):
            raise ValueError(
                f"G0.5 {self.embodiment} is missing camera(s) {missing_cameras}; "
                f"required order is {self.camera_order}."
            )
        if OBS_STATE in observation:
            raw_state = observation[OBS_STATE]
            observation[OBS_STATE] = self._project(raw_state, self.mapping["state"], self.policy_state_dim)
            state_mask = torch.ones(
                *raw_state.shape[:-1],
                self.policy_state_dim,
                dtype=torch.bool,
                device=observation[OBS_STATE].device,
            )
            state_mask[..., list(self.mapping["state"])] = False
            complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
            complementary["proprio_dim_is_pad"] = state_mask
            complementary["g05_camera_order"] = self.camera_order
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            transition[TransitionKey.ACTION] = self._project(
                action, self.mapping["action"], self.policy_action_dim
            )
            batch_shape = action.shape[:1] if action.ndim >= 3 else ()
            action_mask = torch.ones(
                *batch_shape, self.policy_action_dim, dtype=torch.bool, device=action.device
            )
            action_mask[..., list(self.mapping["action"])] = False
            complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
            complementary["action_dim_is_pad"] = action_mask
            if "action_is_pad" not in complementary:
                complementary["action_is_pad"] = torch.zeros(
                    *action.shape[:-1], dtype=torch.bool, device=action.device
                )
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = _copy_feature_tree(features)
        observations = result.setdefault(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in observations:
            observations[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.policy_state_dim,))
        actions = result.setdefault(PipelineFeatureType.ACTION, {})
        if ACTION in actions:
            actions[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.policy_action_dim,))
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "embodiment": self.embodiment,
            "policy_state_dim": self.policy_state_dim,
            "policy_action_dim": self.policy_action_dim,
            "camera_order": list(self.camera_order),
        }


@dataclass
@ProcessorStepRegistry.register(name="g05_inverse_action_projection")
class G05InverseActionProjectionStep(ProcessorStep):
    """Project policy-layout actions back to the environment's exact raw layout."""

    embodiment: str
    policy_action_dim: int

    @property
    def indices(self) -> tuple[int, ...]:
        return G05_EMBODIMENT_MAPPINGS[self.embodiment]["action"]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition
        if action.shape[-1] != self.policy_action_dim:
            raise ValueError(
                f"G0.5 policy action has {action.shape[-1]} dimensions, expected {self.policy_action_dim}."
            )
        transition = transition.copy()
        transition[TransitionKey.ACTION] = action[..., list(self.indices)]
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = _copy_feature_tree(features)
        actions = result.setdefault(PipelineFeatureType.ACTION, {})
        actions[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(len(self.indices),))
        return result

    def get_config(self) -> dict[str, Any]:
        return {"embodiment": self.embodiment, "policy_action_dim": self.policy_action_dim}


def _normalization_mode(config: G05Config) -> NormalizationMode:
    if config.normalization_mode == "q01_q99":
        return NormalizationMode.QUANTILES
    if config.normalization_mode == "z_score":
        return NormalizationMode.MEAN_STD
    return NormalizationMode.IDENTITY


def _project_stats(
    config: G05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    if not dataset_stats:
        return dataset_stats
    result: dict[str, dict[str, torch.Tensor]] = {}
    mapping = G05_EMBODIMENT_MAPPINGS[config.embodiment]
    widths = {OBS_STATE: config.policy_state_dim, ACTION: config.policy_action_dim}
    index_maps = {OBS_STATE: mapping["state"], ACTION: mapping["action"]}
    for feature_name, stats in dataset_stats.items():
        if feature_name not in widths:
            result[feature_name] = stats
            continue
        projected_stats: dict[str, torch.Tensor] = {}
        for stat_name, raw_value in stats.items():
            value = torch.as_tensor(raw_value)
            if value.shape[-1] != len(index_maps[feature_name]):
                raise ValueError(
                    f"{feature_name}.{stat_name} has width {value.shape[-1]}, "
                    f"expected {len(index_maps[feature_name])} for {config.embodiment}."
                )
            fill = {
                "std": 1.0,
                "q01": -1.0,
                "q99": 1.0,
                "min": -1.0,
                "max": 1.0,
            }.get(stat_name, 0.0)
            projected = value.new_full((*value.shape[:-1], widths[feature_name]), fill)
            projected[..., list(index_maps[feature_name])] = value
            projected_stats[stat_name] = projected
        result[feature_name] = projected_stats
    return result


def make_g05_pre_post_processors(
    config: G05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build serializable G0.5 pipelines from checkpoint-authoritative metadata."""

    if config.normalization_mode == "checkpoint" and not config.processor_metadata:
        raise ValueError(
            "normalization_mode='checkpoint' requires processor_metadata from the converted checkpoint."
        )
    mode = _normalization_mode(config)
    if mode is NormalizationMode.QUANTILES and dataset_stats:
        for key in (OBS_STATE, ACTION):
            if key in dataset_stats and not {"q01", "q99"} <= set(dataset_stats[key]):
                raise ValueError(f"{key} requires real q01/q99 statistics; min/max must not be substituted.")
    policy_features = dict(config.input_features or {})
    policy_features.update(config.output_features or {})
    policy_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(config.policy_state_dim,))
    policy_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(config.policy_action_dim,))
    projected_stats = _project_stats(config, dataset_stats)
    norm_map = {
        FeatureType.STATE: mode,
        FeatureType.ACTION: mode,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
    }

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            AddBatchDimensionProcessorStep(),
            G05ImageTransformStep(
                camera_order=config.camera_order,
                camera_sizes=config.camera_sizes,
                mean=config.image_mean,
                std=config.image_std,
            ),
            G05EmbodimentProjectionStep(
                embodiment=config.embodiment,
                policy_state_dim=config.policy_state_dim,
                policy_action_dim=config.policy_action_dim,
                camera_order=config.camera_order,
            ),
            NormalizerProcessorStep(
                features=policy_features,
                norm_map=norm_map,
                stats=projected_stats,
            ),
            DeviceProcessorStep(device=config.device),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[
            UnnormalizerProcessorStep(
                features={ACTION: policy_features[ACTION]},
                norm_map={FeatureType.ACTION: mode},
                stats=projected_stats,
            ),
            G05InverseActionProjectionStep(
                embodiment=config.embodiment, policy_action_dim=config.policy_action_dim
            ),
            DeviceProcessorStep(device="cpu"),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
