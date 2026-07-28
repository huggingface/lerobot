# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Serializable preprocessing and inverse projection for G0.5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torchvision.transforms.functional as vision_functional

from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.relative_action_processor import to_relative_actions
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_g05 import G05_EMBODIMENT_MAPPINGS, G05_POLICY_PARTS, G05Config


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
    optional_camera_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.camera_order = tuple(self.camera_order)
        self.camera_sizes = {key: tuple(value) for key, value in self.camera_sizes.items()}
        self.mean = tuple(self.mean)
        self.std = tuple(self.std)
        self.optional_camera_keys = tuple(self.optional_camera_keys)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition
        missing = [key for key in self.camera_order if key not in observation]
        required_missing = [key for key in missing if key not in self.optional_camera_keys]
        if required_missing:
            raise ValueError(f"G0.5 is missing camera(s) {missing}; required order is {self.camera_order}.")
        transition = transition.copy()
        observation = dict(observation)
        if missing:
            reference = next(
                (torch.as_tensor(observation[key]) for key in self.camera_order if key in observation),
                None,
            )
            if reference is None:
                raise ValueError("G0.5 cannot pad optional cameras without one available reference camera.")
            for key in missing:
                height, width = self.camera_sizes[key]
                observation[key] = reference.new_zeros(*reference.shape[:-2], height, width)
        for key in self.camera_order:
            image = torch.as_tensor(observation[key])
            if image.ndim < 3 or image.shape[-3] != 3:
                raise ValueError(f"G0.5 camera {key!r} must end in [3,H,W], got {image.shape}.")
            was_floating_point = torch.is_floating_point(image)
            flat = image.reshape(-1, *image.shape[-3:])
            target_size = self.camera_sizes[key]
            if tuple(flat.shape[-2:]) != target_size:
                # The author prepends torchvision Resize before its uint8-to-float
                # transform. Preserve its antialiasing and uint8 quantization.
                flat = vision_functional.resize(flat, list(target_size))
            flat = flat.float()
            if not was_floating_point:
                flat = flat / 255.0
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
            height, width = self.camera_sizes[key]
            observations[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_order": list(self.camera_order),
            "camera_sizes": {key: list(value) for key, value in self.camera_sizes.items()},
            "mean": list(self.mean),
            "std": list(self.std),
            "optional_camera_keys": list(self.optional_camera_keys),
        }


@dataclass
@ProcessorStepRegistry.register(name="g05_action_operation_mask")
class G05ActionOperationMaskStep(ProcessorStep):
    """Reproduce the author's per-part R1 movement mask used by ActionCodec."""

    action_parts: tuple[tuple[str, int], ...]
    joint_threshold: float | None = None
    gripper_threshold: float | None = None
    velocity_threshold: float | None = None
    eef_threshold: float | None = 1e-3
    dim_thresholds: dict[str, tuple[float, ...]] = field(default_factory=dict)

    _VELOCITY_KEYS = ("torso", "chassis")

    def __post_init__(self) -> None:
        self.action_parts = tuple((key, int(width)) for key, width in self.action_parts)
        self.dim_thresholds = {
            key: tuple(float(value) for value in values) for key, values in self.dim_thresholds.items()
        }

    def _threshold(self, key: str, width: int, action: torch.Tensor) -> torch.Tensor:
        if key in self.dim_thresholds:
            values = self.dim_thresholds[key]
            if len(values) != width:
                raise ValueError(
                    f"G0.5 action filter threshold {key!r} has {len(values)} values, expected {width}."
                )
            return action.new_tensor(values)
        if "gripper" in key:
            value = self.gripper_threshold
        elif "eef" in key or "ee_pose" in key:
            value = self.eef_threshold
        elif any(velocity_key in key for velocity_key in self._VELOCITY_KEYS):
            value = self.velocity_threshold
        else:
            value = self.joint_threshold
        return action.new_full((width,), float(value or 0.0))

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition
        if action.shape[-1] != sum(width for _, width in self.action_parts):
            raise ValueError("G0.5 action filter metadata does not match the raw action width.")
        half = action.shape[-2] // 2
        masks = []
        offset = 0
        for key, width in self.action_parts:
            part = action[..., offset : offset + width]
            threshold = self._threshold(key, width, action)
            if "hand" in key:
                flag = torch.ones(*part.shape[:-2], width, dtype=torch.bool, device=part.device)
            elif any(velocity_key in key for velocity_key in self._VELOCITY_KEYS):
                flag = (part[..., :half, :].abs() >= threshold).any(dim=-2)
            else:
                deviation = (part[..., :half, :] - part[..., :1, :]).abs()
                flag = (deviation >= threshold).any(dim=-2)
            masks.append(flag)
            offset += width
        transition = transition.copy()
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary["action_op_mask"] = torch.cat(masks, dim=-1)
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "action_parts": [list(part) for part in self.action_parts],
            "joint_threshold": self.joint_threshold,
            "gripper_threshold": self.gripper_threshold,
            "velocity_threshold": self.velocity_threshold,
            "eef_threshold": self.eef_threshold,
            "dim_thresholds": {key: list(values) for key, values in self.dim_thresholds.items()},
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
            if self.embodiment == "libero" and raw_state.shape[-1] == 8:
                # LeRobot's generic LIBERO env exposes both parallel-jaw qpos
                # values. The author evaluator consumes only qpos[0].
                raw_state = torch.cat((raw_state[..., :6], raw_state[..., 6:7]), dim=-1)
            observation[OBS_STATE] = self._project(raw_state, self.mapping["state"], self.policy_state_dim)
            # Masks describe feature dimensions, not observation-history timesteps.
            batch_shape = raw_state.shape[:-2] if raw_state.ndim >= 3 else raw_state.shape[:-1]
            state_mask = torch.ones(
                *batch_shape,
                self.policy_state_dim,
                dtype=torch.bool,
                device=observation[OBS_STATE].device,
            )
            state_mask[..., list(self.mapping["state"])] = False
            complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
            complementary["proprio_dim_is_pad"] = state_mask
            action_mask = torch.ones(
                *batch_shape,
                self.policy_action_dim,
                dtype=torch.bool,
                device=observation[OBS_STATE].device,
            )
            action_mask[..., list(self.mapping["action"])] = False
            complementary["action_dim_is_pad"] = action_mask
            raw_action_op_mask = complementary.get("action_op_mask")
            if isinstance(raw_action_op_mask, torch.Tensor):
                complementary["action_op_mask"] = self._project(
                    raw_action_op_mask, self.mapping["action"], self.policy_action_dim
                ).bool()
            else:
                complementary["action_op_mask"] = ~action_mask
            complementary["action_parts_meta"] = G05_POLICY_PARTS[self.policy_action_dim].copy()
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
            complementary.setdefault("action_dim_is_pad", action_mask)
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
@ProcessorStepRegistry.register(name="g05_relative_joint_actions")
class G05RelativeJointActionsStep(RelativeActionsProcessorStep):
    """Author-compatible joint deltas using the last proprio history step."""

    num_obs_steps: int = 1

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if isinstance(state, torch.Tensor) and state.ndim >= 3:
            state = state[..., -1, :]
        elif (
            isinstance(state, torch.Tensor)
            and state.ndim == 2
            and self.num_obs_steps > 1
            and state.shape[-2] == self.num_obs_steps
        ):
            state = state[-1]
        if state is not None:
            self._last_state = state
        if not self.enabled:
            return transition
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None and state is not None:
            mask = self._build_mask(action.shape[-1])
            new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "num_obs_steps": self.num_obs_steps}


@dataclass
@ProcessorStepRegistry.register(name="g05_tail_normalization")
class G05TailNormalizationStep(ProcessorStep):
    """Apply or invert the author's q01/q99 log-tail compression."""

    inverse: bool = False
    tail_scale: float = 0.075
    stats: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _tensor_stats: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict, init=False, repr=False)

    _TAIL_STATS = ("tail_q01", "tail_q99", "tail_mean", "tail_mask")

    def __post_init__(self) -> None:
        self._tensor_stats = {
            key: {
                name: torch.as_tensor(value)
                for name, value in feature_stats.items()
                if name in self._TAIL_STATS
            }
            for key, feature_stats in self.stats.items()
        }

    def _transform(self, value: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self._tensor_stats:
            return value
        stats = {name: tensor.to(value.device) for name, tensor in self._tensor_stats[key].items()}
        if set(stats) != set(self._TAIL_STATS):
            raise ValueError(f"Incomplete G0.5 tail statistics for {key}.")
        q01 = stats["tail_q01"].to(value.dtype)
        q99 = stats["tail_q99"].to(value.dtype)
        mean = stats["tail_mean"].to(value.dtype)
        mask = stats["tail_mask"].bool()
        degenerate = (q99 <= q01) | (mean <= q01) | (mean >= q99)
        mask = mask & ~degenerate
        c_pos = torch.where(mask, self.tail_scale * (q99 - mean), torch.ones_like(q99))
        c_neg = torch.where(mask, self.tail_scale * (mean - q01), torch.ones_like(q01))
        if self.inverse:
            positive = q99 + c_pos * torch.expm1(torch.clamp((value - q99) / c_pos, min=0.0))
            negative = q01 - c_neg * torch.expm1(torch.clamp((q01 - value) / c_neg, min=0.0))
        else:
            positive = q99 + c_pos * torch.log1p(torch.clamp((value - q99) / c_pos, min=0.0))
            negative = q01 - c_neg * torch.log1p(torch.clamp((q01 - value) / c_neg, min=0.0))
        transformed = torch.where(value > q99, positive, torch.where(value < q01, negative, value))
        return torch.where(mask, transformed, value)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        if OBS_STATE in observation:
            observation[OBS_STATE] = self._transform(observation[OBS_STATE], OBS_STATE)
        transition[TransitionKey.OBSERVATION] = observation
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            transition[TransitionKey.ACTION] = self._transform(action, ACTION)
        return transition

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            f"{key}.{name}": tensor.cpu()
            for key, feature_stats in self._tensor_stats.items()
            for name, tensor in feature_stats.items()
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._tensor_stats = {}
        for flat_key, tensor in state.items():
            key, name = flat_key.rsplit(".", 1)
            self._tensor_stats.setdefault(key, {})[name] = tensor

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"inverse": self.inverse, "tail_scale": self.tail_scale}


@dataclass
@ProcessorStepRegistry.register(name="g05_normalization_clamp")
class G05NormalizationClampStep(ProcessorStep):
    """Match the author's finite clamp after normalization."""

    minimum: float = -5.0
    maximum: float = 5.0

    def _clamp(self, value: torch.Tensor) -> torch.Tensor:
        return value.clamp(self.minimum, self.maximum).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        if OBS_STATE in observation:
            observation[OBS_STATE] = self._clamp(observation[OBS_STATE])
        transition[TransitionKey.OBSERVATION] = observation
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            transition[TransitionKey.ACTION] = self._clamp(action)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"minimum": self.minimum, "maximum": self.maximum}


@dataclass
@ProcessorStepRegistry.register(name="g05_stepwise_unnormalizer")
class G05StepwiseUnnormalizerStep(UnnormalizerProcessorStep):
    """Apply the checkpoint's action-timestep statistics to queued single actions."""

    n_action_steps: int = 1
    _action_index: int = field(default=0, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        stats = self._tensor_stats.get(ACTION)
        if not isinstance(action, torch.Tensor) or not stats:
            return super().__call__(transition)

        first_stat = next(iter(stats.values()))
        if first_stat.device != action.device or first_stat.dtype != action.dtype:
            self.to(device=action.device, dtype=action.dtype)
            stats = self._tensor_stats[ACTION]
            first_stat = next(iter(stats.values()))

        stats_steps = first_stat.shape[-2] if first_stat.ndim >= 2 else 1
        is_single_queued_action = action.ndim <= 2 and action.shape[-2] != stats_steps
        if not is_single_queued_action or stats_steps == 1:
            return super().__call__(transition)

        index = self._action_index % min(self.n_action_steps, stats_steps)
        self._tensor_stats[ACTION] = {
            name: value[index] if value.ndim >= 2 and value.shape[-2] == stats_steps else value
            for name, value in stats.items()
        }
        try:
            return super().__call__(transition)
        finally:
            self._tensor_stats[ACTION] = stats
            self._action_index = (self._action_index + 1) % self.n_action_steps

    def reset(self) -> None:
        self._action_index = 0

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "n_action_steps": self.n_action_steps}


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


@dataclass
@ProcessorStepRegistry.register(name="g05_libero_gripper")
class G05LiberoGripperStep(ProcessorStep):
    """Convert the released checkpoint's gripper value to LIBERO's binary command."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition
        if action.shape[-1] != 7:
            raise ValueError(f"G0.5 LIBERO action must have 7 dimensions, got {action.shape[-1]}.")
        value = action[..., -1]
        in_unit_interval = (value >= 0.0) & (value <= 1.0)
        open_gripper = torch.where(in_unit_interval, value > 0.5, value > 0.0)
        transition = transition.copy()
        action = action.clone()
        action[..., -1] = torch.where(
            open_gripper,
            value.new_tensor(-1.0),
            value.new_tensor(1.0),
        )
        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass
@ProcessorStepRegistry.register(name="g05_action_history_crop")
class G05ActionHistoryCropStep(ProcessorStep):
    """Remove the author model's observation-alignment prefix from action chunks."""

    num_obs_steps: int = 1

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor) or self.num_obs_steps <= 1:
            return transition
        start = self.num_obs_steps - 1
        if action.ndim < 2 or action.shape[-2] <= start:
            raise ValueError(
                "G0.5 action history crop requires a full action chunk with at least "
                f"{self.num_obs_steps} steps, got {tuple(action.shape)}."
            )
        transition = transition.copy()
        transition[TransitionKey.ACTION] = action[..., start:, :]
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"num_obs_steps": self.num_obs_steps}


def _normalization_mode(config: G05Config) -> NormalizationMode:
    if config.normalization_mode == "q01_q99":
        return NormalizationMode.QUANTILES
    if config.normalization_mode in {"z_score", "z_score_tail_mixed"}:
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

    relative_step = G05RelativeJointActionsStep(
        enabled=config.use_relative_actions,
        exclude_joints=list(config.relative_exclude_joints),
        action_names=list(config.action_feature_names) or None,
        num_obs_steps=config.n_obs_steps,
    )
    steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        G05ImageTransformStep(
            camera_order=config.camera_order,
            camera_sizes=config.camera_sizes,
            mean=config.image_mean,
            std=config.image_std,
            optional_camera_keys=config.optional_camera_keys,
        ),
    ]
    action_filter = config.processor_metadata.get("action_filter") or {}
    if str(action_filter.get("_target_", "")).endswith("R1LiteJointActionFilter"):
        action_parts = tuple(
            (str(item["key"]), int(item["shape"]))
            for item in (config.processor_metadata.get("shape_meta") or {}).get("action", [])
        )
        steps.append(
            G05ActionOperationMaskStep(
                action_parts=action_parts,
                joint_threshold=action_filter.get("joint_threshold"),
                gripper_threshold=action_filter.get("gripper_threshold"),
                velocity_threshold=action_filter.get("velocity_threshold"),
                eef_threshold=action_filter.get("eef_threshold", 1e-3),
                dim_thresholds=action_filter.get("dim_thresholds") or {},
            )
        )
    steps.extend(
        [
            relative_step,
            G05EmbodimentProjectionStep(
                embodiment=config.embodiment,
                policy_state_dim=config.policy_state_dim,
                policy_action_dim=config.policy_action_dim,
                camera_order=config.camera_order,
            ),
        ]
    )
    if config.normalization_mode == "z_score_tail_mixed":
        steps.append(G05TailNormalizationStep(stats=projected_stats or {}))
    steps.append(
        NormalizerProcessorStep(
            features=policy_features,
            norm_map=norm_map,
            stats=projected_stats,
        )
    )
    if config.normalization_clip is not None:
        steps.append(
            G05NormalizationClampStep(
                minimum=config.normalization_clip[0],
                maximum=config.normalization_clip[1],
            )
        )
    steps.append(DeviceProcessorStep(device=config.device))
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    unnormalizer_cls = (
        G05StepwiseUnnormalizerStep if config.use_stepwise_action_norm else UnnormalizerProcessorStep
    )
    unnormalizer_kwargs = {
        "features": {ACTION: policy_features[ACTION]},
        "norm_map": {FeatureType.ACTION: mode},
        "stats": projected_stats,
    }
    if config.use_stepwise_action_norm:
        unnormalizer_kwargs["n_action_steps"] = config.n_action_steps
    output_steps: list[ProcessorStep] = [
        unnormalizer_cls(
            **unnormalizer_kwargs,
        )
    ]
    if config.normalization_mode == "z_score_tail_mixed":
        output_steps.append(G05TailNormalizationStep(inverse=True, stats=projected_stats or {}))
    output_steps.append(
        G05InverseActionProjectionStep(
            embodiment=config.embodiment, policy_action_dim=config.policy_action_dim
        )
    )
    if config.libero_gripper_binarize:
        output_steps.append(G05LiberoGripperStep())
    output_steps.extend(
        [
            DeviceProcessorStep(device="cpu"),
            AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
            G05ActionHistoryCropStep(num_obs_steps=config.n_obs_steps),
        ]
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
