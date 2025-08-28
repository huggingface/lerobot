from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.pipeline import (
    EnvTransition,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotProcessor,
    TransitionKey,
)


def _to_tensor(value: Any, device: torch.device | None = None) -> Tensor:
    """Convert common python/numpy/torch types to a torch.float32 tensor.

    Always returns float32; preserves device if provided.
    """
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32, device=device)
    if isinstance(value, np.ndarray):
        # ensure contiguous, cast to float32 then convert
        return torch.from_numpy(np.ascontiguousarray(value.astype(np.float32))).to(device=device)
    if isinstance(value, (int, float)):
        return torch.tensor(value, dtype=torch.float32, device=device)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.float32, device=device)
    raise TypeError(f"Unsupported type for stats value: {type(value)}")


def _convert_stats_to_tensors(
    stats: dict[str, dict[str, Any]], device: torch.device | None = None
) -> dict[str, dict[str, Tensor]]:
    """Convert numeric stats values to torch tensors, preserving keys."""
    tensor_stats: dict[str, dict[str, Tensor]] = {}
    for key, sub in (stats or {}).items():
        if sub is None:
            continue
        tensor_stats[key] = {}
        for stat_name, value in sub.items():
            tensor_stats[key][stat_name] = _to_tensor(value, device=device)
    return tensor_stats


@dataclass
class _NormalizationMixin:
    """
    A mixin class providing core functionality for normalization and unnormalization.

    This class manages normalization statistics, their conversion to tensors, device placement,
    and the application of normalization transformations. It is designed to be inherited by
    concrete ProcessorStep implementations.
    """

    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None
    device: torch.device | str | None = None
    eps: float = 1e-8
    normalize_observation_keys: set[str] | None = None

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        # Robust JSON deserialization handling (guard empty maps)
        if self.features:
            first_val = next(iter(self.features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.features = reconstructed

        if self.norm_map:
            # if keys are strings (JSON), rebuild enum map
            if all(isinstance(k, str) for k in self.norm_map.keys()):
                reconstructed = {}
                for ft_type_str, norm_mode_str in self.norm_map.items():
                    reconstructed[FeatureType(ft_type_str)] = NormalizationMode(norm_mode_str)
                self.norm_map = reconstructed

        # Convert stats to tensors and move to the target device once during initialization.
        self.stats = self.stats or {}
        self._tensor_stats = _convert_stats_to_tensors(self.stats, device=self.device)

    def to(self, device: torch.device | str) -> _NormalizationMixin:
        """Moves the processor's normalization stats to the specified device and returns self."""
        self.device = device
        self._tensor_stats = _convert_stats_to_tensors(self.stats, device=self.device)
        return self

    def state_dict(self) -> dict[str, Tensor]:
        flat: dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor.cpu()  # Always save to CPU
        return flat

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            # Load to the processor's configured device.
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(
                dtype=torch.float32, device=self.device
            )

    def get_config(self) -> dict[str, Any]:
        config = {
            "eps": self.eps,
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {ft_type.value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()},
        }
        if self.normalize_observation_keys is not None:
            config["normalize_observation_keys"] = sorted(self.normalize_observation_keys)
        return config

    def _normalize_observation(self, observation: dict[str, Any], inverse: bool) -> dict[str, Tensor]:
        new_observation = dict(observation)
        for key, feature in self.features.items():
            if self.normalize_observation_keys is not None and key not in self.normalize_observation_keys:
                continue
            if feature.type != FeatureType.ACTION and key in new_observation:
                tensor = torch.as_tensor(new_observation[key], dtype=torch.float32)
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=inverse)
        return new_observation

    def _normalize_action(self, action: Any, inverse: bool) -> Tensor:
        tensor = torch.as_tensor(action, dtype=torch.float32)
        processed_action = self._apply_transform(tensor, "action", FeatureType.ACTION, inverse=inverse)
        return processed_action

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> Tensor:
        """Core logic to apply normalization or unnormalization."""
        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        if norm_mode not in (NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX):
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        # Ensure input tensor is on the same device as the stats.
        if self.device and tensor.device != self.device:
            tensor = tensor.to(self.device)

        # For Accelerate compatibility: move stats to match input tensor device
        input_device = tensor.device
        stats = self._tensor_stats[key]
        tensor = tensor.to(dtype=torch.float32)

        # Move stats to input device if needed
        stats_device = next(iter(stats.values())).device
        if stats_device != input_device:
            stats = _convert_stats_to_tensors({key: self._tensor_stats[key]}, device=input_device)[key]

        if norm_mode == NormalizationMode.MEAN_STD and "mean" in stats and "std" in stats:
            mean, std = stats["mean"], stats["std"]
            # Avoid division by zero by adding a small epsilon.
            denom = std + self.eps
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / denom

        if norm_mode == NormalizationMode.MIN_MAX and "min" in stats and "max" in stats:
            min_val, max_val = stats["min"], stats["max"]
            denom = max_val - min_val
            # When min_val == max_val, substitute the denominator with a small epsilon
            # to prevent division by zero. This consistently maps an input equal to
            # min_val to -1, ensuring a stable transformation.
            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=self.device, dtype=torch.float32), denom
            )
            if inverse:
                # Map from [-1, 1] back to [min, max]
                return (tensor + 1) / 2 * denom + min_val
            # Map from [min, max] to [-1, 1]
            return 2 * (tensor - min_val) / denom - 1

        # If necessary stats are missing, return input unchanged.
        return tensor


@dataclass
@ProcessorStepRegistry.register(name="normalizer_processor")
class NormalizerProcessor(_NormalizationMixin, ProcessorStep):
    """
    A processor that applies normalization to observations and actions in a transition.

    This class directly implements the normalization logic for both observation and action
    components of an `EnvTransition`, using statistics (mean/std or min/max) provided at
    initialization.
    """

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        normalize_observation_keys: set[str] | None = None,
        eps: float = 1e-8,
        device: torch.device | str | None = None,
    ) -> NormalizerProcessor:
        return cls(
            features=features,
            norm_map=norm_map,
            stats=dataset.meta.stats,
            normalize_observation_keys=normalize_observation_keys,
            eps=eps,
            device=device,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        # Handle observation normalization.
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(
                observation, inverse=False
            )

        # Handle action normalization.
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=False)

        return new_transition


@dataclass
@ProcessorStepRegistry.register(name="unnormalizer_processor")
class UnnormalizerProcessor(_NormalizationMixin, ProcessorStep):
    """
    A processor that applies unnormalization (the inverse of normalization) to
    observations and actions in a transition.

    This is typically used to transform actions from a normalized policy output back into
    the original scale for execution in an environment.
    """

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        device: torch.device | str | None = None,
    ) -> UnnormalizerProcessor:
        return cls(features=features, norm_map=norm_map, stats=dataset.meta.stats, device=device)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        # Handle observation unnormalization.
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(observation, inverse=True)

        # Handle action unnormalization.
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=True)

        return new_transition


def hotswap_stats(robot_processor: RobotProcessor, stats: dict[str, dict[str, Any]]) -> RobotProcessor:
    """
    Replaces normalization statistics in a RobotProcessor pipeline.

    This function creates a deep copy of the provided `RobotProcessor` and updates the
    statistics of any `NormalizerProcessor` or `UnnormalizerProcessor` steps within it.
    It's useful for adapting a trained policy to a new environment or dataset with
    different data distributions.
    """
    rp = deepcopy(robot_processor)
    for step in rp.steps:
        if isinstance(step, _NormalizationMixin):
            step.stats = stats
            # Re-initialize tensor_stats on the correct device.
            step._tensor_stats = _convert_stats_to_tensors(stats, device=step.device)
    return rp
