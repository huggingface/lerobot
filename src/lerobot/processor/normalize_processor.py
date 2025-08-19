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
    ActionProcessor,
    EnvTransition,
    ObservationProcessor,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotProcessor,
)


def _convert_stats_to_tensors(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Tensor]]:
    """Convert numpy arrays and other types to torch tensors."""
    tensor_stats: dict[str, dict[str, Tensor]] = {}
    for key, sub in stats.items():
        tensor_stats[key] = {}
        for stat_name, value in sub.items():
            if isinstance(value, np.ndarray):
                tensor_val = torch.from_numpy(value.astype(np.float32))
            elif isinstance(value, torch.Tensor):
                tensor_val = value.to(dtype=torch.float32)
            elif isinstance(value, (int, float, list, tuple)):
                tensor_val = torch.tensor(value, dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported type for stats['{key}']['{stat_name}']: {type(value)}")
            tensor_stats[key][stat_name] = tensor_val
    return tensor_stats


@dataclass
class _NormalizationMixin:
    # Features and normalisation map are mandatory to match the design of normalize.py
    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]

    # Pre-computed statistics coming from dataset.meta.stats for instance.
    stats: dict[str, dict[str, Any]] | None = None

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)

    eps: float = 1e-8

    normalize_keys: set[str] | None = None

    def __post_init__(self):
        # Handle deserialization from JSON config
        if self.features and isinstance(next(iter(self.features.values())), dict):
            # Features came from JSON - need to reconstruct PolicyFeature objects
            reconstructed_features = {}
            for key, ft_dict in self.features.items():
                reconstructed_features[key] = PolicyFeature(
                    type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                )
            self.features = reconstructed_features

        if self.norm_map and isinstance(list(self.norm_map.keys())[0], str):
            # norm_map came from JSON - need to reconstruct enum keys and values
            reconstructed_norm_map = {}
            for ft_type_str, norm_mode_str in self.norm_map.items():
                reconstructed_norm_map[FeatureType(ft_type_str)] = NormalizationMode(norm_mode_str)
            self.norm_map = reconstructed_norm_map

        # Convert statistics once so we avoid repeated numpy→Tensor conversions
        # during runtime.
        self.stats = self.stats or {}
        self._tensor_stats = _convert_stats_to_tensors(self.stats)

    def state_dict(self) -> dict[str, Tensor]:
        flat = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> Tensor:
        """Core logic to apply normalization or unnormalization."""

        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        if norm_mode not in (NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX):
            raise ValueError(f"Unsupported normalization mode : {norm_mode}")
        stats = {k: v.to(tensor.device) for k, v in self._tensor_stats[key].items()}
        tensor = tensor.to(dtype=torch.float32)

        if norm_mode is NormalizationMode.MEAN_STD and "mean" in stats and "std" in stats:
            mean, std = stats["mean"], stats["std"]
            return (tensor * std + mean) if inverse else ((tensor - mean) / (std + self.eps))

        elif norm_mode is NormalizationMode.MIN_MAX and "min" in stats and "max" in stats:
            min_val, max_val = stats["min"], stats["max"]
            if inverse:
                return (tensor + 1) / 2 * (max_val - min_val) + min_val
            return 2 * (tensor - min_val) / (max_val - min_val + self.eps) - 1

        return tensor


@dataclass
class ObservationNormalizer(ObservationProcessor, _NormalizationMixin):
    def observation(self, observation) -> dict[str, Any]:
        new_observation = dict(observation)
        for key, feature in self.features.items():
            if self.normalize_keys is not None and key not in self.normalize_keys:
                continue
            if feature.type is not FeatureType.ACTION and key in new_observation:
                tensor = torch.as_tensor(new_observation[key])
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=False)
        return new_observation


@dataclass
class ActionNormalizer(ActionProcessor, _NormalizationMixin):
    """A processor that normalizes only the action component."""

    def action(self, action: Tensor) -> Tensor:
        tensor = torch.as_tensor(action)
        return self._apply_transform(tensor, "action", FeatureType.ACTION, inverse=False)


@dataclass
class ObservationUnnormalizer(ObservationProcessor, _NormalizationMixin):
    """A processor that unnormalizes only the observation component."""

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        new_observation = dict(observation)
        for key, feature in self.features.items():
            if feature.type is not FeatureType.ACTION and key in new_observation:
                tensor = torch.as_tensor(new_observation[key])
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=True)
        return new_observation


@dataclass
class ActionUnnormalizer(ActionProcessor, _NormalizationMixin):
    """A processor that unnormalizes only the action component."""

    def action(self, action: Tensor) -> Tensor:
        tensor = torch.as_tensor(action)
        return self._apply_transform(tensor, "action", FeatureType.ACTION, inverse=True)


@dataclass
class _BaseNormalizerProcessor(ProcessorStep, _NormalizationMixin):
    _obs_processor: ObservationProcessor = field(init=False, repr=False)
    _action_processor: ActionProcessor = field(init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = self._obs_processor(transition)
        transition = self._action_processor(transition)
        return transition

    def get_config(self) -> dict[str, Any]:
        config = {
            "eps": self.eps,
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {ft_type.value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()},
        }
        if self.normalize_keys is not None:
            config["normalize_keys"] = sorted(self.normalize_keys)
        return config


@dataclass
@ProcessorStepRegistry.register(name="normalizer_processor")
class NormalizerProcessor(_BaseNormalizerProcessor):
    """A composite processor that normalizes both observations and actions."""

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        normalize_keys: set[str] | None = None,
        eps: float = 1e-8,
    ) -> NormalizerProcessor:
        """Factory helper that pulls statistics from a :class:`LeRobotDataset`.

        The features and norm_map parameters are mandatory to match the design
        pattern used in normalize.py.
        """

        return cls(
            features=features,
            norm_map=norm_map,
            stats=dataset.meta.stats,
            normalize_keys=normalize_keys,
            eps=eps,
        )

    def __post_init__(self):
        # The parent's __post_init__ handles stats and JSON deserialization
        super().__post_init__()

        self._obs_processor = ObservationNormalizer(
            features=self.features,
            norm_map=self.norm_map,
            stats=self.stats,
            eps=self.eps,
            normalize_keys=self.normalize_keys,
        )
        self._action_processor = ActionNormalizer(
            features=self.features, norm_map=self.norm_map, stats=self.stats, eps=self.eps
        )


@dataclass
@ProcessorStepRegistry.register(name="unnormalizer_processor")
class UnnormalizerProcessor(_BaseNormalizerProcessor):
    """A composite processor that unnormalizes both observations and actions."""

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
    ) -> UnnormalizerProcessor:
        return cls(features=features, norm_map=norm_map, stats=dataset.meta.stats)

    def __post_init__(self):
        super().__post_init__()

        self._obs_processor = ObservationUnnormalizer(
            features=self.features, norm_map=self.norm_map, stats=self.stats
        )
        self._action_processor = ActionUnnormalizer(
            features=self.features, norm_map=self.norm_map, stats=self.stats
        )


def hotswap_stats(robot_processor: RobotProcessor, stats: dict[str, dict[str, Any]]) -> RobotProcessor:
    """Replaces normalization statistics in a RobotProcessor pipeline."""
    robot_processor = deepcopy(robot_processor)
    for step in robot_processor.steps:
        # Check if the step is a normalizer/unnormalizer
        if isinstance(step, _BaseNormalizerProcessor):
            # Update the state in one place
            step.stats = stats
            step._tensor_stats = _convert_stats_to_tensors(stats)
    return robot_processor


def rename_stats(stats: dict[str, dict[str, Any]], rename_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Rename keys in the stats dictionary according to the provided mapping.

    Args:
        stats: The statistics dictionary with structure {feature_key: {stat_name: value}}
        rename_map: Dictionary mapping old key names to new key names

    Returns:
        A new stats dictionary with renamed keys

    Example:
        >>> stats = {"observation.state": {"mean": 0.0, "std": 1.0}, "action": {"mean": 0.5, "std": 0.5}}
        >>> rename_map = {"observation.state": "observation.robot_state"}
        >>> new_stats = rename_stats(stats, rename_map)
        >>> # new_stats will have "observation.robot_state" instead of "observation.state"
    """
    renamed_stats = {}

    for old_key, sub_stats in stats.items():
        # Use the new key if it exists in the rename map, otherwise keep the old key
        new_key = rename_map.get(old_key, old_key)
        renamed_stats[new_key] = deepcopy(sub_stats)

    return renamed_stats
