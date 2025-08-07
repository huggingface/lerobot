from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, RobotProcessor, TransitionKey


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
@ProcessorStepRegistry.register(name="normalizer_processor")
class NormalizerProcessor:
    """Normalizes observations and actions in a single processor step.

    This processor handles normalization of both observation and action tensors
    using either mean/std normalization or min/max scaling to a [-1, 1] range.

    For each tensor key in the stats dictionary, the processor will:
    - Use mean/std normalization if those statistics are provided: (x - mean) / std
    - Use min/max scaling if those statistics are provided: 2 * (x - min) / (max - min) - 1

    The processor can be configured to normalize only specific keys by setting
    the normalize_keys parameter.
    """

    # Features and normalisation map are mandatory to match the design of normalize.py
    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]

    # Pre-computed statistics coming from dataset.meta.stats for instance.
    stats: dict[str, dict[str, Any]] | None = None

    # Explicit subset of keys to normalise. If ``None`` every key (except
    # "action") found in ``stats`` will be normalised. Using a ``set`` makes
    # membership checks O(1).
    normalize_keys: set[str] | None = None

    eps: float = 1e-8

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)

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
        # Handle deserialization from JSON config
        if self.features and isinstance(list(self.features.values())[0], dict):
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

        # Ensure *normalize_keys* is a set for fast look-ups and compare by
        # value later when returning the configuration.
        if self.normalize_keys is not None and not isinstance(self.normalize_keys, set):
            self.normalize_keys = set(self.normalize_keys)

    def _normalize_obs(self, observation, normalized_info):
        if observation is None:
            return None

        # Decide which keys should be normalised for this call.
        if self.normalize_keys is not None:
            keys_to_norm = self.normalize_keys
        else:
            # Use feature map to skip action keys.
            keys_to_norm = {k for k, ft in self.features.items() if ft.type is not FeatureType.ACTION}

        processed = dict(observation)
        for key in keys_to_norm:
            if key not in processed or key not in self.features:
                continue

            # Check the normalization mode for this feature type
            feature = self.features[key]
            norm_mode = self.norm_map.get(feature.type, NormalizationMode.IDENTITY)

            # Skip normalization if mode is IDENTITY
            if norm_mode is NormalizationMode.IDENTITY:
                normalized_info[key] = "IDENTITY"
                continue

            # Skip if no stats available for this key
            if key not in self._tensor_stats:
                continue

            orig_val = processed[key]
            tensor = (
                orig_val.to(dtype=torch.float32)
                if isinstance(orig_val, torch.Tensor)
                else torch.as_tensor(orig_val, dtype=torch.float32)
            )
            stats = {k: v.to(tensor.device) for k, v in self._tensor_stats[key].items()}

            if norm_mode is NormalizationMode.MEAN_STD:
                if "mean" in stats and "std" in stats:
                    mean, std = stats["mean"], stats["std"]
                    processed[key] = (tensor - mean) / (std + self.eps)
                    normalized_info[key] = "MEAN_STD"
            elif norm_mode is NormalizationMode.MIN_MAX:
                if "min" in stats and "max" in stats:
                    min_val, max_val = stats["min"], stats["max"]
                    processed[key] = 2 * (tensor - min_val) / (max_val - min_val + self.eps) - 1
                    normalized_info[key] = "MIN_MAX"
            else:
                raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        return processed

    def _normalize_action(self, action, normalized_info):
        if action is None:
            return action

        # Check the normalization mode for actions
        norm_mode = self.norm_map.get(FeatureType.ACTION, NormalizationMode.IDENTITY)

        # Skip normalization if mode is IDENTITY
        if norm_mode is NormalizationMode.IDENTITY:
            normalized_info["action"] = "IDENTITY"
            return action

        # Skip if no stats available for actions
        if "action" not in self._tensor_stats:
            return action

        tensor = (
            action.to(dtype=torch.float32)
            if isinstance(action, torch.Tensor)
            else torch.as_tensor(action, dtype=torch.float32)
        )
        stats = {k: v.to(tensor.device) for k, v in self._tensor_stats["action"].items()}

        if norm_mode is NormalizationMode.MEAN_STD:
            if "mean" in stats and "std" in stats:
                mean, std = stats["mean"], stats["std"]
                normalized_info["action"] = "MEAN_STD"
                return (tensor - mean) / (std + self.eps)
        elif norm_mode is NormalizationMode.MIN_MAX:
            if "min" in stats and "max" in stats:
                min_val, max_val = stats["min"], stats["max"]
                normalized_info["action"] = "MIN_MAX"
                return 2 * (tensor - min_val) / (max_val - min_val + self.eps) - 1
        else:
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        # If we reach here, the required stats for the normalization mode are not available
        raise ValueError(f"Action stats must contain appropriate values for {norm_mode} normalization")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Track what was normalized
        normalized_info = {}

        observation = self._normalize_obs(transition.get(TransitionKey.OBSERVATION), normalized_info)
        action = self._normalize_action(transition.get(TransitionKey.ACTION), normalized_info)

        # Create a new transition with normalized values
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = observation
        new_transition[TransitionKey.ACTION] = action

        # Add normalization info to complementary data
        if normalized_info:
            comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            comp_data = {} if comp_data is None else dict(comp_data)
            comp_data["normalized_keys"] = normalized_info
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        config = {
            "eps": self.eps,
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {ft_type.value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()},
        }
        if self.normalize_keys is not None:
            # Serialise as a list for YAML / JSON friendliness
            config["normalize_keys"] = sorted(self.normalize_keys)
        return config

    def state_dict(self) -> dict[str, Tensor]:
        flat = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor

    def reset(self):
        pass

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="unnormalizer_processor")
class UnnormalizerProcessor:
    """Inverse normalisation for observations and actions.

    Exactly mirrors :class:`NormalizerProcessor` but applies the inverse
    transform.
    """

    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
    ) -> UnnormalizerProcessor:
        return cls(features=features, norm_map=norm_map, stats=dataset.meta.stats)

    def __post_init__(self):
        # Handle deserialization from JSON config
        if self.features and isinstance(list(self.features.values())[0], dict):
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

        self.stats = self.stats or {}
        self._tensor_stats = _convert_stats_to_tensors(self.stats)

    def _unnormalize_obs(self, observation, unnormalized_info):
        if observation is None:
            return None
        keys = [k for k, ft in self.features.items() if ft.type is not FeatureType.ACTION]
        processed = dict(observation)
        for key in keys:
            if key not in processed or key not in self.features:
                continue

            # Check the normalization mode for this feature type
            feature = self.features[key]
            norm_mode = self.norm_map.get(feature.type, NormalizationMode.IDENTITY)

            # Skip unnormalization if mode is IDENTITY
            if norm_mode is NormalizationMode.IDENTITY:
                unnormalized_info[key] = "IDENTITY"
                continue

            # Skip if no stats available for this key
            if key not in self._tensor_stats:
                continue

            orig_val = processed[key]
            tensor = (
                orig_val.to(dtype=torch.float32)
                if isinstance(orig_val, torch.Tensor)
                else torch.as_tensor(orig_val, dtype=torch.float32)
            )
            stats = {k: v.to(tensor.device) for k, v in self._tensor_stats[key].items()}

            if norm_mode is NormalizationMode.MEAN_STD:
                if "mean" in stats and "std" in stats:
                    mean, std = stats["mean"], stats["std"]
                    processed[key] = tensor * std + mean
                    unnormalized_info[key] = "MEAN_STD"
            elif norm_mode is NormalizationMode.MIN_MAX:
                if "min" in stats and "max" in stats:
                    min_val, max_val = stats["min"], stats["max"]
                    processed[key] = (tensor + 1) / 2 * (max_val - min_val) + min_val
                    unnormalized_info[key] = "MIN_MAX"
            else:
                raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        return processed

    def _unnormalize_action(self, action, unnormalized_info):
        if action is None:
            return action

        # Check the normalization mode for actions
        norm_mode = self.norm_map.get(FeatureType.ACTION, NormalizationMode.IDENTITY)

        # Skip unnormalization if mode is IDENTITY
        if norm_mode is NormalizationMode.IDENTITY:
            unnormalized_info["action"] = "IDENTITY"
            return action

        # Skip if no stats available for actions
        if "action" not in self._tensor_stats:
            return action

        tensor = (
            action.to(dtype=torch.float32)
            if isinstance(action, torch.Tensor)
            else torch.as_tensor(action, dtype=torch.float32)
        )
        stats = {k: v.to(tensor.device) for k, v in self._tensor_stats["action"].items()}

        if norm_mode is NormalizationMode.MEAN_STD:
            if "mean" in stats and "std" in stats:
                mean, std = stats["mean"], stats["std"]
                unnormalized_info["action"] = "MEAN_STD"
                return tensor * std + mean
        elif norm_mode is NormalizationMode.MIN_MAX:
            if "min" in stats and "max" in stats:
                min_val, max_val = stats["min"], stats["max"]
                unnormalized_info["action"] = "MIN_MAX"
                return (tensor + 1) / 2 * (max_val - min_val) + min_val
        else:
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        # If we reach here, the required stats for the normalization mode are not available
        raise ValueError(f"Action stats must contain appropriate values for {norm_mode} normalization")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Track what was unnormalized
        unnormalized_info = {}

        observation = self._unnormalize_obs(transition.get(TransitionKey.OBSERVATION), unnormalized_info)
        action = self._unnormalize_action(transition.get(TransitionKey.ACTION), unnormalized_info)

        # Create a new transition with unnormalized values
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = observation
        new_transition[TransitionKey.ACTION] = action

        # Add unnormalization info to complementary data
        if unnormalized_info:
            comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            comp_data = {} if comp_data is None else dict(comp_data)
            comp_data["unnormalized_keys"] = unnormalized_info
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {ft_type.value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()},
        }

    def state_dict(self) -> dict[str, Tensor]:
        flat = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor

    def reset(self):
        pass

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


def hotswap_stats(robot_processor: RobotProcessor, stats: dict[str, dict[str, Any]]) -> RobotProcessor:
    robot_processor = deepcopy(robot_processor)
    for step in robot_processor.steps:
        if isinstance(step, NormalizerProcessor) or isinstance(step, UnnormalizerProcessor):
            step: NormalizerProcessor | UnnormalizerProcessor
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
