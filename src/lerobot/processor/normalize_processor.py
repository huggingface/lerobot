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
    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)

    eps: float = 1e-8
    normalize_keys: set[str] | None = None

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

        # convert stats once; leave device unspecified (converted later when used)
        self.stats = self.stats or {}
        # store as float32 CPU tensors by default; they will be moved to the runtime device in _apply_transform
        self._tensor_stats = _convert_stats_to_tensors(self.stats, device=None)

    def state_dict(self) -> dict[str, Tensor]:
        flat: dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(dtype=torch.float32)

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> Tensor:
        """Core logic to apply normalization or unnormalization.

        - Moves stats to the input tensor's device to avoid device mismatch.
        - Uses numeric safeguards to avoid division by zero.
        """
        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        if norm_mode not in (NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX):
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        # move stats to input device and ensure float32
        device = tensor.device
        stats = {k: v.to(device=device, dtype=torch.float32) for k, v in self._tensor_stats[key].items()}
        tensor = tensor.to(dtype=torch.float32)

        if norm_mode == NormalizationMode.MEAN_STD and "mean" in stats and "std" in stats:
            mean, std = stats["mean"], stats["std"]
            # avoid dividing by zero
            denom = std + self.eps
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / denom

        if norm_mode == NormalizationMode.MIN_MAX and "min" in stats and "max" in stats:
            min_val, max_val = stats["min"], stats["max"]
            denom = max_val - min_val
            # replace zero denom with eps to avoid NaNs
            denom = torch.where(denom == 0, torch.tensor(self.eps, device=device, dtype=torch.float32), denom)
            if inverse:
                # map from [-1, 1] back to [min, max]
                return (tensor + 1) / 2 * denom + min_val
            # map from [min, max] to [-1, 1]
            return 2 * (tensor - min_val) / denom - 1

        # if necessary stats are missing, return input unchanged
        return tensor


@dataclass
class ObservationNormalizer(ObservationProcessor, _NormalizationMixin):
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        new_observation = dict(observation)
        for key, feature in self.features.items():
            if self.normalize_keys is not None and key not in self.normalize_keys:
                continue
            if feature.type != FeatureType.ACTION and key in new_observation:
                tensor = torch.as_tensor(new_observation[key], dtype=torch.float32)
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=False)
        return new_observation


@dataclass
class ObservationUnnormalizer(ObservationProcessor, _NormalizationMixin):
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        new_observation = dict(observation)
        for key, feature in self.features.items():
            if self.normalize_keys is not None and key not in self.normalize_keys:
                continue
            if feature.type != FeatureType.ACTION and key in new_observation:
                tensor = torch.as_tensor(new_observation[key], dtype=torch.float32)
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=True)
        return new_observation


@dataclass
class ActionNormalizer(ActionProcessor, _NormalizationMixin):
    def action(self, action: Tensor) -> Tensor:
        tensor = torch.as_tensor(action, dtype=torch.float32)
        return self._apply_transform(tensor, "action", FeatureType.ACTION, inverse=False)


@dataclass
class ActionUnnormalizer(ActionProcessor, _NormalizationMixin):
    def action(self, action: Tensor) -> Tensor:
        tensor = torch.as_tensor(action, dtype=torch.float32)
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
        return cls(
            features=features,
            norm_map=norm_map,
            stats=dataset.meta.stats,
            normalize_keys=normalize_keys,
            eps=eps,
        )

    def __post_init__(self):
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
    """Replaces normalization statistics in a RobotProcessor pipeline.

    Updates both the composite step and any inner sub-processors that hold stats.
    """
    rp = deepcopy(robot_processor)
    for step in rp.steps:
        if isinstance(step, _BaseNormalizerProcessor):
            # update composite step stats
            step.stats = stats
            step._tensor_stats = _convert_stats_to_tensors(stats)
            # ensure inner processors are in sync if present
            if getattr(step, "_obs_processor", None):
                step._obs_processor.stats = stats
                step._obs_processor._tensor_stats = _convert_stats_to_tensors(stats)
            if getattr(step, "_action_processor", None):
                step._action_processor.stats = stats
                step._action_processor._tensor_stats = _convert_stats_to_tensors(stats)
    return rp


def rename_stats(stats: dict[str, dict[str, Any]], rename_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Rename keys in the stats dictionary according to rename_map (defensive copy)."""
    if not stats:
        return {}
    renamed: dict[str, dict[str, Any]] = {}
    for old_key, sub_stats in stats.items():
        new_key = rename_map.get(old_key, old_key)
        renamed[new_key] = deepcopy(sub_stats) if sub_stats is not None else {}
    return renamed
