from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionIndex


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

    stats: dict[str, dict[str, Any]]
    normalize_keys: set[str] | None = None
    eps: float = 1e-8

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        *,
        normalize_keys: set[str] | None = None,
        eps: float = 1e-8,
    ) -> NormalizerProcessor:
        return cls(stats=dataset.meta.stats, normalize_keys=normalize_keys, eps=eps)

    def __post_init__(self):
        self._tensor_stats = _convert_stats_to_tensors(self.stats)

    def _normalize_obs(self, observation):
        if observation is None:
            return None

        keys_to_norm = (
            self.normalize_keys
            if self.normalize_keys is not None
            else {k for k in self._tensor_stats if k != "action"}
        )
        processed = dict(observation)
        for key in keys_to_norm:
            if key not in processed or key not in self._tensor_stats:
                continue

            orig_val = processed[key]
            tensor = (
                orig_val.to(dtype=torch.float32)
                if isinstance(orig_val, torch.Tensor)
                else torch.as_tensor(orig_val, dtype=torch.float32)
            )
            stats = {k: v.to(tensor.device) for k, v in self._tensor_stats[key].items()}

            if "mean" in stats and "std" in stats:
                mean, std = stats["mean"], stats["std"]
                processed[key] = (tensor - mean) / (std + self.eps)
            elif "min" in stats and "max" in stats:
                min_val, max_val = stats["min"], stats["max"]
                processed[key] = 2 * (tensor - min_val) / (max_val - min_val + self.eps) - 1
        return processed

    def _normalize_action(self, action):
        if action is None or "action" not in self._tensor_stats:
            return action

        tensor = (
            action.to(dtype=torch.float32)
            if isinstance(action, torch.Tensor)
            else torch.as_tensor(action, dtype=torch.float32)
        )
        stats = {k: v.to(tensor.device) for k, v in self._tensor_stats["action"].items()}
        if "mean" in stats and "std" in stats:
            mean, std = stats["mean"], stats["std"]
            return (tensor - mean) / (std + self.eps)
        if "min" in stats and "max" in stats:
            min_val, max_val = stats["min"], stats["max"]
            return 2 * (tensor - min_val) / (max_val - min_val + self.eps) - 1
        raise ValueError("Action stats must contain either ('mean','std') or ('min','max')")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = self._normalize_obs(transition[TransitionIndex.OBSERVATION])
        action = self._normalize_action(transition[TransitionIndex.ACTION])
        return (
            observation,
            action,
            transition[TransitionIndex.REWARD],
            transition[TransitionIndex.DONE],
            transition[TransitionIndex.TRUNCATED],
            transition[TransitionIndex.INFO],
            transition[TransitionIndex.COMPLEMENTARY_DATA],
        )

    def get_config(self) -> dict[str, Any]:
        return {"normalize_keys": list(self.normalize_keys) if self.normalize_keys else None, "eps": self.eps}

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


@dataclass
@ProcessorStepRegistry.register(name="unnormalizer_processor")
class UnnormalizerProcessor:
    """Inverse normalisation for observations and actions.

    Exactly mirrors :class:`NormalizerProcessor` but applies the inverse
    transform.
    """

    stats: dict[str, dict[str, Any]]
    unnormalize_keys: set[str] | None = None
    eps: float = 1e-8

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        *,
        unnormalize_keys: set[str] | None = None,
        eps: float = 1e-8,
    ) -> UnnormalizerProcessor:
        return cls(stats=dataset.meta.stats, unnormalize_keys=unnormalize_keys, eps=eps)

    def __post_init__(self):
        self._tensor_stats = _convert_stats_to_tensors(self.stats)

    def _unnormalize_obs(self, observation):
        if observation is None:
            return None
        keys = (
            self.unnormalize_keys
            if self.unnormalize_keys is not None
            else {k for k in self._tensor_stats if k != "action"}
        )
        processed = dict(observation)
        for key in keys:
            if key not in processed or key not in self._tensor_stats:
                continue
            orig_val = processed[key]
            tensor = (
                orig_val.to(dtype=torch.float32)
                if isinstance(orig_val, torch.Tensor)
                else torch.as_tensor(orig_val, dtype=torch.float32)
            )
            stats = {k: v.to(tensor.device) for k, v in self._tensor_stats[key].items()}
            if "mean" in stats and "std" in stats:
                mean, std = stats["mean"], stats["std"]
                processed[key] = tensor * std + mean
            elif "min" in stats and "max" in stats:
                min_val, max_val = stats["min"], stats["max"]
                processed[key] = (tensor + 1) / 2 * (max_val - min_val) + min_val
        return processed

    def _unnormalize_action(self, action):
        if action is None or "action" not in self._tensor_stats:
            return action
        tensor = (
            action.to(dtype=torch.float32)
            if isinstance(action, torch.Tensor)
            else torch.as_tensor(action, dtype=torch.float32)
        )
        stats = {k: v.to(tensor.device) for k, v in self._tensor_stats["action"].items()}
        if "mean" in stats and "std" in stats:
            mean, std = stats["mean"], stats["std"]
            return tensor * std + mean
        if "min" in stats and "max" in stats:
            min_val, max_val = stats["min"], stats["max"]
            return (tensor + 1) / 2 * (max_val - min_val) + min_val
        raise ValueError("Action stats must contain either ('mean','std') or ('min','max')")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = self._unnormalize_obs(transition[TransitionIndex.OBSERVATION])
        action = self._unnormalize_action(transition[TransitionIndex.ACTION])
        return (
            observation,
            action,
            transition[TransitionIndex.REWARD],
            transition[TransitionIndex.DONE],
            transition[TransitionIndex.TRUNCATED],
            transition[TransitionIndex.INFO],
            transition[TransitionIndex.COMPLEMENTARY_DATA],
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "unnormalize_keys": list(self.unnormalize_keys) if self.unnormalize_keys else None,
            "eps": self.eps,
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
