from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import Tensor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionIndex


def _convert_stats_to_tensors(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Tensor]]:
    """Convert numpy arrays and other types to torch tensors."""
    tensor_stats: Dict[str, Dict[str, Tensor]] = {}
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
@ProcessorStepRegistry.register(name="observation_normalizer")
class ObservationNormalizer:
    """Normalize observations using dataset statistics.

    This processor normalizes selected observation keys using either:
    - Standard normalization: ``(x - mean) / (std + eps)``
    - Min-Max normalization to [-1, 1]: ``2 * (x - min) / (max - min + eps) - 1``

    Parameters
    ----------
    stats : Dict[str, Dict[str, np.ndarray | Tensor]]
        Dataset statistics. Each entry must provide either
        ``{"mean", "std"}`` or ``{"min", "max"}``.
    normalize_keys : set[str] | None, default=None
        Observation keys to normalize. ``None`` means all keys
        present in both the observation and stats.
    eps : float, default=1e-8
        Small constant to avoid division by zero.
    """

    stats: Dict[str, Dict[str, Any]]
    normalize_keys: set[str] | None = None
    eps: float = 1e-8

    # Cached tensors for performance
    _tensor_stats: Dict[str, Dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        *,
        normalize_keys: set[str] | None = None,
        eps: float = 1e-8,
    ) -> ObservationNormalizer:
        """Create from a LeRobotDataset."""
        # Filter stats to only include observation keys
        obs_stats = {k: v for k, v in dataset.meta.stats.items() if k != "action"}
        return cls(stats=obs_stats, normalize_keys=normalize_keys, eps=eps)

    def __post_init__(self):
        self._tensor_stats = _convert_stats_to_tensors(self.stats)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition[TransitionIndex.OBSERVATION]

        if observation is None:
            return transition

        # Determine which keys to normalize
        keys_to_norm = (
            self.normalize_keys if self.normalize_keys is not None else set(self._tensor_stats.keys())
        )

        # Create a copy to avoid mutating input
        processed_obs = dict(observation)

        for key in keys_to_norm:
            if key not in processed_obs:
                continue

            if key not in self._tensor_stats:
                if self.normalize_keys is not None:
                    # User explicitly requested this key but stats are missing
                    raise KeyError(f"Stats not found for requested key '{key}'")
                continue

            # Convert to tensor if needed
            orig_val = processed_obs[key]
            if isinstance(orig_val, torch.Tensor):
                tensor = orig_val.to(dtype=torch.float32)
            elif isinstance(orig_val, np.ndarray):
                tensor = torch.from_numpy(orig_val.astype(np.float32))
            else:
                # For lists, tuples, scalars, etc.
                tensor = torch.as_tensor(orig_val, dtype=torch.float32)

            stats = self._tensor_stats[key]
            # Move stats to same device as data
            stats = {k: v.to(device=tensor.device) for k, v in stats.items()}

            # Apply normalization
            if "mean" in stats and "std" in stats:
                mean, std = stats["mean"], stats["std"]
                processed_obs[key] = (tensor - mean) / (std + self.eps)
            elif "min" in stats and "max" in stats:
                min_val, max_val = stats["min"], stats["max"]
                # Normalize to [0, 1] then to [-1, 1]
                processed_obs[key] = 2 * (tensor - min_val) / (max_val - min_val + self.eps) - 1
            else:
                raise ValueError(
                    f"Stats for key '{key}' must contain either ('mean', 'std') or ('min', 'max')"
                )

        # Return new transition with normalized observation
        return (
            processed_obs,
            transition[TransitionIndex.ACTION],
            transition[TransitionIndex.REWARD],
            transition[TransitionIndex.DONE],
            transition[TransitionIndex.TRUNCATED],
            transition[TransitionIndex.INFO],
            transition[TransitionIndex.COMPLEMENTARY_DATA],
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "normalize_keys": list(self.normalize_keys) if self.normalize_keys is not None else None,
            "eps": self.eps,
        }

    def state_dict(self) -> Dict[str, Tensor]:
        flat_state: Dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat_state[f"{key}.{stat_name}"] = tensor
        return flat_state

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.split(".", 1)
            if key not in self._tensor_stats:
                self._tensor_stats[key] = {}
            self._tensor_stats[key][stat_name] = tensor

    def reset(self) -> None:
        """Nothing to reset for this stateless processor."""
        pass


@dataclass
@ProcessorStepRegistry.register(name="action_unnormalizer")
class ActionUnnormalizer:
    """Un-normalize actions using dataset statistics.

    This processor un-normalizes actions using the inverse of normalization:
    - Standard: ``action * std + mean``
    - Min-Max from [-1, 1]: ``(action + 1) / 2 * (max - min) + min``

    Parameters
    ----------
    action_stats : Dict[str, np.ndarray | Tensor]
        Action statistics containing either ``{"mean", "std"}`` or ``{"min", "max"}``.
    eps : float, default=1e-8
        Small constant used during normalization (not used in unnormalization).
    """

    action_stats: Dict[str, Any]
    eps: float = 1e-8  # Kept for consistency, not used in unnormalization

    # Cached tensors for performance
    _tensor_stats: Dict[str, Tensor] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        *,
        eps: float = 1e-8,
    ) -> ActionUnnormalizer:
        """Create from a LeRobotDataset."""
        if "action" not in dataset.meta.stats:
            raise ValueError("Dataset does not contain action statistics")
        return cls(action_stats=dataset.meta.stats["action"], eps=eps)

    def __post_init__(self):
        # Convert action stats to tensors
        tensor_stats = _convert_stats_to_tensors({"action": self.action_stats})
        self._tensor_stats = tensor_stats["action"]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition[TransitionIndex.ACTION]

        if action is None:
            return transition

        # Convert to tensor if needed
        if isinstance(action, torch.Tensor):
            action = action.to(dtype=torch.float32)
        else:
            action = torch.as_tensor(action, dtype=torch.float32)

        # Move stats to same device as action
        stats = {k: v.to(device=action.device) for k, v in self._tensor_stats.items()}

        # Apply unnormalization
        if "mean" in stats and "std" in stats:
            mean, std = stats["mean"], stats["std"]
            unnormalized_action = action * std + mean
        elif "min" in stats and "max" in stats:
            min_val, max_val = stats["min"], stats["max"]
            # Map from [-1, 1] to [0, 1] then to [min, max]
            unnormalized_action = (action + 1) / 2 * (max_val - min_val) + min_val
        else:
            raise ValueError("Action stats must contain either ('mean', 'std') or ('min', 'max')")

        # Return new transition with unnormalized action
        return (
            transition[TransitionIndex.OBSERVATION],
            unnormalized_action,
            transition[TransitionIndex.REWARD],
            transition[TransitionIndex.DONE],
            transition[TransitionIndex.TRUNCATED],
            transition[TransitionIndex.INFO],
            transition[TransitionIndex.COMPLEMENTARY_DATA],
        )

    def get_config(self) -> Dict[str, Any]:
        return {"eps": self.eps}

    def state_dict(self) -> Dict[str, Tensor]:
        return dict(self._tensor_stats.items())

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self._tensor_stats = dict(state)

    def reset(self) -> None:
        """Nothing to reset for this stateless processor."""
        pass


@dataclass
@ProcessorStepRegistry.register(name="normalization_processor")
class NormalizationProcessor:
    """Combined processor that normalizes observations and/or un-normalizes actions.

    This processor combines the functionality of ObservationNormalizer and
    ActionUnnormalizer for convenience when both operations are needed.

    Parameters
    ----------
    stats : Dict[str, Dict[str, np.ndarray | Tensor]]
        Dataset statistics as returned by ``LeRobotDataset.meta.stats``.
    normalize_keys : set[str] | None, default=None
        Observation keys to normalize. ``None`` means all keys
        present in both the observation and stats.
    unnormalize_action : bool, default=True
        Whether to un-normalize actions.
    eps : float, default=1e-8
        Small constant to avoid division by zero.
    """

    stats: Dict[str, Dict[str, Any]]
    normalize_keys: set[str] | None = None
    unnormalize_action: bool = True
    eps: float = 1e-8

    # Cached tensors for performance
    _tensor_stats: Dict[str, Dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        *,
        normalize_keys: set[str] | None = None,
        unnormalize_action: bool = True,
        eps: float = 1e-8,
    ) -> NormalizationProcessor:
        """Create from a LeRobotDataset."""
        return cls(
            stats=dataset.meta.stats,
            normalize_keys=normalize_keys,
            unnormalize_action=unnormalize_action,
            eps=eps,
        )

    def __post_init__(self):
        self._tensor_stats = _convert_stats_to_tensors(self.stats)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition[TransitionIndex.OBSERVATION]
        action = transition[TransitionIndex.ACTION]

        # Normalize observations
        if observation is not None:
            processed_obs = dict(observation)
            keys_to_norm = (
                self.normalize_keys
                if self.normalize_keys is not None
                else {k for k in self._tensor_stats if k != "action"}
            )

            for key in keys_to_norm:
                if key not in processed_obs or key not in self._tensor_stats:
                    continue

                orig_val = processed_obs[key]
                if isinstance(orig_val, torch.Tensor):
                    tensor = orig_val.to(dtype=torch.float32)
                elif isinstance(orig_val, np.ndarray):
                    tensor = torch.from_numpy(orig_val.astype(np.float32))
                else:
                    tensor = torch.as_tensor(orig_val, dtype=torch.float32)

                stats = self._tensor_stats[key]
                stats = {k: v.to(device=tensor.device) for k, v in stats.items()}

                if "mean" in stats and "std" in stats:
                    mean, std = stats["mean"], stats["std"]
                    processed_obs[key] = (tensor - mean) / (std + self.eps)
                elif "min" in stats and "max" in stats:
                    min_val, max_val = stats["min"], stats["max"]
                    processed_obs[key] = 2 * (tensor - min_val) / (max_val - min_val + self.eps) - 1

            observation = processed_obs

        # Un-normalize action
        if self.unnormalize_action and action is not None and "action" in self._tensor_stats:
            if isinstance(action, torch.Tensor):
                action = action.to(dtype=torch.float32)
            else:
                action = torch.as_tensor(action, dtype=torch.float32)

            stats = {k: v.to(device=action.device) for k, v in self._tensor_stats["action"].items()}

            if "mean" in stats and "std" in stats:
                mean, std = stats["mean"], stats["std"]
                action = action * std + mean
            elif "min" in stats and "max" in stats:
                min_val, max_val = stats["min"], stats["max"]
                action = (action + 1) / 2 * (max_val - min_val) + min_val

        # Return new transition
        return (
            observation,
            action,
            transition[TransitionIndex.REWARD],
            transition[TransitionIndex.DONE],
            transition[TransitionIndex.TRUNCATED],
            transition[TransitionIndex.INFO],
            transition[TransitionIndex.COMPLEMENTARY_DATA],
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "normalize_keys": list(self.normalize_keys) if self.normalize_keys is not None else None,
            "unnormalize_action": self.unnormalize_action,
            "eps": self.eps,
        }

    def state_dict(self) -> Dict[str, Tensor]:
        flat_state: Dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat_state[f"{key}.{stat_name}"] = tensor
        return flat_state

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.split(".", 1)
            if key not in self._tensor_stats:
                self._tensor_stats[key] = {}
            self._tensor_stats[key][stat_name] = tensor

    def reset(self) -> None:
        """Nothing to reset for this stateless processor."""
        pass
