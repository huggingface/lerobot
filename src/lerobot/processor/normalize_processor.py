#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION

from .converters import from_tensor_to_numpy, to_tensor
from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry


@dataclass
class _NormalizationMixin:
    """
    A mixin class providing core functionality for normalization and unnormalization.

    This class manages normalization statistics (`stats`), converts them to tensors for
    efficient computation, handles device placement, and implements the logic for
    applying normalization transformations (mean/std and min/max). It is designed to
    be inherited by concrete `ProcessorStep` implementations and should not be used
    directly.

    **Stats Override Preservation:**
    When stats are explicitly provided during construction (e.g., via overrides in
    `DataProcessorPipeline.from_pretrained()`), they are preserved even when
    `load_state_dict()` is called. This allows users to override normalization
    statistics from saved models while keeping the rest of the model state intact.

    Examples:
        ```python
        # Common use case: Override with dataset stats
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset("my_dataset")
        pipeline = DataProcessorPipeline.from_pretrained(
            "model_path", overrides={"normalizer_processor": {"stats": dataset.meta.stats}}
        )
        # dataset.meta.stats will be used, not the stats from the saved model

        # Custom stats override
        custom_stats = {"action": {"mean": [0.0], "std": [1.0]}}
        pipeline = DataProcessorPipeline.from_pretrained(
            "model_path", overrides={"normalizer_processor": {"stats": custom_stats}}
        )
        ```

    Attributes:
        features: A dictionary mapping feature names to `PolicyFeature` objects, defining
            the data structure to be processed.
        norm_map: A dictionary mapping `FeatureType` to `NormalizationMode`, specifying
            which normalization method to use for each type of feature.
        stats: A dictionary containing the normalization statistics (e.g., mean, std,
            min, max) for each feature.
        device: The PyTorch device on which to store and perform tensor operations.
        eps: A small epsilon value to prevent division by zero in normalization
            calculations.
        normalize_observation_keys: An optional set of keys to selectively apply
            normalization to specific observation features.
        _tensor_stats: An internal dictionary holding the normalization statistics as
            PyTorch tensors.
        _stats_explicitly_provided: Internal flag tracking whether stats were explicitly
            provided during construction (used for override preservation).
    """

    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    eps: float = 1e-8
    normalize_observation_keys: set[str] | None = None

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)
    _stats_explicitly_provided: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """
        Initializes the mixin after dataclass construction.

        This method handles the robust deserialization of `features` and `norm_map`
        from JSON-compatible formats (where enums become strings and tuples become
        lists) and converts the provided `stats` dictionary into a dictionary of
        tensors (`_tensor_stats`) on the specified device.
        """
        # Track if stats were explicitly provided (not None and not empty)
        self._stats_explicitly_provided = self.stats is not None and bool(self.stats)
        # Robust JSON deserialization handling (guard empty maps).
        if self.features:
            first_val = next(iter(self.features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.features = reconstructed

        # if keys are strings (JSON), rebuild enum map
        if self.norm_map and all(isinstance(k, str) for k in self.norm_map):
            reconstructed = {}
            for ft_type_str, norm_mode_str in self.norm_map.items():
                reconstructed[FeatureType(ft_type_str)] = NormalizationMode(norm_mode_str)
            self.norm_map = reconstructed

        # Convert stats to tensors and move to the target device once during initialization.
        self.stats = self.stats or {}
        if self.dtype is None:
            self.dtype = torch.float32
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> _NormalizationMixin:
        """
        Moves the processor's normalization stats to the specified device.

        Args:
            device: The target PyTorch device.

        Returns:
            The instance of the class, allowing for method chaining.
        """
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        return self

    def state_dict(self) -> dict[str, Tensor]:
        """
        Returns the normalization statistics as a flat state dictionary.

        All tensors are moved to the CPU before being returned, which is standard practice
        for saving state dictionaries.

        Returns:
            A flat dictionary mapping from `'feature_name.stat_name'` to the
            corresponding statistics tensor on the CPU.
        """
        flat: dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor.cpu()  # Always save to CPU
        return flat

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """
        Loads normalization statistics from a state dictionary.

        The loaded tensors are moved to the processor's configured device.

        **Stats Override Preservation:**
        If stats were explicitly provided during construction (e.g., via overrides in
        `DataProcessorPipeline.from_pretrained()`), they are preserved and the state
        dictionary is ignored. This allows users to override normalization statistics
        while still loading the rest of the model state.

        This behavior is crucial for scenarios where users want to adapt a pretrained
        model to a new dataset with different statistics without retraining the entire
        model.

        Args:
            state: A flat state dictionary with keys in the format
                   `'feature_name.stat_name'`.

        Note:
            When stats are preserved due to explicit provision, only the tensor
            representation is updated to ensure consistency with the current device
            and dtype settings.
        """
        # If stats were explicitly provided during construction, preserve them
        if self._stats_explicitly_provided and self.stats is not None:
            # Don't load from state_dict, keep the explicitly provided stats
            # But ensure _tensor_stats is properly initialized
            self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)  # type: ignore[assignment]
            return

        # Normal behavior: load stats from state_dict
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            # Load to the processor's configured device.
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(
                dtype=torch.float32, device=self.device
            )

        # Reconstruct the original stats dict from tensor stats for compatibility with to() method
        # and other functions that rely on self.stats
        self.stats = {}
        for key, tensor_dict in self._tensor_stats.items():
            self.stats[key] = {}
            for stat_name, tensor in tensor_dict.items():
                # Convert tensor back to python/numpy format
                self.stats[key][stat_name] = from_tensor_to_numpy(tensor)

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        This method is used when saving the processor to disk, ensuring that its
        configuration can be reconstructed later.

        Returns:
            A JSON-serializable dictionary containing the configuration.
        """
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
        """
        Applies (un)normalization to all relevant features in an observation dictionary.

        Args:
            observation: The observation dictionary to process.
            inverse: If `True`, applies unnormalization; otherwise, applies normalization.

        Returns:
            A new observation dictionary with the transformed tensor values.
        """
        new_observation = dict(observation)
        for key, feature in self.features.items():
            if self.normalize_observation_keys is not None and key not in self.normalize_observation_keys:
                continue
            if feature.type != FeatureType.ACTION and key in new_observation:
                # Convert to tensor but preserve original dtype for adaptation logic
                tensor = torch.as_tensor(new_observation[key])
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=inverse)
        return new_observation

    def _normalize_action(self, action: Tensor, inverse: bool) -> Tensor:
        # Convert to tensor but preserve original dtype for adaptation logic
        """
        Applies (un)normalization to an action tensor.

        Args:
            action: The action tensor to process.
            inverse: If `True`, applies unnormalization; otherwise, applies normalization.

        Returns:
            The transformed action tensor.
        """
        processed_action = self._apply_transform(action, ACTION, FeatureType.ACTION, inverse=inverse)
        return processed_action

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> Tensor:
        """
        Core logic to apply a normalization or unnormalization transformation to a tensor.

        This method selects the appropriate normalization mode based on the feature type
        and applies the corresponding mathematical operation.

        Normalization Modes:
          - MEAN_STD: Centers data around zero with unit variance.
          - MIN_MAX: Scales data to [-1, 1] range using actual min/max values.
          - QUANTILES: Scales data to [-1, 1] range using 1st and 99th percentiles (q01/q99).
          - QUANTILE10: Scales data to [-1, 1] range using 10th and 90th percentiles (q10/q90).

        Args:
            tensor: The input tensor to transform.
            key: The feature key corresponding to the tensor.
            feature_type: The `FeatureType` of the tensor.
            inverse: If `True`, applies the inverse transformation (unnormalization).

        Returns:
            The transformed tensor.

        Raises:
            ValueError: If an unsupported normalization mode is encountered.
        """
        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        if norm_mode not in (
            NormalizationMode.MEAN_STD,
            NormalizationMode.MIN_MAX,
            NormalizationMode.QUANTILES,
            NormalizationMode.QUANTILE10,
        ):
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        # For Accelerate compatibility: Ensure stats are on the same device and dtype as the input tensor
        if self._tensor_stats and key in self._tensor_stats:
            first_stat = next(iter(self._tensor_stats[key].values()))
            if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
                self.to(device=tensor.device, dtype=tensor.dtype)

        stats = self._tensor_stats[key]

        if norm_mode == NormalizationMode.MEAN_STD:
            mean = stats.get("mean", None)
            std = stats.get("std", None)
            if mean is None or std is None:
                raise ValueError(
                    "MEAN_STD normalization mode requires mean and std stats, please update the dataset with the correct stats"
                )

            mean, std = stats["mean"], stats["std"]
            # Avoid division by zero by adding a small epsilon.
            denom = std + self.eps
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / denom

        if norm_mode == NormalizationMode.MIN_MAX:
            min_val = stats.get("min", None)
            max_val = stats.get("max", None)
            if min_val is None or max_val is None:
                raise ValueError(
                    "MIN_MAX normalization mode requires min and max stats, please update the dataset with the correct stats"
                )

            min_val, max_val = stats["min"], stats["max"]
            denom = max_val - min_val
            # When min_val == max_val, substitute the denominator with a small epsilon
            # to prevent division by zero. This consistently maps an input equal to
            # min_val to -1, ensuring a stable transformation.
            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:
                # Map from [-1, 1] back to [min, max]
                return (tensor + 1) / 2 * denom + min_val
            # Map from [min, max] to [-1, 1]
            return 2 * (tensor - min_val) / denom - 1

        if norm_mode == NormalizationMode.QUANTILES:
            q01 = stats.get("q01", None)
            q99 = stats.get("q99", None)
            if q01 is None or q99 is None:
                raise ValueError(
                    "QUANTILES normalization mode requires q01 and q99 stats, please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script"
                )

            denom = q99 - q01
            # Avoid division by zero by adding epsilon when quantiles are identical
            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + q01
            return 2.0 * (tensor - q01) / denom - 1.0

        if norm_mode == NormalizationMode.QUANTILE10:
            q10 = stats.get("q10", None)
            q90 = stats.get("q90", None)
            if q10 is None or q90 is None:
                raise ValueError(
                    "QUANTILE10 normalization mode requires q10 and q90 stats, please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script"
                )

            denom = q90 - q10
            # Avoid division by zero by adding epsilon when quantiles are identical
            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + q10
            return 2.0 * (tensor - q10) / denom - 1.0

        # If necessary stats are missing, return input unchanged.
        return tensor


@dataclass
@ProcessorStepRegistry.register(name="normalizer_processor")
class NormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """
    A processor step that applies normalization to observations and actions in a transition.

    This class uses the logic from `_NormalizationMixin` to perform forward normalization
    (e.g., scaling data to have zero mean and unit variance, or to the range [-1, 1]).
    It is typically used in the pre-processing pipeline before feeding data to a policy.
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
    ) -> NormalizerProcessorStep:
        """
        Creates a `NormalizerProcessorStep` instance using statistics from a `LeRobotDataset`.

        Args:
            dataset: The dataset from which to extract normalization statistics.
            features: The feature definition for the processor.
            norm_map: The mapping from feature types to normalization modes.
            normalize_observation_keys: An optional set of observation keys to normalize.
            eps: A small epsilon value for numerical stability.
            device: The target device for the processor.

        Returns:
            A new instance of `NormalizerProcessorStep`.
        """
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

        if action is None:
            return new_transition

        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=False)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="unnormalizer_processor")
class UnnormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """
    A processor step that applies unnormalization to observations and actions.

    This class inverts the normalization process, scaling data back to its original
    range. It is typically used in the post-processing pipeline to convert a policy's
    normalized action output into a format that can be executed by a robot or
    environment.
    """

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        device: torch.device | str | None = None,
    ) -> UnnormalizerProcessorStep:
        """
        Creates an `UnnormalizerProcessorStep` using statistics from a `LeRobotDataset`.

        Args:
            dataset: The dataset from which to extract normalization statistics.
            features: The feature definition for the processor.
            norm_map: The mapping from feature types to normalization modes.
            device: The target device for the processor.

        Returns:
            A new instance of `UnnormalizerProcessorStep`.
        """
        return cls(features=features, norm_map=norm_map, stats=dataset.meta.stats, device=device)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        # Handle observation unnormalization.
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(observation, inverse=True)

        # Handle action unnormalization.
        action = new_transition.get(TransitionKey.ACTION)

        if action is None:
            return new_transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=True)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def hotswap_stats(
    policy_processor: PolicyProcessorPipeline, stats: dict[str, dict[str, Any]]
) -> PolicyProcessorPipeline:
    """
    Replaces normalization statistics in an existing `PolicyProcessorPipeline` instance.

    This function creates a deep copy of the provided pipeline and updates the
    statistics of any `NormalizerProcessorStep` or `UnnormalizerProcessorStep` it
    contains. This is useful for adapting a trained policy to a new environment or
    dataset with different data distributions without having to reconstruct the entire
    pipeline.

    Args:
        policy_processor: The policy processor pipeline to modify.
        stats: The new dictionary of normalization statistics to apply.

    Returns:
        A new `PolicyProcessorPipeline` instance with the updated statistics.
    """
    rp = deepcopy(policy_processor)
    for step in rp.steps:
        if isinstance(step, _NormalizationMixin):
            step.stats = stats
            # Re-initialize tensor_stats on the correct device.
            step._tensor_stats = to_tensor(stats, device=step.device, dtype=step.dtype)  # type: ignore[assignment]
    return rp
