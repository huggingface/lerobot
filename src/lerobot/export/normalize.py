#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Normalizer for applying normalization stats during inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .manifest import NormalizationConfig, NormalizationType

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Normalizer:
    """Handles normalization and denormalization of inputs/outputs.

    Loads statistics from safetensors files and applies normalization
    transformations compatible with numpy arrays for runtime inference.
    """

    def __init__(
        self,
        config: NormalizationConfig,
        stats: dict[str, dict[str, NDArray[np.floating]]],
        eps: float = 1e-8,
    ):
        """Initialize the normalizer.

        Args:
            config: Normalization configuration from manifest.
            stats: Dictionary mapping feature names to their statistics.
            eps: Small epsilon for numerical stability.
        """
        self._config = config
        self._stats = stats
        self._eps = eps
        self._norm_type = config.type
        self._input_features = set(config.input_features)
        self._output_features = set(config.output_features)

    @classmethod
    def from_safetensors(cls, path: Path | str, config: NormalizationConfig, eps: float = 1e-8) -> Normalizer:
        """Load normalizer from a safetensors file.

        Args:
            path: Path to the safetensors file.
            config: Normalization configuration from manifest.
            eps: Small epsilon for numerical stability.

        Returns:
            Initialized Normalizer instance.
        """
        try:
            from safetensors.numpy import load_file
        except ImportError as e:
            raise ImportError("safetensors is required. Install with: pip install safetensors") from e

        path = Path(path)
        flat_stats = load_file(str(path))

        # Reconstruct nested stats dict from flat format: "feature_name.stat_name" -> tensor
        stats: dict[str, dict[str, NDArray[np.floating]]] = {}
        for flat_key, tensor in flat_stats.items():
            key, stat_name = flat_key.rsplit(".", 1)
            if key not in stats:
                stats[key] = {}
            stats[key][stat_name] = tensor.astype(np.float32)

        return cls(config, stats, eps)

    def normalize_inputs(
        self, observation: dict[str, NDArray[np.floating]]
    ) -> dict[str, NDArray[np.floating]]:
        """Apply normalization to input features.

        Args:
            observation: Dictionary mapping feature names to numpy arrays.

        Returns:
            Dictionary with normalized features.
        """
        result = dict(observation)
        for key in self._input_features:
            if key in result and key in self._stats:
                result[key] = self._apply_transform(result[key], key, inverse=False)
        return result

    def denormalize_outputs(self, action: NDArray[np.floating], key: str = "action") -> NDArray[np.floating]:
        """Apply denormalization to output features.

        Args:
            action: Action array to denormalize.
            key: Feature key for looking up stats.

        Returns:
            Denormalized action array.
        """
        if key in self._output_features and key in self._stats:
            return self._apply_transform(action, key, inverse=True)
        return action

    def _apply_transform(
        self,
        tensor: NDArray[np.floating],
        key: str,
        *,
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        """Apply normalization or denormalization transformation.

        Args:
            tensor: Input array.
            key: Feature key for looking up stats.
            inverse: If True, apply denormalization; otherwise normalize.

        Returns:
            Transformed array.
        """
        if self._norm_type == NormalizationType.IDENTITY or key not in self._stats:
            return tensor

        stats = self._stats[key]

        if self._norm_type == NormalizationType.STANDARD:
            return self._apply_standard(tensor, stats, inverse)
        elif self._norm_type == NormalizationType.MIN_MAX:
            return self._apply_min_max(tensor, stats, inverse)
        elif self._norm_type == NormalizationType.QUANTILES:
            return self._apply_quantiles(tensor, stats, inverse, q_low="q01", q_high="q99")
        elif self._norm_type == NormalizationType.QUANTILE10:
            return self._apply_quantiles(tensor, stats, inverse, q_low="q10", q_high="q90")

        return tensor

    def _apply_standard(
        self,
        tensor: NDArray[np.floating],
        stats: dict[str, NDArray[np.floating]],
        inverse: bool,
    ) -> NDArray[np.floating]:
        """Apply mean/std normalization."""
        mean = stats.get("mean")
        std = stats.get("std")

        if mean is None or std is None:
            return tensor

        denom = std + self._eps

        if inverse:
            return tensor * std + mean
        return (tensor - mean) / denom

    def _apply_min_max(
        self,
        tensor: NDArray[np.floating],
        stats: dict[str, NDArray[np.floating]],
        inverse: bool,
    ) -> NDArray[np.floating]:
        """Apply min/max normalization to [-1, 1]."""
        min_val = stats.get("min")
        max_val = stats.get("max")

        if min_val is None or max_val is None:
            return tensor

        denom = max_val - min_val
        # Replace zeros with epsilon to avoid division by zero
        denom = np.where(denom == 0, self._eps, denom)

        if inverse:
            # Map from [-1, 1] back to [min, max]
            return (tensor + 1) / 2 * denom + min_val
        # Map from [min, max] to [-1, 1]
        return 2 * (tensor - min_val) / denom - 1

    def _apply_quantiles(
        self,
        tensor: NDArray[np.floating],
        stats: dict[str, NDArray[np.floating]],
        inverse: bool,
        q_low: str,
        q_high: str,
    ) -> NDArray[np.floating]:
        """Apply quantile-based normalization to [-1, 1]."""
        q_low_val = stats.get(q_low)
        q_high_val = stats.get(q_high)

        if q_low_val is None or q_high_val is None:
            return tensor

        denom = q_high_val - q_low_val
        # Replace zeros with epsilon to avoid division by zero
        denom = np.where(denom == 0, self._eps, denom)

        if inverse:
            return (tensor + 1.0) * denom / 2.0 + q_low_val
        return 2.0 * (tensor - q_low_val) / denom - 1.0


def save_stats_safetensors(
    stats: dict[str, dict[str, Any]],
    path: Path | str,
) -> None:
    """Save normalization statistics to a safetensors file.

    Args:
        stats: Nested dictionary of statistics.
        path: Output path for the safetensors file.
    """
    try:
        from safetensors.numpy import save_file
    except ImportError as e:
        raise ImportError("safetensors is required. Install with: pip install safetensors") from e

    # Flatten nested dict to "feature_name.stat_name" format
    flat_stats: dict[str, NDArray[np.floating]] = {}
    for feature_name, feature_stats in stats.items():
        for stat_name, value in feature_stats.items():
            flat_key = f"{feature_name}.{stat_name}"
            if isinstance(value, np.ndarray):
                flat_stats[flat_key] = value.astype(np.float32)
            else:
                flat_stats[flat_key] = np.array(value, dtype=np.float32)

    save_file(flat_stats, str(path))
