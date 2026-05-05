#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Runtime normalization for exported policies.

Each feature is normalized independently according to the mode declared
in its :class:`ProcessorSpec` (``mean_std``, ``min_max``, ``quantiles``,
``quantile10``, or ``identity``). A single policy may mix modes across
features — e.g. Diffusion uses ``min_max`` for state/action and
``mean_std`` for images; PI05 uses ``quantiles`` for state/action.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .manifest import ProcessorSpec

__all__ = ["Normalizer", "save_stats_safetensors"]

NORMALIZE_EPS: float = 1e-8
# Tiny epsilon prevents division-by-zero when stats contain zero-variance features.

_MODE_ALIASES: dict[str, str] = {
    "mean_std": "mean_std",
    "standard": "mean_std",
    "min_max": "min_max",
    "identity": "identity",
    "quantiles": "quantiles",
    "quantile10": "quantile10",
}


def _canonical_mode(mode: str | None) -> str:
    if mode is None:
        return "mean_std"
    key = str(mode).lower()
    if key not in _MODE_ALIASES:
        raise ValueError(
            f"Unknown normalization mode: {mode!r}. Known modes: {sorted(set(_MODE_ALIASES.values()))}."
        )
    return _MODE_ALIASES[key]


class Normalizer:
    """Applies per-feature normalization to inference inputs and outputs.

    Features are registered individually with their canonical mode
    (``mean_std``, ``min_max``, ``quantiles``, ``quantile10``, or
    ``identity``) and the safetensors stats they should consult.
    Unknown features pass through unchanged. All non-``identity`` modes
    fail fast (``ValueError``) if their required stats keys are missing,
    matching the training-side processor contract.
    """

    def __init__(
        self,
        input_specs: dict[str, str],
        output_specs: dict[str, str],
        stats: dict[str, dict[str, NDArray[np.floating]]],
        eps: float = NORMALIZE_EPS,
    ) -> None:
        self._input_specs = {key: _canonical_mode(mode) for key, mode in input_specs.items()}
        self._output_specs = {key: _canonical_mode(mode) for key, mode in output_specs.items()}
        self._stats = stats
        self._eps = eps

    @classmethod
    def from_specs(
        cls,
        preprocessors: list[ProcessorSpec] | None,
        postprocessors: list[ProcessorSpec] | None,
        package_path: Path | str,
        eps: float = NORMALIZE_EPS,
    ) -> Normalizer | None:
        """Construct a :class:`Normalizer` from manifest processor specs.

        Scans the preprocessor list for ``"normalize"`` specs and the
        postprocessor list for ``"denormalize"`` specs, loads the referenced
        stats files, and returns a ready-to-use normalizer.

        Args:
            preprocessors: List of preprocessor specs from the manifest, or
                ``None``.
            postprocessors: List of postprocessor specs from the manifest, or
                ``None``.
            package_path: Root directory of the exported package (used to
                resolve relative artifact paths).
            eps: Small constant added to std/range to avoid division by zero.

        Returns:
            A :class:`Normalizer` instance, or ``None`` if no normalisation
            specs are present.

        Raises:
            ValueError: If non-identity specs are declared but no stats
                artifact is referenced.
            FileNotFoundError: If a referenced stats artifact does not exist.
        """
        package_path = Path(package_path)

        input_specs: dict[str, str] = {}
        output_specs: dict[str, str] = {}
        artifacts: set[str] = set()

        non_identity_seen = False

        for spec in preprocessors or []:
            if spec.type != "normalize":
                continue
            mode = _canonical_mode(spec.mode)
            for feature in spec.features or []:
                input_specs[feature] = mode
            if mode != "identity":
                non_identity_seen = True
            if spec.artifact:
                artifacts.add(spec.artifact)

        for spec in postprocessors or []:
            if spec.type != "denormalize":
                continue
            mode = _canonical_mode(spec.mode)
            for feature in spec.features or []:
                output_specs[feature] = mode
            if mode != "identity":
                non_identity_seen = True
            if spec.artifact:
                artifacts.add(spec.artifact)

        if not input_specs and not output_specs:
            return None

        if not artifacts:
            # Legitimate when every spec is identity (no stats needed).
            # Otherwise the manifest is malformed: non-identity modes must
            # declare a stats artifact.
            if non_identity_seen:
                raise ValueError(
                    "Manifest declares non-identity normalization specs but no stats artifact. "
                    "Every non-identity normalize/denormalize spec must reference a stats file."
                )
            return None

        stats: dict[str, dict[str, NDArray[np.floating]]] = {}
        for artifact in artifacts:
            stats_path = package_path / artifact
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"Normalization stats artifact not found: {stats_path}. "
                    "The manifest declares normalization specs but the stats file is missing."
                )
            stats.update(_load_stats(stats_path))

        return cls(input_specs, output_specs, stats, eps)

    @classmethod
    def from_safetensors(
        cls,
        path: Path | str,
        mode: str = "mean_std",
        input_features: list[str] | None = None,
        output_features: list[str] | None = None,
        eps: float = NORMALIZE_EPS,
    ) -> Normalizer:
        """Construct a :class:`Normalizer` directly from a safetensors stats file.

        Convenience factory for cases where the manifest is not available.

        Args:
            path: Path to the ``safetensors`` stats file.
            mode: Normalisation mode applied to all features (default
                ``"mean_std"``).
            input_features: Feature keys to normalise on input.  ``None``
                means no input normalisation.
            output_features: Feature keys to denormalise on output.  ``None``
                means no output denormalisation.
            eps: Small constant added to std/range to avoid division by zero.

        Returns:
            A :class:`Normalizer` instance.
        """
        stats = _load_stats(path)
        canonical = _canonical_mode(mode)
        input_specs = dict.fromkeys(input_features or [], canonical)
        output_specs = dict.fromkeys(output_features or [], canonical)
        return cls(input_specs, output_specs, stats, eps)

    def normalize_inputs(
        self,
        observation: dict[str, NDArray[np.floating]],
    ) -> dict[str, NDArray[np.floating]]:
        """Normalise registered input features in an observation dict.

        Features not registered in ``input_specs`` pass through unchanged.

        Args:
            observation: Dict mapping feature keys to numpy arrays.

        Returns:
            A new dict with registered features normalised in-place.

        Raises:
            ValueError: If a registered feature is present in the observation
                but its stats are missing.
        """
        result = dict(observation)
        for key, mode in self._input_specs.items():
            if key not in result:
                continue
            if key not in self._stats:
                raise ValueError(f"missing normalization stats for key {key!r}")
            result[key] = self._apply_transform(result[key], key, mode, inverse=False)
        return result

    def denormalize_outputs(
        self,
        action: NDArray[np.floating],
        key: str = "action",
    ) -> NDArray[np.floating]:
        """Denormalise a single output tensor.

        If ``key`` is not registered in ``output_specs`` the tensor is
        returned unchanged.

        Args:
            action: Output tensor to denormalise.
            key: Feature key used to look up the output spec and stats
                (default ``"action"``).

        Returns:
            Denormalised tensor with the same shape as ``action``.

        Raises:
            ValueError: If ``key`` is registered but its stats are missing.
        """
        mode = self._output_specs.get(key)
        if mode is None:
            return action
        if key not in self._stats:
            raise ValueError(f"missing normalization stats for key {key!r}")
        return self._apply_transform(action, key, mode, inverse=True)

    def _apply_transform(
        self,
        tensor: NDArray[np.floating],
        key: str,
        mode: str,
        *,
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        if mode == "identity" or key not in self._stats:
            return tensor

        stats = self._stats[key]

        # Padded-action families emit predictions at max_action_dim while saved
        # stats track the real action_dim. Mirror eager postprocessing behavior
        # by trimming inverse-transform inputs to the stats vector length.
        if inverse:
            tensor = _trim_to_stats_length(tensor, stats)

        if mode == "mean_std":
            return self._apply_mean_std(tensor, stats, inverse)
        if mode == "min_max":
            return self._apply_min_max(tensor, stats, inverse)
        if mode == "quantiles":
            return self._apply_quantile_range(tensor, stats, inverse, lower_key="q01", upper_key="q99")
        if mode == "quantile10":
            return self._apply_quantile_range(tensor, stats, inverse, lower_key="q10", upper_key="q90")

        return tensor

    def _apply_mean_std(
        self,
        tensor: NDArray[np.floating],
        stats: dict[str, NDArray[np.floating]],
        inverse: bool,
    ) -> NDArray[np.floating]:
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None:
            raise ValueError(
                f"mean_std normalization requires 'mean' and 'std' stats; found keys {sorted(stats)}."
            )

        if inverse:
            return tensor * std + mean
        return (tensor - mean) / (std + self._eps)

    def _apply_min_max(
        self,
        tensor: NDArray[np.floating],
        stats: dict[str, NDArray[np.floating]],
        inverse: bool,
    ) -> NDArray[np.floating]:
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is None or max_val is None:
            raise ValueError(
                f"min_max normalization requires 'min' and 'max' stats; found keys {sorted(stats)}."
            )

        denom = max_val - min_val
        denom = np.where(denom == 0, self._eps, denom)

        if inverse:
            return (tensor + 1) / 2 * denom + min_val
        return 2 * (tensor - min_val) / denom - 1

    def _apply_quantile_range(
        self,
        tensor: NDArray[np.floating],
        stats: dict[str, NDArray[np.floating]],
        inverse: bool,
        *,
        lower_key: str,
        upper_key: str,
    ) -> NDArray[np.floating]:
        lower = stats.get(lower_key)
        upper = stats.get(upper_key)
        if lower is None or upper is None:
            raise ValueError(
                f"Quantile normalization requires '{lower_key}' and '{upper_key}' stats; "
                f"found keys {sorted(stats)}."
            )

        denom = upper - lower
        denom = np.where(denom == 0, self._eps, denom)

        if inverse:
            return (tensor + 1.0) / 2.0 * denom + lower
        return 2.0 * (tensor - lower) / denom - 1.0


def _trim_to_stats_length(
    tensor: NDArray[np.floating],
    stats: dict[str, NDArray[np.floating]],
) -> NDArray[np.floating]:
    if tensor.ndim == 0:
        return tensor
    for vec in stats.values():
        if vec.ndim == 1 and tensor.shape[-1] > vec.shape[0]:
            return tensor[..., : vec.shape[0]]
        break
    return tensor


def _load_stats(path: Path | str) -> dict[str, dict[str, NDArray[np.floating]]]:
    try:
        from safetensors.numpy import load_file
    except ImportError as e:
        raise ImportError("safetensors is required. Install with: pip install safetensors") from e

    flat = load_file(str(path))
    stats: dict[str, dict[str, NDArray[np.floating]]] = {}
    for flat_key, tensor in flat.items():
        feature_name, stat_name = flat_key.rsplit("/", 1)
        if feature_name not in stats:
            stats[feature_name] = {}
        stats[feature_name][stat_name] = tensor.astype(np.float32)
    return stats


def save_stats_safetensors(
    stats: dict[str, dict[str, Any]],
    path: Path | str,
) -> None:
    """Serialise a nested stats dict to a flat safetensors file.

    Keys are flattened to ``"<feature_name>/<stat_name>"`` format.  All
    values are cast to ``float32`` before saving.

    Args:
        stats: Nested dict mapping feature name → stat name → array or
            array-like value.
        path: Destination file path.

    Raises:
        ImportError: If ``safetensors`` is not installed.
    """
    try:
        from safetensors.numpy import save_file
    except ImportError as e:
        raise ImportError("safetensors is required. Install with: pip install safetensors") from e

    flat: dict[str, NDArray[np.floating]] = {}
    for feature_name, feature_stats in stats.items():
        for stat_name, value in feature_stats.items():
            flat_key = f"{feature_name}/{stat_name}"
            if isinstance(value, np.ndarray):
                flat[flat_key] = value.astype(np.float32)
            else:
                flat[flat_key] = np.array(value, dtype=np.float32)

    save_file(flat, str(path))
