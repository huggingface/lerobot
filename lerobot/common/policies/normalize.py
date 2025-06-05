#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import numpy as np
import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )


def _initialize_stats_buffers(
    module: nn.Module,
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> None:
    """Register statistics buffers (mean/std or min/max) on the given *module*.

    The logic matches the previous constructors of `NormalizeBuffer` and `UnnormalizeBuffer`,
    but is factored out so it can be reused by both classes and stay in sync.
    """
    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        shape: tuple[int, ...] = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            # reduce spatial dimensions, keep channel dimension only
            c, *_ = shape
            shape = (c, 1, 1)

        prefix = key.replace(".", "_")

        if norm_mode is NormalizationMode.MEAN_STD:
            mean = torch.full(shape, torch.inf, dtype=torch.float32)
            std = torch.full(shape, torch.inf, dtype=torch.float32)

            if stats and key in stats and "mean" in stats[key] and "std" in stats[key]:
                mean_data = stats[key]["mean"]
                std_data = stats[key]["std"]
                if isinstance(mean_data, torch.Tensor):
                    # Note: The clone is needed to make sure that the logic in save_pretrained doesn't see duplicated
                    # tensors anywhere (for example, when we use the same stats for normalization and
                    # unnormalization). See the logic here
                    # https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L97.
                    mean = mean_data.clone().to(dtype=torch.float32)
                    std = std_data.clone().to(dtype=torch.float32)
                elif isinstance(mean_data, np.ndarray):
                    mean = torch.from_numpy(mean_data).to(dtype=torch.float32)
                    std = torch.from_numpy(std_data).to(dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported stats type for key '{key}' (expected ndarray or Tensor).")

            module.register_buffer(f"{prefix}_mean", mean)
            module.register_buffer(f"{prefix}_std", std)
            continue

        if norm_mode is NormalizationMode.MIN_MAX:
            min_val = torch.full(shape, torch.inf, dtype=torch.float32)
            max_val = torch.full(shape, torch.inf, dtype=torch.float32)

            if stats and key in stats and "min" in stats[key] and "max" in stats[key]:
                min_data = stats[key]["min"]
                max_data = stats[key]["max"]
                if isinstance(min_data, torch.Tensor):
                    min_val = min_data.clone().to(dtype=torch.float32)
                    max_val = max_data.clone().to(dtype=torch.float32)
                elif isinstance(min_data, np.ndarray):
                    min_val = torch.from_numpy(min_data).to(dtype=torch.float32)
                    max_val = torch.from_numpy(max_data).to(dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported stats type for key '{key}' (expected ndarray or Tensor).")

            module.register_buffer(f"{prefix}_min", min_val)
            module.register_buffer(f"{prefix}_max", max_val)
            continue

        raise ValueError(norm_mode)


class Normalize(nn.Module):
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map

        _initialize_stats_buffers(self, features, norm_map, stats)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            prefix = key.replace(".", "_")

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = getattr(self, f"{prefix}_mean")
                std = getattr(self, f"{prefix}_std")
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = (batch[key] - mean) / (std + 1e-8)
                continue

            if norm_mode is NormalizationMode.MIN_MAX:
                min_val = getattr(self, f"{prefix}_min")
                max_val = getattr(self, f"{prefix}_max")
                assert not torch.isinf(min_val).any(), _no_stats_error_str("min")
                assert not torch.isinf(max_val).any(), _no_stats_error_str("max")
                batch[key] = (batch[key] - min_val) / (max_val - min_val + 1e-8)
                batch[key] = batch[key] * 2 - 1
                continue

            raise ValueError(norm_mode)

        return batch


class Unnormalize(nn.Module):
    """Inverse operation of `Normalize`. Uses registered buffers for statistics."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map

        _initialize_stats_buffers(self, features, norm_map, stats)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # batch = dict(batch)
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            prefix = key.replace(".", "_")

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = getattr(self, f"{prefix}_mean")
                std = getattr(self, f"{prefix}_std")
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = batch[key] * std + mean
                continue

            if norm_mode is NormalizationMode.MIN_MAX:
                min_val = getattr(self, f"{prefix}_min")
                max_val = getattr(self, f"{prefix}_max")
                assert not torch.isinf(min_val).any(), _no_stats_error_str("min")
                assert not torch.isinf(max_val).any(), _no_stats_error_str("max")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max_val - min_val) + min_val
                continue

            raise ValueError(norm_mode)

        return batch


def convert_normalize_to_buffer_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert state dict from Normalize/Unnormalize classes to NormalizeBuffer/UnnormalizeBuffer format.

    Args:
        state_dict: State dict from a model using Normalize/Unnormalize classes

    Returns:
        Converted state dict compatible with NormalizeBuffer/UnnormalizeBuffer classes

    Example:
        # Old format (Normalize): "buffer_observation_image.mean"
        # New format (NormalizeBuffer): "observation_image_mean"
    """
    converted_state_dict = {}

    for key, value in state_dict.items():
        # Check if this is a normalization buffer parameter
        if key.startswith("buffer_") and ("." in key):
            # Extract the prefix and stat type
            # e.g. "buffer_observation_image.mean" -> "observation_image", "mean"
            buffer_part = key[7:]  # Remove "buffer_" prefix
            prefix, stat_type = buffer_part.rsplit(".", 1)

            # Convert to new format: "observation_image_mean"
            new_key = f"{prefix}_{stat_type}"
            converted_state_dict[new_key] = value
        else:
            # Keep non-normalization keys unchanged
            converted_state_dict[key] = value

    return converted_state_dict
