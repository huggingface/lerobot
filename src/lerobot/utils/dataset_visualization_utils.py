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

"""Shared helpers for visualizing scalar features from a LeRobot dataset."""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

from .constants import ACTION, DEFAULT_FEATURES, DONE, OBS_STATE, REWARD, SUCCESS

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset


METADATA_KEYS = {*DEFAULT_FEATURES, "task"}
KNOWN_SCALAR_KEYS = {DONE, REWARD, SUCCESS}
SCALAR_DTYPE_KINDS = {"b", "i", "u", "f"}


def is_scalar_feature(feature: Mapping) -> bool:
    """Return whether a feature schema describes a numeric or boolean scalar."""

    dtype = feature.get("dtype")
    if not isinstance(dtype, str):
        return False
    try:
        dtype_kind = np.dtype(dtype).kind
    except (TypeError, ValueError):
        return False
    if dtype_kind not in SCALAR_DTYPE_KINDS:
        return False

    shape = feature.get("shape")
    if shape is None:
        return True
    if isinstance(shape, int):
        return shape == 1
    if not isinstance(shape, (list, tuple)):
        return False
    return len(shape) == 0 or (len(shape) == 1 and shape[0] == 1)


def get_extra_scalar_keys(dataset: "LeRobotDataset", additional_known_keys: Iterable[str] = ()) -> list[str]:
    """Return scalar feature keys not handled by the visualizer's standard paths."""

    known_keys = {
        ACTION,
        OBS_STATE,
        *KNOWN_SCALAR_KEYS,
        *METADATA_KEYS,
        *additional_known_keys,
        *dataset.meta.camera_keys,
    }
    return [
        key
        for key, feature in dataset.features.items()
        if key not in known_keys and is_scalar_feature(feature)
    ]


def is_scalar_like(value: object) -> bool:
    """Return whether a runtime value contains exactly one numeric or boolean scalar."""

    if isinstance(value, torch.Tensor):
        return value.numel() == 1 and not value.is_complex()
    if isinstance(value, np.ndarray):
        return value.size == 1 and value.dtype.kind in SCALAR_DTYPE_KINDS
    return np.isscalar(value) and np.asarray(value).dtype.kind in SCALAR_DTYPE_KINDS


def scalar_to_float(value: object) -> float:
    """Convert a scalar-like tensor, array, or Python value to ``float``."""

    return float(value.item() if hasattr(value, "item") else value)


def get_scalar_values(sample: Mapping, keys: Iterable[str]) -> dict[str, float]:
    """Select and convert scalar-like values from ``sample`` for the requested keys."""

    values = {}
    for key in keys:
        value = sample.get(key)
        if value is not None and is_scalar_like(value):
            values[key] = scalar_to_float(value)
    return values
