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

"""Torch-free canonical feature schemas and value validation shared by both services."""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    """Named canonical values, ordered components, and semantic conventions."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str = "tensor"
    names: tuple[str, ...] = ()
    semantics: str = ""

    def __post_init__(self) -> None:
        """Reject unsupported modalities and ambiguous feature metadata."""
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "names", tuple(self.names))
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or not isinstance(self.semantics, str)
            or not self.semantics.strip()
        ):
            raise ValueError("Feature names and explicit semantic conventions are required.")
        if self.kind not in {"tensor", "rgb"}:
            raise ValueError(f"Unsupported feature modality: {self.kind!r}")
        if len(self.shape) > 4 or any(type(dim) is not int or dim <= 0 for dim in self.shape):
            raise ValueError(f"Invalid feature shape: {self.shape!r}")
        if self.dtype not in {"bool", "uint8", "int32", "int64", "float32", "float64"}:
            raise ValueError(f"Unsupported feature dtype: {self.dtype!r}")
        if self.kind == "rgb" and (len(self.shape) != 3 or self.shape[-1] != 3 or self.dtype != "uint8"):
            raise ValueError("RGB features require HWC uint8 with three channels.")
        if self.names and (len(self.shape) != 1 or len(self.names) != self.shape[0]):
            raise ValueError("Ordered component names must match a one-dimensional feature.")
        if any(not isinstance(name, str) or not name.strip() for name in self.names):
            raise ValueError("Feature component names must be nonempty strings.")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Feature component names must be unique.")


def feature_mismatch(actual: FeatureSpec, expected: FeatureSpec) -> str | None:
    """Describe the first incompatible schema field, without assigning service error semantics."""
    for field in fields(FeatureSpec):
        actual_value, expected_value = getattr(actual, field.name), getattr(expected, field.name)
        if actual_value != expected_value:
            return (
                f"Feature {expected.name!r} field {field.name!r} differs: "
                f"client={actual_value!r}, server={expected_value!r}"
            )
    return None


class NonFiniteArrayError(ValueError):
    """Distinguish malformed values from incompatible array schemas."""


def validate_array(value: Any, feature: FeatureSpec, *, leading_shape: tuple[int, ...] = ()) -> None:
    """Validate canonical NumPy values with optional batch or action-horizon dimensions."""
    shape = (*leading_shape, *feature.shape)
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.name != feature.dtype:
        raise ValueError(f"Feature {feature.name!r} shape/dtype mismatch: expected {shape}, {feature.dtype}")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise NonFiniteArrayError(f"Non-finite feature values: {feature.name}")
