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

"""Simulator integrations with one environment factory and observation adapter contract."""

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NativeIntegration:
    """Functions a simulator integration supplies to the shared environment runtime."""

    create_envs: Callable[..., dict[str, dict[int, Any]]]
    canonical_observation: Callable[[Mapping], dict[str, np.ndarray]]


def load_native_integration(name: str) -> NativeIntegration:
    """Load a simulator integration without encoding simulator names in the shared runtime."""
    module = importlib.import_module(f"{__name__}.{name}")
    try:
        create_envs = module.create_envs
        canonical_observation = module.canonical_observation
    except AttributeError as exc:
        raise TypeError(
            f"Simulator integration {name!r} must define create_envs() and canonical_observation()"
        ) from exc
    return NativeIntegration(create_envs, canonical_observation)
