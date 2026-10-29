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

"""Gymnasium envs that their own package registers on import (gym_aloha, gym_pusht, ...)."""

import importlib
from collections.abc import Callable
from functools import partial
from typing import Any

import gymnasium as gym

from .utils import canonical_observation

__all__ = ["canonical_observation", "env_fns"]


def _make(gym_id: str, gym_kwargs: dict[str, Any], disable_env_checker: bool) -> gym.Env:
    if gym_id not in gym.registry:
        # Importing the package named before the slash registers its envs.
        importlib.import_module(gym_id.split("/")[0])
    return gym.make(gym_id, disable_env_checker=disable_env_checker, **gym_kwargs)


def env_fns(
    task: str,
    n_envs: int,
    gym_id: str,
    gym_kwargs: dict[str, Any] | None = None,
    disable_env_checker: bool = True,
) -> dict[str, dict[int, list[Callable[[], gym.Env]]]]:
    """Return ``n_envs`` factories that call ``gym.make(gym_id)``, as ``{task: {0: [factory, ...]}}``."""
    make = partial(_make, gym_id, dict(gym_kwargs or {}), disable_env_checker)
    return {task: {0: [make] * n_envs}}
