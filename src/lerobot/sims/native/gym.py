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

"""Environments registered with Gymnasium by their own package (gym_aloha, gym_pusht, ...)."""

import importlib
from collections.abc import Callable
from functools import partial
from typing import Any

import gymnasium as gym


def _make(gym_id: str, gym_kwargs: dict[str, Any]) -> gym.Env:
    if gym_id not in gym.registry:
        # The package named before the slash registers its environments on import.
        importlib.import_module(gym_id.split("/")[0])
    return gym.make(gym_id, disable_env_checker=True, **gym_kwargs)


def create_gym_envs(
    task: str,
    n_envs: int,
    gym_id: str,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable | None = None,
) -> dict[str, dict[int, Any]]:
    """One suite named ``task`` with one task: ``n_envs`` worlds made with ``gym.make(gym_id)``."""
    if env_cls is None:
        raise ValueError("env_cls must be provided")
    make = partial(_make, gym_id, dict(gym_kwargs or {}))
    return {task: {0: env_cls([make] * n_envs)}}
