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

from __future__ import annotations

import torch

from .base import RLAlgorithm
from .configs import RLAlgorithmConfig


def make_algorithm_config(algorithm_type: str, **kwargs) -> RLAlgorithmConfig:
    """Instantiate an `RLAlgorithmConfig` from its registered type name.

    Args:
        algorithm_type: Registry key of the algorithm (e.g. ``"sac"``).
        **kwargs: Keyword arguments forwarded to the config class constructor.

    Returns:
        An instance of the matching ``RLAlgorithmConfig`` subclass.

    Raises:
        ValueError: If ``algorithm_type`` is not registered.
    """
    try:
        cls = RLAlgorithmConfig.get_choice_class(algorithm_type)
    except KeyError as err:
        raise ValueError(
            f"Algorithm type '{algorithm_type}' is not registered. "
            f"Available: {list(RLAlgorithmConfig.get_known_choices().keys())}"
        ) from err
    return cls(**kwargs)


def get_algorithm_class(name: str) -> type[RLAlgorithm]:
    """
    Retrieves an RL algorithm class by its registered name.

    This function uses dynamic imports to avoid loading all algorithm classes into
    memory at once, improving startup time and reducing dependencies.

    Args:
        name: The name of the algorithm. Supported names are "sac".

    Returns:
        The algorithm class corresponding to the given name.

    Raises:
        ValueError: If the algorithm name is not recognized.
    """
    if name == "sac":
        from .sac.sac_algorithm import SACAlgorithm

        return SACAlgorithm
    raise ValueError(
        f"Algorithm type '{name}' is not available. "
        f"Known: {list(RLAlgorithmConfig.get_known_choices().keys())}"
    )


def make_algorithm(cfg: RLAlgorithmConfig, policy: torch.nn.Module) -> RLAlgorithm:
    """
    Instantiate an RL algorithm.

    This factory function looks up the :class:`RLAlgorithm` subclass that matches
    ``cfg.type`` and instantiates it with the provided policy. It also enforces
    that ``cfg.policy_config`` has been populated before construction (this is
    normally handled by :meth:`TrainRLServerPipelineConfig.validate`).

    Args:
        cfg: The algorithm configuration. Must have ``policy_config`` set.
        policy: The policy module the algorithm will train.

    Returns:
        An instantiated :class:`RLAlgorithm`.

    Raises:
        ValueError: If ``cfg.policy_config`` is ``None`` or ``cfg.type`` is not
            registered.
    """
    if getattr(cfg, "policy_config", None) is None:
        raise ValueError(
            f"{type(cfg).__name__}.policy_config is None. "
            "It must be populated (typically by TrainRLServerPipelineConfig.validate) "
            "before calling make_algorithm()."
        )
    cls = get_algorithm_class(cfg.type)
    return cls(policy=policy, config=cfg)
