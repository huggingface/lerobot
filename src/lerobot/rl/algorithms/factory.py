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


def make_algorithm(cfg: RLAlgorithmConfig, policy: torch.nn.Module) -> RLAlgorithm:
    return cfg.build_algorithm(policy)
