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

from lerobot.rl.algorithms.base import (
    RLAlgorithm,
    RLAlgorithmConfig,
    TrainingStats,
)
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig


def make_algorithm(
    policy: torch.nn.Module,
    policy_cfg,
) -> RLAlgorithm:
    """Construct an :class:`RLAlgorithm` from a policy and its config.

    Uses ``policy_cfg.type`` (the ``PreTrainedConfig`` registry name, e.g.
    ``"sac"``) to look up the matching algorithm in the
    :class:`RLAlgorithmConfig` registry and build it via the config's
    :meth:`~RLAlgorithmConfig.build_algorithm` method.

    This is fully registry-driven — adding a new algorithm only requires
    registering an ``RLAlgorithmConfig`` subclass; no changes here.

    The returned algorithm has **no optimizers** yet.  On the learner side,
    call ``algorithm.make_optimizers()`` afterwards to create them.  On the
    actor side (inference-only), leave them empty.

    Args:
        policy: Instantiated policy (e.g. ``SACPolicy``).
        policy_cfg: The policy's ``PreTrainedConfig`` (must expose a ``.type``
            property and the hyper-parameters expected by the algorithm config's
            ``from_policy_config`` class-method).
    """
    algo_name = getattr(policy_cfg, "type", "sac")

    known = RLAlgorithmConfig.get_known_choices()
    if algo_name not in known:
        raise ValueError(f"No RLAlgorithmConfig registered for '{algo_name}'. Known: {list(known)}")

    config_cls = RLAlgorithmConfig.get_choice_class(algo_name)
    algo_config = config_cls.from_policy_config(policy_cfg)
    return algo_config.build_algorithm(policy)


__all__ = [
    "RLAlgorithm",
    "RLAlgorithmConfig",
    "TrainingStats",
    "SACAlgorithm",
    "SACAlgorithmConfig",
    "make_algorithm",
]
