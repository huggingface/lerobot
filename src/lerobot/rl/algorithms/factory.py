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

from lerobot.rl.algorithms.base import RLAlgorithm
from lerobot.rl.algorithms.configs import RLAlgorithmConfig


def make_algorithm(
    policy: torch.nn.Module,
    policy_cfg,
    *,
    algorithm_name: str,
) -> RLAlgorithm:
    known = RLAlgorithmConfig.get_known_choices()
    if algorithm_name not in known:
        raise ValueError(f"No RLAlgorithmConfig registered for '{algorithm_name}'. Known: {list(known)}")

    config_cls = RLAlgorithmConfig.get_choice_class(algorithm_name)
    algo_config = config_cls.from_policy_config(policy_cfg)
    return algo_config.build_algorithm(policy)
