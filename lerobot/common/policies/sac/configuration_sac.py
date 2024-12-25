#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. 
# All rights reserved.
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

from dataclasses import dataclass


@dataclass
class SACConfig:
    discount = 0.99
    temperature_init = 1.0
    num_critics = 2
    num_subsample_critics = None
    critic_lr = 3e-4
    actor_lr = 3e-4
    temperature_lr = 3e-4
    critic_target_update_weight = 0.005
    utd_ratio = 2
    critic_network_kwargs = {
            "hidden_dims": [256, 256],
            "activate_final": True,
        }
    actor_network_kwargs = {
            "hidden_dims": [256, 256],
            "activate_final": True,
        }
    policy_kwargs = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        }