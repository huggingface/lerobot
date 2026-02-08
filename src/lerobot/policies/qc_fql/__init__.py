#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
QC-FQL: Q-Chunking with Fitted Q-Learning

Implementation of "Reinforcement Learning with Action Chunking" (Li et al., 2025).

QC-FQL is an offline-to-online RL algorithm that combines action chunking with
flow-matching policies and optimal transport for improved sample efficiency on
long-horizon tasks.

Example usage:
    ```python
    from lerobot.policies.qc_fql import QCFQLPolicy, QCFQLConfig
    
    config = QCFQLConfig(
        action_chunk_size=4,
        num_critics=10,
        distillation_weight=1.0,
        state_dim=128,
        action_dim=16,
    )
    
    policy = QCFQLPolicy(config)
    
    # Training loop
    for batch in dataloader:
        losses = policy.forward(batch)
        
        # Optimize behavior policy
        loss_behavior = losses["loss_behavior"]
        loss_behavior.backward()
        optimizer_behavior.step()
        
        # Optimize critic
        loss_critic = losses["loss_critic"]
        loss_critic.backward()
        optimizer_critic.step()
        policy.update_target_networks()
        
        # Optimize policy
        loss_policy = losses["loss_policy"]
        loss_policy.backward()
        optimizer_policy.step()
    
    # Inference
    action = policy.select_action({"observation.state": obs})
    ```
"""

from .configuration_qc_fql import QCFQLConfig
from .modeling_qc_fql import (
    QCFQLPolicy,
    FlowMatchingBehaviorPolicy,
    ChunkedCritic,
    CriticEnsemble,
    NoiseConditionedPolicy,
    MLP,
)
from .processor_qc_fql import QCFQLProcessor, create_action_chunks, compute_n_step_rewards

__all__ = [
    # Main policy
    "QCFQLPolicy",
    "QCFQLConfig",
    "QCFQLProcessor",
    
    # Components
    "FlowMatchingBehaviorPolicy",
    "ChunkedCritic",
    "CriticEnsemble",
    "NoiseConditionedPolicy",
    "MLP",
    
    # Utilities
    "create_action_chunks",
    "compute_n_step_rewards",
]
