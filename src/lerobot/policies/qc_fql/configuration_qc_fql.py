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
Configuration for QC-FQL (Q-Chunking with Fitted Q-Learning).

QC-FQL is an offline-to-online RL algorithm that combines action chunking with
FQL (Flow Q-Learning) to improve sample efficiency on long-horizon tasks.

Reference: "Reinforcement Learning with Action Chunking" (Li et al., 2025)
https://arxiv.org/abs/2507.07969

Key features:
- Action chunking: Predict sequences of actions instead of single actions
- Flow-matching behavior policy for capturing complex behavior distributions
- Noise-conditioned action prediction for policy optimization
- Critic on action chunks for unbiased n-step backups
- Distillation loss as Wasserstein distance upper bound
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("qc_fql")
@dataclass
class QCFQLConfig(PreTrainedConfig):
    """
    Configuration for QC-FQL policy.
    
    QC-FQL combines Q-chunking (action chunking for RL) with FQL's flow-matching
    and optimal transport framework for effective offline-to-online RL.
    
    Args:
        # Action Chunking Parameters
        action_chunk_size: Number of actions in a chunk (k in paper)
        
        # Critic Parameters
        num_critics: Number of critics in ensemble
        critic_hidden_dims: Hidden layer dimensions for critic networks
        critic_learning_rate: Learning rate for critic
        critic_target_update_tau: Exponential moving average weight for target networks
        
        # Actor/Policy Parameters  
        actor_hidden_dims: Hidden layer dimensions for actor networks
        actor_learning_rate: Learning rate for actor
        
        # Flow Matching Parameters
        flow_matching_sigma: Noise scale for flow matching
        flow_matching_num_inference_steps: Number of steps for flow matching inference
        
        # Distillation Loss (Wasserstein constraint)
        distillation_weight: Weight for distillation loss (lambda in paper)
        
        # TD Learning Parameters
        discount: Discount factor gamma
        expectile: Expectile for value estimation (tau in paper)
        
        # Training Parameters
        batch_size: Training batch size
        num_pretrain_steps: Number of offline pretraining steps
        num_online_steps: Number of online training steps
        
        # Observation/Action Parameters
        state_dim: Dimension of state observations
        action_dim: Dimension of actions
        
        # Optional image encoder
        use_image_encoder: Whether to use image observations
        image_encoder_architecture: Architecture for image encoder ('cnn' or 'resnet')
    """
    
    # Action Chunking
    action_chunk_size: int = 4  # k=4 in paper experiments
    
    # Critic Ensemble
    num_critics: int = 10  # Number of Q-networks in ensemble
    num_subsample_critics: int | None = None  # For computational efficiency
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    critic_learning_rate: float = 3e-4
    critic_target_update_tau: float = 0.005
    
    # Actor (Noise-conditioned action predictor)
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    actor_learning_rate: float = 3e-4
    
    # Flow Matching Behavior Policy
    flow_matching_sigma: float = 0.001  # Noise scale for flow matching
    flow_matching_num_inference_steps: int = 10
    flow_matching_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    behavior_policy_learning_rate: float = 3e-4
    
    # Distillation Loss (Wasserstein distance upper bound)
    distillation_weight: float = 1.0  # lambda for distillation loss
    
    # TD Learning
    discount: float = 0.99  # gamma
    expectile: float = 0.7  # tau for expectile regression
    
    # Training
    batch_size: int = 256
    num_pretrain_steps: int = 100000
    num_online_steps: int = 100000
    update_to_data_ratio: int = 1  # UTD ratio
    
    # Dimensions (will be inferred from dataset if not specified)
    state_dim: int | None = None
    action_dim: int | None = None
    
    # Image encoder (optional)
    use_image_encoder: bool = False
    image_encoder_architecture: str = "cnn"  # or 'resnet'
    image_encoder_hidden_dim: int = 128
    
    # Regularization
    dropout_rate: float = 0.0
    layer_norm: bool = True
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.action_chunk_size >= 1, "action_chunk_size must be >= 1"
        assert self.num_critics >= 2, "num_critics must be >= 2 for ensemble"
        assert 0 < self.discount <= 1, "discount must be in (0, 1]"
        assert 0 < self.expectile < 1, "expectile must be in (0, 1)"
        assert self.distillation_weight >= 0, "distillation_weight must be >= 0"
