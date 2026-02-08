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
Data Processor for QC-FQL.

Handles preprocessing of observations and actions for QC-FQL training,
including action chunking and n-step reward computation.
"""

import numpy as np
import torch
from torch import Tensor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.processors.base import Processor


class QCFQLProcessor(Processor):
    """
    Processor for QC-FQL data.
    
    Key functions:
    1. Action chunking: Group consecutive actions into chunks
    2. N-step reward computation: Compute discounted sum of rewards over chunk
    3. State normalization: Optional normalization of observations
    """
    
    name = "qc_fql"
    
    def __init__(
        self,
        action_chunk_size: int = 4,
        discount: float = 0.99,
        normalize_states: bool = True,
        normalize_actions: bool = True,
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.discount = discount
        self.normalize_states = normalize_states
        self.normalize_actions = normalize_actions
        
        # Statistics for normalization
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
    
    def fit_normalization(self, dataset: LeRobotDataset, num_samples: int = 10000):
        """
        Compute normalization statistics from dataset.
        
        Args:
            dataset: LeRobotDataset to compute statistics from
            num_samples: Number of samples to use
        """
        if not self.normalize_states and not self.normalize_actions:
            return
        
        states = []
        actions = []
        
        num_samples = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for idx in indices:
            data = dataset[idx]
            if self.normalize_states and "observation.state" in data:
                states.append(data["observation.state"].numpy())
            if self.normalize_actions and "action" in data:
                actions.append(data["action"].numpy())
        
        if states:
            states = np.stack(states)
            self.state_mean = torch.tensor(states.mean(axis=0), dtype=torch.float32)
            self.state_std = torch.tensor(states.std(axis=0) + 1e-6, dtype=torch.float32)
        
        if actions:
            actions = np.stack(actions)
            self.action_mean = torch.tensor(actions.mean(axis=0), dtype=torch.float32)
            self.action_std = torch.tensor(actions.std(axis=0) + 1e-6, dtype=torch.float32)
    
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Process a batch of data.
        
        Args:
            batch: Dictionary with keys:
                - observation.state: (batch, state_dim)
                - action: (batch, action_dim)
                - reward: (batch,)
                - next_observation.state: (batch, state_dim)
                - done: (batch,)
                
        Returns:
            Processed batch with action chunks and n-step rewards
        """
        processed = {}
        
        # Normalize states
        if self.normalize_states and self.state_mean is not None:
            processed["observation.state"] = (batch["observation.state"] - self.state_mean) / self.state_std
            processed["next_observation.state"] = (batch["next_observation.state"] - self.state_mean) / self.state_std
        else:
            processed["observation.state"] = batch["observation.state"]
            processed["next_observation.state"] = batch["next_observation.state"]
        
        # Create action chunks
        actions = batch["action"]  # (batch, action_dim)
        batch_size = actions.shape[0]
        action_dim = actions.shape[1]
        
        # Stack consecutive actions to form chunks
        # For simplicity, assume batch is sequential
        action_chunks = []
        for i in range(batch_size):
            chunk = []
            for j in range(self.action_chunk_size):
                idx = min(i + j, batch_size - 1)
                chunk.append(actions[idx])
            action_chunks.append(torch.stack(chunk, dim=0))
        action_chunks = torch.stack(action_chunks, dim=0)  # (batch, chunk_size, action_dim)
        
        # Normalize actions
        if self.normalize_actions and self.action_mean is not None:
            action_mean_expanded = self.action_mean.unsqueeze(0).unsqueeze(0)
            action_std_expanded = self.action_std.unsqueeze(0).unsqueeze(0)
            action_chunks = (action_chunks - action_mean_expanded) / action_std_expanded
        
        processed["action"] = action_chunks
        
        # Compute n-step rewards
        rewards = batch["reward"]  # (batch,)
        n_step_rewards = []
        for i in range(batch_size):
            discounted_sum = 0.0
            for j in range(self.action_chunk_size):
                idx = min(i + j, batch_size - 1)
                discounted_sum += (self.discount ** j) * rewards[idx]
            n_step_rewards.append(discounted_sum)
        processed["reward"] = torch.tensor(n_step_rewards, dtype=torch.float32, device=rewards.device)
        
        # Done flags
        processed["done"] = batch["done"]
        
        return processed


def create_action_chunks(
    actions: np.ndarray,
    chunk_size: int,
    pad_last: bool = True,
) -> np.ndarray:
    """
    Create action chunks from sequence of actions.
    
    Args:
        actions: (seq_len, action_dim) array of actions
        chunk_size: Size of each chunk
        pad_last: Whether to pad the last chunk
        
    Returns:
        action_chunks: (num_chunks, chunk_size, action_dim)
    """
    seq_len, action_dim = actions.shape
    num_chunks = seq_len
    
    chunks = []
    for i in range(num_chunks):
        chunk = []
        for j in range(chunk_size):
            idx = min(i + j, seq_len - 1)
            chunk.append(actions[idx])
        chunks.append(np.stack(chunk, axis=0))
    
    return np.stack(chunks, axis=0)


def compute_n_step_rewards(
    rewards: np.ndarray,
    chunk_size: int,
    discount: float = 0.99,
) -> np.ndarray:
    """
    Compute n-step discounted rewards.
    
    Args:
        rewards: (seq_len,) array of rewards
        chunk_size: Number of steps
        discount: Discount factor
        
    Returns:
        n_step_rewards: (seq_len,) array of n-step returns
    """
    seq_len = len(rewards)
    n_step_rewards = np.zeros(seq_len)
    
    for i in range(seq_len):
        discounted_sum = 0.0
        for j in range(chunk_size):
            idx = min(i + j, seq_len - 1)
            discounted_sum += (discount ** j) * rewards[idx]
        n_step_rewards[i] = discounted_sum
    
    return n_step_rewards
