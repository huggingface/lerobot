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
ReWiND Sampler for temporal sequence loading.
"""

import logging
from typing import Iterator, Optional
import numpy as np
import torch
from torch.utils.data import Sampler
import random


class ReWiNDTemporalSampler(Sampler):
    """
    Sampler for ReWiND that samples random temporal windows from episodes.
    
    Matches original ReWiND sampling:
    - Samples random start and end points within episodes
    - Minimum window size of 3 frames
    - Can sample from beginning, middle, or end of episodes
    
    Args:
        dataset_from_index: Start indices of episodes
        dataset_to_index: End indices of episodes  
        sequence_length: Maximum sequence length (for padding/subsampling)
        stride: Not used (kept for API compatibility)
        shuffle: Whether to shuffle sampling order
        seed: Random seed
    """
    
    def __init__(
        self,
        dataset_from_index: np.ndarray,
        dataset_to_index: np.ndarray,
        sequence_length: int = 32,
        stride: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.dataset_from_index = np.array(dataset_from_index)
        self.dataset_to_index = np.array(dataset_to_index)
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = torch.Generator()
        
        # Compute valid episodes (those with at least 3 frames)
        self._compute_valid_episodes()
        
        # Number of samples per epoch (matching original ReWiND)
        self.samples_per_epoch = 100 * 64  # 100 batches of 64
        
        logging.info(
            f"ReWiNDTemporalSampler: {len(self.valid_episodes)} valid episodes, "
            f"{self.samples_per_epoch} samples per epoch"
        )
    
    def _compute_valid_episodes(self):
        """Compute valid episodes (those with at least 3 frames)."""
        self.valid_episodes = []
        
        for ep_idx in range(len(self.dataset_from_index)):
            ep_start = self.dataset_from_index[ep_idx]
            ep_end = self.dataset_to_index[ep_idx]
            episode_length = ep_end - ep_start
            
            if episode_length >= 3:  # Minimum 3 frames
                self.valid_episodes.append((ep_idx, ep_start, ep_end))
        
        self.valid_episodes = np.array(self.valid_episodes)
    
    def __len__(self) -> int:
        return self.samples_per_epoch
    
    def __iter__(self) -> Iterator[int]:
        """
        Yields ONE index per sample (the end of a random window).
        
        Matches original ReWiND behavior:
        1. Pick random episode
        2. Pick random end frame (at least 3 frames from start)
        3. Yield that end frame index
        4. Dataset/processor loads from episode start to this end frame
        5. Model pads/subsamples to sequence_length (32)
        
        This allows sampling from anywhere in episodes:
        - Early frames → short sequences (mostly padding) → low progress
        - Middle frames → medium sequences (some subsampling) → medium progress  
        - End frames → long sequences (full subsampling) → high progress approaching 1.0
        """
        for _ in range(self.samples_per_epoch):
            # Randomly select an episode
            ep_idx, ep_start, ep_end = self.valid_episodes[
                np.random.randint(0, len(self.valid_episodes))
            ]
            
            episode_length = ep_end - ep_start
            
            # Sample a random end point (must be at least 3 frames from start)
            # This matches original: random.randint(start_idx+3, len(progress_dataset))
            end_offset = np.random.randint(3, episode_length + 1)
            end_idx = ep_start + end_offset
            
            # Yield ONLY the end index
            # The dataset will load all frames from ep_start to end_idx
            yield int(end_idx - 1)  # -1 because end_idx is exclusive
