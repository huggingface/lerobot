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
Temporal Sequence Sampler for reward models and temporal policies.

Supports multiple sampling modes:
- "rewind": ReWiND-style sampling (random windows from episode start)
- "sarm": SARM-style sampling (9-frame sequences with specific pattern)
- "custom": Custom temporal sampling
"""

import logging
from typing import Iterator, Optional
import numpy as np
import torch
from torch.utils.data import Sampler
import random


class TemporalSequenceSampler(Sampler):
    """
    Generalized temporal sampler for reward models.
    
    Supports multiple sampling modes:
    - "rewind": Consecutive frames from episode start to random end point (ReWiND: 32 consecutive frames)
    - "sarm": 9-frame sequences with 1 initial + 8 consecutive (SARM)
    - "custom": Custom temporal sampling
    
    Args:
        dataset_from_index: Start indices of episodes
        dataset_to_index: End indices of episodes  
        sequence_length: Maximum sequence length (for padding/subsampling)
        stride: Frame stride for consecutive sampling (SARM mode)
        shuffle: Whether to shuffle sampling order
        seed: Random seed
        sampling_mode: Sampling mode ("rewind", "sarm", or "custom")
        min_frames: Minimum frames per episode (default: 3)
        samples_per_epoch: Number of samples per epoch (default: 6400)
    """
    
    def __init__(
        self,
        dataset_from_index: np.ndarray,
        dataset_to_index: np.ndarray,
        sequence_length: int = 32,
        stride: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
        sampling_mode: str = "rewind",
        min_frames: int = 3,
        samples_per_epoch: int = 6400,
    ):
        self.dataset_from_index = np.array(dataset_from_index)
        self.dataset_to_index = np.array(dataset_to_index)
        self.sequence_length = sequence_length
        self.stride = stride
        self.shuffle = shuffle
        self.sampling_mode = sampling_mode
        self.min_frames = min_frames
        self.samples_per_epoch = samples_per_epoch
        
        if sampling_mode not in ["rewind", "sarm", "custom"]:
            raise ValueError(f"sampling_mode must be 'rewind', 'sarm', or 'custom', got {sampling_mode}")
        
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = torch.Generator()
        
        # Compute valid episodes
        self._compute_valid_episodes()
        
        logging.info(
            f"TemporalSequenceSampler ({sampling_mode} mode): "
            f"{len(self.valid_episodes)} valid episodes, "
            f"{self.samples_per_epoch} samples per epoch"
        )
    
    def _compute_valid_episodes(self):
        """Compute valid episodes based on minimum frame requirement."""
        self.valid_episodes = []
        
        for ep_idx in range(len(self.dataset_from_index)):
            ep_start = self.dataset_from_index[ep_idx]
            ep_end = self.dataset_to_index[ep_idx]
            episode_length = ep_end - ep_start
            
            # For SARM mode, need enough frames for the sequence pattern
            if self.sampling_mode == "sarm":
                # Need at least sequence_length * stride frames
                min_required = self.sequence_length * self.stride
                if episode_length >= min_required:
                    self.valid_episodes.append((ep_idx, ep_start, ep_end))
            else:
                # For rewind mode, use min_frames
                if episode_length >= self.min_frames:
                    self.valid_episodes.append((ep_idx, ep_start, ep_end))
        
        self.valid_episodes = np.array(self.valid_episodes)
    
    def __len__(self) -> int:
        return self.samples_per_epoch
    
    def __iter__(self) -> Iterator[int]:
        """
        Yields ONE index per sample.
        
        Sampling behavior depends on mode:
        
        ReWiND mode:
        1. Pick random episode
        2. Pick random end frame (at least min_frames from start)
        3. Yield that end frame index
        4. Dataset loads from episode start to this end frame
        
        SARM mode:
        1. Pick random episode
        2. Pick random end frame (must allow sequence_length frames with stride)
        3. Yield that end frame index
        4. Dataset loads sequence_length frames with stride spacing ending at this frame
        """
        for _ in range(self.samples_per_epoch):
            # Randomly select an episode
            ep_idx, ep_start, ep_end = self.valid_episodes[
                np.random.randint(0, len(self.valid_episodes))
            ]
            
            episode_length = ep_end - ep_start
            
            if self.sampling_mode == "rewind":
                # ReWiND: Sample random end point (at least min_frames from start)
                end_offset = np.random.randint(self.min_frames, episode_length + 1)
                end_idx = ep_start + end_offset
                
                # Yield the end index (dataset will load from start to this point)
                yield int(end_idx - 1)  # -1 because end_idx is exclusive
            
            elif self.sampling_mode == "sarm":
                # SARM: Sample end point that allows full sequence
                # We need sequence_length frames with stride spacing
                min_end_offset = self.sequence_length * self.stride
                
                if episode_length >= min_end_offset:
                    # Can sample anywhere from min_end_offset to episode_length
                    end_offset = np.random.randint(min_end_offset, episode_length + 1)
                else:
                    # Episode is exactly the minimum length
                    end_offset = episode_length
                
                end_idx = ep_start + end_offset
                
                # Yield the end index (dataset will load sequence with stride)
                yield int(end_idx - 1)  # -1 because end_idx is exclusive
            
            else:  # custom mode
                # Default to rewind-style sampling
                end_offset = np.random.randint(self.min_frames, episode_length + 1)
                end_idx = ep_start + end_offset
                yield int(end_idx - 1)


# Backwards compatibility alias
ReWiNDTemporalSampler = TemporalSequenceSampler

