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
SARM Temporal Sampler for reward model training.

Samples frames uniformly from episodes for SARM's 9-frame symmetric pattern:
- 1 initial frame + 4 frames before + current + 3 frames after

Boundary handling: clamp to first/last frame when indices go out of bounds.
This enables truly uniform sampling across entire episodes.
"""

import logging
from typing import Iterator, Optional
import numpy as np
import torch
from torch.utils.data import Sampler
import random


class SARMTemporalSampler(Sampler):
    """
    Temporal sampler for SARM reward model training with symmetric/bidirectional sampling.
    
    SARM uses 9 frames per sample:
    - Frame 0: Initial frame of the episode (always frame 0)
    - Frames 1-8: Symmetric context around current frame
      Pattern: [t-4*gap, t-3*gap, t-2*gap, t-gap, t, t+gap, t+2*gap, t+3*gap]
    
    Boundary handling:
    - Early frames: backward indices clamp to 0 (e.g., [0,0,0,5,35,65,95,125])
    - Late frames: forward indices clamp to last frame (e.g., [850,880,910,940,970,1000,1000,1000])
    
    This enables truly uniform sampling across entire episodes.
    
    Args:
        dataset_from_index: Start indices of episodes (global dataset indices)
        dataset_to_index: End indices of episodes (global dataset indices)
        frame_gap: Gap between consecutive frames (default: 30 = 1 second at 30fps)
        shuffle: Whether to shuffle sampling order
        seed: Random seed for reproducibility
        samples_per_epoch: Number of samples per epoch (default: 6400)
        min_episode_length: Minimum episode length to include (default: 1)
    """
    
    def __init__(
        self,
        dataset_from_index: np.ndarray,
        dataset_to_index: np.ndarray,
        frame_gap: int = 30,
        shuffle: bool = True,
        seed: Optional[int] = None,
        samples_per_epoch: int = 6400,
        min_episode_length: int = 1,
    ):
        self.dataset_from_index = np.array(dataset_from_index)
        self.dataset_to_index = np.array(dataset_to_index)
        self.frame_gap = frame_gap
        self.shuffle = shuffle
        self.samples_per_epoch = samples_per_epoch
        self.min_episode_length = min_episode_length
        
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = torch.Generator()
        
        # Compute valid episodes and sampling positions (ALL frames for uniform sampling)
        self._compute_valid_positions()
        
        logging.info(
            f"SARMTemporalSampler: {len(self.valid_episodes)} valid episodes, "
            f"{len(self.all_valid_positions)} positions (uniform sampling), "
            f"{self.samples_per_epoch} samples per epoch, "
            f"frame_gap={frame_gap}, symmetric bidirectional pattern"
        )
    
    def _compute_valid_positions(self):
        """Compute valid episodes and ALL sampling positions for uniform sampling.
        
        With symmetric bidirectional sampling, we can sample from ANY frame:
        - Early frames: backward indices clamp to first frame
        - Late frames: forward indices clamp to last frame
        """
        self.valid_episodes = []
        self.all_valid_positions = []
        
        for ep_idx in range(len(self.dataset_from_index)):
            ep_start = self.dataset_from_index[ep_idx]
            ep_end = self.dataset_to_index[ep_idx]
            episode_length = ep_end - ep_start
            
            # Include all episodes with at least min_episode_length frames
            if episode_length >= self.min_episode_length:
                self.valid_episodes.append((ep_idx, ep_start, ep_end))
                
                # Include ALL positions in the episode (truly uniform sampling)
                for pos in range(ep_start, ep_end):
                    self.all_valid_positions.append(pos)
        
        self.valid_episodes = np.array(self.valid_episodes)
        self.all_valid_positions = np.array(self.all_valid_positions)
        
        if len(self.all_valid_positions) == 0:
            raise ValueError(
                f"No valid sampling positions found! "
                f"Check that episodes have at least {self.min_episode_length} frames."
            )
    
    def __len__(self) -> int:
        return self.samples_per_epoch
    
    def __iter__(self) -> Iterator[int]:
        """
        Yields global dataset indices for uniform sampling across episodes.
        
        Each yielded index represents the "current frame" position.
        The dataset's observation_delta_indices then handles loading:
        - Frame 0: Episode initial frame (via large negative delta clamping)
        - Frames 1-8: Symmetric context around current frame (with boundary clamping)
        
        For early frames: backward indices clamp to first frame (progress ~0%)
        For late frames: forward indices clamp to last frame (progress ~100%)
        """
        if self.shuffle:
            # Randomly sample from all valid positions
            for _ in range(self.samples_per_epoch):
                idx = np.random.randint(0, len(self.all_valid_positions))
                yield int(self.all_valid_positions[idx])
        else:
            # Sequential sampling with wrap-around
            for i in range(self.samples_per_epoch):
                idx = i % len(self.all_valid_positions)
                yield int(self.all_valid_positions[idx])
