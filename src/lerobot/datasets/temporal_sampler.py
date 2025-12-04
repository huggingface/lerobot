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

Uniform target sampling: samples any frame from episodes with proper padding
for out-of-bounds access (first frame + progress 0 at start, last frame + progress 1 at end).
"""

import logging
from typing import Iterator, Optional
import numpy as np
import torch
from torch.utils.data import Sampler
import random


class SARMTemporalSampler(Sampler):
    """
    Uniform temporal sampler for SARM reward model training.
    
    SARM uses 9 frames per sample:
    - Frame 0: Initial frame of the episode (always frame 0)
    - Frames 1-8: 8 consecutive frames with frame_gap spacing ending at current frame
    
    This sampler uses UNIFORM target sampling - any frame in an episode can be
    sampled as the target. Out-of-bounds frame indices are handled with padding:
    - Before episode start: pad with first frame and progress 0
    - After episode end: pad with last frame and progress 1
    
    Args:
        dataset_from_index: Start indices of episodes (global dataset indices)
        dataset_to_index: End indices of episodes (global dataset indices)
        frame_gap: Gap between consecutive frames (default: 30 = 1 second at 30fps)
        shuffle: Whether to shuffle sampling order
        seed: Random seed for reproducibility
        samples_per_epoch: Number of samples per epoch (default: 6400)
    """
    
    def __init__(
        self,
        dataset_from_index: np.ndarray,
        dataset_to_index: np.ndarray,
        frame_gap: int = 30,
        shuffle: bool = True,
        seed: Optional[int] = None,
        samples_per_epoch: int = 6400,
    ):
        self.dataset_from_index = np.array(dataset_from_index)
        self.dataset_to_index = np.array(dataset_to_index)
        self.frame_gap = frame_gap
        self.shuffle = shuffle
        self.samples_per_epoch = samples_per_epoch
        
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = torch.Generator()
        
        # Compute all valid sampling positions (uniform sampling from all frames)
        self._compute_valid_positions()
        
        logging.info(
            f"SARMTemporalSampler (uniform): {len(self.valid_episodes)} episodes, "
            f"{len(self.all_valid_positions)} positions (all frames), "
            f"{self.samples_per_epoch} samples per epoch, "
            f"frame_gap={frame_gap}"
        )
    
    def _compute_valid_positions(self):
        """Compute valid episodes and all valid sampling positions (uniform: all frames)."""
        self.valid_episodes = []
        self.all_valid_positions = []
        
        for ep_idx in range(len(self.dataset_from_index)):
            ep_start = self.dataset_from_index[ep_idx]
            ep_end = self.dataset_to_index[ep_idx]
            episode_length = ep_end - ep_start
            
            # Include all episodes with at least 1 frame
            if episode_length >= 1:
                self.valid_episodes.append((ep_idx, ep_start, ep_end))
                
                # Uniform sampling: ALL frames in the episode are valid positions
                for pos in range(ep_start, ep_end):
                    self.all_valid_positions.append(pos)
        
        self.valid_episodes = np.array(self.valid_episodes)
        self.all_valid_positions = np.array(self.all_valid_positions)
        
        if len(self.all_valid_positions) == 0:
            raise ValueError("No valid sampling positions found! Dataset appears to be empty.")
    
    def __len__(self) -> int:
        return self.samples_per_epoch
    
    def __iter__(self) -> Iterator[int]:
        """
        Yields global dataset indices for sampling.
        
        Each yielded index represents the "target frame" position (uniformly sampled).
        Out-of-bounds indices during frame sequence construction are handled with padding:
        - Before episode start: clamp to first frame, progress = 0
        - After episode end: clamp to last frame, progress = 1
        """
        if self.shuffle:
            # Randomly sample from all positions (uniform)
            for _ in range(self.samples_per_epoch):
                idx = np.random.randint(0, len(self.all_valid_positions))
                yield int(self.all_valid_positions[idx])
        else:
            # Sequential sampling with wrap-around
            for i in range(self.samples_per_epoch):
                idx = i % len(self.all_valid_positions)
                yield int(self.all_valid_positions[idx])
