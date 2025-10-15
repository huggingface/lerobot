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

"""Video sampling utilities for temporal data augmentation and frame selection.

This module provides utilities for sampling and augmenting video sequences, particularly
for reward model training. It includes functions for:
- Padding/sampling videos to fixed lengths
- Video rewind augmentation for learning to decrease rewards
"""

import random
from typing import Tuple

import numpy as np
import torch


def sample_video_feature(
    video_feature: torch.Tensor,
    max_length: int = 32,
    random_sample: bool = True
) -> torch.Tensor:
    """
    Sample or pad video features to a fixed length.
    
    Args:
        video_feature: Video features tensor (num_frames, feature_dim)
        max_length: Target sequence length
        random_sample: If True, randomly sample frames. If False, uniformly sample.
        
    Returns:
        Sampled/padded video features (max_length, feature_dim)
    """
    video_length = len(video_feature)
    
    if video_length < max_length:
        # Pad with last frame
        padding_length = max_length - video_length
        last_frame = video_feature[-1].unsqueeze(0)
        padding_frames = last_frame.repeat(padding_length, 1)
        video_feature = torch.cat([video_feature, padding_frames], dim=0)
        
    elif video_length > max_length:
        if random_sample:
            # Random sampling
            frame_idx = sorted(random.sample(range(video_length), max_length))
        else:
            # Uniform sampling
            frame_idx = np.linspace(0, video_length - 1, max_length, dtype=int)
        video_feature = video_feature[frame_idx]
    
    return video_feature


def sample_reverse_video_feature(
    video_feature: torch.Tensor,
    max_length: int = 32,
    random_sample: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample video with reverse augmentation (video rewind).
    
    This function implements the video rewind augmentation described in the ReWiND paper.
    It splits the video at a random point and reverses k frames from that point, creating
    a trajectory that looks like it's making progress then regressing. This trains the
    reward model to properly decrease rewards when the policy fails.
    
    Args:
        video_feature: Video features tensor (num_frames, feature_dim)
        max_length: Target sequence length
        random_sample: If True, use random sampling for frame selection
        
    Returns:
        Tuple of:
            - Rewound video features (max_length, feature_dim)
            - Progress targets for each frame (max_length,)
    """
    video_length = len(video_feature)
    
    # Sample split point (where to start reversing)
    split_idx = random.randint(1, min(video_length - 1, max_length - 1))
    
    # Sample how many frames to reverse (k in the paper)
    max_reverse = min(split_idx, max_length - split_idx)
    if max_reverse > 0:
        reverse_length = random.randint(1, max_reverse)
    else:
        reverse_length = 0
    
    # Create rewound video
    if reverse_length > 0:
        # Forward part: frames 0 to split_idx
        forward_frames = video_feature[:split_idx]
        
        # Reverse part: frames from split_idx-1 going backwards
        reverse_frames = video_feature[split_idx - reverse_length:split_idx].flip(0)
        
        # Combine forward and reverse parts
        rewound_video = torch.cat([forward_frames, reverse_frames], dim=0)
        
        # Create progress targets
        # Forward part has increasing progress
        forward_progress = torch.linspace(0, split_idx / video_length, split_idx)
        # Reverse part has decreasing progress
        reverse_progress = torch.linspace(
            (split_idx - 1) / video_length,
            (split_idx - reverse_length) / video_length,
            reverse_length
        )
        progress_targets = torch.cat([forward_progress, reverse_progress])
        
    else:
        # No reversal, just use original video
        rewound_video = video_feature[:max_length]
        progress_targets = torch.linspace(0, min(max_length, video_length) / video_length, len(rewound_video))
    
    # Pad or sample to target length
    if len(rewound_video) < max_length:
        # Pad with last frame
        padding_length = max_length - len(rewound_video)
        last_frame = rewound_video[-1].unsqueeze(0)
        padding_frames = last_frame.repeat(padding_length, 1)
        rewound_video = torch.cat([rewound_video, padding_frames], dim=0)
        
        # Extend progress targets (stay at last progress value)
        last_progress = progress_targets[-1]
        padding_progress = torch.full((padding_length,), last_progress)
        progress_targets = torch.cat([progress_targets, padding_progress])
        
    elif len(rewound_video) > max_length:
        # Sample frames
        if random_sample:
            frame_idx = sorted(random.sample(range(len(rewound_video)), max_length))
        else:
            frame_idx = np.linspace(0, len(rewound_video) - 1, max_length, dtype=int)
        rewound_video = rewound_video[frame_idx]
        progress_targets = progress_targets[frame_idx]
    
    return rewound_video, progress_targets

