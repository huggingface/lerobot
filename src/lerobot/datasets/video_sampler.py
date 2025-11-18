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
    random_sample: bool = True,
    remaining_length: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample or pad video features to a fixed length with progress targets.
    
    Progress normalization matches original ReWiND implementation:
    - Progress = (position_in_sequence + 1) / remaining_trajectory_length
    - remaining_trajectory_length = frames from first sampled frame to episode end
    
    Original ReWiND logic (dataset.py lines 12493-12499):
        video_frames = frames[start_idx:end_idx]
        full_frames = frames[start_idx:]  # All frames from start to episode end
        progress = [1, 2, ..., len(video_frames)] / len(full_frames)
    
    This ensures all sequences show increasing progress from near-zero, regardless
    of where they're sampled from in the episode.
    
    Uses original ReWiND sampling: random start/end points with minimum 3 frames.
    
    Args:
        video_feature: Video features tensor (num_frames, feature_dim)
        max_length: Target sequence length
        random_sample: If True, randomly sample frames. If False, uniformly sample consecutive frames.
        remaining_length: Remaining trajectory length from first frame to episode end
        
    Returns:
        Tuple of:
            - Sampled/padded video features (max_length, feature_dim)
            - Progress targets for each frame (max_length,)
    """
    video_length = len(video_feature)
    
    # Original ReWiND sampling: random start/end with minimum 3 frames
    if video_length > 3:
        # Sample random start index (ensuring we can get at least 3 frames)
        start_idx = random.randint(0, max(0, video_length - 3))
        # Sample random end index (at least 3 frames after start, up to video_length)
        end_idx = random.randint(min(start_idx + 3, video_length), video_length)
        
        # Extract the sampled segment
        video_feature = video_feature[start_idx:end_idx]
        
        # Update video_length for the sampled segment
        video_length = len(video_feature)
        
        # Adjust remaining_length to be from start_idx to episode end
        if remaining_length is not None:
            # The remaining length should be from start_idx to episode end
            # If we started at start_idx, we've already consumed start_idx frames
            remaining_length = remaining_length - start_idx if remaining_length > start_idx else video_length
    
    # Generate progress targets using ORIGINAL ReWiND formula
    # Progress = (position_in_sequence + 1) / remaining_trajectory_length
    progress_indices = torch.arange(1, video_length + 1, dtype=torch.float32)
    progress_targets = progress_indices / remaining_length
    
    if video_length < max_length:
        # Pad with last frame
        padding_length = max_length - video_length
        last_frame = video_feature[-1].unsqueeze(0)
        padding_frames = last_frame.repeat(padding_length, 1)
        video_feature = torch.cat([video_feature, padding_frames], dim=0)
        
        # Pad progress with last progress value
        last_progress = progress_targets[-1]
        padding_progress = torch.full((padding_length,), last_progress)
        progress_targets = torch.cat([progress_targets, padding_progress])
        
    elif video_length > max_length:
        if random_sample:
            # Random sampling (maintains temporal order via sorted indices)
            frame_idx = sorted(random.sample(range(video_length), max_length))
        else:
            # Uniform sampling (consecutive frames with even spacing)
            frame_idx = np.linspace(0, video_length - 1, max_length, dtype=int)
        video_feature = video_feature[frame_idx]
        progress_targets = progress_targets[frame_idx]
    
    return video_feature, progress_targets


def sample_reverse_video_feature(
    video_feature: torch.Tensor,
    max_length: int = 32,
    random_sample: bool = True,
    remaining_length: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample video with reverse augmentation (video rewind) - ORIGINAL REWIND LOGIC.
    
    This implements the EXACT video rewind augmentation from the original ReWiND paper:
    1. Take forward sequence (sampled with random start/end, min 3 frames)
    2. Append reversed frames from the END backwards
    3. Progress increases then decreases (simulating task completion then failure)
    
    Progress normalization matches original ReWiND (same as sample_video_feature).
    Original ReWiND logic (dataset.py lines 12526-12541):
        progress = [1, 2, ..., len(video_frames)] / len(full_frames)
        reverse_progress = progress[::-1][1:selected_end_point]
    
    Args:
        video_feature: Video features tensor (num_frames, feature_dim)
        max_length: Target sequence length
        random_sample: If True, use random sampling for frame selection
        remaining_length: Remaining trajectory length from first frame to episode end
        
    Returns:
        Tuple of:
            - Rewound video features (max_length, feature_dim)
            - Progress targets for each frame (max_length,)
    """
    video_length = len(video_feature)
    
    # Original logic: start from first half, end in second half, ensure min 3 frames
    if video_length > 3:
        # Sample start from first half
        start_idx = random.randint(0, video_length // 2)
        # Sample end from second half
        end_idx = random.randint(video_length // 2, video_length)
        
        # Ensure minimum 3 frames difference (original uses while loop)
        while end_idx - start_idx < 3:
            start_idx = random.randint(0, video_length // 2)
            end_idx = random.randint(video_length // 2, video_length)
        
        # Extract the forward segment
        video_feature = video_feature[start_idx:end_idx]
        video_length = len(video_feature)
        
        # Adjust remaining_length
        if remaining_length is not None:
            remaining_length = remaining_length - start_idx if remaining_length > start_idx else video_length
    
    # Generate forward progress targets using ORIGINAL ReWiND formula
    # Progress = (position_in_sequence + 1) / remaining_trajectory_length
    progress_indices = torch.arange(1, video_length + 1, dtype=torch.float32)
    forward_progress = progress_indices / remaining_length
    
    # ORIGINAL LOGIC: Reverse from END backwards, then append to forward sequence
    # Example: video=[A,B,C,D,E] -> reversed=[E,D,C,B,A] -> take some from reversed (skip first)
    # Result: [A,B,C,D,E] + [D,C,B] = progress increases then decreases
    
    # Randomly select how many frames to reverse and append
    selected_end_point = random.randint(2, min(video_length, max_length))
    
    # Reverse the entire video and its progress
    reversed_video = video_feature.flip(0)
    reversed_progress = forward_progress.flip(0)
    
    # Take frames from reversed (skip the first frame which is the last frame of original)
    reverse_frames = reversed_video[1:selected_end_point]
    reverse_progress = reversed_progress[1:selected_end_point]
    
    # Concatenate forward + reversed (creates rewind effect)
    rewound_video = torch.cat([video_feature, reverse_frames], dim=0)
    progress_targets = torch.cat([forward_progress, reverse_progress], dim=0)
    
    # Pad or sample to target length
    if len(rewound_video) < max_length:
        # Pad with last frame
        padding_length = max_length - len(rewound_video)
        last_frame = rewound_video[-1].unsqueeze(0)
        padding_frames = last_frame.repeat(padding_length, 1)
        rewound_video = torch.cat([rewound_video, padding_frames], dim=0)
        
        # Pad progress with last progress value
        last_progress = progress_targets[-1]
        padding_progress = torch.full((padding_length,), last_progress)
        progress_targets = torch.cat([progress_targets, padding_progress])
        
    elif len(rewound_video) > max_length:
        # Sample frames to fit max_length
        if random_sample:
            frame_idx = sorted(random.sample(range(len(rewound_video)), max_length))
        else:
            frame_idx = np.linspace(0, len(rewound_video) - 1, max_length, dtype=int)
        rewound_video = rewound_video[frame_idx]
        progress_targets = progress_targets[frame_idx]
    
    return rewound_video, progress_targets


def sample_sarm_video_feature(
    video_feature: torch.Tensor,
    num_frames: int = 9,
    frame_gap: int = 30,
    random_sample: bool = True,
    absolute_indices: torch.Tensor = None,
    episode_length: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample video features for SARM (Stage-Aware Reward Modeling).
    
    SARM uses a specific pattern:
    - 1 initial frame (from episode start)
    - 8 consecutive frames with frame_gap spacing
    
    Progress normalization matches SARM implementation:
    - Progress = absolute_frame_index / total_episode_length
    
    Args:
        video_feature: Video features tensor (num_frames_available, feature_dim)
        num_frames: Target number of frames (default: 9)
        frame_gap: Gap between consecutive frames (default: 30, i.e., 1 second at 30fps)
        random_sample: If True, use random sampling (not used for SARM's fixed pattern)
        absolute_indices: Absolute frame indices in the episode (num_frames_available,)
        episode_length: Total length of the episode
        
    Returns:
        Tuple of:
            - Sampled video features (num_frames, feature_dim)
            - Progress targets for each frame (num_frames,)
    """
    video_length = len(video_feature)
    
    # Generate progress targets based on relative position within sampled sequence
    # Note: SARM paper uses subtask annotations (Equation 2: yt = Pk−1 + ᾱk * τt)
    # Without annotations, we use linear progress relative to sequence position
    if absolute_indices is not None and episode_length is not None:
        # Compute relative progress: position within sequence / remaining trajectory
        # This ensures progress starts near 0 and increases, not starting at 0.8 if sampled from end
        first_frame_idx = absolute_indices[0].item() if isinstance(absolute_indices[0], torch.Tensor) else absolute_indices[0]
        remaining_length = episode_length - first_frame_idx
        
        # Progress = (position_in_sequence + 1) / remaining_trajectory_length
        progress_indices = torch.arange(1, video_length + 1, dtype=torch.float32)
        progress_targets = progress_indices / remaining_length
    else:
        # Fallback: linear progress
        progress_targets = torch.linspace(1.0/video_length, 1.0, video_length)
    
    # SARM pattern: first frame + (num_frames-1) consecutive frames with frame_gap
    # The first frame should be from the beginning of the sequence
    # The remaining frames are sampled with frame_gap spacing
    
    if video_length < num_frames:
        # Not enough frames, pad with last frame
        sampled_video = video_feature
        sampled_progress = progress_targets
        
        padding_length = num_frames - video_length
        last_frame = sampled_video[-1].unsqueeze(0)
        padding_frames = last_frame.repeat(padding_length, 1)
        sampled_video = torch.cat([sampled_video, padding_frames], dim=0)
        
        last_progress = sampled_progress[-1]
        padding_progress = torch.full((padding_length,), last_progress)
        sampled_progress = torch.cat([sampled_progress, padding_progress])
        
    else:
        # Sample frames: first frame + (num_frames-1) with frame_gap
        # The indices should represent: [0, gap, 2*gap, 3*gap, ..., (num_frames-1)*gap]
        # But we need to ensure we don't exceed video_length
        
        frame_indices = [0]  # First frame
        for i in range(1, num_frames):
            idx = i * frame_gap
            if idx >= video_length:
                idx = video_length - 1
            frame_indices.append(idx)
        
        frame_indices = torch.tensor(frame_indices, dtype=torch.long)
        sampled_video = video_feature[frame_indices]
        sampled_progress = progress_targets[frame_indices]
    
    return sampled_video, sampled_progress


def sample_sarm_reverse_video_feature(
    video_feature: torch.Tensor,
    num_frames: int = 9,
    frame_gap: int = 30,
    random_sample: bool = True,
    absolute_indices: torch.Tensor = None,
    episode_length: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample video with reverse augmentation for SARM (rewind augmentation).
    
    Similar to ReWiND's rewind augmentation but adapted for SARM's frame pattern:
    1. Take forward sequence (1 initial + 8 consecutive)
    2. Append some reversed frames from the end backwards
    3. Progress increases then decreases
    
    Args:
        video_feature: Video features tensor (num_frames_available, feature_dim)
        num_frames: Target number of frames (default: 9)
        frame_gap: Gap between consecutive frames (default: 30)
        random_sample: If True, use random sampling for reverse section
        absolute_indices: Absolute frame indices in the episode
        episode_length: Total length of the episode
        
    Returns:
        Tuple of:
            - Rewound video features (num_frames, feature_dim)
            - Progress targets for each frame (num_frames,)
    """
    video_length = len(video_feature)
    
    # Generate forward progress targets (relative to sequence, not absolute)
    if absolute_indices is not None and episode_length is not None:
        # Use same relative progress as normal sampling
        first_frame_idx = absolute_indices[0].item() if isinstance(absolute_indices[0], torch.Tensor) else absolute_indices[0]
        remaining_length = episode_length - first_frame_idx
        progress_indices = torch.arange(1, video_length + 1, dtype=torch.float32)
        forward_progress = progress_indices / remaining_length
    else:
        forward_progress = torch.linspace(1.0/video_length, 1.0, video_length)
    
    # Sample forward sequence first
    forward_video, forward_progress_sampled = sample_sarm_video_feature(
        video_feature, num_frames, frame_gap, random_sample, absolute_indices, episode_length
    )
    
    # Randomly select how many frames to reverse and append
    # For SARM, we append 2-4 reversed frames
    num_reverse = random.randint(2, min(4, num_frames - 1))
    
    # Reverse the video and progress
    reversed_video = video_feature.flip(0)
    reversed_progress = forward_progress.flip(0)
    
    # Take frames from reversed (skip the first frame which is the last frame of original)
    reverse_frames = reversed_video[1:num_reverse+1]
    reverse_progress = reversed_progress[1:num_reverse+1]
    
    # Concatenate forward + reversed (creates rewind effect)
    rewound_video = torch.cat([forward_video, reverse_frames], dim=0)
    progress_targets = torch.cat([forward_progress_sampled, reverse_progress], dim=0)
    
    # Trim to num_frames if necessary
    if len(rewound_video) > num_frames:
        # Keep the first num_frames
        rewound_video = rewound_video[:num_frames]
        progress_targets = progress_targets[:num_frames]
    elif len(rewound_video) < num_frames:
        # Pad if necessary
        padding_length = num_frames - len(rewound_video)
        last_frame = rewound_video[-1].unsqueeze(0)
        padding_frames = last_frame.repeat(padding_length, 1)
        rewound_video = torch.cat([rewound_video, padding_frames], dim=0)
        
        last_progress = progress_targets[-1]
        padding_progress = torch.full((padding_length,), last_progress)
        progress_targets = torch.cat([progress_targets, padding_progress])
    
    return rewound_video, progress_targets

