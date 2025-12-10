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
SARM Processor for encoding images/text and generating stage+tau targets.

Reference: rm_lerobot_dataset.py (FrameGapLeRobotDataset)
Reference: data_utils.py (adapt_lerobot_batch_sarm)
"""

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.policies.sarm.sarm_utils import (
    compute_cumulative_progress_batch,
    compute_tau,
    pad_state_to_max_dim,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
)
from lerobot.processor.converters import (
    from_tensor_to_numpy,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import PipelineFeatureType
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

# Try to import Faker for language perturbation (optional dependency)
try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False


class SARMEncodingProcessorStep(ProcessorStep):
    """ProcessorStep that encodes images and text with CLIP and generates stage and progress labels for SARM."""

    def __init__(
        self,
        config: SARMConfig,
        image_key: str | None = None,
        dataset_meta=None,
        dataset_stats: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.image_key = image_key or config.image_key
        self.dataset_meta = dataset_meta
        self.dataset_stats = dataset_stats
        self.annotation_mode = config.annotation_mode

        # Sparse annotations (always needed)
        self.sparse_temporal_proportions = (
            {
                name: prop
                for name, prop in zip(
                    self.config.sparse_subtask_names, self.config.sparse_temporal_proportions
                )
            }
            if self.config.sparse_subtask_names and self.config.sparse_temporal_proportions
            else None
        )
        self.sparse_subtask_names = self.config.sparse_subtask_names

        # Dense annotations (only for dual mode or dense_only mode)
        self.dense_temporal_proportions = None
        self.dense_subtask_names = None
        if (
            self.config.uses_dual_heads
            and self.config.dense_subtask_names
            and self.config.dense_temporal_proportions
        ):
            self.dense_temporal_proportions = {
                name: prop
                for name, prop in zip(self.config.dense_subtask_names, self.config.dense_temporal_proportions)
            }
            self.dense_subtask_names = self.config.dense_subtask_names

        self.device = torch.device(
            self.config.device if self.config.device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.clip_model.to(self.device)
        self.clip_model.eval()

        # Language perturbation setup (Reference: rm_lerobot_dataset.py lines 27-28)
        self.verbs = ['move', 'grasp', 'rotate', 'push', 'pull', 'slide', 'lift', 'place']
        self.fake = Faker() if FAKER_AVAILABLE else None

    def _find_episode_for_frame(self, frame_idx: int) -> int:
        """Find the episode index for a given frame index."""
        for ep_idx in range(len(self.dataset_meta.episodes)):
            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
            if ep_start <= frame_idx < ep_end:
                return ep_idx
        return 0

    def _get_episode_indices(self, frame_indices: np.ndarray, episode_index) -> np.ndarray:
        """Get episode indices for each frame index."""
        if episode_index is None:
            return np.array([self._find_episode_for_frame(int(f)) for f in frame_indices])

        episode_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(episode_index)))

        # If single episode but multiple frames, compute episode for each frame
        if len(episode_indices) == 1 and len(frame_indices) > 1:
            return np.array([self._find_episode_for_frame(int(f)) for f in frame_indices])

        return episode_indices

    def _compute_absolute_indices(
        self, frame_idx: int, ep_start: int, ep_end: int, n_obs_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute absolute frame indices for a backward-looking sequence.

        Reference: rm_lerobot_dataset.py get_frame_indices()
        
        Backward-looking pattern ending at target frame:
        - Pattern: [idx - n_obs_steps*gap, ..., idx-gap, idx]
        - Adaptive gap when insufficient history (evenly space from ep_start to idx)
        
        Args:
            frame_idx: Target frame index (last frame of sequence)
            ep_start: Episode start index
            ep_end: Episode end index
            n_obs_steps: Number of observation steps (sequence length = n_obs_steps + 1)

        Returns:
            Tuple of (indices, out_of_bounds_flags)
        """
        frame_gap = self.config.frame_gap
        
        # Clamp idx to episode bounds
        idx = min(frame_idx, ep_end - 1)
        idx = max(idx, ep_start)

        gaps = n_obs_steps
        if gaps == 0:
            return torch.tensor([idx]), torch.tensor([0])

        # Check if fixed stride fits entirely inside the episode
        # Reference: rm_lerobot_dataset.py lines 64-80
        total_needed = frame_gap * gaps  # distance from earliest to idx
        available = idx - ep_start

        if available >= total_needed:
            # Use fixed frame_gap
            frames = [idx - frame_gap * (gaps - k) for k in range(gaps)] + [idx]
        else:
            # Not enough history: adapt stride by evenly spacing from ep_start to idx
            # Use integer rounding and enforce monotonicity
            frames = [ep_start + round(available * k / gaps) for k in range(gaps)] + [idx]
            for i in range(1, len(frames)):
                if frames[i] < frames[i - 1]:
                    frames[i] = frames[i - 1]

        # Track out-of-bounds (all should be in bounds with this adaptive method)
        out_of_bounds = [0] * len(frames)

        return torch.tensor(frames), torch.tensor(out_of_bounds)

    def _compute_episode_metadata(
        self,
        frame_indices: np.ndarray,
        episode_indices: np.ndarray,
        num_frames: int,
    ) -> tuple[list | torch.Tensor, list | torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute episode metadata for all samples.

        Returns:
            Tuple of (absolute_frame_indices, out_of_bounds_flags, remaining_lengths, episode_lengths)
        """
        absolute_indices_list = []
        out_of_bounds_list = []
        remaining_lengths = []
        episode_lengths = []

        for ep_idx, frame_idx in zip(episode_indices.tolist(), frame_indices.tolist()):
            ep_idx, frame_idx = int(ep_idx), int(frame_idx)
            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]

            episode_lengths.append(ep_end - ep_start)
            abs_indices, out_of_bounds = self._compute_absolute_indices(
                frame_idx, ep_start, ep_end, num_frames
            )
            absolute_indices_list.append(abs_indices)
            out_of_bounds_list.append(out_of_bounds)
            remaining_lengths.append(ep_end - abs_indices[0].item())

        return (
            absolute_indices_list,
            out_of_bounds_list,
            torch.tensor(remaining_lengths),
            torch.tensor(episode_lengths),
        )

    def _generate_perturbed_task(self) -> str:
        """
        Generate a random perturbed task string for language perturbation.
        
        Reference: rm_lerobot_dataset.py lines 106-111
        
        Returns:
            Random task string made of verb + random words
        """
        if self.fake is None:
            # Fallback if Faker not available
            return " ".join(random.choices(self.verbs, k=random.randint(2, 5)))
        
        num_words = random.randint(1, 5)
        verb = random.choice(self.verbs)
        return " ".join([verb] + self.fake.words(nb=num_words))

    def _apply_rewind_augmentation(
        self,
        obs_indices: list[int],
        frame_idx: int,
        ep_start: int,
        n_obs_steps: int,
        max_rewind_steps: int,
    ) -> tuple[int, list[int]]:
        """
        Generate rewind frame indices for temporal augmentation.
        
        Reference: rm_lerobot_dataset.py _get_rewind() lines 148-171
        
        Rewind simulates going backwards through previously seen frames.
        Appends reversed frames after the observation sequence.
        
        Args:
            obs_indices: Observation frame indices
            frame_idx: Target frame index
            ep_start: Episode start index
            n_obs_steps: Number of observation steps
            max_rewind_steps: Maximum rewind steps
            
        Returns:
            Tuple of (rewind_step, rewind_indices)
        """
        frame_gap = self.config.frame_gap
        
        # Determine valid rewind range
        max_valid_step = (frame_idx - ep_start - frame_gap) // frame_gap
        max_rewind = min(max_rewind_steps, max(1, max_valid_step))
        
        # Sample rewind steps
        rewind_step = random.randint(1, max_rewind) if max_rewind > 0 else 0
        
        if rewind_step == 0:
            return 0, []
        
        # Generate rewind indices (reversed order)
        # Reference: rm_lerobot_dataset.py lines 160-163
        rewind_indices = list(range(frame_idx - rewind_step * frame_gap, frame_idx, frame_gap))
        
        if len(rewind_indices) < rewind_step:
            pad_count = rewind_step - len(rewind_indices)
            rewind_indices += [rewind_indices[-1]] * pad_count if rewind_indices else [ep_start] * pad_count
        
        # Reverse the indices (going backwards)
        rewind_indices = rewind_indices[::-1]
        
        return rewind_step, rewind_indices

    def _compute_stage_tau_target(
        self,
        current_frame: int,
        episode_length: int,
        subtask_names: list | None,
        subtask_start_frames: list | None,
        subtask_end_frames: list | None,
        global_subtask_names: list,
        temporal_proportions: dict,
    ) -> float:
        """
        Compute stage+tau target for a single frame.
        
        Reference: workspace/sarm_ws.py - target format is stage.tau
        
        Returns target in format: stage_idx + tau
        where stage_idx is the integer stage and tau is within-stage progress [0, 1)
        
        Args:
            current_frame: Frame index relative to episode start
            episode_length: Total frames in episode
            subtask_names: List of subtask names for this episode
            subtask_start_frames: List of subtask start frames
            subtask_end_frames: List of subtask end frames
            global_subtask_names: Global list of all subtask names
            temporal_proportions: Dict of temporal proportions
            
        Returns:
            Target value in stage.tau format
        """
        # Single-stage mode: linear progress
        if global_subtask_names == ["task"] and temporal_proportions == {"task": 1.0}:
            progress = current_frame / max(episode_length - 1, 1)
            return min(1.0, max(0.0, progress))  # tau only, stage = 0

        # Annotation-based mode
        if subtask_names is None:
            return 0.0

        # Find which subtask this frame belongs to
        for name, start_frame, end_frame in zip(subtask_names, subtask_start_frames, subtask_end_frames):
            if start_frame <= current_frame <= end_frame:
                stage_idx = global_subtask_names.index(name) if name in global_subtask_names else 0
                tau = compute_tau(current_frame, start_frame, end_frame)
                return stage_idx + tau

        # Frame outside annotated subtasks
        if current_frame < subtask_start_frames[0]:
            return 0.0
        if current_frame > subtask_end_frames[-1]:
            return len(global_subtask_names) - 1 + 0.999  # Last stage, nearly complete

        # Between subtasks - use previous subtask's end state
        for j in range(len(subtask_names) - 1):
            if subtask_end_frames[j] < current_frame < subtask_start_frames[j + 1]:
                name = subtask_names[j]
                stage_idx = global_subtask_names.index(name) if name in global_subtask_names else j
                return stage_idx + 1.0  # End of previous stage

        return 0.0

    def _compute_progress_for_frame(
        self,
        current_frame: int,
        episode_length: int,
        subtask_names: list | None,
        subtask_start_frames: list | None,
        subtask_end_frames: list | None,
        global_subtask_names: list,
        temporal_proportions: dict,
    ) -> tuple[int, float]:
        """Compute stage index and cumulative progress for a single frame.

        Unified method for both annotation-based and single_stage modes.
        Implements SARM Paper Formula (2): y_t = P_{k-1} + ᾱ_k × τ_t

        Args:
            current_frame: Frame index relative to episode start
            episode_length: Total frames in episode (for single_stage linear progress)
            subtask_names: List of subtask names for this episode (None for single_stage)
            subtask_start_frames: List of subtask start frames (None for single_stage)
            subtask_end_frames: List of subtask end frames (None for single_stage)
            global_subtask_names: Global list of all subtask names
            temporal_proportions: Dict of temporal proportions for each subtask

        Returns:
            Tuple of (stage_idx, cumulative_progress)
        """
        # Single-stage mode: linear progress from 0 to 1
        if global_subtask_names == ["task"] and temporal_proportions == {"task": 1.0}:
            progress = current_frame / max(episode_length - 1, 1)
            return 0, min(1.0, max(0.0, progress))

        # Annotation-based mode: find subtask and compute cumulative progress
        if subtask_names is None:
            return 0, 0.0

        temporal_proportions_list = [temporal_proportions.get(name, 0.0) for name in global_subtask_names]

        # Find which subtask this frame belongs to
        for name, start_frame, end_frame in zip(subtask_names, subtask_start_frames, subtask_end_frames):
            if start_frame <= current_frame <= end_frame:
                stage_idx = global_subtask_names.index(name) if name in global_subtask_names else 0
                tau = compute_tau(current_frame, start_frame, end_frame)
                return stage_idx, compute_cumulative_progress_batch(tau, stage_idx, temporal_proportions_list)

        # Frame outside annotated subtasks
        if current_frame < subtask_start_frames[0]:
            return 0, 0.0
        if current_frame > subtask_end_frames[-1]:
            return len(global_subtask_names) - 1, 1.0

        # Between subtasks - use previous subtask's end state
        for j in range(len(subtask_names) - 1):
            if subtask_end_frames[j] < current_frame < subtask_start_frames[j + 1]:
                name = subtask_names[j]
                stage_idx = global_subtask_names.index(name) if name in global_subtask_names else j
                return stage_idx, compute_cumulative_progress_batch(1.0, stage_idx, temporal_proportions_list)

        return 0, 0.0

    def _compute_labels_for_sample(
        self,
        frame_idx: int,
        ep_idx: int,
        apply_rewind: bool,
        episodes_df: pd.DataFrame | None,
        annotation_type: str = "sparse",
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compute stage+tau targets for a sample with optional rewind augmentation.

        Reference: rm_lerobot_dataset.py __getitem__()
        
        Uses backward-looking frame sequence pattern ending at target frame.
        Optionally appends reversed frames for rewind augmentation.

        Args:
            frame_idx: The target frame index for this sample
            ep_idx: The episode index
            apply_rewind: Whether to apply rewind augmentation
            episodes_df: DataFrame with episode metadata
            annotation_type: "sparse" or "dense"

        Returns:
            Tuple of (targets, lengths, rewind_step)
            - targets: stage+tau targets (T,) where T = 1 + n_obs_steps + max_rewind_steps
            - lengths: Valid sequence length (1 + n_obs_steps + rewind_step)
            - rewind_step: Number of rewind frames (0 if no rewind)
        """
        n_obs_steps = self.config.n_obs_steps
        max_rewind_steps = self.config.max_rewind_steps
        frame_gap = self.config.frame_gap
        
        ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
        ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
        ep_length = ep_end - ep_start

        # Get global names and proportions based on annotation type
        if annotation_type == "dense":
            global_names = self.dense_subtask_names
            temporal_props = self.dense_temporal_proportions
        else:
            global_names = self.sparse_subtask_names
            temporal_props = self.sparse_temporal_proportions

        # Load episode-specific annotations
        subtask_names, subtask_start_frames, subtask_end_frames = None, None, None
        if episodes_df is not None and global_names != ["task"]:
            col_names = (
                f"{annotation_type}_subtask_names"
                if f"{annotation_type}_subtask_names" in episodes_df.columns
                else "subtask_names"
            )
            col_start = (
                f"{annotation_type}_subtask_start_frames"
                if f"{annotation_type}_subtask_start_frames" in episodes_df.columns
                else "subtask_start_frames"
            )
            col_end = (
                f"{annotation_type}_subtask_end_frames"
                if f"{annotation_type}_subtask_end_frames" in episodes_df.columns
                else "subtask_end_frames"
            )

            if col_names in episodes_df.columns and ep_idx < len(episodes_df):
                subtask_names = episodes_df.loc[ep_idx, col_names]
                if subtask_names is not None and not (
                    isinstance(subtask_names, float) and pd.isna(subtask_names)
                ):
                    subtask_start_frames = episodes_df.loc[ep_idx, col_start]
                    subtask_end_frames = episodes_df.loc[ep_idx, col_end]
                else:
                    subtask_names = None

        # Get observation frame indices (backward-looking pattern)
        obs_indices, _ = self._compute_absolute_indices(frame_idx, ep_start, ep_end, n_obs_steps)
        obs_indices = obs_indices.tolist()

        # Determine rewind step
        required_history = n_obs_steps * frame_gap
        rewind_step = 0
        rewind_indices = []
        
        if apply_rewind and max_rewind_steps > 0 and frame_idx > ep_start + required_history:
            rewind_step, rewind_indices = self._apply_rewind_augmentation(
                obs_indices, frame_idx, ep_start, n_obs_steps, max_rewind_steps
            )

        # Initialize targets tensor with zeros
        total_length = 1 + n_obs_steps + max_rewind_steps
        targets = torch.zeros(total_length, dtype=torch.float32)

        # Compute targets for observation frames
        for i, idx in enumerate(obs_indices):
            rel_frame = idx - ep_start
            target = self._compute_stage_tau_target(
                rel_frame,
                ep_length,
                subtask_names,
                subtask_start_frames,
                subtask_end_frames,
                global_names,
                temporal_props,
            )
            targets[i] = target

        # Compute targets for rewind frames (reversed progress)
        # Reference: rm_lerobot_dataset.py lines 115-117
        for i, idx in enumerate(rewind_indices):
            rel_frame = idx - ep_start
            target = self._compute_stage_tau_target(
                rel_frame,
                ep_length,
                subtask_names,
                subtask_start_frames,
                subtask_end_frames,
                global_names,
                temporal_props,
            )
            targets[1 + n_obs_steps + i] = target

        # Compute valid length
        valid_length = 1 + n_obs_steps + rewind_step

        return targets, valid_length, rewind_step


    def compute_episode_ground_truth(
        self,
        episode_index: int,
        annotation_type: str = "sparse",
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Compute ground truth progress and stage labels for all frames in an episode.

        Uses the same _compute_progress_for_frame method as training for consistency.

        Args:
            episode_index: Index of the episode
            annotation_type: "sparse" or "dense"

        Returns:
            Tuple of (progress_array, stage_array) with shape (num_frames,), or (None, None)
        """
        if self.dataset_meta is None:
            return None, None

        ep_start = self.dataset_meta.episodes["dataset_from_index"][episode_index]
        ep_end = self.dataset_meta.episodes["dataset_to_index"][episode_index]
        num_frames = ep_end - ep_start

        # Get global names and proportions
        if annotation_type == "dense":
            if self.dense_temporal_proportions is None:
                return None, None
            global_names, temporal_props = self.dense_subtask_names, self.dense_temporal_proportions
        else:
            if self.sparse_temporal_proportions is None:
                return None, None
            global_names, temporal_props = self.sparse_subtask_names, self.sparse_temporal_proportions

        # Load episode-specific annotations (None for single_stage mode)
        subtask_names, subtask_start_frames, subtask_end_frames = None, None, None
        if global_names != ["task"]:
            episodes_df = self.dataset_meta.episodes.to_pandas()
            col_names = (
                f"{annotation_type}_subtask_names"
                if f"{annotation_type}_subtask_names" in episodes_df.columns
                else "subtask_names"
            )
            col_start = (
                f"{annotation_type}_subtask_start_frames"
                if f"{annotation_type}_subtask_start_frames" in episodes_df.columns
                else "subtask_start_frames"
            )
            col_end = (
                f"{annotation_type}_subtask_end_frames"
                if f"{annotation_type}_subtask_end_frames" in episodes_df.columns
                else "subtask_end_frames"
            )

            if col_names in episodes_df.columns:
                subtask_names = episodes_df.loc[episode_index, col_names]
                if subtask_names is not None and not (
                    isinstance(subtask_names, float) and pd.isna(subtask_names)
                ):
                    subtask_start_frames = episodes_df.loc[episode_index, col_start]
                    subtask_end_frames = episodes_df.loc[episode_index, col_end]
                else:
                    subtask_names = None

        # Compute for each frame using the unified method
        gt_progress = np.zeros(num_frames)
        gt_stages = np.zeros(num_frames, dtype=np.int32)

        for frame_rel in range(num_frames):
            stage_idx, progress = self._compute_progress_for_frame(
                frame_rel,
                num_frames,
                subtask_names,
                subtask_start_frames,
                subtask_end_frames,
                global_names,
                temporal_props,
            )
            gt_progress[frame_rel] = progress
            gt_stages[frame_rel] = stage_idx

        return gt_progress, gt_stages

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Encode images, text, and normalize states in the transition.
        
        Implements SARM training data preparation:
        - Applies language perturbation (20% probability)
        - Applies rewind augmentation (80% probability) 
        - Generates stage+tau targets for all frames
        - Outputs lengths tensor for valid sequence masking
        - Supports per-timestep dense text embeddings
        
        Reference: rm_lerobot_dataset.py __getitem__()
        """
        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        # Get indices
        frame_index = comp_data.get("index")
        episode_index = comp_data.get("episode_index")

        if frame_index is None:
            raise ValueError("Frame index ('index') not found in COMPLEMENTARY_DATA")
        if episode_index is None:
            raise ValueError("Episode index ('episode_index') not found in COMPLEMENTARY_DATA")

        frame_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(frame_index)))
        episode_indices = self._get_episode_indices(frame_indices, episode_index)

        # Extract image data
        image = observation.get(self.image_key)
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # If 4D (T, C, H, W) from delta_timestamps, add batch dim
        # If 3D (C, H, W) single frame, add batch and time dims
        if image.ndim == 4:
            image = image[np.newaxis, ...]  # (T, C, H, W) -> (1, T, C, H, W)
        elif image.ndim == 3:
            image = image[np.newaxis, np.newaxis, ...]  # (C, H, W) -> (1, 1, C, H, W)

        batch_size = image.shape[0]
        total_frames = image.shape[1]  # Should be 13: 9 obs + 4 rewind placeholders
        n_obs_steps = self.config.n_obs_steps
        max_rewind_steps = self.config.max_rewind_steps
        n_obs_frames = 1 + n_obs_steps  # 9 observation frames (including current)

        # === REWIND AUGMENTATION ===
        # Reference: rm_lerobot_dataset.py lines 148-171
        rewind_steps = torch.zeros(batch_size, dtype=torch.int32)
        apply_rewind = self.training and random.random() < self.config.rewind_probability

        if apply_rewind and self.dataset_meta is not None:
            for b_idx, (ep_idx, frame_idx) in enumerate(zip(episode_indices.tolist(), frame_indices.tolist())):
                ep_idx, frame_idx = int(ep_idx), int(frame_idx)
                ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
                
                # Determine valid rewind range
                frame_gap = self.config.frame_gap
                required_history = n_obs_steps * frame_gap
                
                if frame_idx > ep_start + required_history:
                    max_valid_step = (frame_idx - ep_start - required_history) // frame_gap
                    max_rewind = min(max_rewind_steps, max(1, max_valid_step))
                    rewind_steps[b_idx] = random.randint(1, max_rewind) if max_rewind > 0 else 0

        # Compute valid lengths: n_obs_frames + rewind_steps
        lengths = n_obs_frames + rewind_steps  # (B,)

        # Apply rewind masking to images
        # For frames beyond valid length, we mask with zeros (or copy last valid frame)
        # The rewind slots in delta_timestamps already loaded the right frames
        # We just need to mask out unused rewind slots
        for b_idx in range(batch_size):
            valid_len = lengths[b_idx].item()
            if valid_len < total_frames:
                # Zero out frames beyond valid length
                image[b_idx, valid_len:] = 0

        # Encode images with CLIP
        video_features = self._encode_images_batch(image)
        observation["video_features"] = video_features

        # === STATE FEATURES ===
        state_key = self.config.state_key
        state_data = observation.get(state_key)

        if isinstance(state_data, torch.Tensor):
            state_tensor = state_data.float()
        else:
            state_tensor = torch.tensor(state_data, dtype=torch.float32)

        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)  # (T, D) -> (1, T, D)
        elif state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (D,) -> (1, 1, D)

        # Apply same rewind masking to state
        for b_idx in range(batch_size):
            valid_len = lengths[b_idx].item()
            if valid_len < state_tensor.shape[1]:
                state_tensor[b_idx, valid_len:] = 0

        observation["state_features"] = pad_state_to_max_dim(state_tensor, self.config.max_state_dim)

        # === LANGUAGE PERTURBATION ===
        # Reference: rm_lerobot_dataset.py lines 106-111
        task = comp_data.get("task")
        if isinstance(task, list):
            task = task[0] if task else ""

        # Apply language perturbation during training (20% probability)
        # When perturbed, targets will be zeroed to train model to output low values for irrelevant text
        apply_perturbation = self.training and random.random() < self.config.language_perturbation_probability
        if apply_perturbation:
            task = self._generate_perturbed_task()

        # Encode text with CLIP (single embedding broadcast across timesteps)
        observation["text_features"] = self._encode_text_clip(task, batch_size)

        # Store lengths for model
        observation["lengths"] = lengths

        # === GENERATE STAGE+TAU TARGETS ===
        # Reference: rm_lerobot_dataset.py lines 112-117
        # When language is perturbed, targets are ZEROED so perturbed samples don't 
        # contribute to progress loss - they only train model to output low values for random text
        if self.dataset_meta is not None:
            episodes_df = None
            if self.sparse_subtask_names != ["task"]:
                episodes_df = self.dataset_meta.episodes.to_pandas()

            # Generate sparse targets
            if self.sparse_temporal_proportions is not None:
                if apply_perturbation:
                    # Zero targets when language is perturbed
                    sparse_targets = torch.zeros(batch_size, total_frames, dtype=torch.float32)
                else:
                    sparse_targets = self._compute_batch_targets(
                        frame_indices, episode_indices, lengths, rewind_steps, episodes_df, "sparse"
                    )
                observation["sparse_targets"] = sparse_targets

            # Generate dense targets (for dual mode)
            if self.config.uses_dual_heads and self.dense_temporal_proportions is not None:
                if apply_perturbation:
                    # Zero targets when language is perturbed
                    dense_targets = torch.zeros(batch_size, total_frames, dtype=torch.float32)
                else:
                    dense_targets = self._compute_batch_targets(
                        frame_indices, episode_indices, lengths, rewind_steps, episodes_df, "dense"
                    )
                observation["dense_targets"] = dense_targets

                # For dense mode, optionally encode per-timestep text embeddings
                # This allows different subtask descriptions per frame
                # Skip dense text embeddings when perturbed (all use the perturbed task string)
                if not apply_perturbation and self.dense_subtask_names and len(self.dense_subtask_names) > 1:
                    # Generate per-timestep text embeddings based on dense subtask names
                    dense_text_features = self._encode_dense_text_embeddings(
                        dense_targets, batch_size, total_frames
                    )
                    observation["dense_text_features"] = dense_text_features

        # Compute additional episode metadata
        if self.dataset_meta is not None:
            if video_features.dim() >= 2:
                num_frames = video_features.shape[1]
            else:
                num_frames = 1

            abs_indices, out_of_bounds, remaining, ep_lengths = self._compute_episode_metadata(
                frame_indices, episode_indices, num_frames
            )
            observation["absolute_frame_indices"] = abs_indices
            observation["out_of_bounds_flags"] = out_of_bounds
            observation["remaining_length"] = remaining
            observation["episode_length"] = ep_lengths

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def _compute_batch_targets(
        self,
        frame_indices: np.ndarray,
        episode_indices: np.ndarray,
        lengths: torch.Tensor,
        rewind_steps: torch.Tensor,
        episodes_df: pd.DataFrame | None,
        annotation_type: str,
    ) -> torch.Tensor:
        """
        Compute stage+tau targets for a batch of samples.
        
        Args:
            frame_indices: Target frame indices (B,)
            episode_indices: Episode indices (B,)
            lengths: Valid sequence lengths (B,)
            rewind_steps: Number of rewind steps per sample (B,)
            episodes_df: DataFrame with episode annotations
            annotation_type: "sparse" or "dense"
            
        Returns:
            Targets tensor (B, T) in stage.tau format
        """
        batch_size = len(frame_indices)
        n_obs_steps = self.config.n_obs_steps
        max_rewind_steps = self.config.max_rewind_steps
        total_frames = 1 + n_obs_steps + max_rewind_steps
        frame_gap = self.config.frame_gap

        # Get annotation config
        if annotation_type == "dense":
            global_names = self.dense_subtask_names
            temporal_props = self.dense_temporal_proportions
        else:
            global_names = self.sparse_subtask_names
            temporal_props = self.sparse_temporal_proportions

        targets = torch.zeros(batch_size, total_frames, dtype=torch.float32)

        for b_idx in range(batch_size):
            ep_idx = int(episode_indices[b_idx])
            frame_idx = int(frame_indices[b_idx])

            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
            ep_length = ep_end - ep_start

            # Load episode-specific annotations
            subtask_names, subtask_start_frames, subtask_end_frames = None, None, None
            if episodes_df is not None and global_names != ["task"]:
                col_names = (
                    f"{annotation_type}_subtask_names"
                    if f"{annotation_type}_subtask_names" in episodes_df.columns
                    else "subtask_names"
                )
                col_start = (
                    f"{annotation_type}_subtask_start_frames"
                    if f"{annotation_type}_subtask_start_frames" in episodes_df.columns
                    else "subtask_start_frames"
                )
                col_end = (
                    f"{annotation_type}_subtask_end_frames"
                    if f"{annotation_type}_subtask_end_frames" in episodes_df.columns
                    else "subtask_end_frames"
                )

                if col_names in episodes_df.columns and ep_idx < len(episodes_df):
                    subtask_names = episodes_df.loc[ep_idx, col_names]
                    if subtask_names is not None and not (
                        isinstance(subtask_names, float) and pd.isna(subtask_names)
                    ):
                        subtask_start_frames = episodes_df.loc[ep_idx, col_start]
                        subtask_end_frames = episodes_df.loc[ep_idx, col_end]
                    else:
                        subtask_names = None

            # Compute observation frame indices (backward-looking)
            obs_indices, _ = self._compute_absolute_indices(frame_idx, ep_start, ep_end, n_obs_steps)
            obs_indices = obs_indices.tolist()

            # Compute targets for observation frames
            for t_idx, abs_idx in enumerate(obs_indices):
                rel_frame = abs_idx - ep_start
                target = self._compute_stage_tau_target(
                    rel_frame, ep_length, subtask_names, subtask_start_frames,
                    subtask_end_frames, global_names, temporal_props
                )
                targets[b_idx, t_idx] = target

            # Compute targets for rewind frames (if any)
            rewind_step = rewind_steps[b_idx].item()
            if rewind_step > 0:
                # Rewind frames go backwards from current position
                # Reference: rm_lerobot_dataset.py lines 160-163
                rewind_indices = list(range(frame_idx - rewind_step * frame_gap, frame_idx, frame_gap))
                if len(rewind_indices) < rewind_step:
                    pad_count = rewind_step - len(rewind_indices)
                    rewind_indices += [rewind_indices[-1]] * pad_count if rewind_indices else [ep_start] * pad_count
                # Reverse for backward direction
                rewind_indices = rewind_indices[::-1]

                for r_idx, abs_idx in enumerate(rewind_indices[:rewind_step]):
                    rel_frame = max(0, abs_idx - ep_start)
                    target = self._compute_stage_tau_target(
                        rel_frame, ep_length, subtask_names, subtask_start_frames,
                        subtask_end_frames, global_names, temporal_props
                    )
                    targets[b_idx, n_obs_steps + 1 + r_idx] = target

        return targets

    @property
    def training(self) -> bool:
        """Check if in training mode (for augmentation decisions)."""
        # Check if we're in a training context by looking at the config or a flag
        return getattr(self, '_training_mode', True)

    def train(self, mode: bool = True):
        """Set training mode for augmentation decisions."""
        self._training_mode = mode
        return self

    def eval(self):
        """Set evaluation mode (disable augmentations)."""
        return self.train(False)

    def _encode_dense_text_embeddings(
        self, targets: torch.Tensor, batch_size: int, total_frames: int
    ) -> torch.Tensor:
        """
        Generate per-timestep text embeddings for dense mode.
        
        Maps each frame's stage to its corresponding subtask description
        and encodes with CLIP.
        
        Args:
            targets: Target tensor (B, T) in stage.tau format
            batch_size: Batch size
            total_frames: Total number of frames
            
        Returns:
            Dense text features (B, T, 512)
        """
        if not self.dense_subtask_names:
            # Fallback: broadcast single embedding
            return self._encode_text_clip(self.dense_subtask_names[0] if self.dense_subtask_names else "", batch_size)

        # Extract stage indices from targets
        stage_indices = targets.long().clamp(0, len(self.dense_subtask_names) - 1)  # (B, T)

        # Pre-encode all subtask names
        subtask_embeddings = {}
        for name in self.dense_subtask_names:
            emb = self._encode_text_clip(name, 1)  # (1, 512)
            subtask_embeddings[name] = emb.squeeze(0)  # (512,)

        # Build per-timestep embeddings
        dense_text_features = torch.zeros(batch_size, total_frames, self.config.text_dim)
        for b_idx in range(batch_size):
            for t_idx in range(total_frames):
                stage_idx = stage_indices[b_idx, t_idx].item()
                name = self.dense_subtask_names[min(stage_idx, len(self.dense_subtask_names) - 1)]
                dense_text_features[b_idx, t_idx] = subtask_embeddings[name]

        return dense_text_features

    @torch.no_grad()
    def _encode_images_batch(self, images: np.ndarray) -> torch.Tensor:
        """Encode a batch of images using CLIP.

        Args:
            images: Batched images with shape: (B, T, C, H, W)

        Returns:
            Encoded feature vectors with shape (B, T, 512)
        """

        batch_size, seq_length = images.shape[0], images.shape[1]
        images = images.reshape(batch_size * seq_length, *images.shape[2:])

        # Convert to list of PIL images
        num_frames = images.shape[0]
        images_list = []
        for i in range(num_frames):
            img = images[i]
            if img.shape[0] in [1, 3]:  # Channel first (C, H, W)
                img = img.transpose(1, 2, 0)

            # Handle single channel
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

            # Convert to uint8
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

            images_list.append(Image.fromarray(img))

        all_embeddings = []
        for i in range(0, num_frames, self.config.clip_batch_size):
            batch_imgs = images_list[i : i + self.config.clip_batch_size]

            # Process with CLIP
            inputs = self.clip_processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get image embeddings
            embeddings = self.clip_model.get_image_features(**inputs).detach().cpu()

            # Handle single frame case
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)

            all_embeddings.append(embeddings)

        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings)  # (B*T, 512)

        # Reshape back
        all_embeddings = all_embeddings.reshape(batch_size, seq_length, -1)  # (B, T, 512)

        return all_embeddings

    @torch.no_grad()
    def _encode_text_clip(self, text: str, batch_size: int) -> torch.Tensor:
        """Encode text using CLIP text encoder (per SARM paper A.4).

        Args:
            text: Task description text to encode
            batch_size: Batch size to replicate for

        Returns:
            Encoded text features with shape (B, 512)
        """
        tokenizer = self.clip_processor.tokenizer
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        text_embedding = self.clip_model.get_text_features(**inputs).detach().cpu()
        text_embedding = text_embedding.expand(batch_size, -1)

        return text_embedding

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Add encoded features to the observation features."""
        features[PipelineFeatureType.OBSERVATION]["video_features"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(self.config.num_frames, self.config.image_dim)
        )
        features[PipelineFeatureType.OBSERVATION]["text_features"] = PolicyFeature(
            type=FeatureType.LANGUAGE, shape=(self.config.text_dim,)
        )
        features[PipelineFeatureType.OBSERVATION]["state_features"] = PolicyFeature(
            type=FeatureType.STATE, shape=(self.config.num_frames, self.config.max_state_dim)
        )
        return features


def make_sarm_pre_post_processors(
    config: SARMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta=None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create pre-processor and post-processor pipelines for SARM.

    The pre-processing pipeline:
    1. Adds batch dimension
    2. Normalizes observation.state using NormalizerProcessorStep (MEAN_STD)
    3. SARMEncodingProcessorStep:
       - Encodes images with CLIP
       - Pads states to max_state_dim
       - Encodes text with CLIP
    4. Moves data to device

    The post-processing pipeline:
    1. Moves data to CPU
    """
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[
                AddBatchDimensionProcessorStep(),
                NormalizerProcessorStep(
                    features={**config.input_features, **config.output_features},
                    norm_map=config.normalization_mapping,
                    stats=dataset_stats,
                ),
                SARMEncodingProcessorStep(
                    config=config, dataset_meta=dataset_meta, dataset_stats=dataset_stats
                ),
                DeviceProcessorStep(device=config.device),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[DeviceProcessorStep(device="cpu")],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
