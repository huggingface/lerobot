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
        self, frame_idx: int, ep_start: int, ep_end: int, num_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute absolute frame indices for a sequence with out-of-bounds tracking.

        Uniform target sampling: any frame can be the target. Out-of-bounds indices
        are clamped to valid range with tracking for proper progress assignment:
        - Before ep_start: clamp to ep_start, mark as out_of_bounds=-1 (progress=0)
        - After ep_end: clamp to ep_end-1, mark as out_of_bounds=+1 (progress=1)

        Pattern (centered around target frame):
        - Frame 0: Initial frame of the episode (ep_start)
        - Frames 1-8: 8 consecutive frames CENTERED at target frame
          [-4*gap, -3*gap, -2*gap, -gap, 0, +gap, +2*gap, +3*gap]

        Returns:
            Tuple of (indices, out_of_bounds_flags) where:
            - indices: clamped frame indices
            - out_of_bounds_flags: -1 for before start, +1 for after end, 0 for in bounds
        """
        indices = []
        out_of_bounds = []

        indices.append(ep_start)  # First frame is always the episode's initial frame
        out_of_bounds.append(0)  # First frame is always in bounds

        # Compute centered deltas: 4 before, target (0), 3 after
        num_consecutive = num_frames - 1  # 8
        half_before = num_consecutive // 2  # 4
        half_after = num_consecutive - half_before - 1  # 3

        # Build deltas: [-4*gap, -3*gap, -2*gap, -gap, 0, +gap, +2*gap, +3*gap]
        deltas = [-self.config.frame_gap * i for i in range(half_before, 0, -1)]  # [-120, -90, -60, -30]
        deltas.append(0)  # Target frame
        deltas.extend([self.config.frame_gap * i for i in range(1, half_after + 1)])  # [30, 60, 90]

        for offset in deltas:
            raw_idx = frame_idx + offset

            if raw_idx < ep_start:
                indices.append(ep_start)
                out_of_bounds.append(-1)  # Before episode start
            elif raw_idx >= ep_end:
                indices.append(ep_end - 1)
                out_of_bounds.append(1)  # After episode end
            else:
                indices.append(raw_idx)
                out_of_bounds.append(0)  # In bounds

        return torch.tensor(indices), torch.tensor(out_of_bounds)

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
        seq_len: int,
        episodes_df: pd.DataFrame | None,
        annotation_type: str = "sparse",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute stage labels and progress targets for a single sample.

        Uniform target sampling: handles out-of-bounds frames with padding.
        - Before episode start: progress = 0, stage = 0 (first stage)
        - After episode end: progress = 1, stage = last stage

        Pattern (centered around target frame):
        - Frame 0: Initial frame of episode
        - Frames 1-8: 8 consecutive frames CENTERED at target frame
          [-4*gap, -3*gap, -2*gap, -gap, 0, +gap, +2*gap, +3*gap]

        Args:
            frame_idx: The frame index for this sample
            ep_idx: The episode index
            seq_len: Number of frames in the sequence
            episodes_df: DataFrame with episode metadata (can be None for single_stage)
            annotation_type: "sparse" or "dense"

        Returns:
            Tuple of (stage_labels, progress_targets) tensors with shapes (T,) and (T, 1)
        """
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

        # Load episode-specific annotations (None for single_stage mode)
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

        # Build centered deltas for frames 1-8: 4 before, target (0), 3 after
        num_consecutive = seq_len - 1  # 8
        half_before = num_consecutive // 2  # 4
        half_after = num_consecutive - half_before - 1  # 3

        # Deltas: [-4*gap, -3*gap, -2*gap, -gap, 0, +gap, +2*gap, +3*gap]
        deltas = [-self.config.frame_gap * i for i in range(half_before, 0, -1)]
        deltas.append(0)  # Target frame
        deltas.extend([self.config.frame_gap * i for i in range(1, half_after + 1)])

        # Generate labels for each frame in the sequence (uniform target sampling with padding)
        stage_labels, progress_targets = [], []
        num_stages = len(global_names)

        for i in range(seq_len):
            if i == 0:
                # First frame is always the initial frame of episode
                current_frame = 0
                out_of_bounds = 0
            else:
                # Use centered deltas for frames 1-8
                offset = deltas[i - 1]
                raw_frame = frame_idx + offset - ep_start

                # Track out-of-bounds status for proper progress assignment
                if raw_frame < 0:
                    current_frame = 0
                    out_of_bounds = -1  # Before episode start
                elif raw_frame >= ep_length:
                    current_frame = ep_length - 1
                    out_of_bounds = 1  # After episode end
                else:
                    current_frame = raw_frame
                    out_of_bounds = 0

            # Assign progress based on out-of-bounds status
            if out_of_bounds == -1:
                # Before episode start: progress = 0, stage = 0
                stage_idx = 0
                progress = 0.0
            elif out_of_bounds == 1:
                # After episode end: progress = 1, stage = last stage
                stage_idx = num_stages - 1
                progress = 1.0
            else:
                # In bounds: compute normally
                stage_idx, progress = self._compute_progress_for_frame(
                    current_frame,
                    ep_length,
                    subtask_names,
                    subtask_start_frames,
                    subtask_end_frames,
                    global_names,
                    temporal_props,
                )

            stage_labels.append(stage_idx)
            progress_targets.append(progress)

        return torch.tensor(stage_labels, dtype=torch.long), torch.tensor(
            progress_targets, dtype=torch.float32
        ).unsqueeze(-1)

    def _generate_stage_and_progress_labels(
        self, frame_index, episode_index, video_features, annotation_type: str = "sparse"
    ):
        """Generate stage labels and progress targets (unified for all annotation modes).

        Args:
            frame_index: Current frame index or tensor of indices
            episode_index: Episode index or tensor of indices
            video_features: Video features tensor to determine sequence length
            annotation_type: "sparse" or "dense"

        Returns:
            Tuple of (stage_labels, progress_targets) with shapes (B, T) and (B, T, 1)
        """
        if episode_index is None:
            return None, None

        # Check if required proportions are available
        if annotation_type == "dense" and self.dense_temporal_proportions is None:
            return None, None
        if annotation_type == "sparse" and self.sparse_temporal_proportions is None:
            return None, None

        frame_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(frame_index)))
        episode_indices = self._get_episode_indices(frame_indices, episode_index)
        seq_len = video_features.shape[1] if video_features is not None and video_features.dim() >= 2 else 1

        # Only load episodes_df if we have annotations (not single_stage mode)
        episodes_df = None
        if annotation_type == "dense" or (
            annotation_type == "sparse" and self.sparse_subtask_names != ["task"]
        ):
            episodes_df = self.dataset_meta.episodes.to_pandas()

        all_stage_labels, all_progress_targets = [], []
        for ep_idx, frame_idx in zip(episode_indices.tolist(), frame_indices.tolist()):
            stage_labels, progress_targets = self._compute_labels_for_sample(
                int(frame_idx), int(ep_idx), seq_len, episodes_df, annotation_type
            )
            all_stage_labels.append(stage_labels)
            all_progress_targets.append(progress_targets)

        return torch.stack(all_stage_labels, dim=0), torch.stack(all_progress_targets, dim=0)

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
        """Encode images, text, and normalize states in the transition."""

        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)

        image = observation.get(self.image_key)

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # If 4D (T, C, H, W) from delta_timestamps, add batch dim
        # If 3D (C, H, W) single frame, add batch and time dims
        if image.ndim == 4:
            image = image[np.newaxis, ...]  # (T, C, H, W) -> (1, T, C, H, W)
        elif image.ndim == 3:
            image = image[np.newaxis, np.newaxis, ...]  # (C, H, W) -> (1, 1, C, H, W)

        video_features = self._encode_images_batch(image)
        observation["video_features"] = video_features

        # Extract state and pad to max_state_dim (already normalized by NormalizerProcessorStep)
        state_key = self.config.state_key
        state_data = observation.get(state_key)

        if isinstance(state_data, torch.Tensor):
            state_tensor = state_data.float()
        else:
            state_tensor = torch.tensor(state_data, dtype=torch.float32)

        # If 2D (T, state_dim) from delta_timestamps, add batch dim
        # If 1D (state_dim) single frame, add batch and time dims
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)  # (T, D) -> (1, T, D)
        elif state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (D,) -> (1, 1, D)

        observation["state_features"] = pad_state_to_max_dim(state_tensor, self.config.max_state_dim)

        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        # Get task description from dataset (complementary_data["task"])
        task = comp_data.get("task")
        if isinstance(task, list):
            # If batch, take first task (assuming same task for all items in batch)
            task = task[0] if task else ""

        # Encode text with CLIP
        batch_size = video_features.shape[0]
        observation["text_features"] = self._encode_text_clip(task, batch_size)

        frame_index = comp_data.get("index")
        episode_index = comp_data.get("episode_index")

        if frame_index is None:
            raise ValueError("Frame index ('index') not found in COMPLEMENTARY_DATA")
        if episode_index is None:
            raise ValueError("Episode index ('episode_index') not found in COMPLEMENTARY_DATA")

        # Compute episode metadata if dataset_meta is available
        if self.dataset_meta is not None:
            frame_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(frame_index)))
            episode_indices = self._get_episode_indices(frame_indices, episode_index)

            # Determine number of frames from video features
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

        # Generate sparse stage labels and progress targets from subtask annotations
        if self.sparse_temporal_proportions is not None and self.dataset_meta is not None:
            sparse_stage_labels, sparse_progress_targets = self._generate_stage_and_progress_labels(
                frame_index, episode_index, video_features, annotation_type="sparse"
            )
            if sparse_stage_labels is not None:
                observation["sparse_stage_labels"] = sparse_stage_labels
                observation["sparse_progress_targets"] = sparse_progress_targets

        # Generate dense stage labels and progress targets (for dual mode)
        if (
            self.config.uses_dual_heads
            and self.dense_temporal_proportions is not None
            and self.dataset_meta is not None
        ):
            dense_stage_labels, dense_progress_targets = self._generate_stage_and_progress_labels(
                frame_index, episode_index, video_features, annotation_type="dense"
            )
            if dense_stage_labels is not None:
                observation["dense_stage_labels"] = dense_stage_labels
                observation["dense_progress_targets"] = dense_progress_targets

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

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
