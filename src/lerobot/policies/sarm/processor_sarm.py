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
import torch
from PIL import Image
import pandas as pd
from transformers import CLIPModel, CLIPProcessor

from lerobot.processor.core import TransitionKey
from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.policies.sarm.sarm_utils import compute_tau, compute_cumulative_progress_batch, pad_state_to_max_dim
from lerobot.processor import (
    ProcessorStep,
    PolicyProcessorPipeline,
    PolicyAction,
    DeviceProcessorStep,
    AddBatchDimensionProcessorStep,
    NormalizerProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
    from_tensor_to_numpy,
)
from lerobot.processor.pipeline import PipelineFeatureType
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


class SARMEncodingProcessorStep(ProcessorStep):
    """ProcessorStep that encodes images and text with CLIP."""
    def __init__(
        self,
        config: SARMConfig,
        image_key: str | None = None,
        dataset_meta = None,
        dataset_stats: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.image_key = image_key or config.image_key
        self.dataset_meta = dataset_meta
        self.dataset_stats = dataset_stats
        self.temporal_proportions = {name: prop for name, prop in zip(self.config.subtask_names, self.config.temporal_proportions)}
        self.subtask_names = self.config.subtask_names

        self.device = torch.device(
            self.config.device if self.config.device 
            else "cuda" if torch.cuda.is_available() else "cpu"
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
    
    def _compute_absolute_indices(self, frame_idx: int, ep_start: int, num_frames: int) -> torch.Tensor:
        """Compute absolute frame indices for a sequence.
        
        (per SARM paper Section A.4):
        - Frame 0: Initial frame of the episode (ep_start)
        - Frames 1-8: 8 consecutive frames with frame_gap spacing ending at current frame
        Pattern: [ep_start, t-(7*gap), t-(6*gap), ..., t-gap, t]

        """
        indices = []
        indices.append(ep_start) # First frame is the episode's initial frame
            
        # Remaining frames are consecutive with frame_gap spacing
        num_consecutive = num_frames - 1
        for i in range(num_consecutive):
            offset = -(num_consecutive - 1 - i) * self.config.frame_gap
            idx = max(ep_start, frame_idx + offset)
            indices.append(idx)

        return torch.tensor(indices)
    
    def _compute_episode_metadata(
        self, 
        frame_indices: np.ndarray, 
        episode_indices: np.ndarray,
        num_frames: int,
    ) -> tuple[list | torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute episode metadata for all samples.
        
        Returns:
            Tuple of (absolute_frame_indices, remaining_lengths, episode_lengths)
        """
        absolute_indices_list = []
        remaining_lengths = []
        episode_lengths = []
        
        for ep_idx, frame_idx in zip(episode_indices.tolist(), frame_indices.tolist()):
            ep_idx, frame_idx = int(ep_idx), int(frame_idx)
            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
            
            episode_lengths.append(ep_end - ep_start)
            abs_indices = self._compute_absolute_indices(frame_idx, ep_start, num_frames)
            absolute_indices_list.append(abs_indices)
            remaining_lengths.append(ep_end - abs_indices[0].item())
        
        return absolute_indices_list, torch.tensor(remaining_lengths), torch.tensor(episode_lengths)
    
    def _compute_stage_and_progress_for_frame(
        self, 
        current_frame: int,
        subtask_names: list,
        subtask_start_frames: list,
        subtask_end_frames: list,
    ) -> tuple[int, float]:
        """Compute stage index and cumulative progress for a single frame.
        
        Implements SARM Paper Formula (2):
            y_t = P_{k-1} + ᾱ_k × τ_t
        
        where:
            - τ_t = (t - s_k) / (e_k - s_k) is within-subtask progress
            - P_{k-1} is cumulative prior (sum of previous subtask proportions)
            - ᾱ_k is the temporal proportion for subtask k
        
        Args:
            current_frame: Frame index relative to episode start
            subtask_names: List of subtask names for this episode
            subtask_start_frames: List of subtask start frames
            subtask_end_frames: List of subtask end frames
            
        Returns:
            Tuple of (stage_idx, cumulative_progress)
        """
        # Get temporal proportions as list for compute_cumulative_progress
        temporal_proportions_list = [
            self.temporal_proportions.get(name, 0.0) for name in self.subtask_names
        ]
        
        # Find which subtask this frame belongs to
        for j, (name, start_frame, end_frame) in enumerate(zip(subtask_names, subtask_start_frames, subtask_end_frames)):
            if current_frame >= start_frame and current_frame <= end_frame:
                # Found the subtask, get its global index
                stage_idx = self.subtask_names.index(name) if name in self.subtask_names else 0
                
                # Compute τ_t using utility function (Paper Formula 2)
                tau = compute_tau(current_frame, start_frame, end_frame)
                
                # Compute cumulative progress using utility function (Paper Formula 2)
                cumulative_progress = compute_cumulative_progress_batch(
                    tau, stage_idx, temporal_proportions_list
                )     
                return stage_idx, cumulative_progress
        
        # No matching subtask found
        if current_frame < subtask_start_frames[0]:
            return 0, 0.0
        elif current_frame > subtask_end_frames[-1]:
            return len(self.subtask_names) - 1, 1.0
        else:
            # Between subtasks - use previous subtask's end state (tau = 1.0)
            for j in range(len(subtask_names) - 1):
                if current_frame > subtask_end_frames[j] and current_frame < subtask_start_frames[j + 1]:
                    name = subtask_names[j]
                    stage_idx = self.subtask_names.index(name) if name in self.subtask_names else j
                    
                    # Completed subtask, so tau = 1.0
                    cumulative_progress = compute_cumulative_progress_batch(
                        1.0, stage_idx, temporal_proportions_list
                    )
                    return stage_idx, cumulative_progress
        
        return 0, 0.0
    
    def _compute_labels_for_sample(
        self,
        frame_idx: int,
        ep_idx: int,
        seq_len: int,
        episodes_df: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Compute stage labels and progress targets for a single sample.
        
        (per SARM paper Section A.4):
        - Frame 0: Initial frame of episode (stage at frame 0, progress at frame 0)
        - Frames 1-8: 8 consecutive frames with frame_gap spacing ending at current frame
        
        Args:
            frame_idx: The frame index for this sample
            ep_idx: The episode index
            seq_len: Number of frames in the sequence
            episodes_df: DataFrame with episode metadata
            
        Returns:
            Tuple of (stage_labels, progress_targets) tensors with shapes (T,) and (T, 1),
            or (None, None) if no valid annotations
        """
        # Check if episode has valid annotations
        if ep_idx >= len(episodes_df):
            return None, None
        
        subtask_names = episodes_df.loc[ep_idx, 'subtask_names']
        if subtask_names is None or (isinstance(subtask_names, float) and pd.isna(subtask_names)):
            return None, None
        
        subtask_start_frames = episodes_df.loc[ep_idx, 'subtask_start_frames']
        subtask_end_frames = episodes_df.loc[ep_idx, 'subtask_end_frames']
        ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
        
        # Generate labels for each frame in the sequence
        stage_labels = []
        progress_targets = []
        
        for i in range(seq_len):
            if i == 0:
                # Position 0: Initial frame of the episode
                current_frame = 0  # Relative to episode start
            else:
                # Positions 1-8: consecutive frames with frame_gap spacing
                num_consecutive = seq_len - 1
                offset = -(num_consecutive - i) * self.config.frame_gap 
                current_frame = max(0, frame_idx + offset - ep_start)

            
            stage_idx, cumulative_progress = self._compute_stage_and_progress_for_frame(
                current_frame, subtask_names, subtask_start_frames, subtask_end_frames
            )
            
            stage_labels.append(stage_idx)
            progress_targets.append(cumulative_progress)
        
        stage_labels = torch.tensor(stage_labels, dtype=torch.long)
        progress_targets = torch.tensor(progress_targets, dtype=torch.float32).unsqueeze(-1)
        
        return stage_labels, progress_targets
    
    def _generate_stage_and_progress_labels(self, frame_index, episode_index, video_features):
        """Generate stage labels and refined progress targets from subtask annotations.
        
        Args:
            frame_index: Current frame index or tensor of indices
            episode_index: Episode index or tensor of indices  
            video_features: Video features tensor to determine sequence length
            
        Returns:
            Tuple of (stage_labels, progress_targets) or (None, None) if no annotations.
        """
        if self.temporal_proportions is None or episode_index is None:
            return None, None
        
        # Normalize inputs to numpy arrays
        frame_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(frame_index)))
        episode_indices = self._get_episode_indices(frame_indices, episode_index)
        
        # Determine sequence length
        if video_features is not None and video_features.dim() >= 2:
            seq_len = video_features.shape[1]
        else:
            seq_len = 1
        
        episodes_df = self.dataset_meta.episodes.to_pandas()
        
        all_stage_labels = []
        all_progress_targets = []
        
        for ep_idx, frame_idx in zip(episode_indices.tolist(), frame_indices.tolist()):
            result = self._compute_labels_for_sample(int(frame_idx), int(ep_idx), seq_len, episodes_df)
            
            if result[0] is None:
                all_stage_labels.append(torch.zeros(seq_len, dtype=torch.long))
                all_progress_targets.append(torch.zeros(seq_len, 1, dtype=torch.float32))
            else:
                all_stage_labels.append(result[0])
                all_progress_targets.append(result[1])
        
        return torch.stack(all_stage_labels, dim=0), torch.stack(all_progress_targets, dim=0)
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Encode images, text, and normalize states in the transition."""

        new_transition = transition.copy() if hasattr(transition, 'copy') else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)
        
        image = observation.get(self.image_key)
        
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        video_features = self._encode_images_batch(image)
        observation['video_features'] = video_features
        
        # Extract state and pad to max_state_dim (already normalized by NormalizerProcessorStep)
        state_key = self.config.state_key
        state_data = observation.get(state_key)
        
        if isinstance(state_data, torch.Tensor):
            state_tensor = state_data.float()
        else:
            state_tensor = torch.tensor(state_data, dtype=torch.float32)
        
        observation['state_features'] = pad_state_to_max_dim(state_tensor, self.config.max_state_dim)
        
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        
        # Get task description from dataset (complementary_data["task"])
        task = comp_data.get('task')
        if isinstance(task, list):
            # If batch, take first task (assuming same task for all items in batch)
            task = task[0] if task else ""
        
        # Encode text with CLIP
        batch_size = video_features.shape[0]
        observation['text_features'] = self._encode_text_clip(task, batch_size)
        
        frame_index = comp_data.get('index')
        episode_index = comp_data.get('episode_index')
        
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
            
            abs_indices, remaining, ep_lengths = self._compute_episode_metadata(
                frame_indices, episode_indices, num_frames
            )
            observation['absolute_frame_indices'] = abs_indices
            observation['remaining_length'] = remaining
            observation['episode_length'] = ep_lengths
        
        # Generate stage labels and progress targets from subtask annotations
        if self.temporal_proportions is not None and self.dataset_meta is not None:
            stage_labels, progress_targets = self._generate_stage_and_progress_labels(
                frame_index, episode_index, video_features
            )
            if stage_labels is not None:
                observation['stage_labels'] = stage_labels
                observation['progress_targets'] = progress_targets
        
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
        
        # Encode each batch
        all_embeddings = []
        for i in range(0, num_frames, self.config.clip_batch_size):
            batch_imgs = images_list[i:i + self.config.clip_batch_size]
            
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
        # Use CLIP's tokenizer directly for text
        tokenizer = self.clip_processor.tokenizer
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get text features from CLIP
        text_embedding = self.clip_model.get_text_features(**inputs).detach().cpu()
        
        # Replicate for batch (B, 512)
        text_embedding = text_embedding.expand(batch_size, -1)
        
        return text_embedding
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Add encoded features to the observation features."""
        features[PipelineFeatureType.OBSERVATION]['video_features'] = PolicyFeature(
            type=FeatureType.VISUAL, 
            shape=(self.config.num_frames, self.config.image_dim)
        )
        features[PipelineFeatureType.OBSERVATION]['text_features'] = PolicyFeature(
            type=FeatureType.LANGUAGE, 
            shape=(self.config.text_dim,)
        )
        features[PipelineFeatureType.OBSERVATION]['state_features'] = PolicyFeature(
            type=FeatureType.STATE, 
            shape=(self.config.num_frames, self.config.max_state_dim)
        )
        return features


def make_sarm_pre_post_processors(
    config: SARMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta = None,
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
                        config=config,
                        dataset_meta=dataset_meta,
                        dataset_stats=dataset_stats
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
