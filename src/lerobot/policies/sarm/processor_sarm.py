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

import logging
from typing import Any
import numpy as np
import torch
from PIL import Image
import pandas as pd
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.processor import (
    ProcessorStep,
    PolicyProcessorPipeline,
    PolicyAction,
    DeviceProcessorStep,
    AddBatchDimensionProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import PipelineFeatureType
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


class SARMEncodingProcessorStep(ProcessorStep):
    """
    ProcessorStep that encodes images and text for SARM training.
    
    This step handles:
    - CLIP (image) encoding
    - MiniLM (text) encoding
    - Joint state normalization
    
    Supports temporal sequences: (B, T, C, H, W) → (B, T, 512) video features
    """
    
    def __init__(
        self,
        config: SARMConfig,
        image_key: str | None = None,
        task_description: str | None = None,
        dataset_meta = None,
        dataset_stats: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.image_key = image_key or config.image_key
        self.task_description = task_description or config.task_description
        self.dataset_meta = dataset_meta
        self.dataset_stats = dataset_stats
        
        # Compute temporal proportions from subtask annotations if available
        self.temporal_proportions = None
        self.subtask_names = None
        if dataset_meta is not None:
            self._compute_temporal_proportions()
        
        self._init_encoders()
    
    def _init_encoders(self):
        """Initialize CLIP and MiniLM encoders."""
        device = torch.device(
            self.config.device if self.config.device 
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logging.info("Initializing CLIP encoder for SARM...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.clip_model.to(device)
        self.clip_model.eval()
        
        logging.info("Initializing MiniLM encoder for SARM...")
        self.minilm_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.minilm_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.minilm_model.to(device)
        self.minilm_model.eval()
        
        self.device = device
    
    def _compute_temporal_proportions(self):
        """Compute temporal proportions for each subtask from dataset annotations."""
        if self.dataset_meta is None or not hasattr(self.dataset_meta, 'episodes'):
            return
        
        # Check if subtask annotations exist
        episodes = self.dataset_meta.episodes
        if episodes is None or len(episodes) == 0:
            return
        
        # Check for subtask_names column
        if 'subtask_names' not in episodes.column_names:
            logging.info("No subtask annotations found in dataset")
            return
        
        # Convert to pandas
        episodes_df = episodes.to_pandas()
        
        # Collect all subtask names and compute average durations
        subtask_durations = {}
        all_subtask_names = set()
        
        for ep_idx in episodes_df.index:
            subtask_names = episodes_df.loc[ep_idx, 'subtask_names']
            
            # Skip episodes without annotations
            if subtask_names is None or (isinstance(subtask_names, float) and pd.isna(subtask_names)):
                continue
            
            start_times = episodes_df.loc[ep_idx, 'subtask_start_times']
            end_times = episodes_df.loc[ep_idx, 'subtask_end_times']
            
            # Track unique subtask names
            all_subtask_names.update(subtask_names)
            
            # Compute durations
            for i, name in enumerate(subtask_names):
                duration = end_times[i] - start_times[i]
                if name not in subtask_durations:
                    subtask_durations[name] = []
                subtask_durations[name].append(duration)
        
        if not all_subtask_names:
            logging.info("No valid subtask annotations found")
            return
        
        # Sort subtask names for consistent ordering
        self.subtask_names = sorted(list(all_subtask_names))
        self.config.num_stages = len(self.subtask_names)
        self.config.subtask_names = self.subtask_names  # Store in config for reference
        
        # Compute average duration for each subtask
        avg_durations = {}
        for name in self.subtask_names:
            if name in subtask_durations:
                avg_durations[name] = np.mean(subtask_durations[name])
            else:
                avg_durations[name] = 0.0
        
        # Normalize to get proportions
        total_duration = sum(avg_durations.values())
        if total_duration > 0:
            self.temporal_proportions = {
                name: avg_durations[name] / total_duration 
                for name in self.subtask_names
            }
        else:
            # Equal proportions if no duration info
            self.temporal_proportions = {
                name: 1.0 / len(self.subtask_names) 
                for name in self.subtask_names
            }
        
        logging.info(f"Computed temporal proportions for {len(self.subtask_names)} subtasks: {self.temporal_proportions}")
    
    def _to_numpy_array(self, x) -> np.ndarray:
        """Convert input to a 1D numpy array."""
        if isinstance(x, torch.Tensor):
            arr = x.cpu().numpy()
        else:
            arr = np.array(x)
        if arr.ndim == 0:
            arr = np.array([arr.item()])
        return arr
    
    def _find_episode_for_frame(self, frame_idx: int) -> int:
        """Find the episode index for a given frame index."""
        for ep_idx in range(len(self.dataset_meta.episodes)):
            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
            if ep_start <= frame_idx < ep_end:
                return ep_idx
        return 0  # Fallback
    
    def _get_episode_indices(self, frame_indices: np.ndarray, episode_index) -> np.ndarray:
        """Get episode indices for each frame index."""
        if episode_index is None:
            return np.array([self._find_episode_for_frame(int(f)) for f in frame_indices])
        
        episode_indices = self._to_numpy_array(episode_index)
        
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
        frame_gap = getattr(self.config, 'frame_gap', 1)
        
        indices = []
        
        
        # First frame is the episode's initial frame
        indices.append(ep_start)
            
        # Remaining frames are consecutive with frame_gap spacing
        num_consecutive = num_frames - 1
        for i in range(num_consecutive):
            offset = -(num_consecutive - 1 - i) * frame_gap
            idx = max(ep_start, frame_idx + offset)
            indices.append(idx)

        
        return torch.tensor(indices)
    
    def _compute_episode_metadata(
        self, 
        frame_indices: np.ndarray, 
        episode_indices: np.ndarray,
        num_frames: int,
        is_batch: bool,
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
        
        if is_batch:
            return absolute_indices_list, torch.tensor(remaining_lengths), torch.tensor(episode_lengths)
        else:
            return absolute_indices_list[0], remaining_lengths[0], episode_lengths[0]
    
    def _compute_stage_and_progress_for_frame(
        self, 
        current_frame: int,
        subtask_names: list,
        subtask_start_frames: list,
        subtask_end_frames: list,
    ) -> tuple[int, float]:
        """Compute stage index and cumulative progress for a single frame.
        
        Args:
            current_frame: Frame index relative to episode start
            subtask_names: List of subtask names for this episode
            subtask_start_frames: List of subtask start frames
            subtask_end_frames: List of subtask end frames
            
        Returns:
            Tuple of (stage_idx, cumulative_progress)
        """
        stage_idx = -1
        cumulative_progress = 0.0
        
        # Find which subtask this frame belongs to
        for j, (name, start_frame, end_frame) in enumerate(zip(subtask_names, subtask_start_frames, subtask_end_frames)):
            if current_frame >= start_frame and current_frame <= end_frame:
                # Found the subtask
                stage_idx = self.subtask_names.index(name) if name in self.subtask_names else 0
                
                # Calculate within-subtask progress
                subtask_duration = end_frame - start_frame
                if subtask_duration > 0:
                    within_subtask_progress = (current_frame - start_frame) / subtask_duration
                else:
                    within_subtask_progress = 1.0
                
                # Calculate cumulative progress from completed subtasks
                for k in range(j):
                    prev_name = subtask_names[k]
                    if prev_name in self.temporal_proportions:
                        cumulative_progress += self.temporal_proportions[prev_name]
                
                # Add current subtask's partial progress
                if name in self.temporal_proportions:
                    cumulative_progress += self.temporal_proportions[name] * within_subtask_progress
                
                return stage_idx, cumulative_progress
        
        # No matching subtask found - estimate based on position
        if current_frame < subtask_start_frames[0]:
            return 0, 0.0
        elif current_frame > subtask_end_frames[-1]:
            return len(self.subtask_names) - 1, 1.0
        else:
            # Between subtasks - use previous subtask's end state
            for j in range(len(subtask_names) - 1):
                if current_frame > subtask_end_frames[j] and current_frame < subtask_start_frames[j + 1]:
                    name = subtask_names[j]
                    stage_idx = self.subtask_names.index(name) if name in self.subtask_names else j
                    # Sum up all completed subtasks
                    for k in range(j + 1):
                        prev_name = subtask_names[k]
                        if prev_name in self.temporal_proportions:
                            cumulative_progress += self.temporal_proportions[prev_name]
                    return stage_idx, cumulative_progress
        
        return 0, 0.0  # Fallback
    
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
        
        # Get episode boundaries
        ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
        
        # Get config values
        frame_gap = self.config.frame_gap if hasattr(self.config, 'frame_gap') else 1
        
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
                offset = -(num_consecutive - i) * frame_gap
                current_frame = max(0, frame_idx + offset - ep_start)

            
            stage_idx, cumulative_progress = self._compute_stage_and_progress_for_frame(
                current_frame, subtask_names, subtask_start_frames, subtask_end_frames
            )
            
            stage_labels.append(stage_idx)
            progress_targets.append(cumulative_progress)
        
        # Convert to tensors
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
        
        is_batch = isinstance(frame_index, torch.Tensor) and frame_index.numel() > 1
        
        # Normalize inputs to numpy arrays
        frame_indices = self._to_numpy_array(frame_index)
        episode_indices = self._get_episode_indices(frame_indices, episode_index)
        
        # Determine sequence length
        if video_features is not None and video_features.dim() >= 2:
            seq_len = video_features.shape[1] if is_batch else video_features.shape[0]
        else:
            seq_len = 1
        
        episodes_df = self.dataset_meta.episodes.to_pandas()
        
        # Process all samples
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
        
        if is_batch:
            return torch.stack(all_stage_labels, dim=0), torch.stack(all_progress_targets, dim=0)
        return all_stage_labels[0], all_progress_targets[0]
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Encode images, text, and normalize states in the transition."""
        from lerobot.processor.core import TransitionKey
        
        new_transition = transition.copy() if hasattr(transition, 'copy') else dict(transition)
        
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a dictionary")
        
        # 1. Encode images with CLIP
        image = observation.get(self.image_key)
        if image is None:
            raise ValueError(f"Image not found in observation for key: {self.image_key}")
        
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        video_features = self._encode_images_batch(image)
        observation['video_features'] = video_features
        
        # 2. Extract and normalize joint states
        state_data = observation.get("state") or observation.get("observation.state")
        if state_data is None:
            raise ValueError("State data not found in observation (expected 'state' or 'observation.state')")
        
        if isinstance(state_data, torch.Tensor):
            state_data = state_data.cpu().numpy()
        
        state_key = "state" if "state" in observation else "observation.state"
        if self.dataset_stats and state_key in self.dataset_stats:
            mean = self.dataset_stats[state_key]['mean']
            std = self.dataset_stats[state_key]['std']
            state_data = (state_data - mean) / (std + 1e-8)
        
        observation['state_features'] = torch.tensor(state_data, dtype=torch.float32)
        
        # 3. Encode text with MiniLM
        batch_size = video_features.shape[0]
        task_descriptions = new_transition.get('task')
        if task_descriptions is not None:
            if isinstance(task_descriptions, str):
                task_descriptions = [task_descriptions] * batch_size
            observation['text_features'] = self._encode_text_batch_list(task_descriptions)
        else:
            observation['text_features'] = self._encode_text_batch(self.task_description, batch_size)
        
        # 4. Extract frame/episode indices from complementary data
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        if not isinstance(comp_data, dict):
            raise ValueError("COMPLEMENTARY_DATA must be a dictionary")
        
        frame_index = comp_data.get('index')
        episode_index = comp_data.get('episode_index')
        
        if frame_index is None:
            raise ValueError("Frame index ('index') not found in COMPLEMENTARY_DATA")
        if episode_index is None:
            raise ValueError("Episode index ('episode_index') not found in COMPLEMENTARY_DATA")
        
        # 5. Compute episode metadata if dataset_meta is available
        if self.dataset_meta is not None:
            is_batch = isinstance(frame_index, torch.Tensor) and frame_index.numel() > 1
            frame_indices = self._to_numpy_array(frame_index)
            episode_indices = self._get_episode_indices(frame_indices, episode_index)
            
            # Determine number of frames from video features
            if video_features.dim() >= 2:
                num_frames = video_features.shape[1] if is_batch else video_features.shape[0]
            else:
                num_frames = 1
            
            abs_indices, remaining, ep_lengths = self._compute_episode_metadata(
                frame_indices, episode_indices, num_frames, is_batch
            )
            observation['absolute_frame_indices'] = abs_indices
            observation['remaining_length'] = remaining
            observation['episode_length'] = ep_lengths
        
        # 6. Generate stage labels and progress targets from subtask annotations
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
            images: Batched images with shape:
                   - (B, C, H, W) for single frames, or
                   - (B, T, C, H, W) for temporal sequences
            
        Returns:
            Encoded feature vectors with shape (B, 512) or (B, T, 512)
        """
        # Check if we have temporal dimension
        has_temporal = len(images.shape) == 5
        
        if has_temporal:
            # Shape: (B, T, C, H, W)
            batch_size, seq_length = images.shape[0], images.shape[1]
            
            # Reshape to (B*T, C, H, W) to process all frames at once
            images = images.reshape(batch_size * seq_length, *images.shape[2:])
        elif len(images.shape) == 4:
            # Shape: (B, C, H, W)
            batch_size = images.shape[0]
            seq_length = 1
        else:
            raise ValueError(f"Expected 4D (B, C, H, W) or 5D (B, T, C, H, W) input, got shape {images.shape}")
        
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
        
        # Reshape back if temporal
        if has_temporal:
            all_embeddings = all_embeddings.reshape(batch_size, seq_length, -1)  # (B, T, 512)
        
        return all_embeddings
    
    @torch.no_grad()
    def _encode_text_batch(self, text: str, batch_size: int) -> torch.Tensor:
        """Encode a text string using MiniLM and replicate for batch.
        
        Args:
            text: Text string to encode
            batch_size: Batch size to replicate for
            
        Returns:
            Encoded feature vectors with shape (B, 384)
        """
        from lerobot.policies.rewind.modeling_rewind import mean_pooling
        
        encoded_input = self.minilm_tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        
        model_output = self.minilm_model(**encoded_input)
        text_embedding = mean_pooling(model_output, encoded_input["attention_mask"])
        text_embedding = text_embedding.squeeze().cpu()
        
        # Replicate for batch (B, 384)
        text_embedding = text_embedding.unsqueeze(0).repeat(batch_size, 1)
        
        return text_embedding
    
    @torch.no_grad()
    def _encode_text_batch_list(self, text_list: list[str]) -> torch.Tensor:
        """Encode a list of text strings using MiniLM.
        
        Args:
            text_list: List of text strings to encode
            
        Returns:
            Encoded feature vectors with shape (B, 384)
        """
        from lerobot.policies.rewind.modeling_rewind import mean_pooling
        
        # Encode all texts in the batch at once
        encoded_input = self.minilm_tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        
        model_output = self.minilm_model(**encoded_input)
        text_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        text_embeddings = text_embeddings.cpu()
        
        return text_embeddings
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Add encoded features to the observation features."""
        # Add the encoded features
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
            shape=(self.config.num_frames, self.config.state_dim)
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
    1. Encodes images with CLIP (512-dim)
    2. Encodes text with MiniLM (384-dim)
    3. Normalizes joint states
    4. Adds batch dimension
    5. Moves data to device
    
    Args:
        config: SARM configuration
        dataset_stats: Dataset statistics for normalization
        dataset_meta: Dataset metadata for computing episode info
    
    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """
    input_steps = [
        AddBatchDimensionProcessorStep(),
        SARMEncodingProcessorStep(
            config=config,
            dataset_meta=dataset_meta,
            dataset_stats=dataset_stats
        ),
        DeviceProcessorStep(device=config.device),
    ]
    
    output_steps = [
        DeviceProcessorStep(device="cpu"),
    ]
    
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )



