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
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from PIL import Image

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
        
        # Initialize encoders
        self._init_encoders()
    
    def _init_encoders(self):
        """Initialize CLIP and MiniLM encoders."""
        from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
        
        device = torch.device(
            self.config.device if self.config.device 
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logging.info("Initializing CLIP encoder for SARM...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Encode images, text, and normalize states in the transition."""
        from lerobot.processor.core import TransitionKey
        
        self._current_transition = transition.copy() if hasattr(transition, 'copy') else dict(transition)
        new_transition = self._current_transition
        
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            return new_transition
        
        # Extract and encode images
        batch_size = 1
        if self.image_key in observation:
            image = observation[self.image_key]
            
            # Handle different image formats
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            # Encode images
            video_features = self._encode_images_batch(image)
            observation['video_features'] = video_features
            
            # Get batch size from encoded features
            batch_size = video_features.shape[0]
        
        # Extract and normalize joint states
        if self.config.use_joint_state:
            # Look for "state" or "observation.state" in observation
            state_key = None
            state_data = None
            
            if "state" in observation:
                state_key = "state"
                state_data = observation["state"]
            elif "observation.state" in observation:
                state_key = "observation.state"
                state_data = observation["observation.state"]
            
            if state_data is not None:
                if isinstance(state_data, torch.Tensor):
                    state_data = state_data.cpu().numpy()
                
                # Normalize if stats available
                if self.dataset_stats and state_key in self.dataset_stats:
                    mean = self.dataset_stats[state_key]['mean']
                    std = self.dataset_stats[state_key]['std']
                    state_data = (state_data - mean) / (std + 1e-8)
                
                observation['state_features'] = torch.tensor(state_data, dtype=torch.float32)
            else:
                # Create dummy state features if not found
                if 'video_features' in observation:
                    num_frames = observation['video_features'].shape[0] if observation['video_features'].dim() == 2 else observation['video_features'].shape[1]
                    observation['state_features'] = torch.zeros(batch_size, num_frames, self.config.state_dim)
        
        # Get task descriptions
        task_descriptions = None
        if 'task' in new_transition:
            task_descriptions = new_transition['task']
            if isinstance(task_descriptions, str):
                task_descriptions = [task_descriptions] * batch_size
        
        # Encode text
        if task_descriptions is not None:
            text_features = self._encode_text_batch_list(task_descriptions)
        else:
            text_features = self._encode_text_batch(self.task_description, batch_size)
        
        observation['text_features'] = text_features
        
        # Compute episode metadata for progress normalization
        # Note: Processor runs BEFORE batching, so we need to extract from raw dataset structure
        # The dataset provides episode_index and index in the raw item
        
        # Extract index and episode_index from COMPLEMENTARY_DATA
        episode_index = None
        frame_index = None
        
        # Primary location: COMPLEMENTARY_DATA (confirmed from debug logs)
        if TransitionKey.COMPLEMENTARY_DATA in new_transition:
            comp_data = new_transition[TransitionKey.COMPLEMENTARY_DATA]
            if isinstance(comp_data, dict):
                frame_index = comp_data.get('index')
                episode_index = comp_data.get('episode_index')
        
        # Fallback: check other locations
        if frame_index is None and TransitionKey.OBSERVATION in new_transition:
            obs = new_transition[TransitionKey.OBSERVATION]
            if isinstance(obs, dict):
                frame_index = obs.get('index')
                if episode_index is None:
                    episode_index = obs.get('episode_index')
        
        # If we have frame_index but no episode_index, compute it from episode boundaries
        if frame_index is not None and episode_index is None and self.dataset_meta is not None:
            # Convert to int if needed
            if isinstance(frame_index, torch.Tensor):
                frame_idx = frame_index.item() if frame_index.numel() == 1 else frame_index[0].item()
            else:
                frame_idx = int(frame_index)
            
            # Search through episodes to find which one this frame belongs to
            for ep_idx in range(len(self.dataset_meta.episodes)):
                ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
                ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
                if ep_start <= frame_idx < ep_end:
                    episode_index = ep_idx
                    break
        
        if self.dataset_meta is not None and frame_index is not None:
            # Handle batch processing
            is_batch = isinstance(frame_index, torch.Tensor) and frame_index.numel() > 1
            
            if is_batch:
                # Batch case: process multiple samples at once
                batch_size = frame_index.shape[0]
                frame_indices = frame_index.cpu().numpy() if isinstance(frame_index, torch.Tensor) else np.array(frame_index)
                
                # Ensure at least 1D
                if frame_indices.ndim == 0:
                    frame_indices = np.array([frame_indices.item()])
                
                # Compute episode_index for each frame if not provided
                if episode_index is None:
                    episode_indices = []
                    for frame_idx in frame_indices:
                        frame_idx = int(frame_idx)
                        # Search through episodes
                        found = False
                        for ep_idx in range(len(self.dataset_meta.episodes)):
                            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
                            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
                            if ep_start <= frame_idx < ep_end:
                                episode_indices.append(ep_idx)
                                found = True
                                break
                        if not found:
                            episode_indices.append(0)  # Fallback
                    episode_indices = np.array(episode_indices)
                else:
                    episode_indices = episode_index.cpu().numpy() if isinstance(episode_index, torch.Tensor) else np.array(episode_index)
                    # Ensure at least 1D
                    if episode_indices.ndim == 0:
                        episode_indices = np.array([episode_indices.item()])
                    
                    # CRITICAL FIX: If we have a single episode_index but multiple frame_indices,
                    # compute the correct episode for each frame (they might be from different episodes)
                    if len(episode_indices) == 1 and len(frame_indices) > 1:
                        episode_indices = []
                        for frame_idx in frame_indices:
                            frame_idx = int(frame_idx)
                            # Search through episodes
                            found = False
                            for ep_idx in range(len(self.dataset_meta.episodes)):
                                ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
                                ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
                                if ep_start <= frame_idx < ep_end:
                                    episode_indices.append(ep_idx)
                                    found = True
                                    break
                            if not found:
                                episode_indices.append(0)  # Fallback
                        episode_indices = np.array(episode_indices)
                
                # Compute metadata for each sample in batch
                absolute_indices_list = []
                remaining_lengths = []
                episode_lengths = []
                
                # Convert to list for safe iteration
                episode_indices_list = episode_indices.tolist() if hasattr(episode_indices, 'tolist') else list(episode_indices)
                frame_indices_list = frame_indices.tolist() if hasattr(frame_indices, 'tolist') else list(frame_indices)
                
                for i, (ep_idx, frame_idx) in enumerate(zip(episode_indices_list, frame_indices_list)):
                    ep_idx = int(ep_idx)
                    frame_idx = int(frame_idx)
                    ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
                    ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
                    episode_length = ep_end - ep_start
                    episode_lengths.append(episode_length)
                    
                    # Compute absolute indices for this sample
                    if 'video_features' in observation and observation['video_features'].dim() > 1:
                        num_loaded_frames = observation['video_features'].shape[1]  # (batch, seq_len, features)
                        frame_gap = self.config.frame_gap if hasattr(self.config, 'frame_gap') else 1
                        
                        if frame_gap > 1:
                            absolute_indices = []
                            for j in range(num_loaded_frames):
                                offset = -(num_loaded_frames - 1 - j) * frame_gap
                                idx = max(ep_start, frame_idx + offset)
                                absolute_indices.append(idx)
                            absolute_indices = torch.tensor(absolute_indices)
                        else:
                            start_idx = max(ep_start, frame_idx - num_loaded_frames + 1)
                            absolute_indices = torch.arange(start_idx, frame_idx + 1)
                        
                        absolute_indices_list.append(absolute_indices)
                        remaining_lengths.append(ep_end - absolute_indices[0].item())
                    else:
                        absolute_indices_list.append(torch.tensor([frame_idx]))
                        remaining_lengths.append(ep_end - frame_idx)
                
                observation['absolute_frame_indices'] = absolute_indices_list
                observation['remaining_length'] = torch.tensor(remaining_lengths)
                observation['episode_length'] = torch.tensor(episode_lengths)
            else:
                # Single sample case
                if isinstance(frame_index, torch.Tensor):
                    frame_idx = frame_index.item()
                else:
                    frame_idx = int(frame_index)
                
                # Get episode_index
                if episode_index is None:
                    # Search through episodes
                    for ep_idx in range(len(self.dataset_meta.episodes)):
                        ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
                        ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
                        if ep_start <= frame_idx < ep_end:
                            episode_index = ep_idx
                            break
                    if episode_index is None:
                        episode_index = 0  # Fallback
                
                ep_idx = int(episode_index) if not isinstance(episode_index, int) else episode_index
                ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
                ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
                episode_length = ep_end - ep_start
                
                # Compute absolute indices
                if 'video_features' in observation and observation['video_features'].dim() > 0:
                    num_loaded_frames = observation['video_features'].shape[0]
                    frame_gap = self.config.frame_gap if hasattr(self.config, 'frame_gap') else 1
                    
                    if frame_gap > 1:
                        absolute_indices = []
                        for i in range(num_loaded_frames):
                            offset = -(num_loaded_frames - 1 - i) * frame_gap
                            idx = max(ep_start, frame_idx + offset)
                            absolute_indices.append(idx)
                        absolute_indices = torch.tensor(absolute_indices)
                    else:
                        start_idx = max(ep_start, frame_idx - num_loaded_frames + 1)
                        absolute_indices = torch.arange(start_idx, frame_idx + 1)
                    
                    observation['absolute_frame_indices'] = absolute_indices
                    observation['remaining_length'] = ep_end - absolute_indices[0].item()
                else:
                    observation['absolute_frame_indices'] = torch.tensor([frame_idx])
                    observation['remaining_length'] = ep_end - frame_idx
                
                observation['episode_length'] = episode_length
        
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
            inputs = self.clip_processor(images=batch_imgs, return_tensors="pt", padding=True)
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
        if self.config.use_joint_state:
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

