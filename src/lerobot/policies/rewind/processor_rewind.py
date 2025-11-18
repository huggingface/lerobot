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

from lerobot.policies.rewind.configuration_rewind import ReWiNDConfig
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

class ReWiNDEncodingProcessorStep(ProcessorStep):
    """
    ProcessorStep that encodes images and text for ReWiND training.
    
    This step handles the DINO (image) and MiniLM (text) encoding that ReWiND needs.
    
    Supports both single-frame and temporal sequence encoding:
    - Single frame: (B, C, H, W) → (B, 768) video features
    - Temporal sequence: (B, T, C, H, W) → (B, T, 768) video features
    
    To use temporal sequences, configure the dataset with delta_timestamps for your image key.
    For example, to encode sequences of 32 frames:
        delta_timestamps = {
            "observation.images.top": [i / fps for i in range(-15, 17)]  # 32 frames centered on current
        }
    """
    
    def __init__(
        self,
        config: ReWiNDConfig,
        image_key: str | None = None,
        task_description: str | None = None,
        dataset_meta = None,
    ):
        super().__init__()
        self.config = config
        self.image_key = image_key or config.image_key
        self.task_description = task_description or config.task_description
        self.dataset_meta = dataset_meta  # Store dataset metadata for episode info
        
        # Initialize encoders
        self._init_encoders()
    
    def _init_encoders(self):
        """Initialize DINO and MiniLM encoders."""
        from transformers import AutoModel, AutoTokenizer
        
        device = torch.device(
            self.config.device if self.config.device 
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logging.info("Initializing DINO encoder for ReWiND...")
        self.dino_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.dino_encoder.to(device)
        self.dino_encoder.eval()
        
        logging.info("Initializing MiniLM encoder for ReWiND...")
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
        """Encode images and text in the transition."""
        self._current_transition = transition.copy() if hasattr(transition, 'copy') else dict(transition)
        new_transition = self._current_transition
        
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            # If no observation, just return the transition as-is
            return new_transition
        
        # Extract images from observation and encode
        # For ReWiND, we need to load the sequence from episode start to current frame
        batch_size = 1
        if self.image_key in observation:
            image = observation[self.image_key]
            
            # Handle different image formats
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            # Check if we have temporal sequences or single frames
            # Temporal sampling: Load from episode start to current frame
            # This will be handled by the dataset if configured with delta_timestamps
            # Otherwise, we just encode the single frame
            video_features = self._encode_images_batch(image)
            observation['video_features'] = video_features
            
            # Get batch size from the encoded features
            batch_size = video_features.shape[0]
        
        # Get task descriptions - check if 'task' field exists in the transition
        # This allows per-episode task descriptions (e.g., for datasets with multiple tasks)
        task_descriptions = None
        if 'task' in new_transition:
            task_descriptions = new_transition['task']
            # Convert to list if it's a single string
            if isinstance(task_descriptions, str):
                task_descriptions = [task_descriptions] * batch_size
        
        # Encode text
        if task_descriptions is not None:
            # Encode per-sample task descriptions
            text_features = self._encode_text_batch_list(task_descriptions)
        else:
            # Fall back to config task description if no task field in transition
            text_features = self._encode_text_batch(self.task_description, batch_size)
        
        observation['text_features'] = text_features
        
        # Compute episode metadata for progress normalization (used by ReWiND)
        # We need to pass absolute frame indices and total episode length for correct progress calculation
        if self.dataset_meta is not None and 'episode_index' in new_transition and 'index' in new_transition:
            episode_indices = new_transition['episode_index']
            frame_indices = new_transition['index']
            
            # Handle both single samples and batches
            if isinstance(episode_indices, (int, np.integer)):
                ep_idx = int(episode_indices)
                frame_idx = int(frame_indices)
                ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
                ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
                episode_length = ep_end - ep_start
                
                # For temporal sequences with observation_delta_indices:
                # If we loaded frames using delta_indices (e.g., [-31, -30, ..., 0]),
                # we need to compute the absolute indices of those frames
                # The current frame is at frame_idx, and we loaded max_length frames before it
                if 'video_features' in observation and len(observation['video_features'].shape) > 1:
                    # We have a temporal sequence
                    num_loaded_frames = observation['video_features'].shape[0] if observation['video_features'].dim() == 2 else observation['video_features'].shape[1]
                    # Absolute indices: from (frame_idx - num_frames + 1) to frame_idx
                    start_idx = max(ep_start, frame_idx - num_loaded_frames + 1)
                    absolute_indices = torch.arange(start_idx, frame_idx + 1)
                    observation['absolute_frame_indices'] = absolute_indices
                    # Compute remaining length: from first loaded frame to episode end
                    observation['remaining_length'] = ep_end - start_idx
                else:
                    # Single frame
                    observation['absolute_frame_indices'] = torch.tensor([frame_idx])
                    # Remaining length from this frame to episode end
                    observation['remaining_length'] = ep_end - frame_idx
                
                observation['episode_length'] = episode_length
            else:
                # Batch case
                absolute_indices_list = []
                episode_lengths = []
                remaining_lengths = []
                for ep_idx, frame_idx in zip(episode_indices, frame_indices):
                    ep_idx = int(ep_idx.item() if hasattr(ep_idx, 'item') else ep_idx)
                    frame_idx = int(frame_idx.item() if hasattr(frame_idx, 'item') else frame_idx)
                    ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
                    ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
                    episode_length = ep_end - ep_start
                    episode_lengths.append(episode_length)
                    
                    # Compute absolute indices for this sample
                    if 'video_features' in observation and len(observation['video_features'].shape) > 1:
                        num_loaded_frames = observation['video_features'].shape[1]
                        start_idx = max(ep_start, frame_idx - num_loaded_frames + 1)
                        absolute_indices = torch.arange(start_idx, frame_idx + 1)
                        absolute_indices_list.append(absolute_indices)
                        # Remaining length from first loaded frame to episode end
                        remaining_lengths.append(ep_end - start_idx)
                    else:
                        absolute_indices_list.append(torch.tensor([frame_idx]))
                        # Remaining length from this frame to episode end
                        remaining_lengths.append(ep_end - frame_idx)
                
                observation['absolute_frame_indices'] = absolute_indices_list
                observation['episode_length'] = torch.tensor(episode_lengths)
                observation['remaining_length'] = torch.tensor(remaining_lengths)
        
        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition
    
    @torch.no_grad()
    def _encode_images_batch(self, images: np.ndarray) -> torch.Tensor:
        """Encode a batch of images (with optional temporal dimension) using DINO.
        
        Args:
            images: Batched images with shape:
                   - (B, C, H, W) for single frames, or
                   - (B, T, C, H, W) for temporal sequences
            
        Returns:
            Encoded feature vectors with shape (B, 768) or (B, T, 768)
        """
        from lerobot.policies.rewind.modeling_rewind import dino_load_image
        
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
        
        # Convert to list of (H, W, C) images
        num_frames = images.shape[0]
        if images.shape[1] in [1, 3]:  # Channel first (N, C, H, W)
            images_list = [images[i].transpose(1, 2, 0) for i in range(num_frames)]
        else:  # Channel last (N, H, W, C)
            images_list = [images[i] for i in range(num_frames)]
        
        # Encode each frame (can batch process with DINO for efficiency)
        all_embeddings = []
        for i in range(0, num_frames, self.config.dino_batch_size):
            batch_imgs = images_list[i:i + self.config.dino_batch_size]
            
            # Prepare images for DINO
            dino_inputs = []
            for img in batch_imgs:
                # Handle single channel
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                
                # Convert to uint8
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                
                dino_inputs.append(dino_load_image(img))
            
            # Batch encode
            dino_batch = torch.cat(dino_inputs).to(self.device)
            embeddings = self.dino_encoder(dino_batch).detach().cpu()
            
            # Handle single frame case
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            
            all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings)  # (B*T, 768)
        
        # Reshape back if temporal
        if has_temporal:
            all_embeddings = all_embeddings.reshape(batch_size, seq_length, -1)  # (B, T, 768)
        
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
        """Encode a list of text strings using MiniLM (one per sample).
        
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
        """
        Adds video_features and text_features to the observation features.
        """
        # Add the encoded features
        features[PipelineFeatureType.OBSERVATION]['video_features'] = PolicyFeature(
            type=FeatureType.VISUAL, 
            shape=(768,)  # DINO embedding dimension
        )
        features[PipelineFeatureType.OBSERVATION]['text_features'] = PolicyFeature(
            type=FeatureType.LANGUAGE, 
            shape=(384,)  # MiniLM embedding dimension
        )
        return features


def make_rewind_pre_post_processors(
    config: ReWiNDConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create pre-processor and post-processor pipelines for ReWiND.
    
    The pre-processing pipeline:
    1. Encodes images with DINO (768-dim)
    2. Encodes text with MiniLM (384-dim)
    3. Computes remaining episode length for progress normalization
    4. Adds batch dimension
    5. Moves data to device
    
    The post-processing pipeline moves data back to CPU.
    
    Args:
        config: ReWiND configuration
        dataset_stats: Dataset statistics (not used for ReWiND)
        dataset_meta: Dataset metadata for computing episode remaining length
    
    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """
    input_steps = [
        AddBatchDimensionProcessorStep(),
        ReWiNDEncodingProcessorStep(config=config, dataset_meta=dataset_meta),
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

