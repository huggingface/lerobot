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
from lerobot.policies.processor import (
    ProcessorStep,
    PolicyProcessorPipeline,
    PolicyAction,
    DeviceProcessorStep,
)
from lerobot.policies.processor.transition import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

class ReWiNDEncodingProcessorStep(ProcessorStep):
    """
    ProcessorStep that encodes images and text for ReWiND training.
    
    This step handles the DINO (image) and MiniLM (text) encoding that ReWiND needs.
    """
    
    def __init__(
        self,
        config: ReWiNDConfig,
        image_key: str | None = None,
        task_description: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.image_key = image_key or config.image_key
        self.task_description = task_description or config.task_description
        
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
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Encode images and text in the batch."""
        # Extract images
        if self.image_key in batch:
            images = batch[self.image_key]
            
            # Handle different image formats
            if isinstance(images, torch.Tensor):
                images = images.cpu().numpy()
            
            # Encode images
            video_features = self._encode_images(images)
            batch['video_features'] = video_features
        
        # Encode text
        batch_size = len(batch.get('video_features', batch.get(list(batch.keys())[0])))
        task_descriptions = [self.task_description] * batch_size
        text_features = self._encode_text(task_descriptions)
        batch['text_features'] = text_features
        
        return batch
    
    @torch.no_grad()
    def _encode_images(self, images: np.ndarray) -> torch.Tensor:
        """Encode images using DINO."""
        from lerobot.policies.rewind.modeling_rewind import dino_load_image
        
        # Handle single frame case
        if len(images.shape) == 4:
            images = images[:, np.newaxis, ...]
            single_frame = True
        else:
            single_frame = False
        
        batch_size, num_frames, C, H, W = images.shape
        
        # Convert to (B, T, H, W, C)
        if C == 3:
            images = images.transpose(0, 1, 3, 4, 2)
        
        all_embeddings = []
        
        for video in images:
            video_embeddings = []
            
            # Convert to uint8
            if video.dtype != np.uint8:
                video = (video * 255).astype(np.uint8) if video.max() <= 1.0 else video.astype(np.uint8)
            
            frames = [frame for frame in video]
            episode_images_dino = [dino_load_image(frame) for frame in frames]
            
            # Batch process
            for i in range(0, len(episode_images_dino), self.config.dino_batch_size):
                dino_batch = torch.cat(episode_images_dino[i:i + self.config.dino_batch_size])
                dino_batch = dino_batch.to(self.device)
                embeddings = self.dino_encoder(dino_batch).squeeze().detach().cpu()
                
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                
                video_embeddings.append(embeddings)
            
            video_embeddings = torch.cat(video_embeddings)
            all_embeddings.append(video_embeddings)
        
        result = torch.stack(all_embeddings)
        
        if single_frame:
            result = result.squeeze(1)
        
        return result
    
    @torch.no_grad()
    def _encode_text(self, text: List[str]) -> torch.Tensor:
        """Encode text using MiniLM."""
        from lerobot.policies.rewind.modeling_rewind import mean_pooling
        
        all_embeddings = []
        
        for i in range(0, len(text), self.config.batch_size):
            batch_text = text[i:i + self.config.batch_size]
            
            encoded_input = self.minilm_tokenizer(
                batch_text, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            
            model_output = self.minilm_model(**encoded_input)
            text_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
            
            all_embeddings.append(text_embeddings.cpu())
        
        result = torch.cat(all_embeddings)
        
        return result


def make_rewind_pre_post_processors(
    config: ReWiNDConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create pre-processor and post-processor pipelines for ReWiND.
    
    The pre-processing pipeline:
    1. Encodes images with DINO (768-dim)
    2. Encodes text with MiniLM (384-dim)
    3. Moves data to device
    
    The post-processing pipeline is minimal (just moves to CPU).
    
    Args:
        config: ReWiND configuration
        dataset_stats: Dataset statistics (not used for ReWiND)
    
    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """
    input_steps = [
        ReWiNDEncodingProcessorStep(config=config),
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

