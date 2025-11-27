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
from typing import List, Union, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch import Tensor

from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.policies.sarm.sarm_utils import compute_cumulative_progress_batch, pad_state_to_max_dim
from lerobot.policies.pretrained import PreTrainedPolicy

class SARMTransformer(nn.Module):
    """
    SARM Transformer model for stage-aware reward prediction.
    
    This model has a dual-head architecture:
    1. Stage estimator: Predicts the high-level task stage (classification)
    2. Subtask estimator: Predicts fine-grained progress within the stage (regression)
    """
    
    def __init__(
        self,
        video_dim: int = 512,  
        text_dim: int = 512, 
        max_state_dim: int = 32, 
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        num_stages: int = 5,
        max_length: int = 9,
        dropout: float = 0.1,
        temporal_proportions: list[float] | None = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_stages = num_stages
        self.max_state_dim = max_state_dim
        
        if temporal_proportions is None:
            raise ValueError(
                "temporal_proportions is required for SARM. "
                "Provide subtask annotations in your dataset or set temporal_proportions in config."
            )
        
        # ᾱ_k: proportion for each stage
        alpha = torch.tensor(temporal_proportions, dtype=torch.float32)
        
        # P_k: cumulative proportion up to stage k (P_0 = 0)
        cumulative = torch.zeros(num_stages + 1, dtype=torch.float32)
        cumulative[1:] = torch.cumsum(alpha, dim=0)
        self.register_buffer('alpha', alpha)
        self.register_buffer('cumulative_prior', cumulative)
        
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.state_proj = nn.Linear(max_state_dim, hidden_dim) 
        
        # Position embedding only for the first frame
        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Stage estimator head (classification)
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_stages)
        )
        
        # Subtask estimator head (regression)
        self.stage_embedding = nn.Embedding(num_stages, hidden_dim // 4)
        subtask_input_dim = hidden_dim + hidden_dim // 4
        self.subtask_head = nn.Sequential(
            nn.Linear(subtask_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Attention mask
        self.register_buffer("attention_mask", None, persistent=False)
    
    def _get_attention_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Generate or retrieve cached causal attention mask."""
        if self.attention_mask is None or self.attention_mask.shape[0] != seq_length:
            # Create causal mask
            mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=device)
            self.attention_mask = mask
        return self.attention_mask
    
    def forward(
        self, 
        video_frames: torch.Tensor, 
        text_embed: torch.Tensor,
        state_features: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SARM transformer.
        
        Args:
            video_frames: Video frame embeddings (batch_size, seq_len, video_dim)
            text_embed: Text embeddings (batch_size, text_dim)
            state_features: Joint state features (batch_size, seq_len, state_dim)
            
        Returns:
            Tuple of:
                - Stage logits for each frame (batch_size, seq_len, num_stages)
                - Stage probabilities (batch_size, seq_len, num_stages)
                - Progress predictions for each frame (batch_size, seq_len, 1)
        """        
        # Project inputs to common dimension
        video_embed = self.video_proj(video_frames)  # [batch_size, seq_len, hidden_dim]
        text_embed = self.text_proj(text_embed).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Pad state features to max_state_dim before projection
        state_features_padded = pad_state_to_max_dim(state_features, self.max_state_dim)

        state_embed = self.state_proj(state_features_padded)  # [batch_size, seq_len, hidden_dim]

        # Fuse video and state features 
        video_embed = video_embed + state_embed
        
        # Add positional embedding to first video frame
        video_embed[:, 0] += self.first_pos_embed
        
        # Combine sequence: [text, video_frames]
        sequence = torch.cat([text_embed, video_embed], dim=1)
        
        # Get causal attention mask
        seq_length = sequence.shape[1]
        attention_mask = self._get_attention_mask(seq_length, sequence.device)
        
        # Pass through transformer with causal masking
        transformed = self.transformer(sequence, mask=attention_mask, is_causal=True)
        
        # Get frame features
        frame_features = transformed[:, 1:]  # [batch_size, seq_len, hidden_dim]
        
        # Stage estimation
        stage_logits = self.stage_head(frame_features)  # [batch_size, seq_len, num_stages]
        stage_probs = F.softmax(stage_logits, dim=-1)  # [batch_size, seq_len, num_stages]
        
        # Get predicted stage indices
        stage_indices = torch.argmax(stage_probs, dim=-1)  # [batch_size, seq_len]
        
        # Get stage embeddings for conditioning
        stage_embeds = self.stage_embedding(stage_indices) 
        
        # Concatenate frame features with stage embeddings
        conditioned_features = torch.cat([frame_features, stage_embeds], dim=-1)
        
        # Subtask progress estimation (conditioned on stage)
        # τ̂ = within-subtask progress (0-1)
        tau_preds = self.subtask_head(conditioned_features)  # [batch_size, seq_len, 1]
        
        # Convert τ̂ to cumulative progress ŷ using Paper Formula (2):
        # ŷ = P_{k-1} + ᾱ_k × τ̂
        progress_preds = compute_cumulative_progress_batch(
            tau_preds, stage_indices, self.alpha, self.cumulative_prior
        )
        
        return stage_logits, stage_probs, progress_preds


class SARMRewardModel(PreTrainedPolicy):
    """
    SARM Reward Model for stage-aware task completion rewards.
    
    Per SARM paper (Appendix A.4): "We employ a frozen clip-vit-base-patch32 encoder 
    to process both RGB image sequences and task descriptions."
    
    This model combines:
    - CLIP for encoding video frames AND text descriptions
    - SARMTransformer for predicting task stage and progress
    - Optional RA-BC (Reward-Aligned Behavior Cloning) for weighted training
    """
    
    name = "sarm"
    config_class = SARMConfig
    
    def __init__(self, config: SARMConfig, dataset_stats: dict | None = None, dataset_meta=None):
        super().__init__(config, dataset_stats)
        config.validate_features() 
        self.config = config
        self.dataset_stats = dataset_stats
        self.device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load temporal proportions from dataset
        if config.temporal_proportions is None and dataset_meta is not None:
            self._load_temporal_proportions(dataset_meta)
        
        logging.info("Loading CLIP encoder")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        self.sarm_transformer = SARMTransformer(
            video_dim=config.image_dim,
            text_dim=config.text_dim,
            max_state_dim=config.max_state_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_stages=config.num_stages,
            max_length=config.max_length,
            dropout=config.dropout,
            temporal_proportions=config.temporal_proportions
        )
        self.sarm_transformer.to(self.device)
        logging.info(f"SARM initialized on {self.device}")
    
    def _load_temporal_proportions(self, dataset_meta) -> None:
        """
        Load pre-computed temporal proportions from dataset metadata JSON file.

        The temporal proportions are computed during dataset annotation using SARM Paper Formula (1):
            ᾱ_k = (1/M) × Σ_i (L_{i,k} / T_i)
        """
        import json
        
        proportions_path = dataset_meta.root / "meta" / "temporal_proportions.json"
        
        if not proportions_path.exists():
            raise ValueError(
                f"Temporal proportions not found at {proportions_path}. "
                "Run the subtask annotation tool first to compute and save temporal proportions."
            )
        
        with open(proportions_path, "r") as f:
            temporal_proportions_dict = json.load(f)
        
        # Sort subtask names for consistent ordering
        subtask_names = sorted(temporal_proportions_dict.keys())
        
        self.config.num_stages = len(subtask_names)
        self.config.subtask_names = subtask_names
        self.config.temporal_proportions = [temporal_proportions_dict[name] for name in subtask_names]
        
        logging.info(f"Loaded {len(subtask_names)} subtasks: {subtask_names}")
        logging.info(f"Temporal proportions: {temporal_proportions_dict}")
    
    def to(self, device):
        """Override to method to ensure all components move together."""
        super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.clip_model.to(device)
        self.sarm_transformer.to(device)
        return self
    
    @torch.no_grad()
    def encode_images(self, images: np.ndarray) -> np.ndarray:
        """
        Encode video frames using CLIP.
        
        Args:
            images: Video frames with shape (num_videos, num_frames, H, W, C) in uint8.
                   Can also be (num_frames, H, W, C) for a single video.
                   
        Returns:
            Encoded image features (num_videos, num_frames, 512) or (num_frames, 512).
        """
        # Handle single video case
        single_video = False
        if len(images.shape) == 4:
            images = images[np.newaxis, ...]
            single_video = True
        
        assert len(images.shape) == 5, f"Expected 5D input (num_videos, num_frames, H, W, C), got {images.shape}"
        
        all_embeddings = []
        
        for video in images:
            video_embeddings = []
            
            # Convert frames to PIL images for CLIP processor
            frames = []
            for frame in video:
                if frame.shape[0] == 3:  # Channel first
                    frame = frame.transpose(1, 2, 0)
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                frames.append(Image.fromarray(frame))
            
            # Batch process frames with CLIP
            for i in range(0, len(frames), self.config.clip_batch_size):
                batch = frames[i:i + self.config.clip_batch_size]
                inputs = self.clip_processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get image embeddings from CLIP
                embeddings = self.clip_model.get_image_features(**inputs).detach().cpu()
                
                # Handle single frame case
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                
                video_embeddings.append(embeddings)
            
            video_embeddings = torch.cat(video_embeddings)
            all_embeddings.append(video_embeddings)
        
        result = torch.stack(all_embeddings).numpy()
        
        if single_video:
            result = result[0]
        
        return result
    
    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text using CLIP text encoder (per SARM paper A.4).
        
        Args:
            text: Text string or list of text strings.
            
        Returns:
            Encoded text features (batch_size, 512) or (512,) for single text.
        """
        if isinstance(text, str):
            text = [text]
            single_text = True
        else:
            single_text = False
        
        # Use CLIP's tokenizer directly (avoids image processor validation issues)
        tokenizer = self.clip_processor.tokenizer
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(text), self.config.batch_size):
            batch_text = text[i:i + self.config.batch_size]
            
            inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_embeddings = self.clip_model.get_text_features(**inputs)
            all_embeddings.append(text_embeddings.cpu())
        
        result = torch.cat(all_embeddings).numpy()
        
        if single_text:
            result = result[0]
        
        return result
    
    @torch.no_grad()
    def calculate_rewards(
        self,
        text_embeddings: Union[np.ndarray, torch.Tensor],
        video_embeddings: Union[np.ndarray, torch.Tensor],
        state_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_all_frames: bool = False,
        return_stages: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Calculate rewards for given text, video, and state representations.
        
        Args:
            text_embeddings: Encoded text representations (batch_size, 512)
            video_embeddings: Encoded video representations (batch_size, num_frames, 512)
            state_features: Joint state features (batch_size, num_frames, state_dim)
            return_all_frames: If True, return rewards for all frames
            return_stages: If True, also return stage predictions
            
        Returns:
            If return_stages=False:
                Reward values (batch_size,) or (batch_size, num_frames)
            If return_stages=True:
                Tuple of (rewards, stage_probs)
        """
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        if isinstance(video_embeddings, np.ndarray):
            video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32)
        if state_features is not None and isinstance(state_features, np.ndarray):
            state_features = torch.tensor(state_features, dtype=torch.float32)
        
        # Handle single sample case
        if text_embeddings.dim() == 1:
            text_embeddings = text_embeddings.unsqueeze(0)
            video_embeddings = video_embeddings.unsqueeze(0)
            if state_features is not None:
                state_features = state_features.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Process in batches
        all_rewards = []
        all_stage_probs = []
        
        for i in range(0, len(video_embeddings), self.config.batch_size):
            batch_texts = text_embeddings[i:i + self.config.batch_size].to(self.device)
            batch_videos = video_embeddings[i:i + self.config.batch_size].to(self.device)
            batch_states = None
            if state_features is not None:
                batch_states = state_features[i:i + self.config.batch_size].to(self.device)
            
            # Get predictions
            stage_logits, stage_probs, progress_preds = self.sarm_transformer(
                batch_videos.float(), batch_texts.float(), batch_states.float() if batch_states is not None else None
            )
            
            if return_all_frames:
                all_rewards.append(progress_preds.squeeze(-1).cpu())
            else:
                # Return only last frame reward
                all_rewards.append(progress_preds[:, -1, 0].cpu())
            
            if return_stages:
                all_stage_probs.append(stage_probs.cpu())
        
        rewards = torch.cat(all_rewards).numpy()
        
        if single_sample:
            rewards = rewards[0] if not return_all_frames else rewards[0]
        
        if return_stages:
            stage_probs = torch.cat(all_stage_probs).numpy()
            if single_sample:
                stage_probs = stage_probs[0]
            return rewards, stage_probs
        
        return rewards
    
    def train(self, mode: bool = True):
        """Overwrite train method to ensure CLIP encoder stays frozen during training"""
        super().train(mode)
        self.clip_model.eval()
        self.sarm_transformer.train(mode)
        return self
    
    def eval(self):
        """Overwrite eval method to ensure CLIP encoder stays frozen during evaluation"""
        return self.train(False)
    
    def parameters(self):
        """Override to return trainable parameters (only SARM transformer, not CLIP encoder)."""
        return self.sarm_transformer.parameters()
    
    def get_optim_params(self):
        """Override to return optimizer parameters (only SARM transformer, not CLIP encoder)."""
        return self.parameters()
    
    def reset(self):
        """Required by PreTrainedPolicy but not used for reward models."""
        pass
    
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Required by PreTrainedPolicy but not used for reward models."""
        raise NotImplementedError("SARM model does not predict action chunks")
    
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Required by PreTrainedPolicy but not used for SARM."""
        raise NotImplementedError("SARM model does not select actions")
    
    def _apply_temporal_augmentation(
        self, 
        video: torch.Tensor, 
        progress: torch.Tensor, 
        state: torch.Tensor | None,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply temporal augmentation by appending reversed frames (SARM paper A.4).
        
        This helps the model learn to handle non-monotonic progress (failures, recoveries).
        Appends 1-4 reversed frames to simulate going backwards in task progress.
        """
        num_reverse = random.randint(1, min(4, max_length - 1))
        
        # Reverse and take frames (skip first which is last of original)
        reversed_video = video.flip(0)[1:num_reverse + 1]
        reversed_progress = progress.flip(0)[1:num_reverse + 1]
        
        # Concatenate and trim
        video = torch.cat([video, reversed_video], dim=0)[:max_length]
        progress = torch.cat([progress, reversed_progress], dim=0)[:max_length]
        
        if state is not None:
            reversed_state = state.flip(0)[1:num_reverse + 1]
            state = torch.cat([state, reversed_state], dim=0)[:max_length]
        
        return video, progress, state
    
    def _ensure_sequence_length(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad or trim tensor to target length."""
        current_len = tensor.shape[0]
        if current_len == target_len:
            return tensor
        if current_len < target_len:
            padding = target_len - current_len
            return torch.cat([tensor, tensor[-1:].expand(padding, *tensor.shape[1:])])
        return tensor[:target_len]
    
    def forward(self, batch):
        """
        Forward pass for SARM reward model training.
        
        Uses annotation-based progress targets following SARM paper Eq. 2:
        yt = Pk-1 + α̅k × τt
        where:
        - τt = (t - sk) / (ek - sk) is within-subtask normalized time
        - Pk-1 is cumulative prior (sum of previous subtask proportions)
        - α̅k is the temporal proportion for subtask k
        
        Args:
            batch: Dictionary with 'observation' containing:
                - 'video_features': (B, T, 512) pre-encoded video features
                - 'text_features': (B, 512) pre-encoded text features (CLIP)
                - 'state_features': (B, T, state_dim) joint state features
                - 'stage_labels': (B, T) stage labels from annotations
                - 'progress_targets': (B, T, 1) progress targets from annotations
        
        Returns:
            Tuple of (total_loss, output_dict with loss components)
        """
        observation = batch.get('observation', batch)
        
        # Extract required features
        video_features = observation['video_features'].to(self.device)
        text_features = observation['text_features'].to(self.device)
        state_features = observation.get('state_features').to(self.device)
        
        batch_size = video_features.shape[0]
        max_length = self.config.num_frames
        
        # Ensure 3D video features (B, T, D)
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(1).expand(-1, max_length, -1)
        if state_features is not None and state_features.dim() == 2:
            state_features = state_features.unsqueeze(1).expand(-1, max_length, -1)
        
        # Get annotation-based progress targets (required for SARM paper formula)
        progress_from_annotations = observation.get('progress_targets')
        if progress_from_annotations is None:
            raise ValueError("progress_targets from annotations is required for SARM training")
        
        progress_from_annotations = progress_from_annotations.to(self.device)
        if progress_from_annotations.dim() == 2:
            progress_from_annotations = progress_from_annotations.unsqueeze(-1)
        if progress_from_annotations.dim() == 3 and progress_from_annotations.shape[0] == 1:
            progress_from_annotations = progress_from_annotations.expand(batch_size, -1, -1)
        
        # Process each sample: apply temporal REWIND augmentation 
        processed_videos = []
        processed_states = []
        progress_targets = []
        
        for i in range(batch_size):
            video = video_features[i]
            state = state_features[i] if state_features is not None else None
            progress = progress_from_annotations[i].squeeze(-1)  # (T,)
            
            # Apply temporal REWIND augmentation with 50% probability: appends up to 4 reversed frames to simulate failures/recoveries
            if random.random() < 0.5:
                video, progress, state = self._apply_temporal_augmentation(video, progress, state, max_length)
            
            # Ensure correct sequence length
            video = self._ensure_sequence_length(video, max_length)
            progress = self._ensure_sequence_length(progress.unsqueeze(-1), max_length).squeeze(-1)
            if state is not None:
                state = self._ensure_sequence_length(state, max_length)
            
            processed_videos.append(video)
            progress_targets.append(progress)
            if state is not None:
                processed_states.append(state)
        
        # Stack into batches
        processed_videos = torch.stack(processed_videos)
        progress_targets = torch.stack(progress_targets).unsqueeze(-1)  # (B, T, 1)
        processed_states = torch.stack(processed_states) if processed_states else None
        
        # Get model predictions
        stage_logits, stage_probs, progress_preds = self.sarm_transformer(
            processed_videos, text_features, processed_states
        )
        
        # Compute progress loss (MSE)
        progress_loss = F.mse_loss(progress_preds, progress_targets)
        output_dict = {'progress_loss': progress_loss.item()}
        total_loss = progress_loss
        
        # Compute stage loss (cross-entropy)
        stage_labels = observation.get('stage_labels')
        if stage_labels is None:
            raise ValueError("stage_labels from annotations is required for SARM training")
        
        stage_labels = stage_labels.to(self.device)
        if stage_labels.dim() == 1:
            stage_labels = stage_labels.unsqueeze(0).expand(batch_size, -1)
        stage_loss = compute_stage_loss(stage_logits, stage_labels)
        total_loss = total_loss + self.config.stage_loss_weight * stage_loss
        output_dict['stage_loss'] = stage_loss.item()
        
        # Misaligned loss: 20% probability
        if random.random() < 0.2:
            shuffle_idx = torch.randperm(batch_size, device=self.device)
            _, _, misaligned_preds = self.sarm_transformer(
                processed_videos, text_features[shuffle_idx], processed_states
            )
            misaligned_loss = F.mse_loss(misaligned_preds, torch.zeros_like(misaligned_preds))
            total_loss = total_loss + misaligned_loss
            output_dict['misaligned_loss'] = misaligned_loss.item()
        
        output_dict['total_loss'] = total_loss.item()
        return total_loss, output_dict

def compute_stage_loss(stage_logits: torch.Tensor, target_stages: torch.Tensor) -> torch.Tensor:
    _, _, num_stages = stage_logits.shape
    stage_logits_flat = stage_logits.reshape(-1, num_stages)
    target_stages_flat = target_stages.reshape(-1)
    
    loss = F.cross_entropy(stage_logits_flat, target_stages_flat)
    return loss
