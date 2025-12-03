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
    
    This model supports two architectures:
    1. Single-head (dual_sparse_dense=False): Sparse head only (high-level stages)
    2. Dual-head (dual_sparse_dense=True): Twin MLP-based output heads for sparse and dense annotations
       - Sparse heads: For high-level stages
       - Dense heads: For fine-grained stages
    
    Per SARM paper: "On top of the backbone, we incorporate twin MLP-based output heads 
    tailored for different annotation types, namely dense and sparse labels"
    """
    
    def __init__(
        self,
        video_dim: int = 512,  
        text_dim: int = 512, 
        max_state_dim: int = 32, 
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        max_length: int = 9,
        dropout: float = 0.1,
        # Dual sparse/dense head parameters
        dual_sparse_dense: bool = False,
        # Sparse parameters (always required)
        num_sparse_stages: int = 5,
        sparse_temporal_proportions: list[float] | None = None,
        # Dense parameters (only required when dual_sparse_dense=True)
        num_dense_stages: int | None = None,
        dense_temporal_proportions: list[float] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.max_state_dim = max_state_dim
        self.dual_sparse_dense = dual_sparse_dense
        self.num_sparse_stages = num_sparse_stages
        self.num_dense_stages = num_dense_stages
        
        # Sparse proportions (always needed)
        if sparse_temporal_proportions is None:
            raise ValueError(
                "sparse_temporal_proportions is required for SARM. "
                "Provide subtask annotations in your dataset or set sparse_temporal_proportions in config."
            )
        sparse_alpha = torch.tensor(sparse_temporal_proportions, dtype=torch.float32)
        sparse_cumulative = torch.zeros(self.num_sparse_stages + 1, dtype=torch.float32)
        sparse_cumulative[1:] = torch.cumsum(sparse_alpha, dim=0)
        self.register_buffer('sparse_alpha', sparse_alpha)
        self.register_buffer('sparse_cumulative_prior', sparse_cumulative)
        
        if dual_sparse_dense:
            # Dual mode: also need dense proportions
            if dense_temporal_proportions is None:
                raise ValueError(
                    "dense_temporal_proportions is required when dual_sparse_dense=True"
                )
            self.num_dense_stages = num_dense_stages or len(dense_temporal_proportions)
            
            # Dense proportions
            dense_alpha = torch.tensor(dense_temporal_proportions, dtype=torch.float32)
            dense_cumulative = torch.zeros(self.num_dense_stages + 1, dtype=torch.float32)
            dense_cumulative[1:] = torch.cumsum(dense_alpha, dim=0)
            self.register_buffer('dense_alpha', dense_alpha)
            self.register_buffer('dense_cumulative_prior', dense_cumulative)
        
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
        
        # Sparse heads 
        self.sparse_stage_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.num_sparse_stages)
        )
        self.sparse_stage_embedding = nn.Embedding(self.num_sparse_stages, hidden_dim // 4)
        subtask_input_dim = hidden_dim + hidden_dim // 4
        self.sparse_subtask_head = nn.Sequential(
            nn.Linear(subtask_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        if dual_sparse_dense:
            # Dense heads
            self.dense_stage_head = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, self.num_dense_stages)
            )
            self.dense_stage_embedding = nn.Embedding(self.num_dense_stages, hidden_dim // 4)
            self.dense_subtask_head = nn.Sequential(
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
    
    def _compute_backbone_features(
        self,
        video_frames: torch.Tensor,
        text_embed: torch.Tensor,
        state_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute shared backbone features from inputs."""
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
        return frame_features
    
    def forward(
        self, 
        video_frames: torch.Tensor, 
        text_embed: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        head_mode: str = "both"  # "sparse", "dense", or "both" (only for dual mode)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict:
        """
        Forward pass through the SARM transformer.
        
        Args:
            video_frames: Video frame embeddings (batch_size, seq_len, video_dim)
            text_embed: Text embeddings (batch_size, text_dim)
            state_features: Joint state features (batch_size, seq_len, state_dim)
            head_mode: Which head(s) to use in dual mode ("sparse", "dense", or "both")
            
        Returns:
            For single-head mode:
                Tuple of (stage_logits, stage_probs, progress_preds)
            For dual-head mode:
                Dict with keys "sparse" and/or "dense", each containing
                (stage_logits, stage_probs, progress_preds)
        """        
        # Compute shared backbone features
        frame_features = self._compute_backbone_features(video_frames, text_embed, state_features)
        
        if not self.dual_sparse_dense:
            # Single head mode: sparse only
            sparse_stage_logits = self.sparse_stage_head(frame_features)
            sparse_stage_probs = F.softmax(sparse_stage_logits, dim=-1)
            sparse_stage_indices = torch.argmax(sparse_stage_probs, dim=-1)
            sparse_stage_embeds = self.sparse_stage_embedding(sparse_stage_indices) 
            conditioned_features = torch.cat([frame_features, sparse_stage_embeds], dim=-1)
            tau_preds = self.sparse_subtask_head(conditioned_features)
            progress_preds = compute_cumulative_progress_batch(
                tau_preds, sparse_stage_indices, self.sparse_alpha, self.sparse_cumulative_prior
            )
            return sparse_stage_logits, sparse_stage_probs, progress_preds
        
        # Dual head mode: compute outputs for requested heads
        results = {}
        
        if head_mode in ["sparse", "both"]:
            # Sparse head
            sparse_stage_logits = self.sparse_stage_head(frame_features)
            sparse_stage_probs = F.softmax(sparse_stage_logits, dim=-1)
            sparse_stage_indices = torch.argmax(sparse_stage_probs, dim=-1)
            sparse_stage_embeds = self.sparse_stage_embedding(sparse_stage_indices)
            sparse_conditioned = torch.cat([frame_features, sparse_stage_embeds], dim=-1)
            sparse_tau_preds = self.sparse_subtask_head(sparse_conditioned)
            sparse_progress_preds = compute_cumulative_progress_batch(
                sparse_tau_preds, sparse_stage_indices, 
                self.sparse_alpha, self.sparse_cumulative_prior
            )
            results["sparse"] = (sparse_stage_logits, sparse_stage_probs, sparse_progress_preds)
        
        if head_mode in ["dense", "both"]:
            # Dense head
            dense_stage_logits = self.dense_stage_head(frame_features)
            dense_stage_probs = F.softmax(dense_stage_logits, dim=-1)
            dense_stage_indices = torch.argmax(dense_stage_probs, dim=-1)
            dense_stage_embeds = self.dense_stage_embedding(dense_stage_indices)
            dense_conditioned = torch.cat([frame_features, dense_stage_embeds], dim=-1)
            dense_tau_preds = self.dense_subtask_head(dense_conditioned)
            dense_progress_preds = compute_cumulative_progress_batch(
                dense_tau_preds, dense_stage_indices,
                self.dense_alpha, self.dense_cumulative_prior
            )
            results["dense"] = (dense_stage_logits, dense_stage_probs, dense_progress_preds)
        
        return results


class SARMRewardModel(PreTrainedPolicy):
    """
    SARM Reward Model for stage-aware task completion rewards.
    
    Per SARM paper (Appendix A.4): "We employ a frozen clip-vit-base-patch32 encoder 
    to process both RGB image sequences and task descriptions."
    
    This model combines:
    - CLIP for encoding video frames AND text descriptions
    - SARMTransformer for predicting task stage and progress
    - Optional RA-BC (Reward-Aligned Behavior Cloning) for weighted training
    
    Supports dual-head mode (dual_sparse_dense=True) with twin MLP-based output heads
    for sparse and dense annotations as described in the SARM paper.
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
        if config.dual_sparse_dense:
            # Dual mode: load both sparse and dense proportions
            if (config.sparse_temporal_proportions is None or config.dense_temporal_proportions is None) and dataset_meta is not None:
                self._load_dual_temporal_proportions(dataset_meta)
        else:
            # Single mode: load sparse proportions only
            if config.sparse_temporal_proportions is None and dataset_meta is not None:
                self._load_sparse_temporal_proportions(dataset_meta)
        
        logging.info("Loading CLIP encoder")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # Initialize transformer with appropriate parameters
        if config.dual_sparse_dense:
            self.sarm_transformer = SARMTransformer(
                video_dim=config.image_dim,
                text_dim=config.text_dim,
                max_state_dim=config.max_state_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                max_length=config.max_length,
                dropout=config.dropout,
                dual_sparse_dense=True,
                num_sparse_stages=config.num_sparse_stages,
                sparse_temporal_proportions=config.sparse_temporal_proportions,
                num_dense_stages=config.num_dense_stages,
                dense_temporal_proportions=config.dense_temporal_proportions,
            )
            logging.info(f"SARM initialized with dual heads: {config.num_sparse_stages} sparse stages, {config.num_dense_stages} dense stages")
        else:
            self.sarm_transformer = SARMTransformer(
                video_dim=config.image_dim,
                text_dim=config.text_dim,
                max_state_dim=config.max_state_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                max_length=config.max_length,
                dropout=config.dropout,
                dual_sparse_dense=False,
                num_sparse_stages=config.num_sparse_stages,
                sparse_temporal_proportions=config.sparse_temporal_proportions,
            )
            logging.info(f"SARM initialized with sparse head only: {config.num_sparse_stages} stages")
        self.sarm_transformer.to(self.device)
        logging.info(f"SARM initialized on {self.device}")
    
    def _load_sparse_temporal_proportions(self, dataset_meta) -> None:
        """
        Load pre-computed sparse temporal proportions from dataset metadata JSON file.

        The temporal proportions are computed during dataset annotation using SARM Paper Formula (1):
            ᾱ_k = (1/M) × Σ_i (L_{i,k} / T_i)
        
        Tries to load from temporal_proportions_sparse.json first, then falls back to 
        temporal_proportions.json for backward compatibility.
        """
        import json
        
        # Try sparse-specific file first, then fallback to legacy name
        sparse_path = dataset_meta.root / "meta" / "temporal_proportions_sparse.json"
        legacy_path = dataset_meta.root / "meta" / "temporal_proportions.json"
        
        if sparse_path.exists():
            proportions_path = sparse_path
        elif legacy_path.exists():
            proportions_path = legacy_path
        else:
            raise ValueError(
                f"Temporal proportions not found at {sparse_path} or {legacy_path}. "
                "Run the subtask annotation tool first to compute and save temporal proportions."
            )
        
        with open(proportions_path, "r") as f:
            temporal_proportions_dict = json.load(f)
        
        # Sort subtask names for consistent ordering
        subtask_names = sorted(temporal_proportions_dict.keys())
        
        self.config.num_sparse_stages = len(subtask_names)
        self.config.sparse_subtask_names = subtask_names
        self.config.sparse_temporal_proportions = [temporal_proportions_dict[name] for name in subtask_names]
        
        logging.info(f"Loaded {len(subtask_names)} sparse subtasks: {subtask_names}")
        logging.info(f"Sparse temporal proportions: {temporal_proportions_dict}")
    
    def _load_dual_temporal_proportions(self, dataset_meta) -> None:
        """
        Load pre-computed temporal proportions for both sparse and dense annotations.
        
        Expects two JSON files in dataset meta:
        - temporal_proportions_sparse.json: For high-level stages (e.g., fold1, fold2, fold3)
        - temporal_proportions_dense.json: For fine-grained stages (e.g., grab, flatten, fold_left, etc.)
        """
        import json
        
        sparse_path = dataset_meta.root / "meta" / "temporal_proportions_sparse.json"
        dense_path = dataset_meta.root / "meta" / "temporal_proportions_dense.json"
        
        # Load sparse proportions
        if not sparse_path.exists():
            raise ValueError(
                f"Sparse temporal proportions not found at {sparse_path}. "
                "Run the subtask annotation tool with --sparse-subtasks to compute and save sparse proportions."
            )
        
        with open(sparse_path, "r") as f:
            sparse_proportions_dict = json.load(f)
        
        sparse_names = sorted(sparse_proportions_dict.keys())
        self.config.num_sparse_stages = len(sparse_names)
        self.config.sparse_subtask_names = sparse_names
        self.config.sparse_temporal_proportions = [sparse_proportions_dict[name] for name in sparse_names]
        
        logging.info(f"Loaded {len(sparse_names)} sparse subtasks: {sparse_names}")
        logging.info(f"Sparse temporal proportions: {sparse_proportions_dict}")
        
        # Load dense proportions
        if not dense_path.exists():
            raise ValueError(
                f"Dense temporal proportions not found at {dense_path}. "
                "Run the subtask annotation tool with --dense-subtasks to compute and save dense proportions."
            )
        
        with open(dense_path, "r") as f:
            dense_proportions_dict = json.load(f)
        
        dense_names = sorted(dense_proportions_dict.keys())
        self.config.num_dense_stages = len(dense_names)
        self.config.dense_subtask_names = dense_names
        self.config.dense_temporal_proportions = [dense_proportions_dict[name] for name in dense_names]
        
        logging.info(f"Loaded {len(dense_names)} dense subtasks: {dense_names}")
        logging.info(f"Dense temporal proportions: {dense_proportions_dict}")
    
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
                - For single mode (dual_sparse_dense=False, sparse only):
                    - 'sparse_stage_labels': (B, T) sparse stage labels from annotations
                    - 'sparse_progress_targets': (B, T, 1) sparse progress targets from annotations
                    (also accepts legacy 'stage_labels' and 'progress_targets' for backward compat)
                - For dual mode (dual_sparse_dense=True):
                    - 'sparse_stage_labels': (B, T) sparse stage labels
                    - 'sparse_progress_targets': (B, T, 1) sparse progress targets
                    - 'dense_stage_labels': (B, T) dense stage labels
                    - 'dense_progress_targets': (B, T, 1) dense progress targets
        
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
        
        if self.config.dual_sparse_dense:
            return self._forward_dual(observation, video_features, text_features, state_features, batch_size, max_length)
        else:
            return self._forward_single(observation, video_features, text_features, state_features, batch_size, max_length)
    
    def _forward_single(self, observation, video_features, text_features, state_features, batch_size, max_length):
        """Forward pass for single-head mode (sparse only)."""
        # Get sparse annotation-based progress targets 
        # Try sparse_progress_targets first, then fall back to legacy progress_targets
        progress_from_annotations = observation.get('sparse_progress_targets') or observation.get('progress_targets')
        if progress_from_annotations is None:
            raise ValueError("sparse_progress_targets (or progress_targets) is required for SARM training")
        
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
        sparse_stage_logits, sparse_stage_probs, sparse_progress_preds = self.sarm_transformer(
            processed_videos, text_features, processed_states
        )
        
        # Compute sparse progress loss (MSE)
        sparse_progress_loss = F.mse_loss(sparse_progress_preds, progress_targets)
        output_dict = {'sparse_progress_loss': sparse_progress_loss.item()}
        total_loss = sparse_progress_loss
        
        # Compute sparse stage loss (cross-entropy)
        # Try sparse_stage_labels first, then fall back to legacy stage_labels
        stage_labels = observation.get('sparse_stage_labels') or observation.get('stage_labels')
        if stage_labels is None:
            raise ValueError("sparse_stage_labels (or stage_labels) is required for SARM training")
        
        stage_labels = stage_labels.to(self.device)
        if stage_labels.dim() == 1:
            stage_labels = stage_labels.unsqueeze(0).expand(batch_size, -1)
        sparse_stage_loss = compute_stage_loss(sparse_stage_logits, stage_labels)
        total_loss = total_loss + self.config.stage_loss_weight * sparse_stage_loss
        output_dict['sparse_stage_loss'] = sparse_stage_loss.item()
        
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
    
    def _forward_dual(self, observation, video_features, text_features, state_features, batch_size, max_length):
        """Forward pass for dual-head mode (sparse and dense annotations)."""
        # Get both sparse and dense annotation targets
        sparse_progress = observation.get('sparse_progress_targets')
        dense_progress = observation.get('dense_progress_targets')
        sparse_stage_labels = observation.get('sparse_stage_labels')
        dense_stage_labels = observation.get('dense_stage_labels')
        
        if sparse_progress is None or dense_progress is None:
            raise ValueError("Both sparse_progress_targets and dense_progress_targets are required for dual mode training")
        
        sparse_progress = sparse_progress.to(self.device)
        dense_progress = dense_progress.to(self.device)
        
        if sparse_progress.dim() == 2:
            sparse_progress = sparse_progress.unsqueeze(-1)
        if dense_progress.dim() == 2:
            dense_progress = dense_progress.unsqueeze(-1)
        
        if sparse_progress.shape[0] == 1:
            sparse_progress = sparse_progress.expand(batch_size, -1, -1)
        if dense_progress.shape[0] == 1:
            dense_progress = dense_progress.expand(batch_size, -1, -1)
        
        # Process each sample with temporal REWIND augmentation
        processed_videos = []
        processed_states = []
        sparse_progress_targets = []
        dense_progress_targets = []
        
        for i in range(batch_size):
            video = video_features[i]
            state = state_features[i] if state_features is not None else None
            sp_progress = sparse_progress[i].squeeze(-1)
            dn_progress = dense_progress[i].squeeze(-1)
            
            # Apply temporal REWIND augmentation with 50% probability
            if random.random() < 0.5:
                video, sp_progress, state = self._apply_temporal_augmentation(video, sp_progress, state, max_length)
                # Apply same augmentation pattern to dense progress
                dn_progress = self._ensure_sequence_length(dn_progress.unsqueeze(-1), max_length).squeeze(-1)
            
            # Ensure correct sequence length
            video = self._ensure_sequence_length(video, max_length)
            sp_progress = self._ensure_sequence_length(sp_progress.unsqueeze(-1), max_length).squeeze(-1)
            dn_progress = self._ensure_sequence_length(dn_progress.unsqueeze(-1), max_length).squeeze(-1)
            if state is not None:
                state = self._ensure_sequence_length(state, max_length)
            
            processed_videos.append(video)
            sparse_progress_targets.append(sp_progress)
            dense_progress_targets.append(dn_progress)
            if state is not None:
                processed_states.append(state)
        
        # Stack into batches
        processed_videos = torch.stack(processed_videos)
        sparse_progress_targets = torch.stack(sparse_progress_targets).unsqueeze(-1)
        dense_progress_targets = torch.stack(dense_progress_targets).unsqueeze(-1)
        processed_states = torch.stack(processed_states) if processed_states else None
        
        # Get model predictions for both heads
        results = self.sarm_transformer(
            processed_videos, text_features, processed_states, head_mode="both"
        )
        
        sparse_stage_logits, sparse_stage_probs, sparse_progress_preds = results["sparse"]
        dense_stage_logits, dense_stage_probs, dense_progress_preds = results["dense"]
        
        output_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Sparse progress loss
        sparse_progress_loss = F.mse_loss(sparse_progress_preds, sparse_progress_targets)
        output_dict['sparse_progress_loss'] = sparse_progress_loss.item()
        total_loss = total_loss + sparse_progress_loss
        
        # Dense progress loss
        dense_progress_loss = F.mse_loss(dense_progress_preds, dense_progress_targets)
        output_dict['dense_progress_loss'] = dense_progress_loss.item()
        total_loss = total_loss + dense_progress_loss
        
        # Sparse stage loss
        if sparse_stage_labels is not None:
            sparse_stage_labels = sparse_stage_labels.to(self.device)
            if sparse_stage_labels.dim() == 1:
                sparse_stage_labels = sparse_stage_labels.unsqueeze(0).expand(batch_size, -1)
            sparse_stage_loss = compute_stage_loss(sparse_stage_logits, sparse_stage_labels)
            total_loss = total_loss + self.config.stage_loss_weight * sparse_stage_loss
            output_dict['sparse_stage_loss'] = sparse_stage_loss.item()
        
        # Dense stage loss
        if dense_stage_labels is not None:
            dense_stage_labels = dense_stage_labels.to(self.device)
            if dense_stage_labels.dim() == 1:
                dense_stage_labels = dense_stage_labels.unsqueeze(0).expand(batch_size, -1)
            dense_stage_loss = compute_stage_loss(dense_stage_logits, dense_stage_labels)
            total_loss = total_loss + self.config.stage_loss_weight * dense_stage_loss
            output_dict['dense_stage_loss'] = dense_stage_loss.item()
        
        # Misaligned loss: 20% probability
        if random.random() < 0.2:
            shuffle_idx = torch.randperm(batch_size, device=self.device)
            misaligned_results = self.sarm_transformer(
                processed_videos, text_features[shuffle_idx], processed_states, head_mode="both"
            )
            sparse_misaligned = misaligned_results["sparse"][2]
            dense_misaligned = misaligned_results["dense"][2]
            misaligned_loss = F.mse_loss(sparse_misaligned, torch.zeros_like(sparse_misaligned))
            misaligned_loss += F.mse_loss(dense_misaligned, torch.zeros_like(dense_misaligned))
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
