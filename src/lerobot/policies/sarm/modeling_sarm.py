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
from typing import List, Union, Dict, Optional
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from torch import Tensor

from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.policies.pretrained import PreTrainedPolicy


def mean_pooling(model_output, attention_mask):
    """
    Mean pooling - take attention mask into account for correct averaging.
    
    Args:
        model_output: Model output containing token embeddings.
        attention_mask: Attention mask for the tokens.
        
    Returns:
        Mean-pooled embeddings.
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class SARMTransformer(nn.Module):
    """
    SARM Transformer model for stage-aware reward prediction.
    
    This model has a dual-head architecture:
    1. Stage estimator: Predicts the high-level task stage (classification)
    2. Subtask estimator: Predicts fine-grained progress within the stage (regression)
    
    The subtask estimator is conditioned on the stage prediction.
    """
    
    def __init__(
        self,
        video_dim: int = 512,  
        text_dim: int = 384, 
        state_dim: int = 14,
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
        
        # Store temporal proportions for progress conversion (Paper Eq. 4)
        # ŷ = P_{k-1} + ᾱ_k × τ̂
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
        
        # Project video, text, and state to same dimension
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        
        # Position embedding only for the first frame
        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Stage estimator head (classification)
        # Paper A.4: "2 layers with hidden dimension of 512"
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_stages)
        )
        
        # Subtask estimator head (regression, conditioned on stage)
        # Takes concatenated [features, stage_embedding]
        # Paper A.4: "2 layers with hidden dimension of 512"
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
        
        # Attention mask for causal self-attention
        self.register_buffer("attention_mask", None, persistent=False)
    
    def _get_attention_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Generate or retrieve cached causal attention mask."""
        if self.attention_mask is None or self.attention_mask.shape[0] != seq_length:
            # Create causal mask (upper triangular with -inf)
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
        batch_size = video_frames.shape[0]
        
        # Project inputs to common dimension
        video_embed = self.video_proj(video_frames)  # [batch_size, seq_len, hidden_dim]
        text_embed = self.text_proj(text_embed).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        state_embed = self.state_proj(state_features)  # [batch_size, seq_len, hidden_dim]
        # Fuse video and state features (simple addition)
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
        
        # Get frame features (exclude text token)
        frame_features = transformed[:, 1:]  # [batch_size, seq_len, hidden_dim]
        
        # Stage estimation
        stage_logits = self.stage_head(frame_features)  # [batch_size, seq_len, num_stages]
        stage_probs = F.softmax(stage_logits, dim=-1)  # [batch_size, seq_len, num_stages]
        
        # Get predicted stage indices
        stage_indices = torch.argmax(stage_probs, dim=-1)  # [batch_size, seq_len]
        
        # Get stage embeddings for conditioning
        stage_embeds = self.stage_embedding(stage_indices)  # [batch_size, seq_len, hidden_dim//4]
        
        # Concatenate frame features with stage embeddings
        conditioned_features = torch.cat([frame_features, stage_embeds], dim=-1)
        
        # Subtask progress estimation (conditioned on stage)
        # τ̂ = within-subtask progress (0-1)
        tau_preds = self.subtask_head(conditioned_features)  # [batch_size, seq_len, 1]
        
        # Convert τ̂ to cumulative progress ŷ using Paper Eq. 4:
        # ŷ = P_{k-1} + ᾱ_k × τ̂
        # P_{k-1} = cumulative prior up to stage k-1
        # ᾱ_k = temporal proportion of stage k
        P_k_minus_1 = self.cumulative_prior[stage_indices]  # [batch_size, seq_len]
        alpha_k = self.alpha[stage_indices]  # [batch_size, seq_len]
        
        progress_preds = P_k_minus_1.unsqueeze(-1) + alpha_k.unsqueeze(-1) * tau_preds
        
        return stage_logits, stage_probs, progress_preds


class SARMRewardModel(PreTrainedPolicy):
    """
    SARM Reward Model for stage-aware task completion rewards.
    
    This model combines:
    - CLIP for encoding video frames
    - MiniLM for encoding text descriptions
    - SARMTransformer for predicting task stage and progress
    - Optional RA-BC (Reward-Aligned Behavior Cloning) for weighted training
    """
    
    name = "sarm"
    config_class = SARMConfig
    
    def __init__(self, config: SARMConfig, dataset_stats: dict | None = None, dataset_meta=None):
        super().__init__(config, dataset_stats)
        self.config = config
        self.dataset_stats = dataset_stats
        self.device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Auto-detect num_stages from dataset annotations before building the model
        if dataset_meta is not None:
            self._update_num_stages_from_dataset(dataset_meta)
        
        # Initialize CLIP encoder for images
        logging.info("Loading CLIP encoder...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # Initialize MiniLM encoder for text
        logging.info("Loading MiniLM encoder...")
        self.minilm_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.minilm_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.minilm_model.to(self.device)
        self.minilm_model.eval()
        
        # Auto-detect state_dim from dataset_stats
        if config.state_dim is None:
            logging.info(f"Attempting to auto-detect state_dim. dataset_stats is None: {dataset_stats is None}")
            
            if dataset_stats is not None:
                if "observation.state" in dataset_stats:
                    config.state_dim = dataset_stats["observation.state"]["mean"].shape[0]
                    logging.info(f"Auto-detected state_dim={config.state_dim} from dataset_stats['observation.state']")
                elif "state" in dataset_stats:
                    config.state_dim = dataset_stats["state"]["mean"].shape[0]
                    logging.info(f"Auto-detected state_dim={config.state_dim} from dataset_stats['state']")
                else:
                    logging.warning(f"State keys not found in dataset_stats. Available keys: {list(dataset_stats.keys())}")
            else:
                logging.warning("dataset_stats is None, cannot auto-detect state_dim")
            
            # Raise explicit error if still None
            if config.state_dim is None:
                raise ValueError(
                    "Could not determine state_dim! "
                    f"dataset_stats={'None' if dataset_stats is None else f'available with keys: {list(dataset_stats.keys())}'}, "
                    "config.state_dim=None. "
                    "Please either:\n"
                    "1. Provide --policy.state_dim=<your_state_dimension> explicitly, or\n"
                    "2. Ensure dataset_stats contains 'observation.state' or 'state' key"
                )
        
        # Initialize SARM transformer with temporal proportions for progress conversion
        temporal_proportions = getattr(config, 'temporal_proportions', None)
        self.sarm_transformer = SARMTransformer(
            video_dim=config.image_dim,
            text_dim=config.text_dim,
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_stages=config.num_stages,
            max_length=config.max_length,
            dropout=config.dropout,
            temporal_proportions=temporal_proportions
        )
        self.sarm_transformer.to(self.device)
        
        
        logging.info(f"SARM Reward Model initialized on {self.device}")
    
    def _update_num_stages_from_dataset(self, dataset_meta) -> None:
        """Update num_stages and temporal_proportions from dataset subtask annotations."""
        episodes = dataset_meta.episodes
        if episodes is None or len(episodes) == 0:
            raise ValueError("No episodes found, using default num_stages")
            
        if 'subtask_names' not in episodes.column_names:
            raise ValueError("No subtask annotations found in dataset, using default num_stages")

        episodes_df = episodes.to_pandas()
        
        # Collect all unique subtask names and compute durations
        all_subtask_names = set()
        subtask_durations = {}
        
        for ep_idx in episodes_df.index:
            subtask_names = episodes_df.loc[ep_idx, 'subtask_names']
            if subtask_names is None or (isinstance(subtask_names, float) and pd.isna(subtask_names)):
                continue
            
            all_subtask_names.update(subtask_names)
            
            # Compute durations if available
            if 'subtask_start_frames' in episodes_df.columns and 'subtask_end_frames' in episodes_df.columns:
                start_frames = episodes_df.loc[ep_idx, 'subtask_start_frames']
                end_frames = episodes_df.loc[ep_idx, 'subtask_end_frames']
                
                for i, name in enumerate(subtask_names):
                    duration = end_frames[i] - start_frames[i]
                    if name not in subtask_durations:
                        subtask_durations[name] = []
                    subtask_durations[name].append(duration)
        
        if not all_subtask_names:
            raise ValueError("No valid subtask names found, using default num_stages")
        
        # Sort subtask names for consistent ordering
        subtask_names = sorted(list(all_subtask_names))
        num_stages = len(subtask_names)
        
        # Compute temporal proportions (Paper Eq. 1: ᾱ_k)
        avg_durations = {}
        for name in subtask_names:
            if name in subtask_durations and subtask_durations[name]:
                avg_durations[name] = np.mean(subtask_durations[name])
            else:
                avg_durations[name] = 1.0  # Default
        
        total_duration = sum(avg_durations.values())
        if total_duration > 0:
            temporal_proportions = [avg_durations[name] / total_duration for name in subtask_names]
        else:
            temporal_proportions = [1.0 / num_stages] * num_stages
    
        self.config.num_stages = num_stages
        self.config.subtask_names = subtask_names
        self.config.temporal_proportions = temporal_proportions
        
        logging.info(f"Auto-detected {num_stages} subtasks: {subtask_names}")
        logging.info(f"Temporal proportions: {dict(zip(subtask_names, temporal_proportions))}")
            
    def to(self, device):
        """Override to method to ensure all components move together."""
        super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.clip_model.to(device)
        self.minilm_model.to(device)
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
        Encode text using MiniLM.
        
        Args:
            text: Text string or list of text strings.
            
        Returns:
            Encoded text features (batch_size, 384) or (384,) for single text.
        """
        if isinstance(text, str):
            text = [text]
            single_text = True
        else:
            single_text = False
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(text), self.config.batch_size):
            batch_text = text[i:i + self.config.batch_size]
            
            encoded_input = self.minilm_tokenizer(
                batch_text, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            
            model_output = self.minilm_model(**encoded_input)
            text_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
            
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
            text_embeddings: Encoded text representations (batch_size, 384)
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
        # Convert to tensors if needed
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
    
    
    def load_pretrained_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """Load pretrained model weights from a checkpoint file."""
        logging.info(f"Loading pretrained checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Load only the SARMTransformer weights
        missing_keys, unexpected_keys = self.sarm_transformer.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            logging.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        
        logging.info("Checkpoint loaded successfully")
    
    def train(self, mode: bool = True):
        """Set training mode. Note: CLIP and MiniLM encoders always stay in eval mode."""
        super().train(mode)
        # Keep encoders in eval mode
        self.clip_model.eval()
        self.minilm_model.eval()
        # Only transformer can be trained
        self.sarm_transformer.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def parameters(self):
        """Return trainable parameters (only SARM transformer, not encoders)."""
        return self.sarm_transformer.parameters()
    
    def get_optim_params(self):
        """Return optimizer parameters for the policy."""
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
    
    def _get_remaining_length(self, observation: dict, idx: int) -> float | None:
        """Extract remaining length for a sample from observation metadata."""
        remaining_lengths = observation.get('remaining_length')
        if remaining_lengths is None:
            return None
        if isinstance(remaining_lengths, torch.Tensor):
            return remaining_lengths[idx].item() if remaining_lengths.dim() > 0 else remaining_lengths.item()
        return remaining_lengths
    
    def _compute_progress_targets(self, remaining_length: float | None, seq_len: int) -> torch.Tensor:
        """Compute progress targets based on remaining trajectory length."""
        if remaining_length is not None and remaining_length > 0:
            return torch.arange(1, seq_len + 1, dtype=torch.float32, device=self.device) / remaining_length
        else:
            raise ValueError("Remaining length is None, but is required for progress targets")
    
    def _apply_rewind_augmentation(
        self, 
        video: torch.Tensor, 
        progress: torch.Tensor, 
        state: torch.Tensor | None,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply rewind augmentation: append up to 4 reversed frames (SARM paper A.4)."""
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
        
        Args:
            batch: Dictionary with 'observation' containing:
                - 'video_features': (B, T, 512) pre-encoded video features
                - 'text_features': (B, 384) pre-encoded text features
                - 'state_features': (B, T, state_dim) joint state features
                - 'remaining_length': (B,) remaining trajectory lengths (optional)
                - 'stage_labels': (B, T) stage labels (optional, from annotations)
                - 'progress_targets': (B, T, 1) progress targets (optional, from annotations)
        
        Returns:
            Tuple of (total_loss, output_dict with loss components)
        """
        observation = batch.get('observation', batch)
        
        # Extract required features
        video_features = observation['video_features'].to(self.device)
        text_features = observation['text_features'].to(self.device)
        state_features = observation.get('state_features')
        if state_features is not None:
            state_features = state_features.to(self.device)
        
        batch_size = video_features.shape[0]
        max_length = self.config.num_frames
        
        # Ensure 3D video features (B, T, D)
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(1).expand(-1, max_length, -1)
        if state_features is not None and state_features.dim() == 2:
            state_features = state_features.unsqueeze(1).expand(-1, max_length, -1)
        
        # Process each sample: compute progress targets and apply rewind augmentation
        processed_videos = []
        processed_states = []
        progress_targets = []
        
        for i in range(batch_size):
            remaining_length = self._get_remaining_length(observation, i)
            progress = self._compute_progress_targets(remaining_length, max_length)
            
            video = video_features[i]
            state = state_features[i] if state_features is not None else None
            
            # Apply rewind augmentation with 50% probability (SARM paper)
            if random.random() < 0.5:
                video, progress, state = self._apply_rewind_augmentation(video, progress, state, max_length)
            
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
        
        # Use annotation-based progress targets
        progress_from_annotations = observation.get('progress_targets')
        if progress_from_annotations is not None:
            progress_from_annotations = progress_from_annotations.to(self.device)
            if progress_from_annotations.dim() == 2:
                progress_from_annotations = progress_from_annotations.unsqueeze(-1)
            if progress_from_annotations.dim() == 3 and progress_from_annotations.shape[0] == 1:
                progress_from_annotations = progress_from_annotations.expand(batch_size, -1, -1)
            progress_targets = progress_from_annotations
        
        # Compute progress loss
        progress_loss = F.mse_loss(progress_preds, progress_targets)
        output_dict = {'progress_loss': progress_loss.item()}
        total_loss = progress_loss
        
        # Compute stage loss if labels available
        stage_labels = observation.get('stage_labels')
        if stage_labels is not None:
            stage_labels = stage_labels.to(self.device)
            if stage_labels.dim() == 1:
                stage_labels = stage_labels.unsqueeze(0).expand(batch_size, -1)
            stage_loss = compute_stage_loss(stage_logits, stage_labels)
            total_loss = total_loss + self.config.stage_loss_weight * stage_loss
            output_dict['stage_loss'] = stage_loss.item()
        else:
            raise ValueError("Stage labels are None, but are required for stage loss")
        
        # Misaligned loss: 20% probability (SARM paper - improve video-language alignment)
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


def compute_progress_loss(progress_preds: torch.Tensor, target_progress: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(progress_preds, target_progress)
    return loss

