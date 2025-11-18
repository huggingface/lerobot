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
        video_dim: int = 512,  # CLIP dimension
        text_dim: int = 384,  # MiniLM dimension
        state_dim: int = 14,  # Joint state dimension
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        num_stages: int = 5,
        max_length: int = 9,
        dropout: float = 0.1,
        use_joint_state: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_stages = num_stages
        self.use_joint_state = use_joint_state
        
        # Project video, text, and state to common dimension
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        if use_joint_state:
            self.state_proj = nn.Linear(state_dim, hidden_dim)
        
        # Position embedding only for the first frame
        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))
        
        # Transformer encoder (shared backbone)
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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_stages)
        )
        
        # Subtask estimator head (regression, conditioned on stage)
        # Takes concatenated [features, stage_embedding]
        self.stage_embedding = nn.Embedding(num_stages, hidden_dim // 4)
        subtask_input_dim = hidden_dim + hidden_dim // 4
        self.subtask_head = nn.Sequential(
            nn.Linear(subtask_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
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
        
        # Add joint state if provided
        if self.use_joint_state and state_features is not None:
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
        progress_preds = self.subtask_head(conditioned_features)  # [batch_size, seq_len, 1]
        
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
    
    def __init__(self, config: SARMConfig, dataset_stats: dict | None = None):
        super().__init__(config, dataset_stats)
        self.config = config
        self.dataset_stats = dataset_stats
        self.device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize CLIP encoder for images
        logging.info("Loading CLIP encoder...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
        
        # Auto-detect state_dim from input_features if not explicitly set
        if config.state_dim is None:
            # Look for "observation.state" or "state" in input_features
            if "observation.state" in config.input_features:
                config.state_dim = config.input_features["observation.state"].shape[0]
                logging.info(f"Auto-detected state_dim={config.state_dim} from input_features['observation.state']")
            elif "state" in config.input_features:
                config.state_dim = config.input_features["state"].shape[0]
                logging.info(f"Auto-detected state_dim={config.state_dim} from input_features['state']")
            else:
                config.state_dim = 14
                logging.warning(f"Could not find state in input_features, using default state_dim={config.state_dim}")
        
        # Initialize SARM transformer
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
            use_joint_state=config.use_joint_state
        )
        self.sarm_transformer.to(self.device)
        
        # RA-BC running statistics (for weighted loss)
        if config.enable_rabc:
            self.register_buffer("rabc_mean", torch.tensor(0.0))
            self.register_buffer("rabc_m2", torch.tensor(0.0))
            self.register_buffer("rabc_count", torch.tensor(0))
        
        logging.info(f"SARM Reward Model initialized on {self.device}")
    
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
                inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True)
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
    
    def _update_rabc_stats(self, progress_deltas: torch.Tensor):
        """Update running statistics for RA-BC using Welford's online algorithm."""
        if not self.config.enable_rabc:
            return
        
        for delta in progress_deltas:
            self.rabc_count += 1
            delta_val = delta.item()
            delta_mean = delta_val - self.rabc_mean
            self.rabc_mean += delta_mean / self.rabc_count
            delta_m2 = delta_val - self.rabc_mean
            self.rabc_m2 += delta_mean * delta_m2
    
    def _compute_rabc_weights(self, progress_deltas: torch.Tensor) -> torch.Tensor:
        """Compute RA-BC weights for progress deltas."""
        if not self.config.enable_rabc or self.rabc_count < 2:
            return torch.ones_like(progress_deltas)
        
        # Get running statistics
        mean = max(self.rabc_mean.item(), 0.0)  # Clamp mean to non-negative
        variance = self.rabc_m2 / (self.rabc_count - 1)
        std = torch.sqrt(variance).item()
        
        # Compute soft weights
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        weights = (progress_deltas - lower_bound) / (4 * std + self.config.rabc_epsilon)
        weights = torch.clamp(weights, 0.0, 1.0)
        
        # Apply hard threshold
        high_quality_mask = progress_deltas > self.config.rabc_kappa
        weights = torch.where(high_quality_mask, torch.ones_like(weights), weights)
        
        return weights
    
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
    
    def forward(self, batch):
        """
        Forward pass compatible with lerobot training pipeline.
        
        Args:
            batch: Dictionary containing observation with:
                - 'video_features': Pre-encoded video features (B, T, 512)
                - 'text_features': Pre-encoded text features (B, 384)
                - 'state_features': Joint state features (B, T, state_dim)
        
        Returns:
            loss: Total training loss
            output_dict: Dictionary of loss components for logging
        """
        # Extract from observation dict
        observation = batch.get('observation', batch)
        video_features = observation['video_features'].to(self.device)
        text_features = observation['text_features'].to(self.device)
        state_features = observation.get('state_features', None)
        if state_features is not None:
            state_features = state_features.to(self.device)
        
        # Extract stage labels and progress targets if available (from subtask annotations)
        stage_labels = observation.get('stage_labels', None)
        if stage_labels is not None:
            stage_labels = stage_labels.to(self.device)
        
        progress_targets_from_annotations = observation.get('progress_targets', None)
        if progress_targets_from_annotations is not None:
            progress_targets_from_annotations = progress_targets_from_annotations.to(self.device)
        
        batch_size = video_features.shape[0]
        max_length = self.config.num_frames
        
        # Handle both single frames and sequences
        if video_features.dim() == 2:
            # Single frames: replicate to create pseudo-sequences
            video_features = video_features.unsqueeze(1).repeat(1, max_length, 1)
        
        if state_features is not None and state_features.dim() == 2:
            # Single state: replicate to match sequence length
            state_features = state_features.unsqueeze(1).repeat(1, max_length, 1)
        
        # Apply rewind augmentation (following SARM paper: up to 4 reversed frames)
        # Note: video_features are already sampled by dataset (9 frames with 30-frame gaps)
        # We just need to compute progress targets and optionally apply rewind
        
        processed_videos = []
        processed_states = []
        progress_targets = []
        
        # Extract episode metadata for correct progress normalization
        absolute_frame_indices = observation.get('absolute_frame_indices', None)
        episode_lengths = observation.get('episode_length', None)
        remaining_lengths = observation.get('remaining_length', None)
        
        for i in range(batch_size):
            # Get metadata for this sample
            current_absolute_indices = None
            current_episode_length = None
            current_remaining_length = None
            
            if absolute_frame_indices is not None:
                if isinstance(absolute_frame_indices, list):
                    current_absolute_indices = absolute_frame_indices[i]
                else:
                    current_absolute_indices = absolute_frame_indices
            
            if episode_lengths is not None:
                if isinstance(episode_lengths, torch.Tensor) and episode_lengths.dim() > 0:
                    current_episode_length = episode_lengths[i].item()
                else:
                    current_episode_length = episode_lengths.item() if isinstance(episode_lengths, torch.Tensor) else episode_lengths
            
            if remaining_lengths is not None:
                if isinstance(remaining_lengths, torch.Tensor) and remaining_lengths.dim() > 0:
                    current_remaining_length = remaining_lengths[i].item()
                else:
                    current_remaining_length = remaining_lengths.item() if isinstance(remaining_lengths, torch.Tensor) else remaining_lengths
            
            # Compute progress targets directly from metadata (frames already loaded by dataset)
            # Progress = (position_in_sequence + 1) / remaining_trajectory_length
            if current_remaining_length is not None and current_remaining_length > 0:
                # Correct: relative progress from first loaded frame to episode end
                progress_indices = torch.arange(1, max_length + 1, dtype=torch.float32, device=self.device)
                progress = progress_indices / current_remaining_length
            else:
                # Fallback: linear progress (when metadata is not available)
                logging.warning(f"Sample {i}: No remaining_length metadata, using linear progress fallback")
                progress = torch.linspace(1.0/max_length, 1.0, max_length, device=self.device)
            
            # Apply rewind augmentation with 50% probability (following SARM paper)
            # Paper specifies: "appending up to four frames from earlier timestamps with reversed order"
            if random.random() < 0.5:
                # Rewind: append 2-4 reversed frames, trim to max_length
                num_reverse = random.randint(2, min(4, max_length - 1))
                
                # Reverse video and progress
                reversed_video = video_features[i].flip(0)
                reversed_progress = progress.flip(0)
                
                # Take frames from reversed (skip first which is last of original)
                reverse_frames = reversed_video[1:num_reverse+1]
                reverse_progress = reversed_progress[1:num_reverse+1]
                
                # Concatenate forward + reversed
                rewound_video = torch.cat([video_features[i], reverse_frames], dim=0)
                rewound_progress = torch.cat([progress, reverse_progress], dim=0)
                
                # Trim to max_length
                rewound_video = rewound_video[:max_length]
                rewound_progress = rewound_progress[:max_length]
                
                processed_videos.append(rewound_video)
                progress_targets.append(rewound_progress)
                
                # Process state features if available
                if state_features is not None:
                    reversed_state = state_features[i].flip(0)
                    reverse_state_frames = reversed_state[1:num_reverse+1]
                    rewound_state = torch.cat([state_features[i], reverse_state_frames], dim=0)
                    rewound_state = rewound_state[:max_length]
                    processed_states.append(rewound_state)
            else:
                # Normal: use frames as-is with forward progress
                processed_videos.append(video_features[i])
                progress_targets.append(progress)
                
                # Process state features if available
                if state_features is not None:
                    processed_states.append(state_features[i])
        
        # Ensure all sequences have the same length before stacking
        # (sampling functions should return max_length, but double-check)
        validated_videos = []
        validated_progress = []
        for i, (vid, prog) in enumerate(zip(processed_videos, progress_targets)):
            if len(vid) != max_length:
                logging.warning(f"Sample {i}: video length {len(vid)} != {max_length}, padding/trimming")
                if len(vid) < max_length:
                    # Pad
                    padding = max_length - len(vid)
                    vid = torch.cat([vid, vid[-1:].repeat(padding, 1)])
                    prog = torch.cat([prog, torch.full((padding,), prog[-1], device=prog.device)])
                else:
                    # Trim
                    vid = vid[:max_length]
                    prog = prog[:max_length]
            validated_videos.append(vid)
            validated_progress.append(prog)
        
        # Stack processed features
        processed_videos = torch.stack(validated_videos)
        progress_targets = torch.stack(validated_progress)
        
        # Ensure progress_targets has the same shape as progress_preds
        # progress_preds is (batch_size, num_frames, 1)
        # progress_targets is (batch_size, num_frames) -> add last dimension
        if progress_targets.dim() == 2:
            progress_targets = progress_targets.unsqueeze(-1)  # (batch_size, num_frames, 1)
        
        if state_features is not None and len(processed_states) > 0:
            processed_states = torch.stack(processed_states)
        else:
            processed_states = None
        
        # Get predictions
        stage_logits, stage_probs, progress_preds = self.sarm_transformer(
            processed_videos, text_features, processed_states
        )
        
        # Use annotation-based progress targets if available, otherwise use computed ones
        if progress_targets_from_annotations is not None and len(processed_videos) == 1:
            # Use refined progress from subtask annotations (single sample case)
            # Ensure shapes match
            if progress_targets_from_annotations.shape != progress_preds.shape:
                if progress_targets_from_annotations.dim() == 2:
                    progress_targets_from_annotations = progress_targets_from_annotations.unsqueeze(0)
            progress_targets = progress_targets_from_annotations
        
        # Compute progress loss using targets
        progress_loss = F.mse_loss(progress_preds, progress_targets)
        
        # Compute stage loss if labels are available
        stage_loss = None
        if stage_labels is not None and len(processed_videos) == 1:
            # Ensure stage_labels matches the sequence length
            if stage_labels.dim() == 1 and stage_logits.dim() == 3:
                # stage_labels: (seq_len,) -> need to expand to (batch, seq_len)
                stage_labels = stage_labels.unsqueeze(0).expand(stage_logits.shape[0], -1)
            elif stage_labels.shape[0] != stage_logits.shape[0]:
                # Single label for batch - expand
                stage_labels = stage_labels.expand(stage_logits.shape[0], stage_logits.shape[1])
            
            # Compute cross-entropy loss for stage classification
            stage_loss = compute_stage_loss(stage_logits, stage_labels)
        
        # Combine losses
        if stage_loss is not None:
            total_loss = progress_loss + self.config.stage_loss_weight * stage_loss
            output_dict = {
                'progress_loss': progress_loss.item(),
                'stage_loss': stage_loss.item(),
            }
        else:
            total_loss = progress_loss
            output_dict = {
                'progress_loss': progress_loss.item(),
            }
        
        # Compute misaligned loss (following SARM paper and ReWiND)
        # "To improve video-language alignment, task descriptions are occasionally perturbed"
        if random.random() < 0.2:  # 20% probability (matching ReWiND)
            # Create misaligned pairs by shuffling text features
            shuffle_idx = torch.randperm(batch_size, device=self.device)
            misaligned_texts = text_features[shuffle_idx]
            
            # Get predictions for misaligned pairs (should predict zero progress)
            _, _, misaligned_preds = self.sarm_transformer(
                processed_videos, misaligned_texts, processed_states
            )
            
            # Target is zero progress for misaligned pairs
            target_zeros = torch.zeros_like(misaligned_preds)
            misaligned_loss = F.mse_loss(misaligned_preds, target_zeros)
            
            # Add to total loss
            total_loss = total_loss + misaligned_loss
            output_dict['misaligned_loss'] = misaligned_loss.item()
        
        # RA-BC weighted loss (if enabled)
        if self.config.enable_rabc:
            # Compute progress deltas (simplified: use consecutive frame differences)
            progress_deltas = progress_preds[:, 1:, 0] - progress_preds[:, :-1, 0]
            progress_deltas = progress_deltas.mean(dim=1)  # Average over sequence
            
            # Update running statistics
            self._update_rabc_stats(progress_deltas)
            
            # Compute weights
            weights = self._compute_rabc_weights(progress_deltas)
            
            # Apply weighted loss
            weighted_loss = (total_loss * weights.mean()).sum()
            total_loss = weighted_loss
        
        # Add final total loss to output dict
        output_dict['total_loss'] = total_loss.item()
        
        return total_loss, output_dict


# Loss utilities
def compute_stage_loss(
    stage_logits: torch.Tensor,
    target_stages: torch.Tensor
) -> torch.Tensor:
    """
    Compute stage classification loss.
    
    Args:
        stage_logits: Stage predictions (batch_size, num_frames, num_stages)
        target_stages: Target stage indices (batch_size, num_frames)
        
    Returns:
        Cross-entropy loss
    """
    batch_size, num_frames, num_stages = stage_logits.shape
    stage_logits_flat = stage_logits.reshape(-1, num_stages)
    target_stages_flat = target_stages.reshape(-1)
    
    loss = F.cross_entropy(stage_logits_flat, target_stages_flat)
    return loss


def compute_progress_loss(
    progress_preds: torch.Tensor,
    target_progress: torch.Tensor
) -> torch.Tensor:
    """
    Compute progress regression loss.
    
    Args:
        progress_preds: Progress predictions (batch_size, num_frames, 1)
        target_progress: Target progress values (batch_size, num_frames, 1)
        
    Returns:
        Mean squared error loss
    """
    loss = F.mse_loss(progress_preds, target_progress)
    return loss

