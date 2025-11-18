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
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torch import Tensor

from lerobot.policies.rewind.configuration_rewind import ReWiNDConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.datasets.video_sampler import sample_video_feature, sample_reverse_video_feature


# Helper functions for encoding
def dino_load_image(img: np.ndarray) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    
    Args:
        img: Input image as numpy array (H, W, C) in uint8 format.
        
    Returns:
        Transformed image tensor ready for DINO encoder (1, 3, 224, 224).
    """
    # Define transform: center crop to 224x224, normalize to [-1, 1]
    dino_transform = T.Compose([
        T.ToTensor(),
        T.CenterCrop(224),
        T.Normalize([0.5], [0.5])
    ])
    
    img_pil = Image.fromarray(img)
    transformed_img = dino_transform(img_pil)[:3].unsqueeze(0)
    
    return transformed_img


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


class ReWiNDTransformer(nn.Module):
    """
    ReWiND Transformer model for predicting task progress from video and text.
    
    This model takes video frame embeddings and text embeddings as input,
    and predicts a progress score (0-1) for each frame indicating how much
    of the task has been completed.
    """
    
    def __init__(
        self, 
        video_dim: int = 768,
        text_dim: int = 384, 
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        max_length: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Project video and text to common dimension
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Position embeddings for video sequence
        # We only add positional embedding to the first frame as in the original
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
        
        # Progress prediction head (applied to each frame)
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Attention mask for causal self-attention
        # Will be created on-demand based on sequence length
        self.register_buffer("attention_mask", None, persistent=False)
    
    def _get_attention_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Generate or retrieve cached causal attention mask."""
        if self.attention_mask is None or self.attention_mask.shape[0] != seq_length:
            # Create causal mask (upper triangular with -inf)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=device)
            self.attention_mask = mask
        return self.attention_mask
    
    def forward(self, video_frames: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ReWiND transformer.
        
        Args:
            video_frames: Video frame embeddings (batch_size, seq_len, video_dim)
            text_embed: Text embeddings (batch_size, text_dim)
            
        Returns:
            Progress predictions for each frame (batch_size, seq_len, 1)
        """
        batch_size = video_frames.shape[0]
        
        # Project inputs to common dimension
        video_embed = self.video_proj(video_frames)  # [batch_size, seq_len, hidden_dim]
        text_embed = self.text_proj(text_embed).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Add positional embedding to first video frame
        video_embed[:, 0] += self.first_pos_embed
        
        # Combine sequence: [text, video_frames]
        sequence = torch.cat([text_embed, video_embed], dim=1)
        
        # Get causal attention mask
        seq_length = sequence.shape[1]
        attention_mask = self._get_attention_mask(seq_length, sequence.device)
        
        # Pass through transformer with causal masking
        transformed = self.transformer(sequence, mask=attention_mask, is_causal=True)
        
        # Get progress predictions for each frame (exclude text token)
        progress_preds = self.progress_head(transformed[:, 1:])
        
        return progress_preds


class ReWiNDRewardModel(PreTrainedPolicy):
    """
    ReWiND Reward Model for computing task completion rewards from video and text.
    
    This model combines:
    - DINO (DINOv2) for encoding video frames
    - MiniLM for encoding text descriptions
    - ReWiNDTransformer for predicting task progress
    """
    
    name = "rewind"
    config_class = ReWiNDConfig
    
    def __init__(self, config: ReWiNDConfig, dataset_stats: dict | None = None):
        super().__init__(config, dataset_stats)
        self.config = config
        self.dataset_stats = dataset_stats
        self.device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize DINO encoder for images
        logging.info("Loading DINO encoder...")
        self.dino_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.dino_encoder.to(self.device)
        self.dino_encoder.eval()
        
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
        
        # Initialize ReWiND transformer with explicit architecture parameters
        self.rewind_transformer = ReWiNDTransformer(
            video_dim=config.video_dim,
            text_dim=config.text_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_length=config.max_length,
            dropout=config.dropout
        )
        self.rewind_transformer.to(self.device)
        
        logging.info(f"ReWiND Reward Model initialized on {self.device}")
    
    def to(self, device):
        """Override to method to ensure all components move together."""
        super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dino_encoder.to(device)
        self.minilm_model.to(device)
        self.rewind_transformer.to(device)
        return self
    
    @torch.no_grad()
    def encode_images(self, images: np.ndarray) -> np.ndarray:
        """
        Encode video frames using DINO.
        
        Args:
            images: Video frames with shape (num_videos, num_frames, H, W, C) in uint8.
                   Can also be (num_frames, H, W, C) for a single video.
                   
        Returns:
            Encoded image features (num_videos, num_frames, 768) or (num_frames, 768).
        """
        # Handle single video case
        single_video = False
        if len(images.shape) == 4:
            images = images[np.newaxis, ...]
            single_video = True
        
        assert len(images.shape) == 5, f"Expected 5D input (num_videos, num_frames, H, W, C), got {images.shape}"
        
        # Ensure channels are in correct position
        if images.shape[-1] == 3 and images.shape[2] != 3:
            images = np.transpose(images, (0, 1, 4, 2, 3))
        
        all_embeddings = []
        
        for video in images:
            # Process each video
            video_embeddings = []
            
            # Convert frames to list of numpy arrays
            frames = [frame.transpose(1, 2, 0).astype(np.uint8) if frame.shape[0] == 3 else frame for frame in video]
            
            # Batch process frames with DINO
            episode_images_dino = [dino_load_image(frame) for frame in frames]
            
            # Process in batches
            for i in range(0, len(episode_images_dino), self.config.dino_batch_size):
                batch = torch.cat(episode_images_dino[i:i + self.config.dino_batch_size])
                batch = batch.to(self.device)
                embeddings = self.dino_encoder(batch).squeeze().detach().cpu()
                
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
    
    def padding_video(self, video_frames: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Pad or subsample video frames to a fixed length.
        
        Args:
            video_frames: Video frames tensor (num_frames, embedding_dim)
            max_length: Target sequence length
            
        Returns:
            Padded/subsampled video frames (max_length, embedding_dim)
        """
        video_length = len(video_frames)
        
        if isinstance(video_frames, np.ndarray):
            video_frames = torch.tensor(video_frames)
        
        if video_length < max_length:
            # Pad with last frame
            padding_length = max_length - video_length
            last_frame = video_frames[-1].unsqueeze(0)
            padding_frames = last_frame.repeat(padding_length, 1)
            video_frames = torch.cat([video_frames, padding_frames], dim=0)
        
        elif video_length > max_length:
            # Subsample uniformly
            frame_idx = np.linspace(0, video_length - 1, max_length).astype(int)
            video_frames = video_frames[frame_idx]
        
        return video_frames
    
    @torch.no_grad()
    def calculate_rewards(
        self,
        text_embeddings: Union[np.ndarray, torch.Tensor],
        video_embeddings: Union[np.ndarray, torch.Tensor],
        return_all_frames: bool = False
    ) -> np.ndarray:
        """
        Calculate rewards for given text and video representations.
        
        Args:
            text_embeddings: Encoded text representations (batch_size, 384)
            video_embeddings: Encoded video representations (batch_size, num_frames, 768)
            return_all_frames: If True, return rewards for all frames. If False, return only last frame.
            
        Returns:
            Reward values (batch_size,) or (batch_size, num_frames) if return_all_frames=True
        """
        # Convert to tensors if needed
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        if isinstance(video_embeddings, np.ndarray):
            video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32)
        
        # Handle single sample case
        if text_embeddings.dim() == 1:
            text_embeddings = text_embeddings.unsqueeze(0)
            video_embeddings = video_embeddings.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Process in batches
        all_rewards = []
        for i in range(0, len(video_embeddings), self.config.batch_size):
            batch_texts = text_embeddings[i:i + self.config.batch_size].to(self.device)
            batch_videos = video_embeddings[i:i + self.config.batch_size].to(self.device)
            
            # Pad/subsample videos if needed
            if self.config.subsample_video:
                padded_videos = []
                for video in batch_videos:
                    padded_video = self.padding_video(video, self.config.max_length)
                    padded_videos.append(padded_video)
                batch_videos = torch.stack(padded_videos).to(self.device)
            
            # Get progress predictions
            rewards = self.rewind_transformer(batch_videos.float(), batch_texts.float())
            
            if return_all_frames:
                all_rewards.append(rewards.squeeze(-1).cpu())
            else:
                # Return only last frame reward
                all_rewards.append(rewards[:, -1, 0].cpu())
        
        result = torch.cat(all_rewards).numpy()
        
        if single_sample:
            result = result[0] if not return_all_frames else result[0]
        
        return result
    
    def load_pretrained_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained model weights from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the .pth checkpoint file
            strict: Whether to strictly enforce that the keys match
        """
        logging.info(f"Loading pretrained checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            
            # Check for architecture parameters in checkpoint
            if "args" in checkpoint:
                args = checkpoint["args"]
                logging.info(f"Checkpoint was trained with: max_length={args.max_length}")
                
                # Warn if max_length differs
                if hasattr(args, 'max_length') and args.max_length != self.config.max_length:
                    logging.warning(
                        f"Checkpoint max_length ({args.max_length}) differs from config ({self.config.max_length}). "
                        "This may cause issues if sequence lengths don't match."
                    )
        else:
            state_dict = checkpoint
        
        # Load only the ReWiNDTransformer weights
        missing_keys, unexpected_keys = self.rewind_transformer.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            logging.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        
        logging.info("Checkpoint loaded successfully")
    
    def train(self, mode: bool = True):
        """Set training mode. Note: DINO and MiniLM encoders always stay in eval mode."""
        super().train(mode)
        # Keep encoders in eval mode
        self.dino_encoder.eval()
        self.minilm_model.eval()
        # Only transformer can be trained
        self.rewind_transformer.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def parameters(self):
        """Return trainable parameters (only ReWiND transformer, not encoders)."""
        return self.rewind_transformer.parameters()
    
    def get_optim_params(self):
        """Return optimizer parameters for the policy."""
        return self.parameters()
    
    def reset(self):
        """
        This method is required by PreTrainedPolicy but not used for reward models.
        The reward model does not maintain state between episodes.
        """
        pass
    
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for reward models.
        The rewind model is not an actor and does not produce action chunks.
        """
        raise NotImplementedError("Rewind model does not predict action chunks")
    
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for rewind.
        The rewind model is not an actor and does not select actions.
        """
        raise NotImplementedError("Rewind model does not select actions")
    
    def forward(self, batch):
        """
        Forward pass compatible with lerobot training pipeline.

        Args:
            batch: Dictionary containing observation with:
                - 'video_features': Pre-encoded video features (B, 768) or (B, T, 768)
                - 'text_features': Pre-encoded text features (B, 384)
        
        Returns:
            loss: Total training loss
            output_dict: Dictionary of loss components for logging
        """
        # Extract from observation dict
        observation = batch.get('observation', batch)
        video_features = observation['video_features'].to(self.device)
        text_features = observation['text_features'].to(self.device)
        
        batch_size = video_features.shape[0]
        max_length = self.config.max_length
        
        # Handle both single frames (B, 768) and sequences (B, T, 768)
        if video_features.dim() == 2:
            # Single frames: replicate to create pseudo-sequences
            video_features = video_features.unsqueeze(1).repeat(1, max_length, 1)  # (B, max_length, 768)
        
        # Now video_features is (B, T, 768) where T might be > max_length
        
        # Process videos (with potential rewind augmentation)
        import random
        from lerobot.datasets.video_sampler import sample_video_feature, sample_reverse_video_feature
        
        processed_videos = []
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
            
            if random.random() < 0.5:  # 50% chance of rewind
                # Apply video rewind augmentation (now returns tuple)
                rewound_video, progress = sample_reverse_video_feature(
                    video_features[i],
                    max_length=max_length,
                    random_sample=False,  # Use consecutive frames, not random sampling
                    remaining_length=current_remaining_length,
                    absolute_indices=current_absolute_indices,
                    episode_length=current_episode_length
                )
                processed_videos.append(rewound_video.to(self.device))
                progress_targets.append(progress.to(self.device))
            else:
                # Normal video sampling (now returns tuple with progress targets)
                sampled_video, progress = sample_video_feature(
                    video_features[i],
                    max_length=max_length,
                    random_sample=False,  # Use consecutive frames, not random sampling
                    remaining_length=current_remaining_length,
                    absolute_indices=current_absolute_indices,
                    episode_length=current_episode_length
                )
                processed_videos.append(sampled_video.to(self.device))
                progress_targets.append(progress.to(self.device))
        
        processed_videos = torch.stack(processed_videos)
        progress_targets = torch.stack(progress_targets)
        
        # Compute progress loss
        progress_loss = compute_progress_loss(
            self.rewind_transformer,
            processed_videos,
            text_features,
            progress_targets
        )
        
        total_loss = progress_loss
        output_dict = {'progress_loss': progress_loss.item()}
        
        # Compute misaligned loss if requested (20% probability to match original)
        if random.random() < 0.2:  # 20% chance of adding misalignment loss (original ReWiND uses 20%)
            if 'misaligned_video_features' in batch and 'misaligned_text_features' in batch:
                misaligned_videos = batch['misaligned_video_features'].to(self.device)
                misaligned_texts = batch['misaligned_text_features'].to(self.device)
            else:
                # Create misaligned pairs by shuffling
                shuffle_idx = torch.randperm(batch_size)
                misaligned_videos = processed_videos[shuffle_idx]
                misaligned_texts = text_features
            
            # Sample misaligned videos (function now returns tuple)
            # For misaligned pairs, we don't need correct progress targets (will be set to 0)
            misaligned_videos_sampled = []
            for i in range(batch_size):
                sampled, _ = sample_video_feature(
                    misaligned_videos[i],
                    max_length=max_length,
                    random_sample=True,
                    absolute_indices=None,
                    episode_length=None
                )
                misaligned_videos_sampled.append(sampled.to(self.device))
            misaligned_videos_sampled = torch.stack(misaligned_videos_sampled)
            
            misaligned_loss = compute_misaligned_loss(
                self.rewind_transformer,
                misaligned_videos_sampled,
                misaligned_texts
            )
            
            total_loss = total_loss + misaligned_loss
            output_dict['misaligned_loss'] = misaligned_loss.item()
        
        output_dict['total_loss'] = total_loss.item()
        
        return total_loss, output_dict


# Loss utilities
def compute_progress_loss(
    model: ReWiNDTransformer,
    video_features: torch.Tensor,
    text_features: torch.Tensor,
    target_progress: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute progress prediction loss.
    
    Args:
        model: ReWiNDTransformer model
        video_features: Batch of video features (batch_size, max_length, feature_dim)
        text_features: Batch of text features (batch_size, text_dim)
        target_progress: Optional target progress values (batch_size, max_length).
                        If None, uses linear progress from 0 to 1.
        
    Returns:
        Mean squared error loss
    """
    # Get predictions
    progress_preds = model(video_features, text_features)
    
    # Create target progress if not provided
    if target_progress is None:
        batch_size, max_length = video_features.shape[:2]
        target_progress = torch.linspace(0, 1, max_length, device=video_features.device)
        target_progress = target_progress.unsqueeze(0).repeat(batch_size, 1)
    
    # Ensure target has correct shape
    if target_progress.dim() == 2:
        target_progress = target_progress.unsqueeze(-1)
    
    # Compute MSE loss
    loss = F.mse_loss(progress_preds, target_progress)
    
    return loss


def compute_misaligned_loss(
    model: ReWiNDTransformer,
    video_features: torch.Tensor,
    misaligned_text_features: torch.Tensor
) -> torch.Tensor:
    """
    Compute loss for misaligned video-text pairs (should predict 0 progress).
    
    Args:
        model: ReWiNDTransformer model
        video_features: Batch of video features (batch_size, max_length, feature_dim)
        misaligned_text_features: Batch of misaligned text features (batch_size, text_dim)
        
    Returns:
        Mean squared error loss (predictions should be close to 0)
    """
    # Get predictions
    progress_preds = model(video_features, misaligned_text_features)
    
    # Target is all zeros
    target_zeros = torch.zeros_like(progress_preds)
    
    # Compute MSE loss
    loss = F.mse_loss(progress_preds, target_zeros)
    
    return loss
