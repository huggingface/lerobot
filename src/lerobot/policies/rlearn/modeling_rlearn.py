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

"""
RLearN: Video-Language Conditioned Reward Model (ReWiND Implementation)

This implementation follows the ReWiND paper approach (arXiv:2505.10911v1):
- Automatically generates linear progress labels (0 to 1) for each episode
- No need for pre-annotated rewards in the dataset
- Applies video rewinding augmentation to create synthetic failure trajectories

Inputs
  - images: (B, T, C, H, W)  sequence of frames (or single frame with T=1)
  - language: list[str] of length B (goal/instruction)

High-level Architecture

  images (B,T,C,H,W)
        |
        |  per-frame encode
        v
  +------------------------------+
  |  Vision Encoder (frozen)     |  e.g. SigLIP2 (base)
  +------------------------------+
        |s
        |  pooled per-frame embeddings (BT, H_v)
        v
  reshape -> (B, T, H_v) -- Linear proj --> (B, T, D)
                                    +  Positional Encoding [0..T)
                                    +  Optional first-frame bias
                                    |
                                    |          language (B, str)
                                    |                 |
                                    |                 v
                                    |      +------------------------------+
                                    |      |  Text Encoder (frozen)       |  e.g. SigLIP2
                                    |      +------------------------------+
                                    |                 |
                                    |                 |  pooled text embedding (B, H_t)
                                    |                 v
                                    |           Linear proj -> (B, D)
                                    |                 |
                                    +-----------------v----------------------+
                                                      |
                           +--------------------------v---------------------------+
                           |  Temporal Causal Transformer (n_layers, n_heads)     |
                           |    - self-attention over time with causal mask       |
                           |    - cross-attention to a single language token      |
                           +--------------------------+---------------------------+
                                                      |
                                            LayerNorm + Linear Head (D -> 1)
                                                      |
                                                      v
  Output
    - reward_logits: (B, T', 1)  with T' ≤ T (affected by stride and frame dropout)

Notes
  - Uses SigLIP2 for both vision and text encoding.
  - Backbones (vision/text) are frozen by default; only projections, temporal module, and head are trainable.
  - Stride/frame dropout applied during training can subsample timesteps.
"""

from __future__ import annotations

import math
import numpy as np
from contextlib import nullcontext
from itertools import chain
from operator import truediv

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

# ReWiND dependencies
try:
    from x_transformers import Decoder
    import einx
    from einops import rearrange, repeat, pack, unpack
except ImportError as e:
    raise ImportError(
        "ReWiND dependencies not installed. Please install: "
        "pip install x-transformers einx einops x-mlps-pytorch"
    ) from e

from lerobot.constants import OBS_IMAGE, OBS_IMAGES, OBS_LANGUAGE, REWARD
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rlearn.configuration_rlearn import RLearNConfig


class RLearNPolicy(PreTrainedPolicy):
    """Video-language conditioned reward model following ReWiND architecture exactly: https://github.com/lucidrains/rewind-reward-pytorch/blob/main/rewind_reward_pytorch/rewind_reward.py#L11.

    - Visual encoder: frozen SigLIP2, returns per-frame embeddings.
    - Text encoder: frozen SigLIP2, returns a language embedding.
    - Temporal module: x_transformers Decoder with packed tokens [lang | register | video].
    - Output: per-timestep rewards via simple linear regression head.
    """

    config_class = RLearNConfig
    name = "rlearn"

    def __init__(self, config: RLearNConfig, episode_data_index: dict = None):
        super().__init__(config)
        self.config = config
        self.episode_data_index = episode_data_index  # Store episode boundaries for progress calculation

        # Encoders - DINOv3 for vision, SigLIP2 for text
        from transformers import AutoProcessor, AutoModel, AutoImageProcessor
        
        # Load DINOv3 processor and model for vision
        self.vision_processor = AutoImageProcessor.from_pretrained(config.vision_model_name)
        self.vision_model = AutoModel.from_pretrained(config.vision_model_name)
        
        # Load SigLIP2 processor and model for text
        self.text_processor = AutoProcessor.from_pretrained(config.text_model_name, use_fast=True)
        self.text_model = AutoModel.from_pretrained(config.text_model_name)
        
        # Move encoders to GPU if available
        if torch.cuda.is_available():
            self.vision_model = self.vision_model.to('cuda')
            self.text_model = self.text_model.to('cuda')
        
        # Get hidden sizes from models
        # DINOv3-ViTL16 has hidden_size directly in config
        self.vision_hidden = getattr(self.vision_model.config, 'hidden_size', 1024)  # DINOv3-large default
        
        # SigLIP2 text model
        th = getattr(getattr(self.text_model, 'config', None), 'text_config', None)
        self.text_hidden = getattr(th, 'hidden_size', 512)

        # Freeze encoders if requested
        if config.freeze_backbones:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            for p in self.text_model.parameters():
                p.requires_grad = False
        
        # Ensure frozen encoders run in eval mode (no dropout, stable outputs)
        self.vision_model.eval()
        self.text_model.eval()

        # x_transformers Decoder (matching ReWiND exactly)
        self.decoder = Decoder(
            dim=config.dim_model,
            depth=config.n_layers,
            heads=config.n_heads,
            attn_dim_head=64,  # ReWiND default
            ff_mult=config.dim_feedforward // config.dim_model,  # Convert to multiplier
            # Note: x_transformers uses attn_dropout and ff_dropout separately
            attn_dropout=config.dropout,
            ff_dropout=config.dropout,
        )
        
        # Linear projections to the shared temporal model dimension
        self.to_lang_tokens = nn.Linear(self.text_hidden, config.dim_model)
        self.to_video_tokens = nn.Linear(self.vision_hidden, config.dim_model)

        # Stronger temporal positional encoding
        self.temporal_pos_embedding = nn.Parameter(torch.randn(config.max_seq_len, config.dim_model) * 0.1)
        
        # Single MLP processes all frames
        self.frame_mlp = nn.Linear(config.dim_model, config.dim_model)
        
        # Register / memory / attention sink tokens
        self.num_register_tokens = config.num_register_tokens
        self.register_tokens = nn.Parameter(torch.randn(config.num_register_tokens, config.dim_model) * 1e-2)

        # MLP predictor (matching ReWiND's Feedforwards)
        from x_mlps_pytorch import Feedforwards
        self.mlp_predictor = Feedforwards(
            dim=config.dim_model,
            dim_out=None,
            depth=config.mlp_predictor_depth
        )
        
        # FIXED: Simpler head architecture to prevent constant output pathology
        # Remove LayerNorm (causes flat basin), reduce depth, larger init, less dropout
        
        # Simple 2-layer MLP with larger initialization to encourage exploration
        self.reward_head = nn.Sequential(
            nn.Linear(config.dim_model + 1, config.head_hidden_dim),  # +1 for temporal position  
            nn.ReLU(),
            nn.Dropout(0.05),  # Reduced dropout to prevent noise-induced conservatism
            nn.Linear(config.head_hidden_dim, 1)
        )
        
        # FIXED: Larger weight initialization to escape flat basins
        with torch.no_grad():
            for module in self.reward_head:
                if isinstance(module, nn.Linear):
                    # Use Xavier/Glorot initialization for better gradient flow
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    nn.init.zeros_(module.bias)
        
        # Simple frame dropout probability
        self.frame_dropout_p = config.frame_dropout_p
        self.stride = max(1, config.stride)

        # Auto-load episode_data_index from episodes.jsonl if not provided
        if self.episode_data_index is None and getattr(config, "episodes_jsonl_path", None):
            try:
                self.episode_data_index = self._load_episode_index_from_jsonl(config.episodes_jsonl_path)
            except Exception:
                # Defer to runtime error with guidance if loading fails
                self.episode_data_index = None
        
        # Apply torch.compile for additional speedup if enabled
        if getattr(config, "compile_model", False):
            try:
                self.vision_model = torch.compile(self.vision_model, mode="reduce-overhead")
                self.text_model = torch.compile(self.text_model, mode="reduce-overhead")
                self.decoder = torch.compile(self.decoder, mode="reduce-overhead")
                print("✅ Applied torch.compile to encoders and transformer")
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}")
                # Continue without compilation

    def get_optim_params(self) -> list:
        """Return parameter groups with head LR boost."""
        base_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "reward_head" in name:
                    head_params.append(param)
                else:
                    base_params.append(param)
        
        return [
            {"params": base_params},
            {"params": head_params, "lr": self.config.learning_rate * self.config.head_lr_multiplier}
        ]

    def reset(self):
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:  # Required by base class
        raise NotImplementedError("RLearN is a reward model and does not predict actions")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:  # Required by base class
        raise NotImplementedError("RLearN is a reward model and does not select actions")

    @torch.no_grad()
    def predict_rewards(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict per-timestep rewards for evaluation using ReWiND architecture.

        Args:
            batch: Input batch with OBS_IMAGES and optionally OBS_LANGUAGE

        Returns:
            Predicted rewards tensor of shape (B, T)
        """
        batch = self.normalize_inputs(batch)

        # Extract frames and form (B, T, C, H, W)
        frames = extract_visual_sequence(batch, target_seq_len=self.config.max_seq_len)
        B, T, C, H, W = frames.shape

        # CRITICAL FIX: Do NOT apply stride during evaluation
        # During evaluation, we want to process all frames in the sliding window
        # Stride should only be used during training to reduce computational cost
        T_eff = T  # Use all frames during evaluation

        # Get language commands
        commands = batch.get(OBS_LANGUAGE, None)
        if commands is None:
            commands = [""] * B
        elif not isinstance(commands, list):
            commands = [str(commands)] * B

        # Forward through ReWiND model (inference mode)
        device = next(self.parameters()).device
        frames = frames.to(device)
        
        # Process video frames
        video_embeds = self._encode_video_frames(frames).to(device)  # (B, T, D_vision)
        
        # Language embeddings + mask
        lang_embeds, mask = self._encode_language_tokens(commands, device)
        
        # Register tokens
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b=B)
        
        # Project embeddings
        lang_tokens = self.to_lang_tokens(lang_embeds)
        video_tokens = self.to_video_tokens(video_embeds)
        # Add temporal positional encoding (window-relative only)
        T_video = video_tokens.shape[1] 
        video_tokens = video_tokens + self.temporal_pos_embedding[:T_video]
        
        # Pack all tokens for attention
        tokens, lang_video_packed_shape = pack((lang_tokens, register_tokens, video_tokens), 'b * d')
        
        # Extend mask for register and video tokens
        mask = F.pad(mask, (0, register_tokens.shape[1] + video_tokens.shape[1]), value=True)
        
        # Forward through decoder
        attended = self.decoder(tokens, mask=mask)
        
        # Unpack and get video token features
        _, _, attended_video_tokens = unpack(attended, lang_video_packed_shape, 'b * d')
        
        # Process all frames with single MLP
        frame_tokens = self.frame_mlp(attended_video_tokens)  # (B, T, D)
        
        # MLP predictor
        video_frame_embeds = self.mlp_predictor(frame_tokens)
        
        # Get rewards via temporal-aware logit regression head (no pre-normalization)
        
        # Add temporal position information
        B, T_pred = video_frame_embeds.shape[:2]
        temporal_pos = torch.linspace(0, 1, T_pred, device=video_frame_embeds.device)
        temporal_pos = temporal_pos.unsqueeze(0).unsqueeze(-1).expand(B, T_pred, 1)  # (B, T, 1)
        
        # Concatenate embeddings with temporal position
        temporal_input = torch.cat([video_frame_embeds, temporal_pos], dim=-1)  # (B, T, D+1)
        
        # Forward through temporal-aware head
        raw_logits = self.reward_head(temporal_input).squeeze(-1)  # (B, T)
        return torch.sigmoid(raw_logits)  # Apply sigmoid for final predictions

    def normalize_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Initial version: no-op; rely on upstream processors if any
        return batch

    def normalize_targets(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Initial version: no-op
        return batch

    def _encode_video_frames(self, frames: Tensor) -> Tensor:
        """Encode video frames through SigLIP2 to get per-frame embeddings.
        
        Args:
            frames: (B, T, C, H, W)
        
        Returns:
            (B, T, D_vision)
        """
        B, T, C, H, W = frames.shape
        flat = rearrange(frames, 'b t c h w -> (b t) c h w')
        
        # Optimized: Process tensor directly without numpy conversion
        device = next(self.vision_model.parameters()).device
        
        # Normalize to [0, 1] if needed and ensure correct format for DINOv3
        if flat.dtype != torch.float32:
            flat = flat.float()
        if flat.max() > 1.0:
            flat = flat / 255.0
            
        # DINOv3 expects images in [0, 1] range, RGB format
        # Convert tensor to list of PIL-like arrays for processor
        flat_numpy = flat.permute(0, 2, 3, 1).cpu().numpy()  # (BT, H, W, C)
        images_list = [flat_numpy[i] for i in range(B * T)]
        
        # Process through DINOv3 processor and model
        inputs = self.vision_processor(images=images_list, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Process in batch through DINOv3 model
        # Use inference mode for stable, fast frozen encoder forward
        with torch.inference_mode():
            vision_outputs = self.vision_model(**inputs)
        
        # Prefer mean-pooled patch tokens over pooler/CLS to ensure input-dependent variation
        if hasattr(vision_outputs, 'last_hidden_state') and vision_outputs.last_hidden_state is not None:
            tokens = vision_outputs.last_hidden_state  # (BT, N_tokens, D)
            if tokens.dim() == 3 and tokens.shape[1] > 1:
                # Exclude CLS/reg token at index 0, average over patch tokens
                vision_features_flat = tokens[:, 1:, :].mean(dim=1)
            else:
                # Fallback to first token if only one token is present
                vision_features_flat = tokens[:, 0]
        elif hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
            vision_features_flat = vision_outputs.pooler_output  # (BT, D)
        else:
            raise RuntimeError("DINOv3 outputs do not contain last_hidden_state or pooler_output")
        
        # Reshape to (B, T, D) for analysis
        vision_features = rearrange(vision_features_flat, '(b t) d -> b t d', b=B, t=T)
        
        # DEBUG: Analyze vision feature variability
        if self.training and torch.rand(1).item() < 0.1:  # 10% of training steps for more frequent debugging
            with torch.no_grad():
                print(f"\n🔍 DINOv3 VISION FEATURE DEBUG (B={B}, T={T}):")
                
                # CRITICAL: Check if input frames are actually different
                print(f"Raw frame tensor stats: mean={frames.mean():.6f}, std={frames.std():.6f}")
                
                # Check frame-to-frame differences in raw input
                if T > 1:
                    # FIXED: Use proper tensor operations for difference calculation
                    frame_diffs = (frames[:, 1:, :, :, :] - frames[:, :-1, :, :, :]).pow(2).sum(dim=(2, 3, 4)).sqrt()
                    raw_frame_diffs = frame_diffs.mean()
                    print(f"Raw input frame differences: {raw_frame_diffs:.6f}")
                    
                    if raw_frame_diffs < 0.001:
                        print(f"  ⚠️  INPUT FRAMES ARE NEARLY IDENTICAL! Diff: {raw_frame_diffs:.8f}")
                    else:
                        print(f"  ✓ Input frames are different. Diff: {raw_frame_diffs:.6f}")
                
                # Check processed pixel values
                first_sample_pixels = inputs['pixel_values'][:T]  # First sample's pixels
                if T > 1:
                    # FIXED: Use proper tensor operations 
                    pixel_frame_diffs = (first_sample_pixels[1:] - first_sample_pixels[:-1]).pow(2).sum(dim=(1, 2, 3)).sqrt()
                    pixel_diffs = pixel_frame_diffs.mean()
                    print(f"Processed pixel_values differences: {pixel_diffs:.6f}")
                    
                    if pixel_diffs < 0.001:
                        print(f"  ⚠️  PROCESSED PIXELS ARE NEARLY IDENTICAL! Diff: {pixel_diffs:.8f}")
                    else:
                        print(f"  ✓ Processed pixels are different. Diff: {pixel_diffs:.6f}")
                
                # Check if all samples in batch have same first frame
                if B > 1:
                    # FIXED: Use proper tensor operations
                    batch_first_frames = inputs['pixel_values'][::T]  # Every T-th frame (first frame of each sample)
                    if len(batch_first_frames) > 1:
                        first_frame_diffs = (batch_first_frames[1:] - batch_first_frames[0].unsqueeze(0)).pow(2).sum(dim=(1, 2, 3)).sqrt()
                        batch_first_frame_diff = first_frame_diffs.mean()
                        print(f"Batch first-frame differences: {batch_first_frame_diff:.6f}")
                        
                        if batch_first_frame_diff < 0.001:
                            print(f"  ⚠️  ALL BATCH SAMPLES HAVE SAME FIRST FRAME! Diff: {batch_first_frame_diff:.8f}")
                        else:
                            print(f"  ✓ Batch samples have different first frames. Diff: {batch_first_frame_diff:.6f}")
                
                # Check feature statistics
                feature_mean = vision_features.mean().item()
                feature_std = vision_features.std().item()
                print(f"Feature stats: mean={feature_mean:.4f}, std={feature_std:.4f}")
                
                # Check temporal variance for each sample
                for b_idx in range(min(B, 2)):  # Debug first 2 samples
                    sample_features = vision_features[b_idx]  # (T, D)
                    
                    # Variance across time dimension
                    temporal_variance = sample_features.var(dim=0).mean().item()
                    temporal_std = sample_features.std(dim=0).mean().item()
                    print(f"Sample {b_idx} temporal variance: {temporal_variance:.6f} (std: {temporal_std:.6f})")
                    
                    # Frame-to-frame differences
                    if T > 1:
                        frame_diffs = (sample_features[1:] - sample_features[:-1]).pow(2).sum(dim=-1).sqrt()
                        avg_frame_diff = frame_diffs.mean().item()
                        max_frame_diff = frame_diffs.max().item()
                        min_frame_diff = frame_diffs.min().item()
                        print(f"Sample {b_idx} frame differences: avg={avg_frame_diff:.6f}, "
                              f"max={max_frame_diff:.6f}, min={min_frame_diff:.6f}")
                        
                        # Check if features are nearly identical
                        if avg_frame_diff < 0.001:
                            print(f"  ⚠️  FEATURES BARELY CHANGING! Avg diff: {avg_frame_diff:.8f}")
                        elif avg_frame_diff < 0.01:
                            print(f"  ⚠️  Features changing slowly. Avg diff: {avg_frame_diff:.6f}")
                        else:
                            print(f"  ✓ Features changing normally. Avg diff: {avg_frame_diff:.6f}")
                
                # Overall batch statistics
                if B > 1 and T > 1:
                    all_diffs = (
                        vision_features[:, 1:, :] - vision_features[:, :-1, :]
                    ).pow(2).sum(dim=-1).sqrt().flatten()
                    print(f"Batch-wide frame differences: mean={all_diffs.mean():.6f}, "
                          f"std={all_diffs.std():.6f}")
                    
                    # Check percentage of very small differences
                    small_diffs = (all_diffs < 0.001).float().mean().item() * 100
                    print(f"Percentage of tiny differences (<0.001): {small_diffs:.1f}%")
                    
                print("=" * 50)
        
        return vision_features
    
    def _mask_from_lens(self, lens: Tensor) -> Tensor:
        """Create mask from sequence lengths."""
        seq = torch.arange(lens.amax().item(), device=lens.device)
        return einx.less('n, b -> b n', seq, lens)
    
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Compute ReWiND training loss with on-the-fly progress label generation.

        Expected batch keys:
          - OBS_IMAGES: list[Tensor] of shape [(B, C, H, W), ...] per time step or stacked (B, T, C, H, W)
          - OBS_LANGUAGE: optional string tokens already tokenized externally or raw strings

        Note: Progress labels (0 to 1) are generated automatically for each episode.
              No REWARD key is needed in the batch.
        """
        import time
        forward_start = time.perf_counter()
        
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Always use random anchor window sampling
        frames, anchor_stats = self._sample_random_anchor_windows(batch)
            
        B, T, C, H, W = frames.shape
        device = next(self.parameters()).device
        frames = frames.to(device)

        # Apply video rewinding augmentation (always enabled during training)
        augmented_target = None
        if self.training:
            frames, augmented_target = apply_video_rewind(
                frames,
                rewind_prob=self.config.rewind_prob,
                last3_prob=self.config.rewind_last3_prob,
                anchor_stats=anchor_stats,
            )

        # Apply stride and frame dropout
        idx = torch.arange(0, T, self.stride, device=frames.device)
        if self.training and self.frame_dropout_p > 0.0 and T > 1:
            mask = torch.rand_like(idx.float()) > self.frame_dropout_p
            idx = idx[mask.long().bool()]
            if idx.numel() == 0:
                idx = torch.tensor([0], device=frames.device)
        frames = frames[:, idx]
        T_eff = frames.shape[1]

        # Get language commands
        commands = batch.get(OBS_LANGUAGE, None)
        if commands is None:
            commands = [""] * B
        elif not isinstance(commands, list):
            commands = [str(commands)] * B

        # Process video frames through SigLIP2
        vision_start = time.perf_counter()
        video_embeds = self._encode_video_frames(frames).to(device)  # (B, T_eff, D_vision)
        vision_time = time.perf_counter() - vision_start
        
        # Language embeddings + mask
        lang_start = time.perf_counter()
        lang_embeds, mask = self._encode_language_tokens(commands, device)
        lang_time = time.perf_counter() - lang_start
        
        # Token preparation
        # Register tokens
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b=B)
        
        # Project embeddings
        lang_tokens = self.to_lang_tokens(lang_embeds)
        video_tokens = self.to_video_tokens(video_embeds)
        

        # Add temporal positional encoding (window-relative only)  
        T_video = video_tokens.shape[1]
        video_tokens = video_tokens + self.temporal_pos_embedding[:T_video]
        
        # Pack all tokens for attention [lang | register | video]
        tokens, lang_video_packed_shape = pack((lang_tokens, register_tokens, video_tokens), 'b * d')
        
        # Extend mask for register and video tokens
        mask = F.pad(mask, (0, register_tokens.shape[1] + video_tokens.shape[1]), value=True)
        
        # Forward through x_transformers Decoder
        transformer_start = time.perf_counter()
        attended = self.decoder(tokens, mask=mask)
        
        # Unpack and get video token features
        _, _, attended_video_tokens = unpack(attended, lang_video_packed_shape, 'b * d')
        
        # Process all frames with single MLP
        frame_tokens = self.frame_mlp(attended_video_tokens)  # (B, T, D)
        
        # MLP predictor
        video_frame_embeds = self.mlp_predictor(frame_tokens)
        transformer_time = time.perf_counter() - transformer_start

        # Generate progress labels on-the-fly (ReWiND approach)
        # IMPORTANT: Progress should be 0-1 across the ENTIRE EPISODE, not just the temporal window
        loss_dict: dict[str, float] = {}

        # Generate progress targets based on episode-relative positions
        if self.training and augmented_target is not None:
            # For rewind augmentation, the augmented_target already contains proper progress values
            # But we need to handle potential stride/dropout
            target = augmented_target[:, :T_eff] if augmented_target.shape[1] > T_eff else augmented_target
            if target.shape[1] < T_eff:
                # This shouldn't happen but handle it gracefully
                target = torch.linspace(0, 1, T_eff, device=device).unsqueeze(0).expand(B, -1)
        else:
            # Use anchor-based episode-relative progress
            if anchor_stats.get("fallback_used", False):
                raise ValueError(
                    "Anchor-based sampling failed. Ensure 'episode_index', 'frame_index' are in batch "
                    "and 'episode_data_index' is loaded from episodes.jsonl"
                )
            target = self._calculate_anchor_based_progress(T_eff, anchor_stats)

        # During inference, we might not want to compute loss
        if not self.training and target is None:
            # Return predictions without loss using temporal-aware head
            
            # Add temporal position information
            B_inf, T_inf = video_frame_embeds.shape[:2]
            temporal_pos = torch.linspace(0, 1, T_inf, device=video_frame_embeds.device)
            temporal_pos = temporal_pos.unsqueeze(0).unsqueeze(-1).expand(B_inf, T_inf, 1)
            
            # Concatenate and forward through temporal-aware head
            temporal_input = torch.cat([video_frame_embeds, temporal_pos], dim=-1)
            raw_logits = self.reward_head(temporal_input).squeeze(-1)
            rewards = torch.sigmoid(raw_logits)
            return rewards.mean() * 0.0, {"rewards_mean": rewards.mean().item()}

        # Calculate loss using logit regression
        loss_start = time.perf_counter()
        
        # Get model outputs with temporal-aware head
        
        # Add temporal position information
        temporal_pos = torch.linspace(0, 1, T_eff, device=video_frame_embeds.device)
        temporal_pos = temporal_pos.unsqueeze(0).unsqueeze(-1).expand(B, T_eff, 1)  # (B, T_eff, 1)
        
        # Concatenate embeddings with temporal position
        temporal_input = torch.cat([video_frame_embeds, temporal_pos], dim=-1)  # (B, T_eff, D+1)
        
        # Forward through temporal-aware head
        raw_logits = self.reward_head(temporal_input).squeeze(-1)  # (B, T_eff)
        
        # FIXED: More robust logit regression with gradient protection
        eps = self.config.logit_eps
        target_expanded = target.expand(B, -1)[:, :T_eff]  # Expand and trim to T_eff
        target_clamped = torch.clamp(target_expanded, eps, 1 - eps)
        target_logits = torch.logit(target_clamped)
        
        # Use Smooth L1 loss instead of MSE for better gradient stability
        loss = F.smooth_l1_loss(raw_logits, target_logits, reduction='mean', beta=1.0)
        
        # Clip gradients specifically for the reward head during backward pass
        # This prevents extreme gradients from corrupting AdamW momentum
        if self.training:
            raw_logits.register_hook(lambda grad: torch.clamp(grad, -10.0, 10.0))
        
        # For logging, compute sigmoid predictions
        predicted_rewards = torch.sigmoid(raw_logits)

        # Mismatched video-language pairs loss (only when languages actually differ)
        L_mismatch = torch.zeros((), device=device)
        if self.training and B > 1 and torch.rand(1, device=device).item() < self.config.mismatch_prob:
            # Create actual mismatches - ensure shuffled language != original language
            shuffled_indices = torch.randperm(B, device=device)
            
            # Find which samples actually got different languages
            mismatch_mask = []
            shuffled_commands = []
            for i in range(B):
                shuffled_idx = shuffled_indices[i].item()
                original_cmd = commands[i]
                shuffled_cmd = commands[shuffled_idx]
                
                # Only count as mismatch if languages are actually different
                is_mismatch = original_cmd != shuffled_cmd
                mismatch_mask.append(is_mismatch)
                shuffled_commands.append(shuffled_cmd)
            
            # Only apply mismatch loss if we have actual mismatches
            if any(mismatch_mask):
                print("Applying mismatch loss!!!")
                # Re-encode with mismatched language
                lang_embeds_mm, mask_mm = self._encode_language_tokens(shuffled_commands, device)
                lang_tokens_mm = self.to_lang_tokens(lang_embeds_mm)
                
                # Pack and forward
                tokens_mm, lang_video_packed_shape_mm = pack((lang_tokens_mm, register_tokens, video_tokens), 'b * d')
                mask_mm = F.pad(mask_mm, (0, register_tokens.shape[1] + video_tokens.shape[1]), value=True)
                attended_mm = self.decoder(tokens_mm, mask=mask_mm)
                _, _, attended_video_mm = unpack(attended_mm, lang_video_packed_shape_mm, 'b * d')
                
                # Process mismatch frames with single MLP
                mismatch_tokens = self.frame_mlp(attended_video_mm)  # (B, T, D)
                mismatch_embeds = self.mlp_predictor(mismatch_tokens)
                
                # Predict near-zero progress for mismatched pairs with temporal awareness
                
                # Add temporal position information for mismatch computation
                T_mismatch = mismatch_embeds.shape[1]
                temporal_pos_mm = torch.linspace(0, 1, T_mismatch, device=mismatch_embeds.device)
                temporal_pos_mm = temporal_pos_mm.unsqueeze(0).unsqueeze(-1).expand(B, T_mismatch, 1)
                
                # Concatenate mismatch embeddings with temporal position
                temporal_input_mm = torch.cat([mismatch_embeds, temporal_pos_mm], dim=-1)
                
                # Forward through temporal-aware head
                mismatch_raw_logits = self.reward_head(temporal_input_mm).squeeze(-1)
                
                # Create mask tensor for loss calculation
                mismatch_tensor = torch.tensor(mismatch_mask, device=device, dtype=torch.bool)
                
                if mismatch_tensor.any():
                    # Target logit corresponding to sigmoid ≈ 0
                    eps = self.config.logit_eps
                    zeros_target_logits = torch.logit(torch.full_like(target_expanded[:, :T_eff], eps))
                    
                    # Only compute loss for samples that are actually mismatched
                    mismatch_loss_per_sample = F.mse_loss(
                        mismatch_raw_logits, zeros_target_logits, reduction='none'
                    ).mean(dim=1)  # (B,)
                    
                    # Apply mask and average only over true mismatches
                    L_mismatch = mismatch_loss_per_sample[mismatch_tensor].mean()

        # Total loss
        total_loss = loss + L_mismatch
        loss_time = time.perf_counter() - loss_start
        
        # DEBUG: Clean logit regression monitoring with full array printing
        if self.training and torch.rand(1).item() < 0.03:
            with torch.no_grad():
                sample_idx = torch.randint(0, B, (1,)).item()
                sample_targets = target_expanded[sample_idx, :T_eff].cpu().numpy()
                sample_preds = predicted_rewards[sample_idx].cpu().numpy()
                
                print(f"\n=== LOGIT REGRESSION DEBUG ===")
                print(f"Target: min={target_expanded.min():.3f}, max={target_expanded.max():.3f}, mean={target_expanded.mean():.3f}")
                has_high_targets = (target_expanded > 0.8).any().item()
                print(f"✓ Has targets >0.8: {has_high_targets} | T_eff: {T_eff}")
                print(f"Logits: min={raw_logits.min():.3f}, max={raw_logits.max():.3f}, mean={raw_logits.mean():.3f}")
                print(f"Preds:  min={predicted_rewards.min():.3f}, max={predicted_rewards.max():.3f}, mean={predicted_rewards.mean():.3f}")
                
                # Show full arrays occasionally (25% chance within debug)
                show_full = torch.rand(1).item() < 0.25
                if show_full:
                    print(f"\n📊 FULL SAMPLE {sample_idx} ARRAYS (T_eff={T_eff}):")
                    # Always show full arrays up to 16 frames
                    if T_eff <= 16:
                        print(f"  Targets: {sample_targets}")
                        print(f"  Preds:   {sample_preds}")
                        
                        # Show differences and error metrics
                        diffs = sample_preds - sample_targets
                        print(f"  Errors:  {diffs}")
                        mae = np.abs(diffs).mean()
                        mse = (diffs ** 2).mean()
                        max_error = np.abs(diffs).max()
                        print(f"  MAE: {mae:.4f} | MSE: {mse:.4f} | Max Error: {max_error:.4f}")
                        
                        # Check if predictions are stuck or varying
                        pred_std = sample_preds.std()
                        target_std = sample_targets.std()
                        print(f"  Variation - Target std: {target_std:.4f} | Pred std: {pred_std:.4f}")
                    else:
                        # For longer sequences, show first 8 and last 8
                        print(f"  Targets: {sample_targets[:8]} ... {sample_targets[-8:]}")
                        print(f"  Preds:   {sample_preds[:8]} ... {sample_preds[-8:]}")
                else:
                    print(f"Sample {sample_idx}: T_eff={T_eff}, target ∈ [{sample_targets.min():.3f}, {sample_targets.max():.3f}], pred ∈ [{sample_preds.min():.3f}, {sample_preds.max():.3f}]")
                
                print(f"Loss: {loss:.6f}")
                print("=" * 60)
        
        total_forward_time = time.perf_counter() - forward_start

        # Log individual loss components
        loss_dict.update({
            "loss": float(total_loss.detach().item()),
            "loss_main": float(loss.detach().item()),
            "loss_mismatch": float(L_mismatch.detach().item()),
            "t_eff": float(T_eff),
            "lang_len_mean": float(mask.sum().float().mean().item()), # Use mask to get actual lengths
            # Target statistics for monitoring
            "target_min": float(target.min().item()),
            "target_max": float(target.max().item()),
            "target_mean": float(target.mean().item()),
            "target_std": float(target.std().item()),
            # Prediction statistics
            "pred_mean": float(predicted_rewards.mean().item()),
            "pred_std": float(predicted_rewards.std().item()),
            # Raw logits statistics (useful for monitoring head behavior)
            "raw_logits_mean": float(raw_logits.mean().item()),
            "raw_logits_std": float(raw_logits.std().item()),
            # Anchor sampling statistics
            "anchor_mean": float(anchor_stats.get('anchor_mean', 0.0)),
            "anchor_std": float(anchor_stats.get('anchor_std', 0.0)),
            "oob_fraction": float(anchor_stats.get('oob_fraction', 0.0)),
            "padded_fraction": float(anchor_stats.get('padded_fraction', 0.0)),
            # Mismatch loss statistics
            "mismatch_applied": float(L_mismatch.item() > 0),
            # Timing information
            "timing_vision_ms": float(vision_time * 1000),
            "timing_language_ms": float(lang_time * 1000),
            "timing_transformer_ms": float(transformer_time * 1000),
            "timing_loss_ms": float(loss_time * 1000),
            "timing_total_forward_ms": float(total_forward_time * 1000),
        })
        
        # Collect timing statistics for averaged reporting every minute
        if self.training:
            # Initialize timing accumulator if not exists
            if not hasattr(self, '_timing_stats'):
                self._timing_stats = {
                    'vision_times': [],
                    'language_times': [],
                    'transformer_times': [],
                    'loss_times': [],
                    'total_forward_times': [],
                    'throughputs': [],
                    'batch_sizes': [],
                    't_effs': [],
                    'last_print_time': time.perf_counter()
                }
            
            # Accumulate current step's timings
            stats = self._timing_stats
            stats['vision_times'].append(vision_time * 1000)
            stats['language_times'].append(lang_time * 1000)
            stats['transformer_times'].append(transformer_time * 1000)
            stats['loss_times'].append(loss_time * 1000)
            stats['total_forward_times'].append(total_forward_time * 1000)
            stats['throughputs'].append(B * T_eff / total_forward_time)
            stats['batch_sizes'].append(B)
            stats['t_effs'].append(T_eff)
            
            # Print averaged stats every minute (60 seconds)
            current_time = time.perf_counter()
            if current_time - stats['last_print_time'] >= 60.0:
                n_samples = len(stats['vision_times'])
                if n_samples > 0:
                    avg_b = sum(stats['batch_sizes']) / n_samples
                    avg_t_eff = sum(stats['t_effs']) / n_samples
                    
                    print(f"\nRLearN Average Timing (last {n_samples} steps, avg B={avg_b:.1f}, avg T_eff={avg_t_eff:.1f}):")
                    print(f"  Vision encoding:    {sum(stats['vision_times'])/n_samples:.2f} ms")
                    print(f"  Language encoding:  {sum(stats['language_times'])/n_samples:.2f} ms")
                    print(f"  Transformer:        {sum(stats['transformer_times'])/n_samples:.2f} ms")
                    print(f"  Loss computation:   {sum(stats['loss_times'])/n_samples:.2f} ms")
                    print(f"  Total forward pass: {sum(stats['total_forward_times'])/n_samples:.2f} ms")
                    print(f"  Avg throughput:     {sum(stats['throughputs'])/n_samples:.1f} frames/sec")
                    print("-" * 60)
                
                # Reset stats for next minute
                for key in stats:
                    if key != 'last_print_time':
                        stats[key] = []
                stats['last_print_time'] = current_time

        return total_loss, loss_dict

    def _encode_language_tokens(self, commands: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
        """Return (embeddings, mask) for language tokens using SigLIP2.
        embeddings: (B, L, D); mask: (B, L) True for valid tokens.
        """
        # Optimized: Process all commands in batch (much faster than individual processing)
        proc = self.text_processor(
            text=commands, 
            return_tensors='pt', 
            padding='max_length', 
            max_length=64,
            truncation=True  # Ensure we don't exceed max length
        )
        
        # Simplified access - SigLIP2 processor should return these directly
        input_ids = proc.get('input_ids')
        attention_mask = proc.get('attention_mask')
        
        if input_ids is None:
            # Fallback for different processor structures
            if hasattr(proc, 'input_ids'):
                input_ids = proc.input_ids
                attention_mask = getattr(proc, 'attention_mask', None)
            else:
                raise ValueError(f"Could not find input_ids in SigLIP processor output. Keys: {list(proc.keys())}")
        
        # Move to device efficiently
        input_ids = input_ids.to(device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True) 
        else:
            attention_mask = torch.ones_like(input_ids, device=device)
        
        # Batch encode through text model
        outputs = self.text_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.bool()
        return last_hidden, mask

    def _extract_episode_and_frame_indices(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        """Try to extract (episode_index, frame_index) tensors from batch or complementary data.

        Accepts shapes (B,) or (B,1) and returns 1D long tensors on the model device.
        """
        device = next(self.parameters()).device

        ep = batch.get("episode_index")
        fr = batch.get("frame_index")

        # Try complementary_data
        if (ep is None or fr is None) and isinstance(batch.get("complementary_data"), dict):
            comp = batch["complementary_data"]
            ep = comp.get("episode_index", ep)
            fr = comp.get("frame_index", fr)

        # Fallback: derive from global dataset index using episode_data_index
        if (ep is None or fr is None) and self.episode_data_index is not None:
            glob_idx = batch.get("index")
            if glob_idx is None and isinstance(batch.get("complementary_data"), dict):
                glob_idx = batch["complementary_data"].get("index")

            if glob_idx is not None:
                if torch.is_tensor(glob_idx):
                    if glob_idx.dim() == 2 and glob_idx.shape[1] == 1:
                        glob_idx = glob_idx.squeeze(1)
                    glob_idx = glob_idx.to(device=device, dtype=torch.long)
                else:
                    glob_idx = torch.as_tensor(glob_idx, device=device, dtype=torch.long)

                # Compute episode_index by bucketizing absolute indices into episode 'to' boundaries
                ep_to = self.episode_data_index["to"].to(device=device)
                ep_from = self.episode_data_index["from"].to(device=device)
                # torch.bucketize returns positions in [0, num_episodes]
                ep_idx = torch.bucketize(glob_idx, ep_to, right=False)
                # Clamp to valid range just in case
                ep_idx = ep_idx.clamp(min=0, max=ep_from.numel() - 1)
                fr_idx = glob_idx - ep_from[ep_idx]

                return ep_idx, fr_idx

        if ep is None or fr is None:
            return None, None

        # Convert to 1D long tensors on device
        if torch.is_tensor(ep):
            if ep.dim() == 2 and ep.shape[1] == 1:
                ep = ep.squeeze(1)
            ep = ep.to(device=device, dtype=torch.long)
        else:
            ep = torch.as_tensor(ep, device=device, dtype=torch.long)

        if torch.is_tensor(fr):
            if fr.dim() == 2 and fr.shape[1] == 1:
                fr = fr.squeeze(1)
            fr = fr.to(device=device, dtype=torch.long)
        else:
            fr = torch.as_tensor(fr, device=device, dtype=torch.long)

        return ep, fr

    def _sample_random_anchor_windows(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Sample random anchor windows for training and compute episode-relative progress."""
        # Extract episode and frame indices - required for proper anchor sampling
        episode_indices, frame_indices = self._extract_episode_and_frame_indices(batch)
        
        if episode_indices is None or frame_indices is None or self.episode_data_index is None:
            raise ValueError(
                "Random anchor sampling requires 'episode_index', 'frame_index' in batch "
                "and loaded 'episode_data_index'. Ensure episodes.jsonl is available."
            )
        
        device = next(self.parameters()).device
        B = len(episode_indices)
        T = self.config.max_seq_len
        
        # Get raw image data - this contains the window of frames provided by the dataset
        raw_frames = extract_visual_sequence(batch, target_seq_len=None)
        available_T = raw_frames.shape[1]
        
        # Sample random anchors and build windows
        sampled_frames = []
        anchor_positions = []
        window_frame_indices = []  # Store actual frame indices for progress calculation
        episode_lengths = []  # Store episode lengths for progress calculation
        oob_count = 0
        
        for b_idx in range(B):
            ep_idx = episode_indices[b_idx].item()
            
            # Get episode boundaries
            ep_start = self.episode_data_index["from"][ep_idx].item()
            ep_end = self.episode_data_index["to"][ep_idx].item() 
            ep_length = ep_end - ep_start
            episode_lengths.append(ep_length)
            
            # Proper window-relative stride sampling within available frames
            stride = self.config.temporal_sampling_stride
            # Ensure we have room for T frames at given stride; shrink stride if needed
            if available_T <= 1:
                effective_stride = 1
            else:
                effective_stride = max(1, min(stride, (available_T - 1) // max(T - 1, 1) if (T - 1) > 0 else 1))
            min_anchor_in_window = (T - 1) * effective_stride
            max_anchor_in_window = max(min_anchor_in_window, available_T - 1)
            anchor_in_window = torch.randint(min_anchor_in_window, max_anchor_in_window + 1, (1,)).item()

            # Convert window-anchor to episode-anchor (absolute frame index within episode)
            cur_frame_idx = frame_indices[b_idx].item()
            anchor_abs = cur_frame_idx + (anchor_in_window - (available_T - 1))
            anchor_abs = int(max(0, min(anchor_abs, ep_length - 1)))
            anchor_positions.append(anchor_abs)

            # Build window indices with stride and reflection within [0, available_T)
            window_indices = []
            frame_indices_for_progress = []  # Episode-relative absolute indices for progress
            had_oob = False
            for i in range(T):
                delta = -(T - 1 - i) * effective_stride
                w_idx = anchor_in_window + delta
                if w_idx < 0:
                    w_idx = -w_idx
                    had_oob = True
                elif w_idx >= available_T:
                    w_idx = 2 * (available_T - 1) - w_idx
                    had_oob = True
                w_idx = max(0, min(w_idx, available_T - 1))
                window_indices.append(w_idx)

                # Map window index back to episode-relative absolute frame index
                abs_idx = cur_frame_idx + (w_idx - (available_T - 1))
                abs_idx = int(max(0, min(abs_idx, ep_length - 1)))
                frame_indices_for_progress.append(abs_idx)

            if had_oob:
                oob_count += 1
                
            # Extract frames
            frame_tensors = [raw_frames[b_idx, idx] for idx in window_indices]
            sampled_frames.append(torch.stack(frame_tensors))
            window_frame_indices.append(frame_indices_for_progress)
            
            # DEBUG: Check if stride sampling is producing different frames
            if torch.rand(1).item() < 0.1 and b_idx == 0:  # Debug first sample occasionally
                print(f"\n🔍 STRIDE SAMPLING DEBUG (Sample {b_idx}):")
                print(f"Episode length: {ep_length}, Anchor(abs): {anchor_abs}, Anchor(win): {anchor_in_window}, eff_stride: {effective_stride}")
                print(f"Window indices: {window_indices[:5]}...{window_indices[-5:]}")  # First and last 5
                print(f"Frame indices for progress: {frame_indices_for_progress[:5]}...{frame_indices_for_progress[-5:]}")
                
                # Check if window indices are all the same
                unique_indices = len(set(window_indices))
                print(f"Unique window indices: {unique_indices} out of {len(window_indices)}")
                if unique_indices == 1:
                    print(f"  ⚠️  ALL WINDOW INDICES ARE THE SAME! Index: {window_indices[0]}")
                elif unique_indices < T // 2:
                    print(f"  ⚠️  TOO FEW UNIQUE INDICES! Only {unique_indices} unique frames")
                else:
                    print(f"  ✓ Good frame diversity: {unique_indices} unique frames")
                
                # Check frame tensor differences
                if len(frame_tensors) > 1:
                    frame_diff = (frame_tensors[1] - frame_tensors[0]).pow(2).sum().sqrt().item()
                    print(f"First vs second frame difference: {frame_diff:.6f}")
                    if frame_diff < 0.001:
                        print(f"  ⚠️  CONSECUTIVE SAMPLED FRAMES ARE NEARLY IDENTICAL!")
                    else:
                        print(f"  ✓ Frames are different")
                print("-" * 50)
        
        frames = torch.stack(sampled_frames, dim=0)
        
        anchor_stats = {
            "anchor_mean": float(torch.tensor(anchor_positions).float().mean()),
            "anchor_std": float(torch.tensor(anchor_positions).float().std()),
            "oob_fraction": float(oob_count) / B,
            "padded_fraction": 0.0,  # No padding with reflection approach
            "fallback_used": False,
            "window_frame_indices": window_frame_indices,  # Pass frame indices for progress calculation
            "episode_lengths": episode_lengths  # Pass episode lengths for progress calculation
        }
        
        return frames, anchor_stats
    
    def _calculate_anchor_based_progress(self, T_eff: int, anchor_stats: dict) -> Tensor:
        """Generate episode-relative progress based on actual frame positions within episodes."""
        device = next(self.parameters()).device
        
        # Extract frame indices and episode lengths from anchor_stats
        window_frame_indices = anchor_stats.get("window_frame_indices")
        episode_lengths = anchor_stats.get("episode_lengths")
        
        if window_frame_indices is None or episode_lengths is None:
            # Fallback to window-relative progress if episode info not available
            # This should not happen in normal training
            if T_eff == 1:
                progress = torch.tensor([0.5], device=device)
            else:
                progress = torch.linspace(0, 1, T_eff, device=device)
            return progress.unsqueeze(0)
        
        B = len(window_frame_indices)
        T = len(window_frame_indices[0])  # Original window size (16)
        
        # Calculate episode-relative progress for each sample
        all_progress = []
        for b_idx in range(B):
            frame_indices = window_frame_indices[b_idx]
            ep_length = episode_lengths[b_idx]
            
            # Calculate progress as frame_index / (episode_length - 1)
            # This gives us progress from 0.0 to 1.0 across the episode
            progress = torch.tensor([
                frame_idx / max(ep_length - 1, 1) for frame_idx in frame_indices
            ], device=device, dtype=torch.float32)
            
            # If we have stride/dropout (T_eff < T), subsample the progress values
            if T_eff < T:
                # Subsample evenly from the progress values
                indices = torch.linspace(0, T - 1, T_eff, dtype=torch.long)
                progress = progress[indices]
            
            all_progress.append(progress)
        
        return torch.stack(all_progress)  # (B, T_eff)
    


    def _load_episode_index_from_jsonl(self, path: str) -> dict[str, Tensor]:
        import json
        lengths: list[int] = []
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Expect keys: episode_index, length
                lengths.append(int(obj["length"]))

        # Build cumulative from/to (exclusive)
        starts = [0]
        for L in lengths[:-1]:
            starts.append(starts[-1] + L)
        ends = []
        for i, L in enumerate(lengths):
            ends.append(starts[i] + L)

        device = next(self.parameters()).device
        return {
            "from": torch.tensor(starts, device=device, dtype=torch.long),
            "to": torch.tensor(ends, device=device, dtype=torch.long),
        }

# Helper functions for ReWiND architecture
def extract_visual_sequence(batch: dict[str, Tensor], target_seq_len: int = None) -> Tensor:
    """Extract visual sequence from batch and ensure it has the expected temporal length.

    Args:
        batch: Input batch containing image data
        target_seq_len: Expected sequence length. If provided and the actual sequence is shorter,
                       it will be padded by repeating the first frame.

    Returns:
        Tensor of shape (B, T, C, H, W)
    """
    # Accept various image key formats from datasets
    # With delta_indices, the dataset provides temporal sequences automatically

    # List of possible image keys to check, in order of preference
    possible_keys = [
        OBS_IMAGES,  # 'observation.images'
        OBS_IMAGE,  # 'observation.image'
        "observation.images.image",  # nested format from some datasets
        "observation.images.front", 
    ]

    frames = None
    for key in possible_keys:
        if key in batch:
            image_val = batch[key]

            if isinstance(image_val, list) and len(image_val) > 0:
                # List of (B, C, H, W) -> stack over time
                # This happens when dataset provides temporal sequence as list
                frames = torch.stack(image_val, dim=1)
                break
            elif torch.is_tensor(image_val):
                # Tensor of shape (B, T, C, H, W) or (B, C, H, W)
                if image_val.dim() == 5:
                    # Already has time dimension - this is what we expect with delta_indices
                    frames = image_val
                    break
                elif image_val.dim() == 4:
                    # Add time dimension (single frame) - fallback for datasets without temporal sequences
                    frames = image_val.unsqueeze(1)
                    break
                else:
                    raise ValueError(
                        f"'{key}' must be a Tensor of shape (B,T,C,H,W) or (B,C,H,W), got shape {image_val.shape}"
                    )

    if frames is None:
        # If no image key found, provide helpful error with available keys
        available_keys = list(batch.keys())
        image_like_keys = [k for k in available_keys if "image" in k.lower()]
        raise ValueError(
            f"Could not find image data in batch. Looked for keys: {possible_keys}. "
            f"Available keys with 'image': {image_like_keys}. "
            f"All keys: {available_keys}"
        )

    # Adjust sequence length if needed
    if target_seq_len is not None:
        B, T, C, H, W = frames.shape
        if T < target_seq_len:
            # Pad by repeating the first frame (assumes first frame in sequence is the earliest)
            padding_needed = target_seq_len - T
            first_frame = frames[:, :1]  # (B, 1, C, H, W)
            padding = first_frame.expand(B, padding_needed, C, H, W)
            frames = torch.cat([padding, frames], dim=1)  # Prepend padding

            import logging

            logging.debug(f"Padded sequence from {T} to {target_seq_len} frames by repeating first frame")
        elif T > target_seq_len:
            # Truncate to target length, keeping the most recent frames
            frames = frames[:, -target_seq_len:]

            import logging

            logging.debug(f"Truncated sequence from {T} to {target_seq_len} frames by keeping most recent frames")

    return frames


def apply_video_rewind(frames: Tensor, rewind_prob: float = 0.5, last3_prob: float | None = None, anchor_stats: dict | None = None) -> tuple[Tensor, Tensor]:
    """Apply video rewinding augmentation with episode-relative progress.
    
    This version ensures the rewound sequence is exactly T frames and generates
    episode-relative progress labels based on actual frame positions.

    Args:
        frames: Tensor of shape (B, T, C, H, W)
        rewind_prob: Probability of applying rewind augmentation to each video
        last3_prob: Probability of limiting rewind to last 3 frames
        anchor_stats: Dictionary containing window_frame_indices and episode_lengths for episode-relative progress

    Returns:
        Augmented frames and corresponding episode-relative progress labels
    """
    B, T, C, H, W = frames.shape
    device = frames.device

    # Extract episode information if available
    window_frame_indices = anchor_stats.get("window_frame_indices") if anchor_stats else None
    episode_lengths = anchor_stats.get("episode_lengths") if anchor_stats else None
    
    # Create default progress labels based on episode-relative positions
    if window_frame_indices and episode_lengths:
        # Use actual episode-relative progress
        default_progress = []
        for b_idx in range(B):
            frame_indices = window_frame_indices[b_idx]
            ep_length = episode_lengths[b_idx]
            progress = torch.tensor([
                frame_idx / max(ep_length - 1, 1) for frame_idx in frame_indices
            ], device=device, dtype=torch.float32)
            default_progress.append(progress)
        default_progress = torch.stack(default_progress)
    else:
        # Fallback to window-relative progress
        default_progress = torch.linspace(0, 1, T, device=device).unsqueeze(0).expand(B, -1)

    # Apply rewind augmentation to each sample in batch independently
    augmented_frames = []
    augmented_progress = []

    for b in range(B):
        # Each video has independent chance of being rewound
        should_rewind = torch.rand(1).item() < rewind_prob

        if not should_rewind or T < 3:
            # Keep original sequence
            augmented_frames.append(frames[b])
            augmented_progress.append(default_progress[b])
            continue

        # Apply rewinding - but ensure we get exactly T frames
        max_attempts = 10  # Limit resampling attempts
        success = False
        
        for attempt in range(max_attempts):
            # Split point i: between frame 2 and T-1
            i = torch.randint(2, T, (1,)).item()

            # Rewind length k: between 1 and i-1 frames
            if last3_prob is not None and torch.rand(1).item() < last3_prob and i >= 3:
                k = min(3, i - 1)
            else:
                k = torch.randint(1, i, (1,)).item()
                k = min(k, i - 1)

            # Create rewound sequence: frames[0:i] + reversed frames[i-k:i]
            forward_length = i
            reverse_length = k
            total_length = forward_length + reverse_length
            
            # Check if we can make exactly T frames
            if total_length == T:
                # Perfect fit!
                forward_frames = frames[b, :i]
                reverse_frames = frames[b, max(0, i - k):i].flip(dims=[0])
                rewound_seq = torch.cat([forward_frames, reverse_frames], dim=0)
                
                # Create corresponding progress labels based on episode-relative positions
                if window_frame_indices and episode_lengths:
                    # Use episode-relative progress for rewind
                    frame_indices = window_frame_indices[b]
                    ep_length = episode_lengths[b]
                    # Forward part: use actual frame indices
                    forward_progress = torch.tensor([
                        frame_indices[idx] / max(ep_length - 1, 1) for idx in range(i)
                    ], device=device, dtype=torch.float32)
                    # Reverse part: use reversed frame indices
                    reverse_indices = list(range(max(0, i - k), i))[::-1]
                    reverse_progress = torch.tensor([
                        frame_indices[idx] / max(ep_length - 1, 1) for idx in reverse_indices
                    ], device=device, dtype=torch.float32)
                    rewound_progress = torch.cat([forward_progress, reverse_progress])
                else:
                    # Fallback to window-relative progress
                    denom = max(T - 1, 1)
                    forward_progress = torch.linspace(0, (i - 1) / denom, i, device=device)
                    reverse_progress = torch.linspace((i - 1) / denom, max(0.0, (i - k) / denom), k, device=device)
                    rewound_progress = torch.cat([forward_progress, reverse_progress])
                
                success = True
                break
            elif total_length < T:
                # Too short - try to extend by adjusting k
                needed = T - total_length
                if i + needed <= T:  # Can we extend k?
                    k_extended = k + needed
                    if i - k_extended >= 0:
                        forward_frames = frames[b, :i]
                        reverse_frames = frames[b, max(0, i - k_extended):i].flip(dims=[0])
                        rewound_seq = torch.cat([forward_frames, reverse_frames], dim=0)
                        
                        if rewound_seq.shape[0] == T:
                            # Create progress labels based on episode-relative positions
                            if window_frame_indices and episode_lengths:
                                frame_indices = window_frame_indices[b]
                                ep_length = episode_lengths[b]
                                # Forward part
                                forward_progress = torch.tensor([
                                    frame_indices[idx] / max(ep_length - 1, 1) for idx in range(i)
                                ], device=device, dtype=torch.float32)
                                # Extended reverse part
                                reverse_indices = list(range(max(0, i - k_extended), i))[::-1]
                                reverse_progress = torch.tensor([
                                    frame_indices[idx] / max(ep_length - 1, 1) for idx in reverse_indices
                                ], device=device, dtype=torch.float32)
                                rewound_progress = torch.cat([forward_progress, reverse_progress])
                            else:
                                # Fallback to window-relative progress
                                denom = max(T - 1, 1)
                                forward_progress = torch.linspace(0, (i - 1) / denom, i, device=device)
                                reverse_progress = torch.linspace((i - 1) / denom, max(0.0, (i - k_extended) / denom), k_extended, device=device)
                                rewound_progress = torch.cat([forward_progress, reverse_progress])
                            
                            success = True
                            break
            # If too long or can't fix, try again with different i,k
        
        if success:
            augmented_frames.append(rewound_seq)
            augmented_progress.append(rewound_progress)
        else:
            # Fallback: use original sequence if we can't create a good rewind
            augmented_frames.append(frames[b])
            augmented_progress.append(default_progress[b])

    return torch.stack(augmented_frames), torch.stack(augmented_progress)