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

        # Encoders - SigLIP2 for both vision and text
        from transformers import AutoProcessor, AutoModel
        
        # Load SigLIP2 processors and models
        self.vision_processor = AutoProcessor.from_pretrained(config.vision_model_name, use_fast=True)
        self.vision_model = AutoModel.from_pretrained(config.vision_model_name)
        
        self.text_processor = AutoProcessor.from_pretrained(config.text_model_name, use_fast=True)
        self.text_model = AutoModel.from_pretrained(config.text_model_name)
        
        # Move encoders to GPU if available
        if torch.cuda.is_available():
            self.vision_model = self.vision_model.to('cuda')
            self.text_model = self.text_model.to('cuda')
        
        # Get hidden sizes from SigLIP2 config
        vh = getattr(getattr(self.vision_model, 'config', None), 'vision_config', None)
        self.vision_hidden = getattr(vh, 'hidden_size', 768)
        
        th = getattr(getattr(self.text_model, 'config', None), 'text_config', None)
        self.text_hidden = getattr(th, 'hidden_size', 512)

        # Freeze encoders if requested
        if config.freeze_backbones:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            for p in self.text_model.parameters():
                p.requires_grad = False

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
        
        # CRITICAL: Frame-specific MLPs prevent temporal over-smoothing
        # Problem: Transformer attention was making all 16 predictions identical (e.g. all 0.34)
        # Solution: Each temporal position gets its own specialized MLP processing
        # Frame 0 → MLP[0], Frame 1 → MLP[1], ..., Frame 15 → MLP[15]
        # This creates distinct pathways for each frame while preserving attention context
        self.frame_specific_mlp = nn.ModuleList([
            nn.Linear(config.dim_model, config.dim_model) 
            for _ in range(config.max_seq_len)  # 16 separate MLPs for 16 frame positions
        ])
        
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
        
        # Layer normalization before reward head to stabilize MLP outputs
        self.pre_reward_norm = nn.LayerNorm(config.dim_model)
        
        # Regression head - supports both logit and sigmoid modes
        self.reward_head = nn.Linear(config.dim_model, 1)
        
        # Initialize head with improved settings
        with torch.no_grad():
            if config.use_logit_regression:
                # Logit regression: can use larger weights since no saturation issues
                self.reward_head.weight.normal_(0.0, config.head_weight_init_std)
                self.reward_head.bias.fill_(0.0)  # Neutral start in logit space
            else:
                # Sigmoid mode: moderate initialization
                self.reward_head.weight.normal_(0.0, 0.02)
                self.reward_head.bias.fill_(0.0)
            
        self.sigmoid = nn.Sigmoid() if not config.use_logit_regression else None
        
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
        """Return parameter groups with custom LR and weight decay settings."""
        # Collect trainable parameters
        base_params = []
        head_weight_params = []
        head_bias_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            if "reward_head" in name:
                if "bias" in name:
                    head_bias_params.append(param)
                else:
                    head_weight_params.append(param)
            else:
                base_params.append(param)
        
        # Create parameter groups with different settings
        param_groups = []
        
        # Base parameters (everything except head)
        if base_params:
            param_groups.append({
                "params": base_params,
                "name": "base"
            })
        
        # Head weight parameters (higher LR)
        if head_weight_params:
            param_groups.append({
                "params": head_weight_params,
                "lr": self.config.learning_rate * self.config.head_lr_multiplier,
                "name": "head_weights"
            })
        
        # Head bias parameters (higher LR, optionally no weight decay)
        if head_bias_params:
            head_bias_group = {
                "params": head_bias_params,
                "lr": self.config.learning_rate * self.config.head_lr_multiplier,
                "name": "head_bias"
            }
            if self.config.remove_head_bias_wd:
                head_bias_group["weight_decay"] = 0.0
            param_groups.append(head_bias_group)
        
        return param_groups

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
        
        # Apply frame-specific processing to prevent over-smoothing
        frame_specific_embeds = []
        T_video = attended_video_tokens.shape[1]
        for t in range(T_video):
            # Apply frame-specific MLP to each temporal position
            frame_embed = self.frame_specific_mlp[t](attended_video_tokens[:, t])
            frame_specific_embeds.append(frame_embed)
        frame_specific_tokens = torch.stack(frame_specific_embeds, dim=1)  # (B, T, D)
        
        # MLP predictor
        video_frame_embeds = self.mlp_predictor(frame_specific_tokens)
        
        # Get rewards via linear head
        normalized_embeds = self.pre_reward_norm(video_frame_embeds)
        raw_logits = self.reward_head(normalized_embeds).squeeze(-1)  # (B, T)
        
        if self.config.use_logit_regression:
            # In logit mode, apply sigmoid at inference
            return torch.sigmoid(raw_logits)
        else:
            # In sigmoid mode, apply sigmoid as usual
            return self.sigmoid(raw_logits)

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
        
        # Normalize to [0, 1] if needed and ensure correct format for SigLIP2
        if flat.dtype != torch.float32:
            flat = flat.float()
        if flat.max() > 1.0:
            flat = flat / 255.0
            
        # SigLIP2 expects images in [0, 1] range, RGB format
        # Resize and normalize in batch - much faster than individual processing
        try:
            # Try direct tensor processing (faster path)
            processed = self.vision_processor(images=flat, return_tensors="pt")
            pixel_values = processed["pixel_values"].to(device)
        except:
            # Fallback to individual processing if needed, but optimized
            # Convert entire batch to numpy at once (much faster)
            flat_numpy = flat.permute(0, 2, 3, 1).cpu().numpy()  # (BT, H, W, C)
            images_list = [flat_numpy[i] for i in range(B * T)]
            
            processed = self.vision_processor(images=images_list, return_tensors="pt") 
            pixel_values = processed["pixel_values"].to(device)
        
        # Process in batch through vision model
        vision_outputs = self.vision_model.vision_model(pixel_values=pixel_values)
        cls_tokens = vision_outputs.last_hidden_state[:, 0]
        
        return rearrange(cls_tokens, '(b t) d -> b t d', b=B, t=T)
    
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

        # NEW: Explicit random anchor window sampling for training
        if self.training:
            frames, anchor_stats = self._sample_random_anchor_windows(batch)
        else:
            # During inference, use the generic extractor
            frames = extract_visual_sequence(batch, target_seq_len=self.config.max_seq_len)
            anchor_stats = None
            
        B, T, C, H, W = frames.shape
        device = next(self.parameters()).device
        frames = frames.to(device)

        # Apply video rewinding augmentation during training (FIXED: no constant padding)
        augmented_target = None
        if self.training and self.config.use_video_rewind:
            frames, augmented_target = apply_video_rewind_fixed(
                frames,
                rewind_prob=self.config.rewind_prob,
                last3_prob=getattr(self.config, "rewind_last3_prob", None),
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
        
        # Apply frame-specific processing to prevent over-smoothing
        frame_specific_embeds = []
        T_video = attended_video_tokens.shape[1]
        for t in range(T_video):
            # Apply frame-specific MLP to each temporal position
            frame_embed = self.frame_specific_mlp[t](attended_video_tokens[:, t])
            frame_specific_embeds.append(frame_embed)
        frame_specific_tokens = torch.stack(frame_specific_embeds, dim=1)  # (B, T, D)
        
        # MLP predictor
        video_frame_embeds = self.mlp_predictor(frame_specific_tokens)
        transformer_time = time.perf_counter() - transformer_start

        # Generate progress labels on-the-fly (ReWiND approach)
        # IMPORTANT: Progress should be 0-1 across the ENTIRE EPISODE, not just the temporal window
        loss_dict: dict[str, float] = {}

        # Check if video rewinding already set the target
        if self.training and self.config.use_video_rewind and augmented_target is not None:
            # Use the augmented target from video rewinding and align with temporal subsampling
            target = augmented_target[:, idx]
        elif self.training and anchor_stats is not None and not anchor_stats.get("fallback_used", False):
            # NEW: Calculate progress using the known random anchors
            target = self._calculate_anchor_based_progress(batch, anchor_stats, T_eff)
        else:
            # Fallback: Calculate episode progress the old way
            episode_indices, frame_indices = self._extract_episode_and_frame_indices(batch)
            if episode_indices is not None and frame_indices is not None and self.episode_data_index is not None:
                target = self._calculate_episode_progress(batch, episode_indices, frame_indices, T_eff, idx)
            else:
                raise ValueError(
                    "No episode information found to build full-episode progress. "
                    "Expected 'episode_index' and 'frame_index' in batch and a valid 'episode_data_index' on the policy. "
                    "Please pass RLearNPolicy(episode_data_index=...) built from episodes.jsonl (per-episode lengths), "
                    "and ensure the dataset exposes 'episode_index' and 'frame_index' (shape (B,) or (B,1))."
                )

        # During inference, we might not want to compute loss
        if not self.training and target is None:
            # Return predictions without loss
            normalized_embeds = self.pre_reward_norm(video_frame_embeds)
            rewards = self.sigmoid(self.reward_head(normalized_embeds)).squeeze(-1)
            return rewards.mean() * 0.0, {"rewards_mean": rewards.mean().item()}

        # Calculate loss using the configured mode (logit regression or sigmoid+MSE)
        loss_start = time.perf_counter()
        assert target.dtype == torch.float, "Continuous rewards require float targets"
        
        # Get model outputs
        normalized_embeds = self.pre_reward_norm(video_frame_embeds)
        raw_logits = self.reward_head(normalized_embeds).squeeze(-1)  # (B, T_eff)
        
        if self.config.use_logit_regression:
            # Logit regression: transform targets to logit space and compute MSE on logits
            eps = self.config.logit_eps
            target_clamped = torch.clamp(target[:, :T_eff], eps, 1 - eps)
            target_logits = torch.logit(target_clamped)
            loss = F.mse_loss(raw_logits, target_logits, reduction='mean')
            # For logging/debug, also compute sigmoid predictions
            predicted_rewards = torch.sigmoid(raw_logits)
        else:
            # Sigmoid mode: apply sigmoid and compute MSE on probabilities
            predicted_rewards = self.sigmoid(raw_logits)
            loss = F.mse_loss(predicted_rewards, target[:, :T_eff], reduction='mean')

        # Optional: Mismatched video-language pairs loss
        L_mismatch = torch.zeros((), device=device)
        if self.training and self.config.use_mismatch_loss and B > 1:
            if torch.rand(1, device=device).item() < getattr(self.config, "mismatch_prob", 0.2):
                # Shuffle language within batch
                shuffled_indices = torch.randperm(B, device=device)
                shuffled_commands = [commands[i] for i in shuffled_indices]
                
                # Re-encode with mismatched language
                lang_embeds_mm, mask_mm = self._encode_language_tokens(shuffled_commands, device)
                lang_tokens_mm = self.to_lang_tokens(lang_embeds_mm)
                
                # Pack and forward
                tokens_mm, lang_video_packed_shape_mm = pack((lang_tokens_mm, register_tokens, video_tokens), 'b * d')
                mask_mm = F.pad(mask_mm, (0, register_tokens.shape[1] + video_tokens.shape[1]), value=True)
                attended_mm = self.decoder(tokens_mm, mask=mask_mm)
                _, _, attended_video_mm = unpack(attended_mm, lang_video_packed_shape_mm, 'b * d')
                
                # Apply frame-specific processing to mismatch embeddings
                mismatch_specific_embeds = []
                T_video_mm = attended_video_mm.shape[1]
                for t in range(T_video_mm):
                    frame_embed = self.frame_specific_mlp[t](attended_video_mm[:, t])
                    mismatch_specific_embeds.append(frame_embed)
                mismatch_specific_tokens = torch.stack(mismatch_specific_embeds, dim=1)
                
                mismatch_embeds = self.mlp_predictor(mismatch_specific_tokens)
                
                # Mismatched pairs should predict zero progress
                normalized_mismatch_embeds = self.pre_reward_norm(mismatch_embeds)
                mismatch_raw_logits = self.reward_head(normalized_mismatch_embeds).squeeze(-1)
                
                if self.config.use_logit_regression:
                    # In logit mode, target logit of ~0 corresponds to sigmoid(x)≈0
                    eps = self.config.logit_eps
                    zeros_target_logits = torch.logit(torch.full_like(target[:, :T_eff], eps))
                    L_mismatch = F.mse_loss(mismatch_raw_logits, zeros_target_logits, reduction='mean')
                else:
                    # In sigmoid mode, target sigmoid output of 0
                    mismatch_predictions = self.sigmoid(mismatch_raw_logits)
                    zeros_target = torch.zeros_like(target[:, :T_eff])
                    L_mismatch = F.mse_loss(mismatch_predictions, zeros_target, reduction='mean')

        # Total loss
        total_loss = loss + L_mismatch
        loss_time = time.perf_counter() - loss_start
        
        # DEBUG: Print targets and predictions occasionally during training
        if self.training and torch.rand(1).item() < 0.02:  # ~2% chance to debug print
            with torch.no_grad():
                # Get raw MLP outputs, normalized outputs, and predictions
                raw_outputs = video_frame_embeds
                normalized_embeds = self.pre_reward_norm(video_frame_embeds)
                raw_logits = self.reward_head(normalized_embeds).squeeze(-1)
                preds = self.sigmoid(raw_logits)
                
                # Randomly sample a sequence from the batch for detailed analysis
                sample_idx = torch.randint(0, B, (1,)).item()
                
                print(f"\n=== DEBUG TRAINING ===")
                # Target statistics
                print(f"Target min: {target.min():.6f}")
                print(f"Target max: {target.max():.6f}")
                print(f"Target mean: {target.mean():.6f}")
                print(f"Target range: [{target.min():.3f}, {target.max():.3f}]")
                # Model output statistics  
                print(f"Raw MLP range: [{raw_outputs.min():.3f}, {raw_outputs.max():.3f}]")
                print(f"Normalized MLP range: [{normalized_embeds.min():.6f}, {normalized_embeds.max():.6f}]")
                print(f"Raw logits range: [{raw_logits.min():.6f}, {raw_logits.max():.6f}]")
                print(f"Raw logits mean: {raw_logits.mean():.6f}")
                print(f"Sigmoid pred range: [{preds.min():.3f}, {preds.max():.3f}]") 
                print(f"Sigmoid pred mean: {preds.mean():.3f}")
                print(f"Loss: {loss:.4f}")
                # Show randomly sampled sequence for comparison
                print(f"Sample {sample_idx} targets (all 16):", target[sample_idx].cpu().numpy())
                print(f"Sample {sample_idx} preds (all 16):  ", preds[sample_idx].cpu().numpy())
                
                # TARGET FIX VERIFICATION: Check if we still have flat/stuck patterns
                sample_targets = target[sample_idx].cpu().numpy()
                # Count consecutive identical values (should be minimal after fix)
                consecutive_same = 0
                max_consecutive = 0
                for i in range(1, len(sample_targets)):
                    if abs(sample_targets[i] - sample_targets[i-1]) < 1e-6:
                        consecutive_same += 1
                        max_consecutive = max(max_consecutive, consecutive_same + 1)
                    else:
                        consecutive_same = 0
                
                if max_consecutive >= 3:
                    print(f"⚠️  STILL STUCK: {max_consecutive} consecutive identical targets!")
                else:
                    print(f"✅ TARGET FIXED: Max consecutive identical = {max_consecutive}")
                print("="*25)
        
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
            # NEW: Anchor sampling statistics if available
            **({
                "anchor_mean": float(anchor_stats['anchor_mean']) if anchor_stats and not anchor_stats.get('fallback_used', False) else 0.0,
                "anchor_std": float(anchor_stats['anchor_std']) if anchor_stats and not anchor_stats.get('fallback_used', False) else 0.0,
                "oob_fraction": float(anchor_stats['oob_fraction']) if anchor_stats and not anchor_stats.get('fallback_used', False) else 0.0,
                "padded_fraction": float(anchor_stats['padded_fraction']) if anchor_stats and not anchor_stats.get('fallback_used', False) else 0.0,
                "use_random_anchors": not (anchor_stats and anchor_stats.get('fallback_used', False)) if anchor_stats else False,
            }),
            # Loss mode indicator
            "logit_regression": bool(self.config.use_logit_regression),
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
        """Sample random anchor windows for training to avoid sampling bias.
        
        Returns:
            frames: (B, T, C, H, W) tensor with T = max_seq_len
            anchor_stats: dict with sampling statistics for logging
        """
        # Extract episode and frame indices
        episode_indices, frame_indices = self._extract_episode_and_frame_indices(batch)
        
        if episode_indices is None or frame_indices is None or self.episode_data_index is None:
            # Fallback to generic extractor if we don't have episode info
            frames = extract_visual_sequence(batch, target_seq_len=self.config.max_seq_len)
            return frames, {"fallback_used": True}
        
        device = next(self.parameters()).device
        B = len(episode_indices)
        T = self.config.max_seq_len
        delta_indices = self.config.observation_delta_indices  # [-15, -14, ..., 0]
        
        # Get raw image data - assume it's already a temporal sequence from dataset
        raw_frames = extract_visual_sequence(batch, target_seq_len=None)  # Don't force padding
        available_T = raw_frames.shape[1]
        
        # For each sample, choose a random anchor and build the window
        sampled_frames = []
        anchor_positions = []
        oob_count = 0
        padded_count = 0
        resampled_count = 0
        
        for b_idx in range(B):
            ep_idx = episode_indices[b_idx].item()
            frame_idx = frame_indices[b_idx].item()  # Current frame position in episode
            
            # Get episode boundaries
            ep_start = self.episode_data_index["from"][ep_idx].item()
            ep_end = self.episode_data_index["to"][ep_idx].item() 
            ep_length = ep_end - ep_start
            
            # Choose random anchor within episode bounds such that we can get a full window
            # The anchor is the "current" frame (delta=0), so we need at least T-1 frames before it
            min_anchor = T - 1  # Need 15 frames before for [-15..0] window
            max_anchor = ep_length - 1  # Episode frame indices are 0-based
            
            if min_anchor > max_anchor:
                # Episode too short for full window - use available frames with padding
                anchor = max_anchor
                padded_count += 1
            else:
                # Sample uniformly from valid range
                anchor = torch.randint(min_anchor, max_anchor + 1, (1,)).item()
            
            anchor_positions.append(anchor)
            
            # Build window indices relative to episode start
            window_indices = [anchor + delta for delta in delta_indices]
            
            # Handle out-of-bounds with reflection or clamping
            valid_indices = []
            had_oob = False
            for idx in window_indices:
                if idx < 0:
                    # Reflect at episode boundary
                    valid_indices.append(-idx)
                    had_oob = True
                elif idx >= ep_length:
                    # Reflect at episode end
                    valid_indices.append(2 * (ep_length - 1) - idx)
                    had_oob = True
                else:
                    valid_indices.append(idx)
            
            if had_oob:
                oob_count += 1
                
            # Extract frames at these indices from the raw temporal sequence
            # Map episode-relative indices to sequence indices
            frame_tensors = []
            for ep_rel_idx in valid_indices:
                if ep_rel_idx < available_T:
                    frame_tensors.append(raw_frames[b_idx, ep_rel_idx])
                else:
                    # Fallback: repeat last available frame
                    frame_tensors.append(raw_frames[b_idx, -1])
                    padded_count += 1
                    
            sampled_frames.append(torch.stack(frame_tensors))  # (T, C, H, W)
        
        frames = torch.stack(sampled_frames, dim=0)  # (B, T, C, H, W)
        
        anchor_stats = {
            "anchor_mean": float(torch.tensor(anchor_positions).float().mean()),
            "anchor_std": float(torch.tensor(anchor_positions).float().std()),
            "oob_fraction": float(oob_count) / B,
            "padded_fraction": float(padded_count) / B,
            "resampled_count": resampled_count,
            "fallback_used": False
        }
        
        return frames, anchor_stats
    
    def _calculate_anchor_based_progress(self, batch: dict[str, Tensor], anchor_stats: dict, T_eff: int) -> Tensor:
        """Calculate progress labels based on known random anchors (more efficient)."""
        episode_indices, _ = self._extract_episode_and_frame_indices(batch)
        if episode_indices is None:
            raise ValueError("Need episode_indices for anchor-based progress calculation")
            
        device = next(self.parameters()).device
        B = len(episode_indices)
        delta_indices = self.config.observation_delta_indices
        
        # Build progress for each anchor position in the batch
        all_progress = []
        
        for i, delta in enumerate(delta_indices[:T_eff]):  # Only compute for frames we'll actually use
            frame_progress = []
            for b_idx in range(B):
                ep_idx = episode_indices[b_idx].item()
                
                # Get episode length
                ep_start = self.episode_data_index["from"][ep_idx].item()
                ep_end = self.episode_data_index["to"][ep_idx].item()
                ep_length = ep_end - ep_start
                
                # The anchor was chosen during window sampling
                # For anchor-based progress, we use window-relative progress to center around 0.5
                # This is more stable and matches ReWiND's simple approach
                window_position = i  # Position in window [0, T_eff-1]
                progress = window_position / max(1, T_eff - 1)  # 0 to 1 across window
                
                frame_progress.append(progress)
            
            all_progress.append(
                torch.tensor(frame_progress, device=device, dtype=torch.float32)
            )
        
        return torch.stack(all_progress, dim=1)  # (B, T_eff)
    
    def _calculate_episode_progress(self, batch: dict[str, Tensor], episode_indices: Tensor, 
                                  frame_indices: Tensor, T_eff: int, idx: Tensor) -> Tensor:
        """Calculate progress labels using episode-relative positions (legacy fallback)."""
        device = next(self.parameters()).device
        B = len(episode_indices)
        delta_indices = self.config.observation_delta_indices
        
        # Calculate progress for each frame in the temporal window
        all_progress = []
        
        # DEBUG: Log indexing details for first sample occasionally
        debug_indexing = torch.rand(1).item() < 0.05  # 5% chance 
        if debug_indexing:
            print(f"\n=== EPISODE PROGRESS DEBUG ===")
            print(f"Delta indices: {delta_indices}")
            print(f"Batch size: {B}, T_eff: {T_eff}")
            
            # Check if batch samples have diverse frame indices
            unique_frames = torch.unique(frame_indices).tolist()
            unique_episodes = torch.unique(episode_indices).tolist()
            print(f"Unique frame indices in batch: {len(unique_frames)} values")
            print(f"Unique episode indices in batch: {len(unique_episodes)} values")
            
            if len(unique_frames) == 1:
                print("🚨 RED FLAG: All samples have IDENTICAL frame index!")
        
        for i, delta in enumerate(delta_indices):
            # For each sample, calculate the progress of the frame at delta offset
            frame_progress = []
            for b_idx in range(B):
                ep_idx = episode_indices[b_idx].item()
                frame_idx = frame_indices[b_idx].item()

                # Calculate the actual frame index with delta
                target_frame_idx = frame_idx + delta

                # Get episode boundaries
                ep_start = self.episode_data_index["from"][ep_idx].item()
                ep_end = self.episode_data_index["to"][ep_idx].item()
                ep_length = ep_end - ep_start

                # Calculate progress with proper boundary handling
                if target_frame_idx < 0:
                    prog = target_frame_idx / max(1, ep_length - 1)
                elif target_frame_idx >= ep_length:
                    prog = target_frame_idx / max(1, ep_length - 1)
                else:
                    prog = target_frame_idx / max(1, ep_length - 1)
                    
                # Clip to reasonable bounds and clamp to [0,1] as recommended
                prog = max(0.0, min(1.0, prog))
                frame_progress.append(prog)

            all_progress.append(
                torch.tensor(frame_progress, device=device, dtype=torch.float32)
            )
        
        if debug_indexing:
            print("=" * 30)

        # Stack to get (B, T) tensor where T is the temporal sequence length
        target = torch.stack(all_progress, dim=1)  # (B, max_seq_len)

        # Apply stride/dropout indexing to match the processed frames
        return target[:, idx]

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


def apply_video_rewind_fixed(frames: Tensor, rewind_prob: float = 0.5, last3_prob: float | None = None) -> tuple[Tensor, Tensor]:
    """Apply video rewinding augmentation WITHOUT constant-value padding (FIXED version).
    
    This version ensures the rewound sequence is exactly T frames without flat plateaus
    that drag down the target mean.

    Args:
        frames: Tensor of shape (B, T, C, H, W)
        rewind_prob: Probability of applying rewind augmentation to each video
        last3_prob: Probability of limiting rewind to last 3 frames

    Returns:
        Augmented frames and corresponding progress labels
    """
    B, T, C, H, W = frames.shape
    device = frames.device

    # Create default progress labels using window-relative progress (0 to 1)
    # This centers the mean around 0.5 and removes episode-length dependence
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
                
                # Create corresponding progress labels without constant padding
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
                            # Create progress labels
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