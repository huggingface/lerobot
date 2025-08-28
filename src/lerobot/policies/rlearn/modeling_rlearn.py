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

This implementation follows the ReWiND paper approach:
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
  |  Vision Encoder (frozen)     |  e.g. SigLIP2 vision tower
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
                                    |      |  Text Encoder (frozen)       |  e.g. SigLIP2 text tower
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

Training
  - Loss: composite loss with progress regression, spatial-aware InfoNCE, and ReWiND reversible ranking

Notes
  - Backbones (vision/text) are frozen by default; only projections, temporal module, and head are trainable.
  - Stride/frame dropout applied during training can subsample timesteps.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.constants import OBS_IMAGE, OBS_IMAGES, OBS_LANGUAGE, REWARD
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rlearn.configuration_rlearn import RLearNConfig


class RLearNPolicy(PreTrainedPolicy):
    """Video-language conditioned reward model.

    - Visual encoder: frozen SigLIP2 (via transformers AutoModel), returns per-frame embeddings.
    - Text encoder: frozen SigLIP2 text tower, returns a language embedding.
    - Temporal module: causal transformer over time that cross-attends to language embedding.
    - Output: per-timestep reward logits; trainable small head.
    """

    config_class = RLearNConfig
    name = "rlearn"

    def __init__(self, config: RLearNConfig, episode_data_index: dict = None):
        super().__init__(config)
        self.config = config
        self.episode_data_index = episode_data_index  # Store episode boundaries for progress calculation

        # Encoders
        from transformers import AutoModel, AutoProcessor

        self.vision_text_model = AutoModel.from_pretrained(config.model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

        # Detect towers
        if hasattr(self.vision_text_model, "vision_model") and hasattr(self.vision_text_model, "text_model"):
            self.vision_encoder = self.vision_text_model.vision_model
            self.text_encoder = self.vision_text_model.text_model
            self.vision_hidden = getattr(self.vision_text_model.config, "vision_config", None).hidden_size
            self.text_hidden = getattr(self.vision_text_model.config, "text_config", None).hidden_size
        else:
            # Fallback if AutoModel exposes pooled outputs directly (rare for SigLIP2)
            self.vision_encoder = self.vision_text_model
            self.text_encoder = self.vision_text_model
            self.vision_hidden = getattr(self.vision_text_model.config, "hidden_size", 768)
            self.text_hidden = getattr(self.vision_text_model.config, "hidden_size", 768)

        if config.freeze_backbones:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # Linear projections to the shared temporal model dimension
        self.visual_proj = nn.Linear(self.vision_hidden, config.dim_model)
        self.text_proj = nn.Linear(self.text_hidden, config.dim_model)

        # Positional encodings over time
        self.register_buffer(
            "positional_encoding",
            create_sinusoidal_pos_encoding(config.max_seq_len, config.dim_model),
            persistent=False,
        )
        # Optional first-frame learned bias to discourage position cheating
        self.first_frame_bias = (
            nn.Parameter(torch.zeros(1, 1, config.dim_model))
            if config.use_first_frame_positional_bias
            else None
        )

        # Temporal aggregator: causal transformer over time with language cross-attention
        self.temporal = TemporalCausalTransformer(
            dim_model=config.dim_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            pre_norm=config.pre_norm,
        )

        # Reward head with proper initialization
        head_linear = nn.Linear(config.dim_model, 1)
        # Initialize with small weights and bias to output values around 0
        nn.init.normal_(head_linear.weight, mean=0.0, std=0.02)
        nn.init.constant_(head_linear.bias, 0.0)  # Start with 0 bias, sigmoid(0) = 0.5

        head_layers: list[nn.Module] = [head_linear]
        if config.use_tanh_head:
            head_layers.append(nn.Tanh())
        self.head = nn.Sequential(*head_layers)
        # Simple frame dropout probability
        self.frame_dropout_p = config.frame_dropout_p
        self.stride = max(1, config.stride)

    def get_optim_params(self) -> dict:
        # Train only projections, temporal module and head by default if backbones are frozen
        return [p for p in self.parameters() if p.requires_grad]

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
        """Predict per-timestep rewards for evaluation.

        Args:
            batch: Input batch with OBS_IMAGES and optionally OBS_LANGUAGE

        Returns:
            Predicted rewards tensor of shape (B, T)
        """

        batch = self.normalize_inputs(batch)

        # Extract frames and form (B, T, C, H, W), padding if needed
        frames = extract_visual_sequence(batch, target_seq_len=self.config.max_seq_len)
        B, T, C, H, W = frames.shape

        # Apply stride (no dropout during eval)
        idx = torch.arange(0, T, self.stride, device=frames.device)
        frames = frames[:, idx]
        B, T_eff, C, H, W = frames.shape  # NEW: effective length after stride

        # Encode language
        lang_emb = encode_language(
            batch.get(OBS_LANGUAGE, None), self.text_encoder, self.processor, batch_size=B
        )
        lang_emb = self.text_proj(lang_emb)  # (B, D)

        # Use the HF processor to standardize size & normalization
        # Flatten (B, T_eff, C, H, W) -> (BT, C, H, W)
        BT = B * T_eff
        flat = frames.reshape(BT, C, H, W).detach().cpu()

        # Convert to uint8 HWC numpy (processor prefers PIL/np)
        # If already in [0,1], scale to [0,255]
        if flat.dtype != torch.uint8:
            if flat.numel() > 0 and float(flat.max()) <= 1.0:
                flat = flat * 255.0
            flat = flat.clamp(0, 255).round().to(torch.uint8)

        images = [flat[k].permute(1, 2, 0).numpy() for k in range(flat.size(0))]

        proc_out = self.processor(images=images, return_tensors="pt")
        pixel_values = proc_out["pixel_values"].to(next(self.vision_encoder.parameters()).device)

        # Encode frames through visual tower per frame
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)

        # Extract CLS tokens for temporal modeling
        if hasattr(vision_outputs, "last_hidden_state"):
            cls_tokens = vision_outputs.last_hidden_state[:, 0]  # (BT, D_vision)
        else:
            raise RuntimeError("Vision encoder must output last_hidden_state")

        # Project CLS tokens for temporal sequence
        visual_seq = self.visual_proj(cls_tokens).reshape(B, T_eff, self.config.dim_model)  # (B, T', D)

        # Add temporal positional encodings and optional first-frame bias
        pe = (
            self.positional_encoding[: visual_seq.shape[1]]
            .unsqueeze(0)
            .to(visual_seq.dtype)
            .to(visual_seq.device)
        )
        visual_seq = visual_seq + pe
        if self.first_frame_bias is not None:
            visual_seq = visual_seq.clone()
            visual_seq[:, :1] = visual_seq[:, :1] + self.first_frame_bias

        # Temporal model with cross-attention to language
        temporal_features = self.temporal(visual_seq, lang_emb, return_features=True)  # (B, T', D)
        values = self.head(temporal_features).squeeze(-1)  # (B, T')

        return values

    def normalize_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Initial version: no-op; rely on upstream processors if any
        return batch

    def normalize_targets(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Initial version: no-op
        return batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Compute ReWiND training loss with on-the-fly progress label generation.

        Expected batch keys:
          - OBS_IMAGES: list[Tensor] of shape [(B, C, H, W), ...] per time step or stacked (B, T, C, H, W)
          - OBS_LANGUAGE: optional string tokens already tokenized externally or raw strings

        Note: Progress labels (0 to 1) are generated automatically for each episode.
              No REWARD key is needed in the batch.
        """
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Extract frames and form (B, T, C, H, W), padding if needed
        frames = extract_visual_sequence(batch, target_seq_len=self.config.max_seq_len)
        B, T, C, H, W = frames.shape

        # Apply video rewinding augmentation during training
        if self.training and self.config.use_video_rewind:
            frames, augmented_target = apply_video_rewind(frames, rewind_prob=self.config.rewind_prob)
            # Use augmented progress labels if rewinding was applied
            if REWARD in batch:
                target = augmented_target

        # Apply stride and frame dropout during training
        idx = torch.arange(0, T, self.stride, device=frames.device)
        if self.training and self.frame_dropout_p > 0.0 and T > 1:
            mask = torch.rand_like(idx.float()) > self.frame_dropout_p
            idx = idx[mask.long().bool()]
            if idx.numel() == 0:
                idx = torch.tensor([0], device=frames.device)
        frames = frames[:, idx]

        # Encode language
        lang_emb = encode_language(
            batch.get(OBS_LANGUAGE, None), self.text_encoder, self.processor, batch_size=B
        )
        lang_emb = self.text_proj(lang_emb)  # (B, D)

        # Encode frames through visual tower per frame
        # Flatten time for batched encode
        BT = B * frames.shape[1]
        flat = frames.reshape(BT, C, H, W)

        # Use HF processor to properly resize and normalize images
        # Convert to CPU for processing, then move back to device
        flat_cpu = flat.detach().cpu()

        # Convert to uint8 HWC numpy format expected by processor
        if flat_cpu.dtype != torch.uint8:
            if flat_cpu.numel() > 0 and float(flat_cpu.max()) <= 1.0:
                flat_cpu = flat_cpu * 255.0
            flat_cpu = flat_cpu.clamp(0, 255).round().to(torch.uint8)

        # Convert to list of numpy arrays
        images = [flat_cpu[k].permute(1, 2, 0).numpy() for k in range(flat_cpu.size(0))]

        # Process with HF processor (resizes to 256x256 and normalizes)
        proc_out = self.processor(images=images, return_tensors="pt")
        pixel_values = proc_out["pixel_values"].to(next(self.vision_encoder.parameters()).device)

        # Encode through vision model
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)

        # Extract CLS token for temporal modeling
        if hasattr(vision_outputs, "last_hidden_state"):
            cls_tokens = vision_outputs.last_hidden_state[:, 0]  # (BT, D) - CLS token
        else:
            raise RuntimeError("Vision encoder must output last_hidden_state")

        # Project CLS tokens for temporal sequence
        visual_seq = self.visual_proj(cls_tokens).reshape(B, -1, self.config.dim_model)  # (B, T', D)

        # Add temporal positional encodings and optional first-frame bias
        pe = (
            self.positional_encoding[: visual_seq.shape[1]]
            .unsqueeze(0)
            .to(visual_seq.dtype)
            .to(visual_seq.device)
        )
        visual_seq = visual_seq + pe
        if self.first_frame_bias is not None:
            visual_seq = visual_seq.clone()
            visual_seq[:, :1] = visual_seq[:, :1] + self.first_frame_bias

        # Temporal model with cross-attention to language
        temporal_features = self.temporal(visual_seq, lang_emb, return_features=True)  # (B, T', D)
        values = self.head(temporal_features).squeeze(-1)  # (B, T')

        # Generate progress labels on-the-fly (ReWiND approach)
        # IMPORTANT: Progress should be 0-1 across the ENTIRE EPISODE, not just the temporal window
        loss_dict: dict[str, float] = {}

        # Check if video rewinding already set the target
        if self.training and self.config.use_video_rewind and "augmented_target" in locals():
            # Use the augmented target from video rewinding
            target = augmented_target
        else:
            # Calculate true episode progress using episode_index and frame_index from batch
            if "episode_index" in batch and "frame_index" in batch and hasattr(self, "episode_data_index"):
                # Get episode indices and frame indices from batch
                episode_indices = batch["episode_index"]  # Shape: (B,)
                frame_indices = batch["frame_index"]  # Shape: (B,)

                # Calculate progress for the current frame in each sample
                progress_values = []

                for b_idx in range(B):
                    ep_idx = episode_indices[b_idx].item()
                    frame_idx = frame_indices[b_idx].item()

                    # Get episode boundaries
                    ep_start = self.episode_data_index["from"][ep_idx].item()
                    ep_end = self.episode_data_index["to"][ep_idx].item()
                    ep_length = ep_end - ep_start

                    # Progress from 0 to 1 within the episode
                    # frame_index is relative to the episode (0-based within episode)
                    progress = frame_idx / max(1, ep_length - 1)
                    progress_values.append(progress)

                # Create progress tensor for the current frame (last in temporal sequence)
                current_progress = torch.tensor(progress_values, device=values.device, dtype=values.dtype)

                # Now calculate progress for ALL frames in the temporal window
                # The observation_delta_indices tell us which frames we're looking at
                delta_indices = self.config.observation_delta_indices  # e.g., [-15, -14, ..., 0]

                # Calculate progress for each frame in the temporal window
                all_progress = []
                for delta in delta_indices:
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

                        # Clamp to episode boundaries (frame_index is relative to episode)
                        target_frame_idx = max(0, min(ep_length - 1, target_frame_idx))

                        # Calculate progress for this frame
                        prog = target_frame_idx / max(1, ep_length - 1)
                        frame_progress.append(prog)

                    all_progress.append(
                        torch.tensor(frame_progress, device=values.device, dtype=values.dtype)
                    )

                # Stack to get (B, T) tensor where T is the temporal sequence length
                target = torch.stack(all_progress, dim=1)  # (B, max_seq_len)

                # Apply stride/dropout indexing to match the processed frames
                target = target[:, idx]

            elif "index" in batch and hasattr(self, "episode_data_index"):
                # Fallback: Use global index if available
                global_indices = batch["index"]  # Shape: (B,)

                # For each index, find which episode it belongs to and its position
                progress_values = []

                for global_idx in global_indices:
                    # Find which episode this index belongs to
                    episode_starts = self.episode_data_index["from"]
                    episode_ends = self.episode_data_index["to"]

                    # Find the episode by checking which range the index falls into
                    episode_idx = None
                    frame_in_episode = None
                    for ep_idx in range(len(episode_starts)):
                        if episode_starts[ep_idx] <= global_idx < episode_ends[ep_idx]:
                            episode_idx = ep_idx
                            frame_in_episode = global_idx.item() - episode_starts[ep_idx].item()
                            break

                    if episode_idx is not None:
                        # Calculate position within episode
                        ep_start = episode_starts[episode_idx].item()
                        ep_end = episode_ends[episode_idx].item()
                        ep_length = ep_end - ep_start

                        # Progress from 0 to 1 within the episode
                        progress = frame_in_episode / max(1, ep_length - 1)
                    else:
                        # Fallback if we can't find the episode (shouldn't happen)
                        progress = 0.5

                    progress_values.append(progress)

                # For temporal window, use simplified linear progress
                # (proper calculation would need all frame indices in the window)
                T_effective = len(idx)
                target = torch.tensor(progress_values, device=values.device, dtype=values.dtype)
                target = target.unsqueeze(1).expand(B, T_effective)  # Simple expansion

            else:
                raise ValueError(
                    "No episode information found in batch. Please ensure 'episode_index' and 'frame_index' keys are present."
                )

        # During inference, we might not want to compute loss
        if not self.training and target is None:
            loss = values.mean() * 0.0
            loss_dict["has_labels"] = 0.0
            return loss, {**loss_dict, "values_mean": values.mean().item()}

        # ReWiND Loss (following the paper exactly)
        # The core loss is progress regression with video rewinding augmentation

        # 1) Main progress regression loss for matched sequences
        # Target should be normalized progress from 0 to 1 (t/T)
        L_progress = F.mse_loss(values, target)

        # 2) Mismatched video-language pairs should predict zero progress
        L_mismatch = torch.zeros((), device=values.device)
        if self.training and self.config.use_mismatch_loss and values.size(0) > 1:
            # Randomly shuffle language instructions within the batch
            shuffled_indices = torch.randperm(B, device=values.device)
            lang_mismatch = lang_emb[shuffled_indices]

            # Forward pass with mismatched language
            mismatch_feat = self.temporal(visual_seq, lang_mismatch, return_features=True)
            mismatch_values = self.head(mismatch_feat).squeeze(-1)

            # Mismatched pairs should predict zero progress
            L_mismatch = F.mse_loss(mismatch_values, torch.zeros_like(target))

        # Total loss is just progress regression (rewinding is handled via data augmentation)
        loss = L_progress + L_mismatch

        # Log individual loss components
        loss_dict.update(
            {
                "loss_progress": L_progress.item(),
                "loss_mismatch": L_mismatch.item(),
            }
        )

        loss_dict["loss"] = loss.item()
        loss_dict["values_mean"] = values.mean().item()
        return loss, loss_dict


class TemporalCausalTransformer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
        pre_norm: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TemporalCausalTransformerLayer(dim_model, n_heads, dim_feedforward, dropout, pre_norm)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim_model)
        self.head = nn.Linear(dim_model, 1)

    def forward(self, x: Tensor, lang_emb: Tensor, return_features: bool = False) -> Tensor:
        # x: (B, T, D), lang_emb: (B, D)
        B, T, D = x.shape
        # Prepare language as a single token for cross-attention context
        lang_token = lang_emb.unsqueeze(1)  # (B, 1, D)

        x = x.transpose(0, 1)  # (T, B, D)
        lang_token = lang_token.transpose(0, 1)  # (1, B, D)
        causal_mask = generate_causal_mask(T, device=x.device)
        for layer in self.layers:
            x = layer(x, lang_token, causal_mask)
        x = self.norm(x)
        x = x.transpose(0, 1)  # (B, T, D)
        if return_features:
            return x
        return self.head(x)  # (B, T, 1)


class TemporalCausalTransformerLayer(nn.Module):
    def __init__(self, dim_model: int, n_heads: int, dim_feedforward: int, dropout: float, pre_norm: bool):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.activation = F.gelu
        self.pre_norm = pre_norm

    def forward(self, x: Tensor, lang_token: Tensor, causal_mask: Tensor) -> Tensor:
        # Self-attention with causal mask
        residual = x
        if self.pre_norm:
            x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask=causal_mask)[0]
        x = residual + self.dropout1(x)
        if not self.pre_norm:
            x = self.norm1(x)

        # Cross-attention to language token (keys/values from language, queries are time tokens)
        residual = x
        if self.pre_norm:
            x = self.norm2(x)
        # Broadcast language token across time
        T = x.shape[0]
        lang_kv = lang_token.expand(1, x.shape[1], x.shape[2])  # (1, B, D)
        x = self.cross_attn(x, lang_kv, lang_kv)[0]
        x = residual + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)

        # Feed-forward
        residual = x
        if self.pre_norm:
            x = self.norm3(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_encoding(max_len: int, dim: int) -> Tensor:
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))  # (D/2)
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L, D)


def generate_causal_mask(T: int, device=None) -> Tensor:
    # (T, T) with True where masking should occur for MultiheadAttention expects float mask or bool?
    mask = torch.full((T, T), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


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

    # Pad sequence if needed
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

    return frames


def encode_language(
    language_input: Tensor | list | str | None, text_encoder, processor, batch_size: int
) -> Tensor:
    # language_input can be: list[str] length B, or None
    if language_input is None:
        texts = [""] * batch_size
    elif isinstance(language_input, list):
        texts = language_input
    else:
        # Single string for the batch
        texts = [str(language_input)] * batch_size

    inputs = processor(text=texts, padding=True, return_tensors="pt")
    inputs = {k: v.to(next(text_encoder.parameters()).device) for k, v in inputs.items()}
    outputs = text_encoder(**inputs)
    if hasattr(outputs, "pooler_output"):
        emb = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state"):
        emb = outputs.last_hidden_state[:, 0]
    else:
        raise RuntimeError("Unsupported text encoder output structure")
    return emb


def apply_video_rewind(frames: Tensor, rewind_prob: float = 0.5) -> tuple[Tensor, Tensor]:
    """Apply video rewinding augmentation as described in ReWiND paper.

    Each video in the batch has an independent chance of being rewound.

    Args:
        frames: Tensor of shape (B, T, C, H, W)
        rewind_prob: Probability of applying rewind augmentation to each video

    Returns:
        Augmented frames and corresponding progress labels
    """
    B, T, C, H, W = frames.shape
    device = frames.device

    # Create default progress labels (linearly increasing from 0 to 1)
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

        # Apply rewinding to this video
        # Split point i: between frame 2 and T-1
        i = torch.randint(2, T, (1,)).item()

        # Rewind length k: between 1 and i-1 frames
        k = torch.randint(1, min(i, T - i + 1), (1,)).item()

        # Create rewound sequence: o1...oi, oi-1, ..., oi-k
        forward_frames = frames[b, :i]  # Frames up to split point
        reverse_frames = frames[b, max(0, i - k) : i].flip(dims=[0])  # Reversed frames

        # Concatenate forward and reverse parts
        rewound_seq = torch.cat([forward_frames, reverse_frames], dim=0)

        # Pad with zeros if needed to maintain shape
        if rewound_seq.shape[0] < T:
            padding = torch.zeros(T - rewound_seq.shape[0], C, H, W, device=device)
            rewound_seq = torch.cat([rewound_seq, padding], dim=0)
        elif rewound_seq.shape[0] > T:
            rewound_seq = rewound_seq[:T]

        # Create corresponding progress labels
        # Forward part: increasing progress
        forward_progress = torch.linspace(0, i / T, i, device=device)
        # Reverse part: decreasing progress
        reverse_progress = torch.linspace(i / T, max(0, (i - k) / T), k, device=device)

        rewound_progress = torch.cat([forward_progress, reverse_progress])

        # Pad progress if needed
        if rewound_progress.shape[0] < T:
            padding = torch.zeros(T - rewound_progress.shape[0], device=device)
            rewound_progress = torch.cat([rewound_progress, padding])
        elif rewound_progress.shape[0] > T:
            rewound_progress = rewound_progress[:T]

        augmented_frames.append(rewound_seq)
        augmented_progress.append(rewound_progress)

    return torch.stack(augmented_frames), torch.stack(augmented_progress)
