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
RLearN: Video-Language Conditioned Reward Model

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
        |
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

    def __init__(self, config: RLearNConfig):
        super().__init__(config)
        self.config = config

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
        # Projection from scalar value summary to text embedding dim for InfoNCE
        self.value_to_text_proj = nn.Linear(1, config.dim_model)

        # Spatial attention for InfoNCE
        self.spatial_cross_attn = nn.MultiheadAttention(
            embed_dim=config.dim_model, num_heads=config.n_heads, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(config.dim_model)

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

        # ---- NEW: use the HF processor to standardize size & normalization ----
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
        # ----------------------------------------------------------------------

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
        """Compute training loss and logs.

        Expected batch keys:
          - OBS_IMAGES: list[Tensor] of shape [(B, C, H, W), ...] per time step or stacked (B, T, C, H, W)
          - OBS_LANGUAGE: optional string tokens already tokenized externally or raw strings
          - REWARD: (B, T) or (B,) target rewards
        """
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Extract frames and form (B, T, C, H, W), padding if needed
        frames = extract_visual_sequence(batch, target_seq_len=self.config.max_seq_len)
        B, T, C, H, W = frames.shape

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

        # Extract BOTH CLS token and spatial patches
        if hasattr(vision_outputs, "last_hidden_state"):
            all_tokens = vision_outputs.last_hidden_state  # (BT, num_tokens, D)
            cls_tokens = all_tokens[:, 0]  # (BT, D) - CLS token for temporal modeling
            spatial_tokens = all_tokens[:, 1:]  # (BT, num_patches, D) - spatial patches
        else:
            raise RuntimeError("Vision encoder must output last_hidden_state with spatial features")

        # Project CLS tokens for temporal sequence
        visual_seq = self.visual_proj(cls_tokens).reshape(B, -1, self.config.dim_model)  # (B, T', D)

        # Keep spatial features for spatial-aware losses (project them too)
        # Assuming 16x16 patches for 256x256 image with patch_size=16
        num_patches = spatial_tokens.shape[1]
        spatial_features = self.visual_proj(spatial_tokens).reshape(
            B, -1, num_patches, self.config.dim_model
        )  # (B, T', num_patches, D)

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

        # Targets
        target = batch.get(REWARD, None)
        loss_dict: dict[str, float] = {}
        if target is None:
            # If no labels, return zeros loss and logits for inference
            loss = values.mean() * 0.0
            loss_dict["has_labels"] = 0.0
            return loss, {**loss_dict, "values_mean": values.mean().item()}

        # Align target with sampled timesteps
        if target.dim() == 1:
            target = target.unsqueeze(1)  # (B, 1)

        # Handle target padding to match frame sequence if needed
        if target.shape[1] < self.config.max_seq_len:
            # Pad targets by repeating the first value (assuming it's the earliest)
            padding_needed = self.config.max_seq_len - target.shape[1]
            first_target = target[:, :1]  # (B, 1)
            padding = first_target.expand(target.shape[0], padding_needed)
            target = torch.cat([padding, target], dim=1)  # Prepend padding

            import logging

            logging.debug(
                f"Padded targets from {target.shape[1] - padding_needed} to {self.config.max_seq_len}"
            )

        # Now safely index with idx
        target = target[:, idx]

        # Composite loss
        # 1) Progress regression on z-scored values to match normalized progress labels y in [0,1]

        # Debug: Check if values have enough variance
        values_std = values.std()
        if values_std < 1e-4:
            # Early in training, model outputs are nearly constant
            # Use direct MSE loss without z-scoring to encourage variance
            import logging

            logging.info(f"Low variance in values (std={values_std:.6f}), using direct MSE")
            # Apply sigmoid directly to raw values to get them in [0,1] range
            prog_pred = torch.sigmoid(values * 10.0)  # Scale up to encourage learning
            L_prog = F.mse_loss(prog_pred, torch.clamp(target, 0.0, 1.0))
        else:
            # Normal case: use z-score normalization
            zV = zscore(values, eps=self.config.zscore_eps)
            # Check for NaN after zscore
            if torch.isnan(zV).any():
                import logging

                logging.warning(f"NaN after zscore. Values: {values}, zV: {zV}")
                # Fallback to direct sigmoid
                prog_pred = torch.sigmoid(values * 10.0)
            else:
                prog_pred = torch.sigmoid(zV)

            L_prog = F.mse_loss(prog_pred, torch.clamp(target, 0.0, 1.0))

        # Mismatched pairs: randomly shuffle language within batch and require near-zero progress
        if self.training and torch.rand(()) < self.config.mismatch_lang_prob and values.size(0) > 1:
            shuffled = torch.randperm(B, device=values.device)
            lang_mismatch = lang_emb[shuffled]
            mismatch_feat = self.temporal(visual_seq, lang_mismatch, return_features=True)
            mismatch_V = self.head(mismatch_feat).squeeze(-1)
            L_prog_mismatch = F.mse_loss(
                torch.sigmoid(zscore(mismatch_V, eps=self.config.zscore_eps)), torch.zeros_like(target)
            )
        else:
            L_prog_mismatch = torch.zeros((), device=values.device)

        # 2) Spatial-Aware InfoNCE: Use language to attend to relevant spatial regions
        # Take late timesteps' spatial features
        k = min(self.config.last_k_for_nce, spatial_features.shape[1])
        late_spatial = spatial_features[:, -k:].mean(dim=1)  # (B, num_patches, D)

        # Language queries spatial patches via cross-attention
        lang_query = lang_emb.unsqueeze(1)  # (B, 1, D)
        attended_spatial, spatial_attn_weights = self.spatial_cross_attn(
            query=lang_query, key=late_spatial, value=late_spatial, need_weights=True
        )
        attended_spatial = self.spatial_norm(attended_spatial).squeeze(1)  # (B, D)

        # Contrastive loss with spatially-attended features
        attended_spatial = F.normalize(attended_spatial, dim=-1)
        lang_norm = F.normalize(lang_emb, dim=-1)
        logits_spatial = (attended_spatial @ lang_norm.t()) / self.config.nce_temperature  # (B, B)
        targets_nce = torch.arange(B, device=values.device)
        L_spatial_nce = F.cross_entropy(logits_spatial, targets_nce)

        # 3) ReWiND Reversible Ranking: Learn from both forward and reversed trajectories
        # This teaches the model what constitutes progress vs undoing progress
        L_rank_forward, L_rank_reverse = reversible_ranking_loss(
            values,
            target,
            margin=self.config.ranking_margin,
            num_pairs=self.config.num_ranking_pairs,
            min_gap=self.config.min_rank_gap,
        )
        L_rewind = L_rank_forward + L_rank_reverse

        # Check for NaNs in individual loss components
        if torch.isnan(L_prog):
            import logging

            logging.warning(f"NaN in L_prog. Values: {values}, Target: {target}")
            # Return a small loss with gradients instead of zero
            L_prog = values.mean() * 0.0 + 0.01

        if torch.isnan(L_spatial_nce):
            import logging

            logging.warning("NaN in L_spatial_nce")
            # Use a dummy loss that maintains gradients
            L_spatial_nce = attended_spatial.mean() * 0.0 + 0.01

        if torch.isnan(L_rewind):
            import logging

            logging.warning("NaN in L_rewind")
            # Use values to maintain gradient flow
            L_rewind = values.mean() * 0.0 + 0.01

        loss = (
            self.config.lambda_prog * (L_prog + L_prog_mismatch)
            + self.config.lambda_spatial_nce * L_spatial_nce
            + self.config.lambda_rewind * L_rewind
        )

        # Final NaN check
        if torch.isnan(loss):
            import logging

            logging.warning("NaN loss detected, using fallback loss")
            # Use a small loss that maintains gradients
            loss = values.mean() * 0.0 + 0.01

        loss_dict.update(
            {
                "loss_prog": L_prog.item() if not torch.isnan(L_prog) else 0.0,
                "loss_prog_mismatch": L_prog_mismatch.item() if not torch.isnan(L_prog_mismatch) else 0.0,
                "loss_spatial_nce": L_spatial_nce.item() if not torch.isnan(L_spatial_nce) else 0.0,
                "loss_rewind_forward": L_rank_forward.item() if not torch.isnan(L_rank_forward) else 0.0,
                "loss_rewind_reverse": L_rank_reverse.item() if not torch.isnan(L_rank_reverse) else 0.0,
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


def pairwise_ranking_loss(logits: Tensor, target: Tensor, margin: float = 0.1, num_pairs: int = 32) -> Tensor:
    # logits, target: (B, T)
    B, T = logits.shape
    if T < 2:
        return logits.mean() * 0.0
    # Sample pairs i<j and enforce r_j > r_i when target_j > target_i
    losses = []
    for _ in range(num_pairs):
        i = torch.randint(0, T - 1, (B,), device=logits.device)
        j = i + torch.randint(1, T - i.max(), (1,), device=logits.device)
        j = j.expand_as(i)
        li = logits[torch.arange(B), i]
        lj = logits[torch.arange(B), j]
        yi = target[torch.arange(B), i]
        yj = target[torch.arange(B), j]
        sign = torch.sign(yj - yi)
        # hinge: max(0, margin - sign*(lj-li))
        loss = F.relu(margin - sign * (lj - li))
        losses.append(loss.mean())
    return torch.stack(losses).mean()


def zscore(x: Tensor, eps: float = 1e-3) -> Tensor:
    """Z-score normalization with numerical stability.

    Args:
        x: Tensor of shape (B, T) where B is batch size, T is sequence length
        eps: Small epsilon for numerical stability

    Returns:
        Z-scored tensor of same shape as input
    """
    # Handle both (B,) and (B, T) shapes
    if x.dim() == 1:
        x = x.unsqueeze(1)  # Make it (B, 1)

    B, T = x.shape

    if T == 1:
        # Single timestep: use tanh to bound values instead of z-score
        return torch.tanh(x * 0.1)

    # Multiple timesteps: compute z-score across time dimension for each batch
    mean = x.mean(dim=1, keepdim=True)  # (B, 1)
    std = x.std(dim=1, keepdim=True, unbiased=False)  # (B, 1)

    # Check if std is valid (not zero or NaN)
    std_is_valid = (std > eps) & (~torch.isnan(std))

    # Safe std for division
    std_safe = torch.where(std_is_valid, std, torch.ones_like(std))

    # Compute z-score where valid
    z = (x - mean) / std_safe

    # For invalid cases (constant values across time), use tanh of centered values
    z_fallback = torch.tanh((x - mean) * 0.1)
    z = torch.where(std_is_valid.expand_as(z), z, z_fallback)

    # Final safety clamp
    z = torch.clamp(z, min=-5.0, max=5.0)

    # Check for any remaining NaNs and replace with 0
    z = torch.nan_to_num(z, nan=0.0)

    return z


def temporal_logistic_ranking(
    values: Tensor, margin: float = 0.1, min_gap: int = 1, num_pairs: int = 64
) -> Tensor:
    """VLC-style temporal monotonicity: encourage V[j] > V[i] for j>i.

    Samples pairs (i<j) with a minimum gap and applies softplus(m - (Vj - Vi)).
    """
    B, T = values.shape
    if T < 2:
        return values.mean() * 0.0
    losses = []
    device = values.device
    for _ in range(num_pairs):
        i = torch.randint(0, max(1, T - min_gap), (B,), device=device)
        j = i + torch.randint(min_gap, T - i.max(), (1,), device=device)
        j = j.expand_as(i)
        vi = values[torch.arange(B), i]
        vj = values[torch.arange(B), j]
        losses.append(F.softplus(margin - (vj - vi)).mean())
    return torch.stack(losses).mean()


def reversible_ranking_loss(
    values: Tensor, target: Tensor, margin: float = 0.1, num_pairs: int = 64, min_gap: int = 1
) -> tuple[Tensor, Tensor]:
    """ReWiND-style reversible ranking: learn from both forward and reversed trajectories.

    Key insight: If a trajectory shows progress forward, its reverse shows undoing progress.
    By training on both, the model learns what constitutes progress vs regression.

    Args:
        values: (B, T) predicted values
        target: (B, T) progress labels (0 to 1 for forward progress)
        margin: Margin for ranking loss
        num_pairs: Number of (far, near) pairs to sample
        min_gap: Minimum temporal gap between pairs

    Returns:
        forward_loss: Loss from forward trajectory pairs
        reverse_loss: Loss from reversed trajectory pairs
    """
    B, T = values.shape
    if T < 2:
        zero_loss = values.mean() * 0.0
        return zero_loss, zero_loss

    device = values.device

    # Forward trajectory ranking: later frames should have higher values
    forward_losses = []
    for _ in range(num_pairs // 2):
        # Sample far-near pairs (far is earlier, near is later)
        far_idx = torch.randint(0, max(1, T - min_gap), (B,), device=device)
        near_idx = far_idx + torch.randint(min_gap, T - far_idx.max(), (1,), device=device)
        near_idx = near_idx.expand_as(far_idx)

        v_far = values[torch.arange(B), far_idx]
        v_near = values[torch.arange(B), near_idx]

        # Near (later) should have higher value than far (earlier)
        forward_losses.append(F.softplus(margin - (v_near - v_far)).mean())

    # Reversed trajectory ranking: treat reversed sequence with inverted progress
    # Reverse both values and targets
    reversed_values = values.flip(dims=[1])  # Reverse time dimension
    reversed_target = 1.0 - target.flip(dims=[1])  # Invert and reverse progress

    reverse_losses = []
    for _ in range(num_pairs // 2):
        # In reversed trajectory, what was "later" is now "earlier"
        far_idx = torch.randint(0, max(1, T - min_gap), (B,), device=device)
        near_idx = far_idx + torch.randint(min_gap, T - far_idx.max(), (1,), device=device)
        near_idx = near_idx.expand_as(far_idx)

        v_far_rev = reversed_values[torch.arange(B), far_idx]
        v_near_rev = reversed_values[torch.arange(B), near_idx]

        # In reversed trajectory with inverted progress,
        # near (which was originally earlier) should still have higher value
        reverse_losses.append(F.softplus(margin - (v_near_rev - v_far_rev)).mean())

    forward_loss = torch.stack(forward_losses).mean() if forward_losses else values.mean() * 0.0
    reverse_loss = torch.stack(reverse_losses).mean() if reverse_losses else values.mean() * 0.0

    return forward_loss, reverse_loss


def intra_trajectory_directional_ranking(
    values: Tensor, progress: Tensor, margin: float = 0.2, num_pairs: int = 64, min_gap: int = 1
) -> Tensor:
    """Directional ranking within trajectory based on progress labels.

    For pairs i<j within a trajectory:
    - If progress increases (y_j > y_i), enforce V_j > V_i
    - If progress decreases (y_j < y_i), enforce V_j < V_i
    - Ignore pairs where progress is unchanged

    Uses logistic loss: log(1 + exp(m - s_ij * (V_j - V_i)))
    where s_ij = sign(y_j - y_i)
    """
    B, T = values.shape
    if T < 2:
        return values.mean() * 0.0

    losses = []
    device = values.device

    for _ in range(num_pairs):
        # Sample time pairs i < j
        i = torch.randint(0, max(1, T - min_gap), (B,), device=device)
        max_j = min(T, i.max() + T - min_gap)
        j = i + torch.randint(min_gap, max_j - i.min(), (1,), device=device)
        j = j.expand_as(i).clamp(max=T - 1)

        # Get values and progress at sampled times
        vi = values[torch.arange(B), i]
        vj = values[torch.arange(B), j]
        yi = progress[torch.arange(B), i]
        yj = progress[torch.arange(B), j]

        # Compute direction sign
        s_ij = torch.sign(yj - yi)

        # Only compute loss for non-zero progress differences
        mask = s_ij != 0
        if mask.any():
            diff = vj - vi
            loss = torch.log1p(torch.exp(margin - s_ij * diff))
            losses.append(loss[mask].mean())

    return torch.stack(losses).mean() if losses else values.mean() * 0.0


def inter_instruction_contrastive_ranking(
    values_correct: Tensor, values_incorrect: Tensor, margin: float = 0.2
) -> Tensor:
    """Ranking between correct and incorrect instructions for same frames.

    Enforces V_t(z) > V_t(z') where z is correct instruction and z' is incorrect.
    Uses logistic loss: log(1 + exp(m - (V_t(z) - V_t(z'))))
    """
    diff = values_correct - values_incorrect
    return torch.log1p(torch.exp(margin - diff)).mean()


def flatness_under_mismatch(values: Tensor, epsilon: float = 0.05, num_pairs: int = 32) -> Tensor:
    """Enforce flat values over time for mismatched instructions.

    For trajectory with wrong instruction, V should not change much over time.
    Uses Huber loss to allow small variations within epsilon band.
    """
    B, T = values.shape
    if T < 2:
        return values.mean() * 0.0

    losses = []
    device = values.device

    for _ in range(num_pairs):
        i = torch.randint(0, T - 1, (B,), device=device)
        j = torch.randint(i.min() + 1, T, (1,), device=device)
        j = j.expand_as(i)

        vi = values[torch.arange(B), i]
        vj = values[torch.arange(B), j]

        # Huber loss with small delta for near-zero target
        diff = vj - vi
        loss = F.huber_loss(diff, torch.zeros_like(diff), delta=epsilon)
        losses.append(loss)

    return torch.stack(losses).mean()
