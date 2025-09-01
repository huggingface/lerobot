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

from __future__ import annotations

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor, nn
try:
    from hl_gauss_pytorch import HLGaussLayer
except Exception:
    HLGaussLayer = None  # Optional dependency; guarded at use sites

from lerobot.constants import OBS_IMAGE, OBS_IMAGES, OBS_LANGUAGE, REWARD
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rlearn.configuration_rlearn import RLearNConfig


class RLearNPolicy(PreTrainedPolicy):
    """Video-language conditioned reward model following ReWiND architecture: https://github.com/lucidrains/rewind-reward-pytorch/blob/main/rewind_reward_pytorch/rewind_reward.py#L11.

    - Visual encoder: frozen SigLIP2 vision tower, returns per-frame patch embeddings.
    - Text encoder: frozen SigLIP2 text tower, returns language token embeddings.

    """

    config_class = RLearNConfig
    name = "rlearn"

    def __init__(self, config: RLearNConfig, episode_data_index: dict = None):
        super().__init__(config)
        self.config = config
        self.episode_data_index = episode_data_index  # Store episode boundaries for progress calculation

        # Encoders - SigLIP2 shared checkpoint for vision and text
        from transformers import AutoProcessor, AutoModel
        
        # Shared processor handles both images and text
        self.processor = AutoProcessor.from_pretrained(config.vision_model_name, use_fast=True)
        # Shared model exposes .vision_model and .text_model
        self.siglip_model = AutoModel.from_pretrained(config.vision_model_name)
        self.vision_model = self.siglip_model.vision_model
        self.text_model = self.siglip_model
        
        # Move encoders to GPU if available
        if torch.cuda.is_available():
            self.vision_model = self.vision_model.to('cuda')
            self.text_model = self.text_model.to('cuda')
        
        # Get hidden sizes from models
        # SigLIP2 hidden sizes
        self.vision_hidden = getattr(getattr(self.siglip_model, 'config', None), 'vision_config', None)
        self.vision_hidden = getattr(self.vision_hidden, 'hidden_size', getattr(self.vision_model.config, 'hidden_size', 768))
        th = getattr(getattr(self.siglip_model, 'config', None), 'text_config', None)
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

        # Linear projections to the shared temporal model dimension
        self.to_lang_tokens = nn.Linear(self.text_hidden, config.dim_model)
        self.to_video_tokens = nn.Linear(self.vision_hidden, config.dim_model)

        # First-frame positional embedding (only applied to the first video frame)
        self.first_frame_pos = nn.Parameter(torch.zeros(1, 1, config.dim_model))

        # Cross-modal sequential aggregator – causal transformer over
        # [language tokens | video frame tokens] using PyTorch TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_model * config.ff_mult,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.aggregator = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Per-frame predictor pre-head
        self.frame_mlp = nn.Sequential(
            nn.LayerNorm(config.dim_model),
            nn.Linear(config.dim_model, config.dim_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Reward heads (mode-aware)
        self.use_categorical = bool(config.use_categorical_rewards)
        if self.use_categorical:
            self.reward_head = nn.Linear(config.dim_model, int(config.num_reward_bins))
            self.hl_gauss_layer = None
        else:
            # HL-Gauss expects per-bin logits; head outputs histogram-bin logits
            self.reward_head = nn.Linear(config.dim_model, int(config.hl_gauss_num_bins))
            if HLGaussLayer is not None:
                self.hl_gauss_layer = HLGaussLayer(
                    dim=int(config.hl_gauss_num_bins),
                    use_regression=not bool(config.use_hl_gauss_loss),
                    hl_gauss_loss=dict(
                        min_value=float(config.reward_min_value),
                        max_value=float(config.reward_max_value),
                        num_bins=int(config.hl_gauss_num_bins),
                    ),
                )
                self.hl_gauss_use_regression = not bool(config.use_hl_gauss_loss)
            else:
                self.hl_gauss_layer = None
                self.hl_gauss_use_regression = False

        # Sampling and regularization knobs
        self.stride = max(1, int(config.inference_stride))
        self.frame_dropout_p = float(config.frame_dropout_p)

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
                self.aggregator = torch.compile(self.aggregator, mode="reduce-overhead")
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

    def _encode_video_frames(self, frames: Tensor) -> Tensor:
        """Encode video frames through SigLIP2 vision tower and return per-frame CLS embeddings.
        
        Args:
            frames: (B, T, C, H, W)
        
        Returns:
            (B, T, D_vision) CLS token per frame
        """
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        
        # Optimized: Process tensor directly without numpy conversion
        device = next(self.vision_model.parameters()).device
        
        # Normalize to [0, 1] if needed and ensure correct format for DINOv3
        if flat.dtype != torch.float32:
            flat = flat.float()
        if flat.max() > 1.0:
            flat = flat / 255.0
            
        # GPU-friendly image preprocessing (resize + normalize) without Python loops
        iproc = getattr(self.processor, 'image_processor', None)
        if iproc is not None:
            size_cfg = getattr(iproc, 'size', {})
            if isinstance(size_cfg, dict):
                target_h = size_cfg.get('height', size_cfg.get('shortest_edge', 224))
                target_w = size_cfg.get('width', target_h)
            else:
                target_h = target_w = 224
            mean = torch.tensor(getattr(iproc, 'image_mean', [0.5, 0.5, 0.5]), device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
            std = torch.tensor(getattr(iproc, 'image_std', [0.5, 0.5, 0.5]), device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
        else:
            target_h = target_w = 224
            mean = torch.tensor([0.5, 0.5, 0.5], device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)

        flat = flat.to(device, non_blocking=True)
        flat = torch.nn.functional.interpolate(flat, size=(target_h, target_w), mode='bilinear', align_corners=False)
        pixel_values = (flat - mean.to(device)) / std.to(device)
        pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)

        use_amp = getattr(self.config, 'use_amp', False) and torch.cuda.is_available()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            vision_outputs = self.vision_model(pixel_values=pixel_values)
        
        # Prefer CLS token from last_hidden_state at index 0
        if hasattr(vision_outputs, 'last_hidden_state') and vision_outputs.last_hidden_state is not None:
            tokens = vision_outputs.last_hidden_state  # (BT, N_tokens, D)
            if tokens.dim() == 3 and tokens.shape[1] >= 1:
                cls_tokens_flat = tokens[:, 0, :]  # (BT, D)
            else:
                # Fallback to pooler if structure unexpected
                cls_tokens_flat = getattr(vision_outputs, 'pooler_output')
        elif hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
            # Use pooled output
            cls_tokens_flat = vision_outputs.pooler_output  # (BT, D)
        else:
            raise RuntimeError("SigLIP2 vision outputs do not contain last_hidden_state or pooler_output")
        
        # Robustly reshape CLS to (B, T, D): detect correct flatten order by maximizing temporal variance
        try:
            D = cls_tokens_flat.shape[-1]
            cand1 = cls_tokens_flat.reshape(B, T, D)
            cand2 = cls_tokens_flat.reshape(T, B, D).permute(1, 0, 2)
            def mean_time_diff_3d(x):
                if T <= 1:
                    return torch.tensor(0.0, device=x.device)
                diffs = (x[:, 1:, :] - x[:, :-1, :]).pow(2).sum(dim=-1).sqrt()
                return diffs.mean()
            diff1 = mean_time_diff_3d(cand1)
            diff2 = mean_time_diff_3d(cand2)
            frame_features = cand1 if diff1 >= diff2 else cand2
            if self.training and torch.rand(1).item() < 0.05:
                print(f"SigLIP reshape choice: {'(b t)->b t' if diff1 >= diff2 else '(t b)->b t'} | diff1={diff1.item():.6f}, diff2={diff2.item():.6f}")
        except Exception:
            # Fallback to default
            frame_features = cls_tokens_flat.view(B, T, -1)
        
        return frame_features
    
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

        # Process video frames through vision encoder (returns patch tokens)
        vision_start = time.perf_counter()
        video_patch_embeds = self._encode_video_frames(frames).to(device)  # (B, T_eff, P, D_vision)
        vision_time = time.perf_counter() - vision_start
        
        # Language embeddings + mask
        lang_start = time.perf_counter()
        lang_embeds, mask = self._encode_language_tokens(commands, device)
        lang_time = time.perf_counter() - lang_start
        
        # Token preparation
        # Project embeddings
        lang_tokens = self.to_lang_tokens(lang_embeds)  # (B, L, D)
        # SigLIP2 CLS per-frame already returned
        video_frame_embeds = video_patch_embeds  # (B, T_eff, D_vision)
        video_tokens = self.to_video_tokens(video_frame_embeds)  # (B, T_eff, D)
        # ReWiND: only add a first-frame positional bias (prevents time cheating)
        video_tokens[:, :1, :] = video_tokens[:, :1, :] + self.first_frame_pos

        # Build masks for TransformerEncoder
        lang_valid = mask  # (B, L) True where valid
        video_valid = torch.ones(B, video_tokens.shape[1], device=device, dtype=torch.bool)
        valid_mask = torch.cat([lang_valid, video_valid], dim=1)  # (B, S)
        key_padding_mask = ~valid_mask  # True -> masked

        tokens_seq = torch.cat([lang_tokens, video_tokens], dim=1)  # (B, S, D)

        # Causal mask (S, S): True masks out future positions
        S = tokens_seq.shape[1]
        causal_mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)

        transformer_start = time.perf_counter()
        attended_all = self.aggregator(tokens_seq, src_key_padding_mask=key_padding_mask, mask=causal_mask)
        transformer_time = time.perf_counter() - transformer_start

        # Split back video part
        L_len = lang_tokens.shape[1]
        attended_video = attended_all[:, L_len:, :]

        # Per-frame prediction
        frame_tokens = self.frame_mlp(attended_video)  # (B, T_eff, D)

        # Optional masking from batch for variable-length sequences
        video_lens = batch.get("video_lens", None)
        video_mask = None
        if video_lens is not None:
            if torch.is_tensor(video_lens):
                video_lens = video_lens.to(frame_tokens.device).long()
            else:
                video_lens = torch.as_tensor(video_lens, device=frame_tokens.device, dtype=torch.long)
            video_mask = self._lens_to_mask(video_lens, frame_tokens.shape[1])

        if self.use_categorical:
            # classification over bins
            video_frame_logits = self.reward_head(frame_tokens)  # (B,T,L)
            raw_like_logits = video_frame_logits.max(dim=-1).values
            predicted_rewards = torch.softmax(video_frame_logits, dim=-1)
        else:
            # embeddings for HL-Gauss (or regression)
            video_frame_embeds = self.reward_head(frame_tokens)  # (B,T,Bins)
            # derive a scalar proxy for regularizers
            raw_like_logits = torch.tanh(video_frame_embeds).mean(dim=-1)
            # predicted_rewards will be set after loss branch below

        # Regularizers use raw_like_logits for generality
        var_min = 1e-3
        if self.use_categorical:
            # use the max-logit trajectory as a proxy
            pred_proxy = torch.softmax(video_frame_logits, dim=-1).max(dim=-1).values
        else:
            pred_proxy = torch.sigmoid(raw_like_logits)
        L_flat = F.relu(var_min - pred_proxy.var(dim=1, unbiased=False)).mean() if pred_proxy.shape[1] > 1 else torch.zeros((), device=device)
        rank_margin = 0.02
        if raw_like_logits.shape[1] > 1:
            L_rank = F.relu(rank_margin - (raw_like_logits[:, 1:] - raw_like_logits[:, :-1])).mean()
        else:
            L_rank = torch.zeros((), device=device)

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

        # Compute main loss (or just return predictions in eval)
        loss_start = time.perf_counter()
        loss = torch.tensor(0.0, device=device)
        if target is None:
            total_loss = torch.tensor(0.0, device=device)
            loss = total_loss
            predicted_rewards = pred_proxy if self.use_categorical else pred_proxy
        else:
            if self.use_categorical:
                # map targets in [0,1] to bins
                num_bins = int(self.config.num_reward_bins)
                bin_idx = (target.clamp(0, 1) * (num_bins - 1) + 1e-6).long()
                if video_mask is not None:
                    bin_idx = torch.where(video_mask, bin_idx, torch.full_like(bin_idx, -1))
                loss_ce = F.cross_entropy(video_frame_logits.permute(0, 2, 1), bin_idx, ignore_index=-1)
                total_loss = loss_ce
                loss = loss_ce
                predicted_rewards = torch.softmax(video_frame_logits, dim=-1)
            else:
                # HL-Gauss or regression
                if (self.hl_gauss_layer is not None) and (not self.hl_gauss_use_regression):
                    # Ensure targets within configured range
                    t_min = float(self.config.reward_min_value)
                    t_max = float(self.config.reward_max_value)
                    target_clamped = target.clamp(t_min, t_max)
                    loss = self.hl_gauss_layer(video_frame_embeds, target_clamped, mask=video_mask)
                    total_loss = loss
                    predicted_rewards = self.hl_gauss_layer(video_frame_embeds)
                elif (self.hl_gauss_layer is not None) and self.hl_gauss_use_regression:
                    pred_values = self.hl_gauss_layer(video_frame_embeds)  # (B,T)
                    if video_mask is not None:
                        loss = F.smooth_l1_loss(pred_values[video_mask], target[video_mask], beta=0.25)
                    else:
                        loss = F.smooth_l1_loss(pred_values, target, beta=0.25)
                    total_loss = loss
                    predicted_rewards = pred_values
                else:
                    # fall back to existing logit regression path on a scalar proxy
                    target_expanded = target
                    eps = self.config.logit_eps
                    target_logits = torch.logit(target_expanded.clamp(eps, 1 - eps))
                    loss = F.smooth_l1_loss(raw_like_logits, target_logits, beta=0.25)
                    total_loss = loss
                    predicted_rewards = torch.sigmoid(raw_like_logits)
        

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
                
                # Pack and forward with masks
                lang_valid_mm = mask_mm
                valid_mask_mm = torch.cat([lang_valid_mm, video_valid], dim=1)
                key_padding_mask_mm = ~valid_mask_mm
                tokens_seq_mm = torch.cat([lang_tokens_mm, video_tokens], dim=1)
                attended_all_mm = self.aggregator(tokens_seq_mm, src_key_padding_mask=key_padding_mask_mm, mask=causal_mask)
                attended_video_mm = attended_all_mm[:, L_len:, :]
                
                # Process mismatch frames with single MLP
                mismatch_tokens = self.frame_mlp(attended_video_mm)  # (B, T, D)
                mismatch_raw_logits = self.reward_head(mismatch_tokens).squeeze(-1)

                mismatch_tensor = torch.tensor(mismatch_mask, device=device, dtype=torch.bool)
                if mismatch_tensor.any():
                    eps = self.config.logit_eps
                    zeros_target_logits = torch.logit(torch.full_like(mismatch_raw_logits, eps))
                    mismatch_loss_per_sample = F.smooth_l1_loss(
                        mismatch_raw_logits, zeros_target_logits, beta=0.25, reduction='none'
                    ).mean(dim=1)
                    L_mismatch = mismatch_loss_per_sample[mismatch_tensor].mean()

        # Total loss
        total_loss = total_loss + L_mismatch
        loss_time = time.perf_counter() - loss_start
        
        # DEBUG: Clean logit regression monitoring with full array printing
        if self.training and torch.rand(1).item() < 0.03:
            with torch.no_grad():
                sample_idx = torch.randint(0, B, (1,)).item()
                debug_target = target if target is not None else torch.zeros((B, T_eff), device=device)
                sample_targets = debug_target[sample_idx, :T_eff].detach().cpu().numpy()
                # If categorical, collapse to max-prob over bins for readability
                if predicted_rewards.dim() == 3:
                    sample_preds = predicted_rewards.max(dim=-1).values[sample_idx].detach().cpu().numpy()
                else:
                    sample_preds = predicted_rewards[sample_idx].detach().cpu().numpy()
                
                print(f"\n=== LOGIT REGRESSION DEBUG ===")
                print(f"Target: min={debug_target.min():.3f}, max={debug_target.max():.3f}, mean={debug_target.mean():.3f}")
                has_high_targets = (debug_target > 0.8).any().item()
                print(f"✓ Has targets >0.8: {has_high_targets} | T_eff: {T_eff}")
                print(f"Logits(proxy): min={raw_like_logits.min():.3f}, max={raw_like_logits.max():.3f}, mean={raw_like_logits.mean():.3f}")
                # For categorical, report max-prob stats
                preds_scalar = predicted_rewards.max(dim=-1).values if predicted_rewards.dim() == 3 else predicted_rewards
                print(f"Preds:  min={preds_scalar.min():.3f}, max={preds_scalar.max():.3f}, mean={preds_scalar.mean():.3f}")
                
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
                
                print(f"Sample {sample_idx}: T_eff={T_eff}, target ∈ [{sample_targets.min():.3f}, {sample_targets.max():.3f}], pred ∈ [{sample_preds.min():.3f}, {sample_preds.max():.3f}]")
                
                print(f"Loss: {total_loss:.6f}")
                print("=" * 60)
        
        total_forward_time = time.perf_counter() - forward_start

        # Log individual loss components
        loss_dict.update({
            "loss": float(total_loss.detach().item()),
            "loss_main": float(loss.detach().item() if isinstance(loss, torch.Tensor) else 0.0),
            "loss_mismatch": float(L_mismatch.detach().item()),
            
            "t_eff": float(T_eff),
            "lang_len_mean": float(mask.sum().float().mean().item()), # Use mask to get actual lengths
            # Target statistics for monitoring
            "target_min": float(target.min().item()) if target is not None else 0.0,
            "target_max": float(target.max().item()) if target is not None else 0.0,
            "target_mean": float(target.mean().item()) if target is not None else 0.0,
            "target_std": float(target.std().item()) if target is not None else 0.0,
            # Prediction statistics
            "pred_mean": float(predicted_rewards.mean().item()),
            "pred_std": float(predicted_rewards.std().item()),
            # Raw logits statistics (useful for monitoring head behavior)
            "raw_logits_mean": float(raw_like_logits.mean().item()),
            "raw_logits_std": float(raw_like_logits.std().item()),
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
        # Optimized: Process all commands in batch and take CLS token
        proc = self.processor(
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
        # Use CLS token (position 0) as single language token
        cls_only = outputs.last_hidden_state[:, :1, :]
        mask = torch.ones(cls_only.shape[:2], device=device, dtype=torch.bool)
        return cls_only, mask

    def _lens_to_mask(self, lens: Tensor, T: int) -> Tensor:
        rng = torch.arange(T, device=lens.device)[None, :]
        return rng < lens[:, None]

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
            
            # Proper window-relative stride sampling; allow clamping for out-of-bounds
            effective_stride = max(1, int(self.config.temporal_sampling_stride))
            min_anchor_in_window = (T - 1) * effective_stride
            max_anchor_in_window = available_T - 1
            if min_anchor_in_window <= max_anchor_in_window:
                anchor_in_window = torch.randint(min_anchor_in_window, max_anchor_in_window + 1, (1,)).item()
            else:
                # Not enough frames to satisfy stride; anchor at the last available frame,
                # earlier indices will clamp to 0 (repeat first frame)
                anchor_in_window = max_anchor_in_window

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
                # Lower-bound OOB: clamp to 0 (repeat first frame)
                if w_idx < 0:
                    w_idx = 0
                    had_oob = True
                # Upper-bound OOB (shouldn't happen when sampling past): clamp to last
                elif w_idx >= available_T:
                    w_idx = available_T - 1
                    print(f"  ⚠️  OOB: {w_idx} >= {available_T}, this should not happen!")
                    had_oob = True
                window_indices.append(w_idx)

                # Map window index back to episode-relative absolute frame index and clamp to 0..ep_length-1
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