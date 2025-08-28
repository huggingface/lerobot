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
  |  Vision Encoder (frozen)     |  e.g. DINOv2 (base)
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
                                    |      |  Text Encoder (frozen)       |  e.g. sentence-transformers
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
  - Uses DINOv2 (base, ~ViT-B) for vision and sentence-transformers (all-MiniLM-L12-v2) for text encoding.
  - Backbones (vision/text) are frozen by default; only projections, temporal module, and head are trainable.
  - Stride/frame dropout applied during training can subsample timesteps.
"""

from __future__ import annotations

import math
from itertools import chain

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

# ReWiND dependencies
try:
    from x_transformers import Decoder
    from hl_gauss_pytorch import HLGaussLayer
    import einx
    from einops import rearrange, repeat, pack, unpack
except ImportError as e:
    raise ImportError(
        "ReWiND dependencies not installed. Please install: "
        "pip install x-transformers hl-gauss-pytorch einx einops"
    ) from e

from lerobot.constants import OBS_IMAGE, OBS_IMAGES, OBS_LANGUAGE, REWARD
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rlearn.configuration_rlearn import RLearNConfig


class RLearNPolicy(PreTrainedPolicy):
    """Video-language conditioned reward model following ReWiND architecture exactly: https://github.com/lucidrains/rewind-reward-pytorch/blob/main/rewind_reward_pytorch/rewind_reward.py#L11.

    - Visual encoder: frozen DINOv2 (base), returns per-frame embeddings.
    - Text encoder: frozen sentence-transformers (all-MiniLM-L12-v2), returns a language embedding.
    - Temporal module: x_transformers Decoder with packed tokens [lang | register | video].
    - Output: per-timestep rewards via HLGauss layer or categorical bins.
    """

    config_class = RLearNConfig
    name = "rlearn"

    def __init__(self, config: RLearNConfig, episode_data_index: dict = None):
        super().__init__(config)
        self.config = config
        self.episode_data_index = episode_data_index  # Store episode boundaries for progress calculation
        self.categorical_rewards = config.categorical_rewards

        # Encoders - ReWiND paper setup: DINOv2 for vision, sentence-transformers for text
        from transformers import AutoImageProcessor, AutoModel
        from sentence_transformers import SentenceTransformer
        
        # Load DINOv2 (base) vision encoder with its processor
        self.vision_processor = AutoImageProcessor.from_pretrained(config.vision_model_name)
        self.vision_encoder = AutoModel.from_pretrained(config.vision_model_name)
        
        # Load sentence-transformers text encoder
        self.text_encoder = SentenceTransformer(config.text_model_name)
        
        # DINOv2-base has 768 hidden size, all-MiniLM-L12-v2 has 384
        self.vision_hidden = 768  # DINOv2-base
        self.text_hidden = 384  # all-MiniLM-L12-v2

        if config.freeze_backbones:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
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

        # Only first frame gets a positional embed (no cheating on progress)
        self.first_pos_emb = nn.Parameter(torch.randn(config.dim_model) * 1e-2)
        
        # Register / memory / attention sink tokens
        self.num_register_tokens = config.num_register_tokens
        self.register_tokens = nn.Parameter(torch.randn(config.num_register_tokens, config.dim_model) * 1e-2)

        # MLP predictor (matching ReWiND's Feedforwards)
        from x_mlps_pytorch import Feedforwards
        self.mlp_predictor = Feedforwards(
            dim=config.dim_model,
            dim_out=config.reward_bins if config.categorical_rewards else None,
            depth=config.mlp_predictor_depth
        )
        
        # HLGauss layer or plain regression
        self.hl_gauss_layer = HLGaussLayer(
            dim=config.dim_model,
            use_regression=not config.use_hl_gauss_loss,
            hl_gauss_loss=dict(
                min_value=config.reward_min_value,
                max_value=config.reward_max_value,
                num_bins=config.reward_hl_gauss_loss_num_bins,
            ) if config.use_hl_gauss_loss else None
        )
        
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

        # Apply stride (no dropout during eval)
        idx = torch.arange(0, T, self.stride, device=frames.device)
        frames = frames[:, idx]
        T_eff = frames.shape[1]

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
        video_embeds = self._encode_video_frames(frames)  # (B, T, D_vision)
        
        # Language embeddings
        lang_embeds = self.text_encoder.encode(
            commands,
            output_value='token_embeddings',
            convert_to_tensor=True,
            device=device
        )
        lang_embeds = pad_sequence(lang_embeds, batch_first=True).to(device)
        lens = torch.tensor([le.shape[0] for le in lang_embeds], device=device)
        mask = self._mask_from_lens(lens)
        
        # Register tokens
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b=B)
        
        # Project embeddings
        lang_tokens = self.to_lang_tokens(lang_embeds)
        video_tokens = self.to_video_tokens(video_embeds)
        
        # Add first frame positional embedding
        first_video_token, rest_video_tokens = video_tokens[:, :1], video_tokens[:, 1:]
        first_video_token = first_video_token + repeat(self.first_pos_emb, 'd -> b 1 d', b=B)
        video_tokens = torch.cat((first_video_token, rest_video_tokens), dim=1)
        
        # Pack all tokens for attention
        tokens, lang_video_packed_shape = pack((lang_tokens, register_tokens, video_tokens), 'b * d')
        
        # Extend mask for register and video tokens
        mask = F.pad(mask, (0, register_tokens.shape[1] + video_tokens.shape[1]), value=True)
        
        # Forward through decoder
        attended = self.decoder(tokens, mask=mask)
        
        # Unpack and get video token features
        _, _, attended_video_tokens = unpack(attended, lang_video_packed_shape, 'b * d')
        
        # MLP predictor
        video_frame_embeds = self.mlp_predictor(attended_video_tokens)
        
        # Get rewards via HLGauss layer
        if self.categorical_rewards:
            return video_frame_embeds  # Return logits directly
        else:
            return self.hl_gauss_layer(video_frame_embeds).squeeze(-1)  # (B, T)

    def normalize_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Initial version: no-op; rely on upstream processors if any
        return batch

    def normalize_targets(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Initial version: no-op
        return batch

    def _encode_video_frames(self, frames: Tensor) -> Tensor:
        """Encode video frames through DINOv2 to get per-frame embeddings.
        
        Args:
            frames: (B, T, C, H, W)
        
        Returns:
            (B, T, D_vision)
        """
        B, T, C, H, W = frames.shape
        flat = rearrange(frames, 'b t c h w -> (b t) c h w')
        
        # Process with DINOv2
        images_list = []
        for i in range(B * T):
            img = flat[i].permute(1, 2, 0)  # CHW -> HWC
            if img.dtype == torch.uint8:
                img = img.cpu().numpy()
            else:
                img = (img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            images_list.append(img)
        
        processed = self.vision_processor(images=images_list, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(next(self.vision_encoder.parameters()).device)
        vision_outputs = self.vision_encoder(pixel_values)
        
        # Extract CLS tokens
        cls_tokens = vision_outputs.last_hidden_state[:, 0]  # (BT, D_vision)
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
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Extract frames and form (B, T, C, H, W)
        frames = extract_visual_sequence(batch, target_seq_len=self.config.max_seq_len)
        B, T, C, H, W = frames.shape
        device = next(self.parameters()).device
        frames = frames.to(device)

        # Apply video rewinding augmentation during training
        augmented_target = None
        if self.training and self.config.use_video_rewind:
            frames, augmented_target = apply_video_rewind(
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

        # Process video frames through DINOv2
        video_embeds = self._encode_video_frames(frames)  # (B, T_eff, D_vision)
        
        # Language embeddings
        lang_embeds = self.text_encoder.encode(
            commands,
            output_value='token_embeddings',
            convert_to_tensor=True,
            device=device
        )
        lang_embeds = pad_sequence(lang_embeds, batch_first=True).to(device)
        lens = torch.tensor([le.shape[0] for le in lang_embeds], device=device)
        mask = self._mask_from_lens(lens)
        
        # Register tokens
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b=B)
        
        # Project embeddings
        lang_tokens = self.to_lang_tokens(lang_embeds)
        video_tokens = self.to_video_tokens(video_embeds)
        
        # Add first frame positional embedding
        first_video_token, rest_video_tokens = video_tokens[:, :1], video_tokens[:, 1:]
        first_video_token = first_video_token + repeat(self.first_pos_emb, 'd -> b 1 d', b=B)
        video_tokens = torch.cat((first_video_token, rest_video_tokens), dim=1)
        
        # Pack all tokens for attention [lang | register | video]
        tokens, lang_video_packed_shape = pack((lang_tokens, register_tokens, video_tokens), 'b * d')
        
        # Extend mask for register and video tokens
        mask = F.pad(mask, (0, register_tokens.shape[1] + video_tokens.shape[1]), value=True)
        
        # Forward through x_transformers Decoder
        attended = self.decoder(tokens, mask=mask)
        
        # Unpack and get video token features
        _, _, attended_video_tokens = unpack(attended, lang_video_packed_shape, 'b * d')
        
        # MLP predictor
        video_frame_embeds = self.mlp_predictor(attended_video_tokens)

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
            else:
                raise ValueError(
                    "No episode information found in batch. Please ensure 'episode_index' and 'frame_index' keys are present."
                )

        # During inference, we might not want to compute loss
        if not self.training and target is None:
            # Return predictions without loss
            if self.categorical_rewards:
                return video_frame_embeds.mean() * 0.0, {"has_labels": 0.0}
            else:
                rewards = self.hl_gauss_layer(video_frame_embeds)
                return rewards.mean() * 0.0, {"rewards_mean": rewards.mean().item()}

        # Calculate loss using HLGauss or categorical
        if self.categorical_rewards:
            # Categorical cross-entropy loss
            assert target.dtype in (torch.long, torch.int), "Categorical rewards require integer targets"
            loss = F.cross_entropy(
                rearrange(video_frame_embeds, 'b t l -> b l t'),
                target.long(),
                ignore_index=-1
            )
        else:
            # HLGauss loss or MSE regression
            assert target.dtype == torch.float, "Continuous rewards require float targets"
            # Create video mask for variable length support
            video_mask = torch.ones(B, T_eff, dtype=torch.bool, device=device)
            loss = self.hl_gauss_layer(video_frame_embeds, target[:, :T_eff], mask=video_mask)

        # Optional: Mismatched video-language pairs loss
        L_mismatch = torch.zeros((), device=device)
        if self.training and self.config.use_mismatch_loss and B > 1:
            if torch.rand(1, device=device).item() < getattr(self.config, "mismatch_prob", 0.2):
                # Shuffle language within batch
                shuffled_indices = torch.randperm(B, device=device)
                shuffled_commands = [commands[i] for i in shuffled_indices]
                
                # Re-encode with mismatched language
                lang_embeds_mm = self.text_encoder.encode(
                    shuffled_commands,
                    output_value='token_embeddings',
                    convert_to_tensor=True,
                    device=device
                )
                lang_embeds_mm = pad_sequence(lang_embeds_mm, batch_first=True).to(device)
                lang_tokens_mm = self.to_lang_tokens(lang_embeds_mm)
                
                # Pack and forward
                tokens_mm, _ = pack((lang_tokens_mm, register_tokens, video_tokens), 'b * d')
                attended_mm = self.decoder(tokens_mm, mask=mask)
                _, _, attended_video_mm = unpack(attended_mm, lang_video_packed_shape, 'b * d')
                mismatch_embeds = self.mlp_predictor(attended_video_mm)
                
                # Mismatched pairs should predict zero progress
                zeros_target = torch.zeros_like(target[:, :T_eff])
                if self.categorical_rewards:
                    L_mismatch = F.cross_entropy(
                        rearrange(mismatch_embeds, 'b t l -> b l t'),
                        zeros_target.long(),
                        ignore_index=-1
                    )
                else:
                    L_mismatch = self.hl_gauss_layer(mismatch_embeds, zeros_target, mask=video_mask)

        # Total loss
        total_loss = loss + L_mismatch

        # Log individual loss components
        loss_dict.update({
            "loss": total_loss.item(),
            "loss_main": loss.item(),
            "loss_mismatch": L_mismatch.item(),
        })

        return total_loss, loss_dict


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





def apply_video_rewind(frames: Tensor, rewind_prob: float = 0.5, last3_prob: float | None = None) -> tuple[Tensor, Tensor]:
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

        # Rewind length k: between 1 and i-1 frames, with option to force last-3 frames occasionally
        if last3_prob is not None and torch.rand(1).item() < last3_prob and i >= 3:
            k = min(3, i - 1)
        else:
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