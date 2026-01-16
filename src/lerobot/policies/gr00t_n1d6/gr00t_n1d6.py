#!/usr/bin/env python

# Copyright 2025 Nvidia and The HuggingFace Inc. team. All rights reserved.
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
Groot N1.6 core model implementation.

This module contains the core Gr00tN1d6 model, ported from:
- gr00t-orig/model/gr00t_n1d6/gr00t_n1d6.py

Key classes:
- Gr00tN1d6ActionHead: Action head with AlternateVLDiT support
- Gr00tN1d6: Main model class with collator integration

Key differences from N1.5:
- Uses AlternateVLDiT with image/text separation
- 32 DiT layers instead of 16
- State-relative action chunks
- New CategorySpecificMLP and MultiEmbodimentActionEncoder modules
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from torch import nn
from torch.distributions import Beta
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from lerobot.policies.gr00t_n1d6.configuration_gr00t_n1d6 import Gr00tN1d6Config
from lerobot.policies.gr00t_n1d6.eagle3_model.eagle_backbone import EagleBackbone
from lerobot.policies.gr00t_n1d6.modules import (
    AlternateVLDiT,
    CategorySpecificMLP,
    DiT,
    MultiEmbodimentActionEncoder,
)


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize diffusion model based on config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            print("Using DiT for diffusion model")

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # State encoder with category-specific MLP for multi-embodiment support
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )

        # Action encoder with multi-embodiment support
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )

        # Action decoder with category-specific MLP
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # Vision-Language Layer Norm
        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()

        # Positional embeddings
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        # Beta distribution for noise scheduling
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        # Set trainable parameters
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model, config.tune_vlln)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool):
        """Configure which parameters are trainable."""
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln

        # Start with all parameters trainable
        for p in self.parameters():
            p.requires_grad = True

        # Freeze projector components if not tuning
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)

        # Freeze diffusion model if not tuning
        if not tune_diffusion_model:
            self.model.requires_grad_(False)

        # Freeze VLLN if not tuning
        if not tune_vlln:
            self.vlln.requires_grad_(False)

        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")

        # Check if any parameters are still trainable
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")

        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        HuggingFace will call model.train() at each training step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        """Sample timesteps from beta distribution."""
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """Apply VLLN to backbone features."""
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head (training).

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
                - action_loss: per-element action loss
                - action_mask: action mask
                - backbone_features: processed backbone features
                - state_features: encoded state features
        """
        # Set frozen modules to eval mode
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get state and actions
        state = action_input.state
        actions = action_input.action

        # Get batch size from state (the authoritative source for training batch size)
        state_batch_size = state.shape[0]

        # Ensure actions batch size matches state batch size
        # This handles cases where action processing in modeling file creates mismatched batches
        action_batch_size = actions.shape[0]
        if action_batch_size != state_batch_size:
            if action_batch_size == 1 and state_batch_size > 1:
                # Actions have batch 1 but state has full batch - expand actions
                # This can happen when actions were reshaped incorrectly
                actions = actions.expand(state_batch_size, -1, -1)
                action_batch_size = state_batch_size
            elif state_batch_size == 1 and action_batch_size > 1:
                # Unusual case - state has batch 1, use action batch size
                state_batch_size = action_batch_size

        # Use state batch size as the canonical batch size
        batch_size = state_batch_size

        # Get embodiment ID
        embodiment_id = action_input.embodiment_id

        # Convert to tensor if it's a Python int/float
        if not isinstance(embodiment_id, torch.Tensor):
            embodiment_id = torch.full((batch_size,), embodiment_id, device=device, dtype=torch.long)
        # Ensure embodiment_id is at least 1D [B] for proper indexing
        if embodiment_id.ndim == 0:
            embodiment_id = embodiment_id.unsqueeze(0).expand(batch_size)
        elif embodiment_id.ndim == 1 and embodiment_id.shape[0] != batch_size:
            # Batch size mismatch - expand or truncate to match batch_size
            if embodiment_id.shape[0] == 1:
                embodiment_id = embodiment_id.expand(batch_size)
            else:
                # Use first embodiment ID for all samples (common in single-embodiment training)
                embodiment_id = embodiment_id[:1].expand(batch_size)
        elif embodiment_id.ndim > 1:
            # Flatten if needed (shouldn't happen, but be defensive)
            embodiment_id = embodiment_id.flatten()
            if embodiment_id.shape[0] != batch_size:
                if embodiment_id.shape[0] == 1:
                    embodiment_id = embodiment_id.expand(batch_size)
                else:
                    embodiment_id = embodiment_id[:1].expand(batch_size)

        # Embed state
        # Handle 2D state tensors [B, state_dim] by expanding to 3D [B, 1, state_dim]
        # The state encoder expects 3D input [B, T, state_dim]
        if state.ndim == 2:
            state = state.unsqueeze(1)  # [B, state_dim] -> [B, 1, state_dim]
        state_features = self.state_encoder(state, embodiment_id)

        # Apply state dropout during training
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device) < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features during training
        if self.training and self.state_additive_noise_scale > 0:
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory (flow matching)
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B, 1, 1) for broadcast

        # Interpolate between noise and actions
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert continuous t to discrete timesteps
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Add position embedding
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Concatenate state and action embeddings
        sa_embs = torch.cat((state_features, action_features), dim=1)

        # Ensure vl_embeds batch size matches sa_embs batch size
        # The backbone might output batch size 1 if it processes the batch as a single item
        sa_batch_size = sa_embs.shape[0]
        vl_batch_size = vl_embeds.shape[0]
        if vl_batch_size == 1 and sa_batch_size > 1:
            # Expand vl_embeds to match sa_embs batch size
            # Repeat the single batch item for all batches
            vl_embeds = vl_embeds.expand(sa_batch_size, -1, -1)
            # Also expand attention mask if it exists
            if (
                hasattr(backbone_output, "backbone_attention_mask")
                and backbone_output.backbone_attention_mask is not None
            ):
                vl_attn_mask = backbone_output.backbone_attention_mask
                if vl_attn_mask.shape[0] == 1:
                    vl_attn_mask = vl_attn_mask.expand(sa_batch_size, -1)
            else:
                vl_attn_mask = backbone_output.backbone_attention_mask
        else:
            vl_attn_mask = backbone_output.backbone_attention_mask

        # Forward through DiT
        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            # Expand image_mask and backbone_attention_mask if needed
            if image_mask is not None and image_mask.shape[0] == 1 and sa_batch_size > 1:
                image_mask = image_mask.expand(sa_batch_size, -1)
            if (
                backbone_attention_mask is not None
                and backbone_attention_mask.shape[0] == 1
                and sa_batch_size > 1
            ):
                backbone_attention_mask = backbone_attention_mask.expand(sa_batch_size, -1)
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            # Ensure vl_embeds batch size matches sa_embs batch size (same fix as above)
            sa_batch_size = sa_embs.shape[0]
            vl_batch_size = vl_embeds.shape[0]
            if vl_batch_size == 1 and sa_batch_size > 1:
                vl_embeds = vl_embeds.expand(sa_batch_size, -1, -1)
                if vl_attn_mask is not None and vl_attn_mask.shape[0] == 1:
                    vl_attn_mask = vl_attn_mask.expand(sa_batch_size, -1)
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        # Decode actions
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Compute masked MSE loss
        # Get action_mask from input, or create default (all valid) if missing
        action_mask = getattr(action_input, "action_mask", None)
        if action_mask is None:
            # Create default mask (all valid) matching pred_actions shape
            action_mask = torch.ones_like(pred_actions)
            logging.warning(
                f"action_mask missing in action_input, created default mask with shape {action_mask.shape}"
            )
        else:
            # Expand action_mask to match batch size if needed (fixes batch size mismatch)
            if action_mask.shape[0] != pred_actions.shape[0]:
                # action_mask has batch_size=1 but pred_actions has batch_size=B
                # Expand action_mask: [1, T, D] -> [B, T, D]
                action_mask = action_mask.expand(pred_actions.shape[0], -1, -1)
        # Ensure velocity matches pred_actions shape (in case actions were truncated)
        if velocity.shape[1] != pred_actions.shape[1]:
            velocity = velocity[:, :pred_actions.shape[1], :]
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Encode features for the action head (inference helper).

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        embodiment_id = action_input.embodiment_id
        # Convert to tensor if it's a Python int/float
        if not isinstance(embodiment_id, torch.Tensor):
            # Get batch size from state to expand scalar embodiment_id
            batch_size = action_input.state.shape[0] if hasattr(action_input.state, "shape") else 1
            embodiment_id = torch.full((batch_size,), embodiment_id, device=device, dtype=torch.long)
        # Ensure embodiment_id is at least 1D [B] for proper indexing
        if embodiment_id.ndim == 0:
            # Get batch size from state to expand scalar tensor
            batch_size = action_input.state.shape[0] if hasattr(action_input.state, "shape") else 1
            embodiment_id = embodiment_id.unsqueeze(0).expand(batch_size)
        elif embodiment_id.ndim > 1:
            # Flatten if needed (shouldn't happen, but be defensive)
            embodiment_id = embodiment_id.flatten()

        # Embed state
        # Handle 2D state tensors [B, state_dim] by expanding to 3D [B, 1, state_dim]
        # The state encoder expects 3D input [B, T, state_dim]
        state = action_input.state
        if state.ndim == 2:
            state = state.unsqueeze(1)  # [B, state_dim] -> [B, 1, state_dim]
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process with pre-encoded features.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model (for masks)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
                - backbone_features: processed backbone features
                - state_features: encoded state features
        """
        vl_embeds = backbone_features

        # Initialize actions as random noise
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Run flow matching denoising steps
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            # Add position embedding
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Concatenate state and action embeddings
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Forward through DiT
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )

            # Decode velocity prediction
            pred = self.action_decoder(model_output, embodiment_id)
            # Ensure pred_velocity matches actions shape exactly
            # pred has shape [B, state_horizon + action_horizon, action_dim]
            # We need to extract only the action part (last action_horizon timesteps)
            # Use actions.shape[1] to ensure exact match with actions tensor
            pred_velocity = pred[:, -actions.shape[1] :]

            # Euler integration update
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)

        # Convert embodiment_id to tensor if needed (must match what _encode_features does)
        embodiment_id = action_input.embodiment_id
        device = features.backbone_features.device
        if not isinstance(embodiment_id, torch.Tensor):
            batch_size = action_input.state.shape[0] if hasattr(action_input.state, "shape") else 1
            embodiment_id = torch.full((batch_size,), embodiment_id, device=device, dtype=torch.long)
        elif embodiment_id.ndim == 0:
            batch_size = action_input.state.shape[0] if hasattr(action_input.state, "shape") else 1
            embodiment_id = embodiment_id.unsqueeze(0).expand(batch_size)
        elif embodiment_id.ndim > 1:
            embodiment_id = embodiment_id.flatten()

        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=embodiment_id,
            backbone_output=backbone_output,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config):
    """Get backbone class based on config.

    Args:
        config: Either Gr00tN1d6Config or Gr00tN1d6ModelConfig
    """
    # N1.6 uses Eagle backbone from eagle3_model
    return EagleBackbone


# Internal transformers-compatible config for Gr00tN1d6 model
@dataclass
class Gr00tN1d6ModelConfig(PretrainedConfig):
    """Transformers-compatible config for Gr00tN1d6 model (internal use).

    This config is used internally by the Gr00tN1d6 PreTrainedModel.
    The LeRobot policy wrapper uses Gr00tN1d6Config instead.

    This config stores all parameters needed by the model, either loaded from HF
    or converted from LeRobot config.
    """

    model_type = "Gr00tN1d6"

    # Core model parameters loaded from HF
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})
    action_horizon: int = 50  # Default from HF config
    max_action_dim: int = 128  # Default from HF config

    # Backbone parameters (used by __init__)
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    backbone_model_type: str = "eagle"
    backbone_embedding_dim: int = 2048  # project_to_dim
    select_layer: int = 16
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True
    tune_top_llm_layers: int = 4
    backbone_trainable_params_fp32: bool = True

    # Fine-tuning flags (used by __init__)
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # Action head configuration parameters
    max_state_dim: int = 29  # Default from state_shape
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # Global parameters
    add_pos_embed: bool = True
    use_vlln: bool = True
    max_seq_len: int = 1024

    # Diffusion model type selection
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    # Diffusion model configuration with 32 layers (main difference from N15)
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,  # 32 layers instead of 16
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # State Augmentation parameters
    state_dropout_prob: float = 0.0  # State dropout probability
    state_additive_noise_scale: float = 0.0  # Scale for additive Gaussian noise on state features

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone.

    This is the main model class that combines the Eagle backbone
    with the flow matching action head for action prediction.
    """

    config_class = Gr00tN1d6ModelConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6ModelConfig,
        transformers_loading_kwargs: dict = None,
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - trust_remote_code: Whether to trust remote code when loading from HF Hub
                - local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - cache_dir: Directory to cache downloaded models
                - token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}

        # Initialize backbone
        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name="nvidia/Eagle-Block2A-2B-v2",  # Fixed for N1.6
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6ActionHead(config)

        # Collator will be initialized lazily when needed
        self._collator = None
        self._transformers_loading_kwargs = transformers_loading_kwargs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load pretrained Gr00tN1d6 model from HuggingFace Hub or local path.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path
            **kwargs: Additional arguments including:
                - tune_llm: Whether to fine-tune the LLM backbone
                - tune_visual: Whether to fine-tune the vision tower
                - tune_projector: Whether to fine-tune the projector
                - tune_diffusion_model: Whether to fine-tune the diffusion model
                - tune_vlln: Whether to fine-tune the VLLN (N1.6 specific)
                - tune_top_llm_layers: Number of top LLM layers to tune (N1.6 specific)
                - transformers_loading_kwargs: Dict with transformers loading parameters

        Returns:
            Gr00tN1d6: The loaded pretrained model
        """
        tune_llm = kwargs.pop("tune_llm", False)
        tune_visual = kwargs.pop("tune_visual", False)
        tune_projector = kwargs.pop("tune_projector", True) # Set to False #FIXME: debug cuda memory issue
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        tune_vlln = kwargs.pop("tune_vlln", True)
        tune_top_llm_layers = kwargs.pop("tune_top_llm_layers", 4)
        transformers_loading_kwargs = kwargs.pop("transformers_loading_kwargs", None)

        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}

        print(f"Loading pretrained Gr00tN1d6 from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune top {tune_top_llm_layers} LLM layers: {tune_top_llm_layers > 0}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head diffusion model: {tune_diffusion_model}")
        print(f"Tune action head vlln: {tune_vlln}")

        # Get the current model path being downloaded
        try:
            # This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or available in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        # Load the model using transformers' from_pretrained
        pretrained_model = super().from_pretrained(
            local_model_path, transformers_loading_kwargs=transformers_loading_kwargs, **kwargs
        )

        # Set trainable parameters for backbone
        pretrained_model.backbone.set_trainable_parameters(
            tune_llm=tune_llm, tune_visual=tune_visual, tune_top_llm_layers=tune_top_llm_layers
        )

        # Set trainable parameters for action head
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            tune_vlln=tune_vlln,
        )

        return pretrained_model

    @property
    def collator(self):
        """Lazy initialization of collator to avoid circular imports.

        Note: The Gr00tN1d6DataCollator must be implemented in processor_gr00t_n1d6.py
        for this to work. If not available, a NotImplementedError will be raised.
        """
        if self._collator is None:
            try:
                from lerobot.policies.gr00t_n1d6.processor_gr00t_n1d6 import Gr00tN1d6DataCollator

                self._collator = Gr00tN1d6DataCollator(
                    model_name="nvidia/Eagle-Block2A-2B-v2",
                    model_type=self.config.backbone_model_type,
                    transformers_loading_kwargs=self._transformers_loading_kwargs,
                )
            except ImportError as e:
                raise NotImplementedError(
                    "Gr00tN1d6DataCollator is not yet implemented. "
                    "Please implement it in processor_gr00t_n1d6.py first."
                ) from e
        return self._collator

    def prepare_input(self, inputs: dict) -> tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # Process VLM content through collator if present
        if "vlm_content" in inputs:
            # Handle multiple environments: ensure vlm_content_list is always a list
            vlm_content_list = inputs["vlm_content"]
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if not isinstance(x, torch.Tensor):
                return x
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model (training).

        Args:
            inputs: Dictionary containing:
                - VLM inputs (pixel_values, input_ids, attention_mask, etc.)
                - Action inputs (state, action, embodiment_id, action_mask, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """
        Generate actions using the complete model (inference).

        Args:
            inputs: Dictionary containing:
                - VLM inputs (pixel_values, input_ids, attention_mask, etc.)
                - Action inputs (state, embodiment_id, etc.)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6ModelConfig)
AutoModel.register(Gr00tN1d6ModelConfig, Gr00tN1d6)
