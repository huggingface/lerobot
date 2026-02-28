#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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

"""Multi-Task Diffusion Transformer (DiT) Policy

Transformer-based diffusion policy for multi-task robot learning with text and vision conditioning.
Supports both diffusion and flow matching objectives for action generation.

References:
- https://arxiv.org/abs/2507.05331
- https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/
- https://brysonkjones.substack.com/p/dissecting-and-open-sourcing-multitask-diffusion-transformer-policy
"""

import math
from collections import deque
from typing import TYPE_CHECKING

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor

from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import CLIPTextModel, CLIPVisionModel
else:
    CLIPTextModel = None
    CLIPVisionModel = None
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

# -- Policy --


class MultiTaskDiTPolicy(PreTrainedPolicy):
    config_class = MultiTaskDiTConfig
    name = "multi_task_dit"

    def __init__(self, config: MultiTaskDiTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None

        self.observation_encoder = ObservationEncoder(config)
        conditioning_dim = self.observation_encoder.conditioning_dim
        self.noise_predictor = DiffusionTransformer(config, conditioning_dim=conditioning_dim)

        action_dim = config.action_feature.shape[0]
        horizon = config.horizon

        if config.is_diffusion:
            self.objective = DiffusionObjective(
                config,
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
            )
        elif config.is_flow_matching:
            self.objective = FlowMatchingObjective(
                config,
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
            )
        else:
            raise ValueError(f"Unsupported objective: {config.objective}")

        self.reset()

    def get_optim_params(self) -> list:
        """Returns parameter groups with different learning rates for vision vs non-vision parameters"""
        non_vision_params = []
        vision_encoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "observation_encoder.vision_encoder" in name:
                vision_encoder_params.append(param)
            else:
                non_vision_params.append(param)

        return [
            {"params": non_vision_params},
            {
                "params": vision_encoder_params,
                "lr": self.config.optimizer_lr * self.config.vision_encoder_lr_multiplier,
            },
        ]

    def _generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        conditioning_vec = self.observation_encoder.encode(batch)
        actions = self.objective.conditional_sample(self.noise_predictor, batch_size, conditioning_vec)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]
        return actions

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations"""
        self.eval()

        for k in batch:
            if k in self._queues:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        actions = self._generate_actions(batch)
        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare batch by stacking image features if needed."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy to avoid modifying original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        return batch

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations"""
        if ACTION in batch:
            batch = dict(batch)  # shallow copy to avoid modifying original
            batch.pop(ACTION)

        batch = self._prepare_batch(batch)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Run the batch through the model and compute the loss for training"""
        batch = self._prepare_batch(batch)

        conditioning_vec = self.observation_encoder.encode(batch)
        loss = self.objective.compute_loss(self.noise_predictor, batch, conditioning_vec)

        return loss, None


# -- Observation Encoders --


class CLIPVisionEncoder(nn.Module):
    """CLIP vision encoder using the CLS token for global image representation."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = CLIPVisionModel.from_pretrained(self.model_name)
        self.num_non_spatial_tokens = 1
        self.embed_dim = self.model.config.hidden_size

    def forward(self, x: Tensor) -> Tensor:
        """Encode RGB image to CLS token."""
        outputs = self.model(pixel_values=x, output_hidden_states=False)
        cls_token = outputs.last_hidden_state[:, 0]
        b, embed_dim = cls_token.shape
        return cls_token.reshape(b, embed_dim, 1, 1)

    def get_output_shape(self) -> tuple:
        return (self.embed_dim, 1, 1)


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder with frozen weights and a learnable projection layer.

    Accepts pre-tokenized inputs (input_ids and attention_mask) from the processor pipeline. See the processor
    pipeline to see how the tokenization is handled.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", projection_dim: int = 512):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_embed_dim = self.text_encoder.config.hidden_size
        self.projection = nn.Linear(self.text_embed_dim, projection_dim)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode pre-tokenized text to feature vectors."""
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            clip_features = outputs.pooler_output

        return self.projection(clip_features)


class ObservationEncoder(nn.Module):
    """Handles all observation processing for the conditioning vector."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_preprocessing(config)

        if config.image_features:
            self.num_cameras = len(config.image_features)
            self.camera_names = list(config.image_features.keys())

            if config.use_separate_rgb_encoder_per_camera:
                self.vision_encoders = nn.ModuleList(
                    [CLIPVisionEncoder(model_name=config.vision_encoder_name) for _ in self.camera_names]
                )
                self.vision_encoder = None
            else:
                self.vision_encoder = CLIPVisionEncoder(model_name=config.vision_encoder_name)
                self.vision_encoders = None
        else:
            self.vision_encoder = None
            self.vision_encoders = None
            self.camera_names = []
            self.num_cameras = 0

        if hasattr(config, "robot_state_feature") and config.robot_state_feature:
            self.robot_state_dim = config.robot_state_feature.shape[0]
        else:
            self.robot_state_dim = 0

        self.text_dim = config.hidden_dim
        self.text_encoder = CLIPTextEncoder(model_name=config.text_encoder_name, projection_dim=self.text_dim)

        self._setup_vector_output()

    def _apply_preprocessing(self, images: Tensor) -> Tensor:
        if self.do_resize:
            images = self.resize(images)
        if self.do_crop:
            images = self.maybe_random_crop(images) if self.training else self.center_crop(images)
        return images

    def _setup_preprocessing(self, config):
        if config.image_resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(
                size=config.image_resize_shape,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            self.do_resize = False

        if config.image_crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.image_crop_shape)
            if config.image_crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.image_crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

    def _setup_vector_output(self):
        total_dim = 0

        if self.vision_encoder is not None or self.vision_encoders is not None:
            encoder_to_check = self.vision_encoder or next(iter(self.vision_encoders))
            feature_map_shape = encoder_to_check.get_output_shape()
            c, h, w = feature_map_shape
            spatial_feature_dim = c * h * w
            total_dim += spatial_feature_dim * self.num_cameras

        total_dim += self.robot_state_dim
        total_dim += self.text_dim

        self.conditioning_dim = total_dim * self.config.n_obs_steps

    def encode(self, batch: dict) -> Tensor:
        """Encode observations to vector format."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        conditioning_feats = []

        conditioning_feats.append(batch[OBS_STATE])

        if self.vision_encoder is not None or self.vision_encoders is not None:
            images = batch[OBS_IMAGES]

            if len(images.shape) == 5:
                images = images.unsqueeze(1)

            if self.config.use_separate_rgb_encoder_per_camera:
                camera_features = []
                for cam_idx in range(self.num_cameras):
                    cam_images = images[:, :, cam_idx]
                    cam_images_flat = einops.rearrange(cam_images, "b s c h w -> (b s) c h w")
                    cam_images_flat = self._apply_preprocessing(cam_images_flat)
                    cam_features = self.vision_encoders[cam_idx](cam_images_flat)
                    cam_visual_features = cam_features.flatten(start_dim=1)
                    cam_features_reshaped = einops.rearrange(
                        cam_visual_features, "(b s) f -> b s f", b=batch_size, s=n_obs_steps
                    )
                    camera_features.append(cam_features_reshaped)
                img_features = torch.cat(camera_features, dim=-1)
                conditioning_feats.append(img_features)
            else:
                images_flat = einops.rearrange(images, "b s n c h w -> (b s n) c h w")
                images_flat = self._apply_preprocessing(images_flat)
                visual_features = self.vision_encoder(images_flat).flatten(start_dim=1)
                img_features = einops.rearrange(
                    visual_features, "(b s n) f -> b s (n f)", b=batch_size, s=n_obs_steps, n=self.num_cameras
                )
                conditioning_feats.append(img_features)

        if self.text_encoder is not None and OBS_LANGUAGE_TOKENS in batch:
            input_ids = batch[OBS_LANGUAGE_TOKENS]  # [batch_size, seq_length]
            attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]  # [batch_size, seq_length]

            text_features = self.text_encoder(input_ids, attention_mask)

            text_features = text_features.unsqueeze(1).expand(-1, n_obs_steps, -1)
            conditioning_feats.append(text_features)

        combined_features = torch.cat(conditioning_feats, dim=-1)
        return combined_features.flatten(start_dim=1)


# -- Transformer Components --


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Modulate input with shift and scale for AdaLN-Zero."""
    return x * (1 + scale) + shift


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformers."""

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("_sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.")

        cos = self._cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self._sin_cached[:, :, :seq_len, :].to(q.dtype)

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        return q_rotated, k_rotated


class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rope = RotaryPositionalEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape  # noqa: N806

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) and self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.hidden_size)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """DiT-style transformer block with AdaLN-Zero."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_features: int = 128,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.use_rope = use_rope

        if use_rope:
            self.attn = RoPEAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
            )
        else:
            self.multihead_attn = nn.MultiheadAttention(
                hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
            )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(num_features, 6 * hidden_size, bias=True))

    def forward(self, x: Tensor, features: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            features
        ).chunk(6, dim=1)

        attn_input = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))

        if self.use_rope:
            attn_out = self.attn(attn_input)
        else:
            attn_out, _ = self.multihead_attn(attn_input, attn_input, attn_input)

        x = x + gate_msa.unsqueeze(1) * attn_out

        mlp_input = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        mlp_out = self.mlp(mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class DiffusionTransformer(nn.Module):
    """Transformer-based diffusion noise prediction model."""

    def __init__(self, config, conditioning_dim: int):
        super().__init__()
        self.config = config
        self.conditioning_dim = conditioning_dim

        self.action_dim = config.action_feature.shape[0]
        self.horizon = config.horizon
        self.hidden_size = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        self.timestep_embed_dim = config.timestep_embed_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.timestep_embed_dim),
            nn.Linear(self.timestep_embed_dim, 2 * self.timestep_embed_dim),
            nn.GELU(),
            nn.Linear(2 * self.timestep_embed_dim, self.timestep_embed_dim),
            nn.GELU(),
        )

        self.cond_dim = self.timestep_embed_dim + conditioning_dim
        self.input_proj = nn.Linear(self.action_dim, self.hidden_size)

        if config.use_positional_encoding:
            self.pos_embedding = nn.Parameter(
                torch.empty(1, self.horizon, self.hidden_size).normal_(std=0.02)
            )
        else:
            self.pos_embedding = None

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    num_features=self.cond_dim,
                    dropout=self.dropout,
                    use_rope=self.use_rope,
                    max_seq_len=self.horizon,
                    rope_base=config.rope_base,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_proj = nn.Linear(self.hidden_size, self.action_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, timestep: Tensor, conditioning_vec: Tensor) -> Tensor:
        _, seq_len, _ = x.shape

        timestep_features = self.time_mlp(timestep)
        cond_features = torch.cat([timestep_features, conditioning_vec], dim=-1)

        hidden_seq = self.input_proj(x)

        if self.pos_embedding is not None:
            hidden_seq = hidden_seq + self.pos_embedding[:, :seq_len, :]

        for block in self.transformer_blocks:
            hidden_seq = block(hidden_seq, cond_features)

        return self.output_proj(hidden_seq)


# -- Objectives --


class DiffusionObjective(nn.Module):
    """Standard diffusion (DDPM/DDIM) objective implementation."""

    def __init__(self, config, action_dim: int, horizon: int, do_mask_loss_for_padding: bool = False):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.horizon = horizon
        self.do_mask_loss_for_padding = do_mask_loss_for_padding

        scheduler_kwargs = {
            "num_train_timesteps": config.num_train_timesteps,
            "beta_start": config.beta_start,
            "beta_end": config.beta_end,
            "beta_schedule": config.beta_schedule,
            "clip_sample": config.clip_sample,
            "clip_sample_range": config.clip_sample_range,
            "prediction_type": config.prediction_type,
        }

        if config.noise_scheduler_type == "DDPM":
            self.noise_scheduler: DDPMScheduler | DDIMScheduler = DDPMScheduler(**scheduler_kwargs)
        elif config.noise_scheduler_type == "DDIM":
            self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported noise scheduler type {config.noise_scheduler_type}")

        self.num_inference_steps = (
            config.num_inference_steps
            if config.num_inference_steps is not None
            else self.noise_scheduler.config.num_train_timesteps
        )

    def compute_loss(self, model: nn.Module, batch: dict[str, Tensor], conditioning_vec: Tensor) -> Tensor:
        clean_actions = batch[ACTION]
        noise = torch.randn_like(clean_actions)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(clean_actions.shape[0],),
            device=clean_actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(clean_actions, noise, timesteps)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = clean_actions
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        predicted = model(noisy_actions, timesteps, conditioning_vec=conditioning_vec)
        loss = F.mse_loss(predicted, target, reduction="none")

        if self.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid_actions = ~batch["action_is_pad"]
            loss = loss * valid_actions.unsqueeze(-1)

        return loss.mean()

    def conditional_sample(self, model: nn.Module, batch_size: int, conditioning_vec: Tensor) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        sample = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = model(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                conditioning_vec=conditioning_vec,
            )
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample

        return sample


class FlowMatchingObjective(nn.Module):
    """Flow matching objective: trains a model to predict velocity fields."""

    def __init__(self, config, action_dim: int, horizon: int, do_mask_loss_for_padding: bool = False):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.horizon = horizon
        self.do_mask_loss_for_padding = do_mask_loss_for_padding

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        if self.config.timestep_sampling_strategy == "uniform":
            return torch.rand(batch_size, device=device)
        elif self.config.timestep_sampling_strategy == "beta":
            beta_dist = torch.distributions.Beta(
                self.config.timestep_sampling_alpha, self.config.timestep_sampling_beta
            )
            u = beta_dist.sample((batch_size,)).to(device)
            return self.config.timestep_sampling_s * (1.0 - u)
        else:
            raise ValueError(f"Unknown timestep strategy: {self.config.timestep_sampling_strategy}")

    def compute_loss(self, model: nn.Module, batch: dict[str, Tensor], conditioning_vec: Tensor) -> Tensor:
        data = batch[ACTION]
        batch_size = data.shape[0]
        device = data.device

        noise = torch.randn_like(data)
        t = self._sample_timesteps(batch_size, device)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * data + (1 - (1 - self.config.sigma_min) * t_expanded) * noise

        target_velocity = data - (1 - self.config.sigma_min) * noise
        predicted_velocity = model(x_t, t, conditioning_vec=conditioning_vec)
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")

        if self.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid_mask = ~batch["action_is_pad"]
            loss = loss * valid_mask.unsqueeze(-1)

        return loss.mean()

    def conditional_sample(self, model: nn.Module, batch_size: int, conditioning_vec: Tensor) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        x = torch.randn((batch_size, self.horizon, self.action_dim), dtype=dtype, device=device)

        num_steps = self.config.num_integration_steps
        time_grid = torch.linspace(0, 1, num_steps + 1, device=device)

        if self.config.integration_method == "euler":
            x = self._euler_integrate(model, x, time_grid, conditioning_vec)
        elif self.config.integration_method == "rk4":
            x = self._rk4_integrate(model, x, time_grid, conditioning_vec)
        else:
            raise ValueError(f"Unknown integration method: {self.config.integration_method}")

        return x

    def _euler_integrate(
        self, model: nn.Module, x_init: Tensor, time_grid: Tensor, conditioning_vec: Tensor
    ) -> Tensor:
        x = x_init
        for i in range(len(time_grid) - 1):
            t_scalar = time_grid[i].item()
            dt = (time_grid[i + 1] - time_grid[i]).item()
            t_batch = torch.full((x.shape[0],), t_scalar, dtype=x.dtype, device=x.device)
            with torch.no_grad():
                velocity = model(x, t_batch, conditioning_vec=conditioning_vec)
            x = x + dt * velocity
        return x

    def _rk4_integrate(
        self, model: nn.Module, x_init: Tensor, time_grid: Tensor, conditioning_vec: Tensor
    ) -> Tensor:
        x = x_init

        def dynamics(x_val: Tensor, t_scalar: float) -> Tensor:
            t_batch = torch.full((x_val.shape[0],), t_scalar, dtype=x_val.dtype, device=x_val.device)
            with torch.no_grad():
                return model(x_val, t_batch, conditioning_vec=conditioning_vec)

        for i in range(len(time_grid) - 1):
            t = time_grid[i].item()
            dt = (time_grid[i + 1] - time_grid[i]).item()

            k1 = dynamics(x, t)
            k2 = dynamics(x + dt * k1 / 2, t + dt / 2)
            k3 = dynamics(x + dt * k2 / 2, t + dt / 2)
            k4 = dynamics(x + dt * k3, t + dt)

            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return x
