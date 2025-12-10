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

"""Observation encoding for Multi-Task DiT policy.

Handles vision encoding, text encoding, robot state, and environment state.
"""

import einops
import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class CLIPVisionEncoder(nn.Module):
    """CLIP vision encoder using the CLS token for global image representation."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

        # Load CLIP vision model from transformers
        self.model = CLIPVisionModel.from_pretrained(self.model_name)

        # CLIP models have 1 CLS token
        self.num_non_spatial_tokens = 1

        # Get embed_dim from model config
        self.embed_dim = self.model.config.hidden_size

    def forward(self, x: Tensor) -> Tensor:
        """Encode RGB image to CLS token.

        Preprocessing (resize, crop) is handled by ObservationEncoder
        """
        # Extract features using CLIPVisionModel
        # Input: (B, C, H, W) - already preprocessed
        outputs = self.model(pixel_values=x, output_hidden_states=False)

        # Extract CLS token from last_hidden_state (first token)
        # last_hidden_state shape: (B, sequence_length, hidden_size)
        cls_token = outputs.last_hidden_state[:, 0]  # (B, embed_dim)
        b, embed_dim = cls_token.shape

        # Reshape to spatial format (B, C, H, W) with H=W=1 for compatibility
        cls_features = cls_token.reshape(b, embed_dim, 1, 1)
        return cls_features

    def get_output_shape(self) -> tuple:
        return (self.embed_dim, 1, 1)


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder with frozen weights and a learnable projection layer."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", projection_dim: int = 512):
        super().__init__()

        self.model_name = model_name
        self.projection_dim = projection_dim

        # Load CLIP text encoder and tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        # Freeze all CLIP text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_embed_dim = self.text_encoder.config.hidden_size

        # Learnable projection layer (always present, only trainable component)
        self.projection = nn.Linear(self.text_embed_dim, projection_dim)

    def forward(self, text: str | list[str]) -> Tensor:
        """Encode text to feature vectors.

        Args:
            text: Single string or list of strings

        Returns:
            Text features of shape (B, projection_dim)
        """
        # handle single string input
        if isinstance(text, str):
            text = [text]

        text_inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        text_inputs = {k: v.to(next(self.parameters()).device) for k, v in text_inputs.items()}

        # encode text through CLIP (frozen)
        with torch.no_grad():
            outputs = self.text_encoder(**text_inputs)
            # Extract pooled output (EOS token embedding)
            clip_features = outputs.pooler_output  # (B, text_embed_dim)

        # project to desired dimension (trainable)
        projected_features = self.projection(clip_features)  # (B, projection_dim)

        return projected_features


class ObservationEncoder(nn.Module):
    """Handles all observation processing for the conditioning vector."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self._setup_preprocessing(config)

        if config.image_features:
            self.num_cameras = len(config.image_features)
            self.camera_names = list(config.image_features.keys())

            if config.use_separate_encoder_per_camera:
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

        if hasattr(config, "env_state_feature") and config.env_state_feature:
            self.env_state_dim = config.env_state_feature.shape[0]
        else:
            self.env_state_dim = 0

        self.text_dim = config.hidden_dim
        self.text_encoder = CLIPTextEncoder(model_name=config.text_encoder_name, projection_dim=self.text_dim)

        self._setup_vector_output()

    def _apply_preprocessing(self, images: Tensor) -> Tensor:
        """Apply preprocessing transforms to images."""
        if self.do_resize:
            images = self.resize(images)
        if self.do_crop:
            images = self.maybe_random_crop(images) if self.training else self.center_crop(images)

        return images

    def _setup_preprocessing(self, config):
        """Setup image preprocessing transforms."""
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

        # Vision features - get CLS token feature dimension
        if self.vision_encoder is not None or self.vision_encoders is not None:
            encoder_to_check = self.vision_encoder or next(iter(self.vision_encoders))

            # Get output shape from encoder (deterministic for CLS tokens)
            feature_map_shape = encoder_to_check.get_output_shape()
            c, h, w = feature_map_shape
            spatial_feature_dim = c * h * w  # For CLS token: embed_dim * 1 * 1 = embed_dim

            total_dim += spatial_feature_dim * self.num_cameras

        # State features
        total_dim += self.robot_state_dim
        total_dim += self.env_state_dim

        # Text features
        total_dim += self.text_dim

        # Account for temporal stacking
        self.conditioning_dim = total_dim * self.config.n_obs_steps

    def encode(self, batch: dict) -> Tensor:
        """Encode observations to vector format."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        conditioning_feats = []

        conditioning_feats.append(batch[OBS_STATE])

        if self.vision_encoder is not None or self.vision_encoders is not None:
            images = batch[OBS_IMAGES]  # (B, n_obs_steps, num_cameras, C, H, W)

            # Handle case when n_obs=1 and time dimension might be squeezed
            if len(images.shape) == 5:
                # Shape is (B, N, C, H, W) - add time dimension
                images = images.unsqueeze(1)  # (B, 1, N, C, H, W)

            if self.config.use_separate_encoder_per_camera:
                # Process each camera with its own encoder
                camera_features = []

                for cam_idx in range(self.num_cameras):
                    # Extract images for this camera: (B, n_obs_steps, C, H, W)
                    cam_images = images[:, :, cam_idx]

                    # Rearrange to: (B*n_obs_steps, C, H, W)
                    cam_images_flat = einops.rearrange(cam_images, "b s c h w -> (b s) c h w")

                    # Apply preprocessing
                    cam_images_flat = self._apply_preprocessing(cam_images_flat)

                    # Process with camera-specific encoder (direct index access)
                    cam_features = self.vision_encoders[cam_idx](cam_images_flat)

                    # Apply spatial vectorization (flatten CLS token features)
                    cam_visual_features = cam_features.flatten(start_dim=1)

                    # Reshape back: (B*n_obs_steps, feature_dim) → (B, n_obs_steps, feature_dim)
                    cam_features_reshaped = einops.rearrange(
                        cam_visual_features, "(b s) f -> b s f", b=batch_size, s=n_obs_steps
                    )
                    camera_features.append(cam_features_reshaped)

                # Concatenate features from all cameras: (B, n_obs_steps, total_feature_dim)
                img_features = torch.cat(camera_features, dim=-1)
                conditioning_feats.append(img_features)

            else:
                # Shared encoder for all cameras
                # Rearrange to: (B*n_obs_steps*num_cameras, C, H, W)
                images_flat = einops.rearrange(images, "b s n c h w -> (b s n) c h w")

                images_flat = self._apply_preprocessing(images_flat)

                visual_features = self.vision_encoder(images_flat).flatten(start_dim=1)

                # Reshape back and concatenate camera features
                # (B*n_obs_steps*num_cameras, feature_dim) → (B, n_obs_steps, num_cameras*feature_dim)
                img_features = einops.rearrange(
                    visual_features, "(b s n) f -> b s (n f)", b=batch_size, s=n_obs_steps, n=self.num_cameras
                )

                conditioning_feats.append(img_features)

        if self.env_state_dim > 0 and OBS_ENV_STATE in batch:
            conditioning_feats.append(batch[OBS_ENV_STATE])

        if self.text_encoder is not None and "task" in batch:
            text_features = self.text_encoder(batch["task"])  # (B, text_dim)
            # Expand across temporal dimension to match other features
            text_features = text_features.unsqueeze(1).expand(-1, n_obs_steps, -1)  # (B, T, text_dim)
            conditioning_feats.append(text_features)

        combined_features = torch.cat(conditioning_feats, dim=-1)  # (B, n_obs_steps, total_feature_dim)

        return combined_features.flatten(start_dim=1)  # (B, n_obs_steps * total_feature_dim)
