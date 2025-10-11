#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

import re
from typing import Dict, List, Optional, Tuple
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import AutoTokenizer, T5EncoderModel

from lerobot.policies.octo.base import TokenGroup


# Image processing components
@torch.no_grad()
def normalize_images(img, img_norm_type="default"):
    """Normalize images according to the specified normalization type."""
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.float() / 127.5 - 1.0
    elif img_norm_type == "imagenet":
        # put pixels in [0,1]
        img = img.float() / 255.0
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(1, 1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(1, 1, 1, 3)

        # tile mean and std (to account for stacked early_fusion images)
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = mean.repeat(*num_tile)
        std_tile = std.repeat(*num_tile)

        # tile the mean/std, normalize image, and return
        return (img - mean_tile) / std_tile
    raise ValueError(f"Unknown img_norm_type: {img_norm_type}")


class FilmConditioning(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning layer."""

    def __init__(self):
        super().__init__()

    def forward(self, conv_filters: torch.Tensor, conditioning: torch.Tensor):
        """
        Applies FiLM conditioning to a convolutional feature map.

        Args:
            conv_filters: A tensor of shape [batch_size, height, width, channels].
            conditioning: A tensor of shape [batch_size, conditioning_size].

        Returns:
            A tensor of shape [batch_size, height, width, channels].
        """
        channels = conv_filters.shape[-1]
        cond_size = conditioning.shape[-1]

        self.proj_add = nn.Linear(cond_size, channels)
        self.proj_mult = nn.Linear(cond_size, channels)

        projected_cond_add = self.proj_add(conditioning)
        projected_cond_mult = self.proj_mult(conditioning)

        # Reshape for broadcasting
        projected_cond_add = projected_cond_add.unsqueeze(1).unsqueeze(1)
        projected_cond_mult = projected_cond_mult.unsqueeze(1).unsqueeze(1)

        return conv_filters * (1 + projected_cond_mult) + projected_cond_add


class WeightStandardizedConv2d(nn.Conv2d):
    """Convolution with weight standardization."""

    def forward(self, x):
        weight = self.weight

        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        # NOTE: the use of unbiased estimator
        weight_std = weight.std(dim=(1, 2, 3), keepdim=True, unbiased=False) + 1e-5
        standardized_weight = (weight - weight_mean) / weight_std

        return F.conv2d(
            x, standardized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class SmallStem(nn.Module):
    """Passes the image through a few light-weight convolutional layers,
    before patchifying the image. Empirically useful for many computer vision tasks.

    See Xiao et al: Early Convolutions Help Transformers See Better
    """

    def __init__(
        self,
        use_film: bool = False,
        patch_size: int = 32,
        kernel_sizes: tuple[int, ...] = (3, 3, 3, 3),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        features: tuple[int, ...] = (32, 96, 192, 384),
        padding: tuple[int, ...] = (1, 1, 1, 1),
        num_features: int = 512,
        img_norm_type: str = "default",
    ):
        super().__init__()
        self.use_film = use_film
        self.patch_size = patch_size
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.features = features
        self.padding = padding
        self.num_features = num_features
        self.img_norm_type = img_norm_type

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 6  # Assuming RGB input

        for n, (kernel_size, stride, out_features, conv_padding) in enumerate(
            zip(kernel_sizes, strides, features, padding, strict=True),
        ):
            self.conv_layers.append(
                nn.Sequential(
                    WeightStandardizedConv2d(
                        in_channels=in_channels,
                        out_channels=out_features,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=conv_padding,
                    ),
                    nn.GroupNorm(32, out_features, eps=1e-06),
                    nn.ReLU(),
                )
            )
            in_channels = out_features

        # Final embedding layer
        final_patch_size = patch_size // 16
        # Use the last element of the features tuple
        last_feature = features[-1] if isinstance(features, tuple) else features
        self.embedding = nn.Conv2d(
            in_channels=last_feature,
            out_channels=num_features,
            kernel_size=(final_patch_size, final_patch_size),
            stride=(final_patch_size, final_patch_size),
            padding=0,
        )

        # FiLM conditioning layer
        self.film = FilmConditioning() if use_film else None

    def forward(
        self, observations: torch.Tensor, train: bool = True, cond_var: torch.Tensor | None = None
    ):
        """
        Args:
            observations: Tensor of shape [batch_size, height, width, channels]
            train: Whether in training mode
            cond_var: Optional conditioning variable for FiLM

        Returns:
            Tensor of shape [batch_size, n_patches_h, n_patches_w, num_features]
        """
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert expecting_cond_var == received_cond_var, "Only pass in cond var iff model expecting cond var"

        # Normalize images
        x = normalize_images(observations, self.img_norm_type)

        # Convert from NHWC to NCHW format for PyTorch
        x = x.permute(0, 3, 1, 2)

        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Apply final embedding
        x = self.embedding(x)

        # Convert back to NHWC format
        x = x.permute(0, 2, 3, 1)

        # Apply FiLM conditioning if needed
        if self.use_film:
            assert cond_var is not None, "Cond var is None, nothing to condition on"
            x = self.film(x, cond_var)

        return x


class SmallStem16(SmallStem):
    """SmallStem with patch_size=16."""

    def __init__(
        self,
        use_film: bool = False,
        kernel_sizes: tuple[int, ...] = (3, 3, 3, 3),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        features: tuple[int, ...] = (32, 96, 192, 384),
        padding: tuple[int, ...] = (1, 1, 1, 1),
        num_features: int = 512,
        img_norm_type: str = "default",
    ):
        super().__init__(
            use_film=use_film,
            patch_size=16,  # Fixed to 16
            kernel_sizes=kernel_sizes,
            strides=strides,
            features=features,
            padding=padding,
            num_features=num_features,
            img_norm_type=img_norm_type,
        )


def regex_match(regex_keys, x):
    """Match a string against a list of regex patterns."""
    return any([re.match(r_key, x) for r_key in regex_keys])


def regex_filter(regex_keys, xs):
    """Filter a list of strings using regex patterns."""
    return list(filter(lambda x: regex_match(regex_keys, x), xs))


def generate_proper_pad_mask(
    tokens: torch.Tensor,
    pad_mask_dict: dict[str, torch.Tensor] | None,
    keys: Sequence[str],
) -> torch.Tensor:
    """Generate proper padding mask for tokens."""
    if pad_mask_dict is None:
        print("No pad_mask_dict found. Nothing will be masked.")
        return torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)

    if not all([key in pad_mask_dict for key in keys]):
        print(f"pad_mask_dict missing keys {set(keys) - set(pad_mask_dict.keys())}. Nothing will be masked.")
        return torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)

    pad_mask = torch.stack([pad_mask_dict[key] for key in keys], dim=-1)
    pad_mask = torch.any(pad_mask, dim=-1)
    pad_mask = pad_mask.unsqueeze(-1).expand(tokens.shape[:-1])

    return pad_mask


class ImageTokenizer(nn.Module):
    """Image tokenizer that encodes image stack into tokens."""

    def __init__(
        self,
        encoder: nn.Module,
        num_tokens: int = 8,
        conditioning_type: str = "none",
        obs_stack_keys: Sequence[str] = ("image_.*", "depth_.*"),
        task_stack_keys: Sequence[str] = tuple(),
        task_film_keys: Sequence[str] = tuple(),
        proper_pad_mask: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_tokens = num_tokens
        self.conditioning_type = conditioning_type
        self.obs_stack_keys = obs_stack_keys
        self.task_stack_keys = task_stack_keys
        self.task_film_keys = task_film_keys
        self.proper_pad_mask = proper_pad_mask

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        tasks: dict[str, torch.Tensor] | None = None,
    ):
        """Forward pass through image tokenizer."""

        def extract_inputs(keys, inputs, check_spatial=False):
            """Extract and concatenate inputs based on keys."""
            extracted_outputs = []
            for key in keys:
                if check_spatial:
                    assert len(inputs[key].shape) >= 4
                extracted_outputs.append(inputs[key])
            return torch.cat(extracted_outputs, dim=-1)

        # Filter observation keys using regex
        obs_stack_keys = regex_filter(self.obs_stack_keys, sorted(observations.keys()))
        if len(obs_stack_keys) == 0:
            assert self.proper_pad_mask, "Cannot skip unless using proper_pad_mask."
            return None

        # Stack all spatial observation inputs
        enc_inputs = extract_inputs(obs_stack_keys, observations, check_spatial=True)

        # Stack task inputs if specified
        if self.task_stack_keys and tasks is not None:
            needed_task_keys = regex_filter(self.task_stack_keys, observations.keys())
            # If any task inputs are missing, replace with zero padding
            for k in needed_task_keys:
                if k not in tasks:
                    # Create a copy of tasks with the missing key added
                    if isinstance(tasks, dict):
                        tasks = {**tasks, k: torch.zeros_like(observations[k][:, 0])}
                    else:
                        # Handle case where tasks is not a dict (e.g., None)
                        tasks = {k: torch.zeros_like(observations[k][:, 0])}

            task_stack_keys = regex_filter(self.task_stack_keys, sorted(tasks.keys()))
            if len(task_stack_keys) == 0:
                raise ValueError(f"No task inputs matching {self.task_stack_keys} were found.")

            task_inputs = extract_inputs(task_stack_keys, tasks, check_spatial=True)
            # Repeat task inputs for each timestep
            task_inputs = task_inputs.unsqueeze(1).repeat(1, enc_inputs.shape[1], 1, 1, 1)
            enc_inputs = torch.cat([enc_inputs, task_inputs], dim=-1)

        # Get shape information
        b, t, h, w, c = enc_inputs.shape

        # Reshape for encoder
        enc_inputs = enc_inputs.reshape(b * t, h, w, c)

        # Extract non-spatial FiLM inputs
        encoder_input_kwargs = {}

        # Run visual encoder
        image_tokens = self.encoder(enc_inputs, **encoder_input_kwargs)

        # Reshape back to batch, timestep format
        if isinstance(image_tokens, torch.Tensor):
            # Get spatial dimensions from encoder output
            spatial_dims = image_tokens.shape[1:-1]  # Exclude batch and channel dims
            token_dim = image_tokens.shape[-1]

            # Reshape from (b*t, h', w', c) to (b, t, h'*w', c)
            num_spatial_tokens = np.prod(spatial_dims)
            image_tokens = image_tokens.reshape(b, t, num_spatial_tokens, token_dim)

        # Generate padding mask
        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                image_tokens,
                observations.get("pad_mask_dict", None),
                obs_stack_keys,
            )
        else:
            pad_mask = torch.ones(image_tokens.shape[:-1], dtype=torch.bool, device=image_tokens.device)

        # Return TokenGroup
        return TokenGroup(image_tokens, pad_mask)


class LanguageTokenizer(nn.Module):
    """Language tokenizer that embeds text input IDs into continuous language embeddings."""

    def __init__(self, finetune_encoder: bool = False, proper_pad_mask: bool = True):
        super().__init__()
        self.proper_pad_mask = proper_pad_mask

        # Load pretrained weights directly with explicit float32 dtype
        self.t5_encoder = T5EncoderModel.from_pretrained("t5-base", torch_dtype=torch.float32)
        self.finetune_encoder = finetune_encoder

        if not self.finetune_encoder:
            for param in self.t5_encoder.parameters():
                param.requires_grad = False

    def forward(self, language_input: dict[str, torch.Tensor], tasks=None) -> TokenGroup:
        outputs = self.t5_encoder(
            input_ids=language_input["input_ids"], attention_mask=language_input["attention_mask"]
        )
        tokens = outputs.last_hidden_state.float()

        # Generate padding mask
        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                tokens,
                tasks.get("pad_mask_dict", None),
                ("language_instruction",),
            )
        else:
            pad_mask = torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)

        # # TODO (lilkm): check this
        # # All true attention mask, simple torch.ones
        # mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)

        # # TODO (lilkm): this more correct
        # # mask = language_input["attention_mask"].bool()

        return TokenGroup(tokens, pad_mask)


class TextProcessor:
    """HF Tokenizer wrapper."""

    def __init__(self, tokenizer_name: str = "t5-base", tokenizer_kwargs: dict | None = None):
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
            }

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_kwargs = tokenizer_kwargs

    def encode(self, strings: list[str]) -> dict[str, torch.Tensor]:
        """Encode strings to token IDs and attention masks."""
        return self.tokenizer(strings, **self.tokenizer_kwargs)
