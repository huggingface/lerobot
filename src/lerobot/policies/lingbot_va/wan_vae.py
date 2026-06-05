# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Thin helpers around the stock diffusers ``AutoencoderKLWan`` (Wan2.2, ``z_dim=48``).

The VAE class itself is NOT vendored — it lives in ``diffusers>=0.36``. This module
provides:
  * loaders for the VAE / text encoder / tokenizer / transformer sub-checkpoints,
  * the streaming-encoder wrapper used for autoregressive frame-by-frame VAE encoding
    (it caches the causal-conv state across chunks),
  * latent (de)normalization helpers using the VAE's ``latents_mean`` / ``latents_std``.

Vendored and adapted from ``wan_va/modules/utils.py`` upstream.
"""

import torch

__all__ = [
    "WanVAEStreamingWrapper",
    "load_vae",
    "load_text_encoder",
    "load_tokenizer",
    "normalize_latents",
    "denormalize_latents",
    "patchify",
]


def load_vae(vae_path, torch_dtype, torch_device):
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch_dtype)
    return vae.to(torch_device)


def load_text_encoder(text_encoder_path, torch_dtype, torch_device):
    from transformers import UMT5EncoderModel

    text_encoder = UMT5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=torch_dtype)
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path):
    from transformers import T5TokenizerFast

    return T5TokenizerFast.from_pretrained(tokenizer_path)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(
        batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(
        batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size
    )
    return x


def normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    """Apply ``(x - mean) * std`` channel-wise (note: upstream passes ``1/std`` as ``latents_std``)."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents = ((latents.float() - latents_mean) * latents_std).to(latents)
    return latents


def denormalize_latents(latents: torch.Tensor, latents_mean, latents_std, z_dim) -> torch.Tensor:
    """Inverse of the normalization applied at encode time, for VAE decoding of predicted latents."""
    mean = torch.tensor(latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    inv_std = 1.0 / torch.tensor(latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    return latents / inv_std + mean


class WanVAEStreamingWrapper:
    """Wraps an ``AutoencoderKLWan`` encoder to support causal streaming encoding across chunks."""

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
