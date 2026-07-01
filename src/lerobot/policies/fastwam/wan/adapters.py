# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan


class WanVideoVAE38(torch.nn.Module):
    """FastWAM VAE contract over `diffusers.AutoencoderKLWan` (Wan2.2-TI2V-5B).

    16x spatial / 4x temporal compression, 48 latent channels. diffusers'
    `AutoencoderKLWan` returns *raw* latents (it does not apply `latents_mean`/
    `latents_std`), so `encode`/`decode` here apply the same standardization the
    Wan reference uses — `(latents - mean) / std` — done in fp32 for stability.
    `encode` uses the deterministic posterior mode, matching the original VAE
    which returned the latent mean `mu`.
    """

    upsampling_factor = 16
    temporal_downsample_factor = 4
    z_dim = 48

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        *,
        pretrained: AutoencoderKLWan,
    ) -> None:
        super().__init__()
        # The Wan2.2 VAE is a fixed pretrained model — it is never trained from scratch,
        # so a real `AutoencoderKLWan` (with weights) must always be supplied (loaded from
        # the diffusers repo by `load_pretrained_wan_vae`). No random/offline build path.
        self.vae = pretrained.to(device=device, dtype=dtype)

        # Read the standardization stats from the VAE's own config (diffusers populates
        # these from vae/config.json) — single source of truth, no local copy. diffusers'
        # encode/decode return *raw* latents, so we apply (latent - mean) / std ourselves.
        # Non-persistent: kept out of state_dict.
        self.register_buffer(
            "latents_mean",
            torch.tensor(self.vae.config.latents_mean).view(1, self.z_dim, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(self.vae.config.latents_std).view(1, self.z_dim, 1, 1, 1),
            persistent=False,
        )

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.vae.parameters())
        return param.device, param.dtype

    def encode(
        self,
        videos: list[torch.Tensor] | torch.Tensor,
        device: str | torch.device | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        del device, tile_size, tile_stride
        if tiled:
            raise NotImplementedError("Tiled Wan2.2 VAE encoding is not supported by the FastWAM adapter.")
        if isinstance(videos, (list, tuple)):
            videos = torch.stack(list(videos))
        dev, dtype = self._device_dtype()
        mu = self.vae.encode(videos.to(device=dev, dtype=dtype)).latent_dist.mode().float()
        mean = self.latents_mean.float().to(mu.device)
        std = self.latents_std.float().to(mu.device)
        return (mu - mean) / std

    def decode(
        self,
        hidden_states: list[torch.Tensor] | torch.Tensor,
        device: str | torch.device | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        del device, tile_size, tile_stride
        if tiled:
            raise NotImplementedError("Tiled Wan2.2 VAE decoding is not supported by the FastWAM adapter.")
        if isinstance(hidden_states, (list, tuple)):
            hidden_states = torch.stack(list(hidden_states))
        dev, dtype = self._device_dtype()
        z = hidden_states.float()
        z = z * self.latents_std.float().to(z.device) + self.latents_mean.float().to(z.device)
        out = self.vae.decode(z.to(device=dev, dtype=dtype)).sample
        return out.float().clamp_(-1.0, 1.0)
