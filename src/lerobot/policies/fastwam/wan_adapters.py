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

from pathlib import Path
from typing import Any

import torch

from .wan.modules.vae2_2 import Wan2_2_VAE


class WanVideoVAE38(torch.nn.Module):
    """Tensor-batch adapter around the official Wan2.2 VAE wrapper."""

    upsampling_factor = 16
    temporal_downsample_factor = 4
    z_dim = 48

    def __init__(
        self,
        vae_pth: str | Path,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.wan_vae = Wan2_2_VAE(vae_pth=str(vae_pth), dtype=dtype, device=str(device))
        self.model = self.wan_vae.model
        self.dtype = dtype
        self.device = torch.device(device)

    def to(self, *args: Any, **kwargs: Any):
        super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        param = next(self.model.parameters())
        self.device = param.device
        self.dtype = param.dtype
        self.wan_vae.device = self.device
        self.wan_vae.dtype = self.dtype
        self.wan_vae.scale = [scale.to(device=self.device, dtype=self.dtype) for scale in self.wan_vae.scale]
        self.wan_vae.model = self.model
        return self

    def encode(
        self,
        videos: list[torch.Tensor] | torch.Tensor,
        device: str | torch.device | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        del tile_size, tile_stride
        if tiled:
            raise NotImplementedError("Tiled Wan2.2 VAE encoding is not supported by the FastWAM adapter.")
        target_device = self.device if device is None else torch.device(device)
        if target_device != self.device:
            self.to(device=target_device)
        if isinstance(videos, torch.Tensor):
            videos = list(videos)
        hidden_states = self.wan_vae.encode([video.to(self.device) for video in videos])
        if hidden_states is None:
            raise RuntimeError("Wan2.2 VAE encode failed; expected a list of video tensors.")
        return torch.stack(hidden_states)

    def decode(
        self,
        hidden_states: list[torch.Tensor] | torch.Tensor,
        device: str | torch.device | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        del tile_size, tile_stride
        if tiled:
            raise NotImplementedError("Tiled Wan2.2 VAE decoding is not supported by the FastWAM adapter.")
        target_device = self.device if device is None else torch.device(device)
        if target_device != self.device:
            self.to(device=target_device)
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = list(hidden_states)
        videos = self.wan_vae.decode([hidden_state.to(self.device) for hidden_state in hidden_states])
        if videos is None:
            raise RuntimeError("Wan2.2 VAE decode failed; expected a list of latent tensors.")
        return torch.stack(videos)


__all__ = ["WanVideoVAE38"]
