# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
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
#
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/utils/causal_vae.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.activations import get_activation
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings


def _shape_of(x) -> tuple | None:
    """Get shape of tensor if it is a torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    return None


def _compute_conv_output_size(in_size: int, k: int, s: int, p: int, d: int) -> int:
    """Compute output size of a convolution operation."""
    return max((in_size + 2 * p - d * (k - 1) - 1) // s + 1, 0)


class PerChannelRMSNorm(nn.Module):
    """Per-pixel (per-location) RMS normalization layer.

    For each element along the chosen dimension, this layer normalizes the tensor by the root-mean-square of its values
    across that dimension:

        y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)

    Args:
        channel_dim: Dimension along which to compute the RMS (typically channels).
        eps: Small constant added for numerical stability.
    """

    def __init__(self, channel_dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.eps = eps

    def forward(self, x: torch.Tensor, channel_dim: int | None = None) -> torch.Tensor:
        """Apply RMS normalization along the configured dimension."""
        channel_dim = channel_dim or self.channel_dim
        mean_sq = torch.mean(x**2, dim=self.channel_dim, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


class LTX2VideoCausalConv3d(nn.Module):
    """Causal 3D convolution for video processing.

    Like LTXCausalConv3d, but whether causal inference is performed can be
    specified at runtime via the `causal` parameter.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Temporal, height, width kernel size (int or tuple)
        stride: Stride for convolution (int or tuple)
        dilation: Dilation for convolution (int or tuple)
        groups: Number of groups for grouped convolution
        spatial_padding_mode: Padding mode for spatial dimensions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        )

        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            padding_mode=spatial_padding_mode,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional causal convolution and feature caching.

        Args:
            hidden_states: Input tensor [B, C, T, H, W]
            causal: Whether to use causal convolution
            feat_cache: Cache for temporal feature reuse
            feat_idx: Current index into feat_cache (mutated in-place)

        Returns:
            Output tensor after convolution
        """
        time_kernel_size = self.kernel_size[0]
        input_shape = _shape_of(hidden_states)

        # Handle empty temporal chunks
        if hidden_states.shape[2] == 0:
            return self._handle_empty_chunk(hidden_states, causal, feat_cache, feat_idx, time_kernel_size)

        if causal:
            hidden_states = self._apply_causal_padding(
                hidden_states, feat_cache, feat_idx, time_kernel_size, input_shape
            )
        else:
            hidden_states = self._apply_bidirectional_padding(hidden_states, time_kernel_size)

        hidden_states = self.conv(hidden_states)
        return hidden_states

    def _handle_empty_chunk(
        self,
        hidden_states: torch.Tensor,
        causal: bool,
        feat_cache: list | None,
        feat_idx: list | None,
        time_kernel_size: int,
    ) -> torch.Tensor:
        """Handle empty temporal chunks by reserving cache slots."""
        cache_len = max(time_kernel_size - 1, 0)
        if causal and feat_cache is not None and feat_idx is not None and cache_len > 0:
            idx = feat_idx[0]
            while len(feat_cache) <= idx:
                feat_cache.append(None)
            feat_idx[0] += 1

        batch = hidden_states.shape[0]
        out_h = _compute_conv_output_size(
            hidden_states.shape[3],
            self.conv.kernel_size[1],
            self.conv.stride[1],
            self.conv.padding[1],
            self.conv.dilation[1],
        )
        out_w = _compute_conv_output_size(
            hidden_states.shape[4],
            self.conv.kernel_size[2],
            self.conv.stride[2],
            self.conv.padding[2],
            self.conv.dilation[2],
        )
        return hidden_states.new_empty(batch, self.conv.out_channels, 0, out_h, out_w)

    def _apply_causal_padding(
        self,
        hidden_states: torch.Tensor,
        feat_cache: list | None,
        feat_idx: list | None,
        time_kernel_size: int,
        input_shape: tuple | None,
    ) -> torch.Tensor:
        """Apply causal padding using cached history."""
        cache_len = max(time_kernel_size - 1, 0)
        if feat_cache is not None and feat_idx is not None and cache_len > 0:
            idx = feat_idx[0]
            while len(feat_cache) <= idx:
                feat_cache.append(None)

            prefix = feat_cache[idx]
            current_cache = hidden_states[:, :, -cache_len:, :, :].clone()

            # Handle short chunks by preserving history
            if isinstance(prefix, torch.Tensor) and current_cache.shape[2] < cache_len:
                needed = cache_len - current_cache.shape[2]
                carry = prefix.to(hidden_states.device)
                carry = carry[:, :, -needed:, :, :]
                if carry.shape[2] < needed:
                    fill = carry[:, :, :1, :, :].repeat((1, 1, needed - carry.shape[2], 1, 1))
                    carry = torch.cat([fill, carry], dim=2)
                current_cache = torch.cat([carry, current_cache], dim=2)

            # Prepare prefix from cache or fallback
            if prefix is not None:
                prefix = prefix.to(hidden_states.device)
                if prefix.shape[2] > cache_len:
                    prefix = prefix[:, :, -cache_len:, :, :]
                if prefix.shape[2] < cache_len:
                    # Extend by repeating oldest cached frame
                    fallback = prefix[:, :, :1, :, :].repeat((1, 1, cache_len - prefix.shape[2], 1, 1))
                    prefix = torch.concatenate([fallback, prefix], dim=2)
            else:
                # No cache: use first frame as fallback
                prefix = hidden_states[:, :, :1, :, :].repeat((1, 1, cache_len, 1, 1))

            hidden_states = torch.concatenate([prefix, hidden_states], dim=2)
            feat_cache[idx] = current_cache
            feat_idx[0] += 1
        else:
            # No cache: use causal padding
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, cache_len, 1, 1))
            hidden_states = torch.concatenate([pad_left, hidden_states], dim=2)

        return hidden_states

    def _apply_bidirectional_padding(
        self, hidden_states: torch.Tensor, time_kernel_size: int
    ) -> torch.Tensor:
        """Apply symmetric padding for bidirectional mode."""
        pad_size = (time_kernel_size - 1) // 2
        pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, pad_size, 1, 1))
        pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, pad_size, 1, 1))
        return torch.concatenate([pad_left, hidden_states, pad_right], dim=2)


class LTX2VideoResnetBlock3d(nn.Module):
    """A 3D ResNet block used in the LTX 2.0 audiovisual model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels. If None, defaults to `in_channels`.
        dropout: Dropout rate.
        eps: Epsilon value for normalization layers.
        elementwise_affine: Whether to enable elementwise affinity in normalization.
        non_linearity: Activation function to use.
        conv_shortcut: Whether to use a convolution shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        non_linearity: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = PerChannelRMSNorm()
        self.conv1 = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm2 = PerChannelRMSNorm()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = LTX2VideoCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
            # LTX 2.0 uses a normal nn.Conv3d here rather than LTXVideoCausalConv3d
            self.conv_shortcut = nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1
            )

        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        temb: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        causal: bool = True,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        """Forward pass through ResNet block."""
        hidden_states = inputs

        hidden_states = self.norm1(hidden_states)

        if self.scale_shift_table is not None:
            temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]

        hidden_states = self.norm2(hidden_states)

        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]

        if self.norm3 is not None:
            inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class LTXVideoDownsampler3d(nn.Module):
    """3D downsampling layer for spatiotemporal reduction.

    Uses pixel unshuffle pattern for downsampling with causal temporal handling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int] = 1,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.group_size = (in_channels * stride[0] * stride[1] * stride[2]) // out_channels

        out_channels = out_channels // (self.stride[0] * self.stride[1] * self.stride[2])

        self.conv = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        """Forward pass with temporal grouping state management."""
        _shape_of(hidden_states)
        prefix_len = max(self.stride[0] - 1, 0)

        if prefix_len > 0 and feat_cache is not None and feat_idx is not None:
            hidden_states = self._manage_temporal_grouping(hidden_states, feat_cache, feat_idx, prefix_len)
        else:
            hidden_states = torch.cat([hidden_states[:, :, : self.stride[0] - 1], hidden_states], dim=2)

        # Handle empty temporal dimension after grouping
        if hidden_states.shape[2] == 0:
            return self._handle_empty_output(hidden_states, feat_cache, feat_idx)

        # Apply pixel unshuffle downsampling
        residual = self._compute_residual(hidden_states)

        hidden_states = self.conv(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
        hidden_states = self._unshuffle_output(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states

    def _manage_temporal_grouping(
        self, hidden_states: torch.Tensor, feat_cache: list, feat_idx: list, prefix_len: int
    ) -> torch.Tensor:
        """Manage temporal grouping state for chunked processing."""
        idx = feat_idx[0]
        while len(feat_cache) <= idx:
            feat_cache.append(None)

        state = feat_cache[idx]
        if state is None or not isinstance(state, dict):
            state = {"prefixed": False, "pending": None}

        sequence = hidden_states
        if not state["prefixed"]:
            prefix = hidden_states[:, :, :1, :, :].repeat((1, 1, prefix_len, 1, 1))
            sequence = torch.cat([prefix, sequence], dim=2)
            state["prefixed"] = True

        if state["pending"] is not None:
            pending = state["pending"].to(sequence.device)
            sequence = torch.cat([pending, sequence], dim=2)

        stride_t = self.stride[0]
        usable_t = (sequence.shape[2] // stride_t) * stride_t
        state["pending"] = sequence[:, :, usable_t:, :, :].clone() if usable_t < sequence.shape[2] else None
        hidden_states = sequence[:, :, :usable_t, :, :]
        feat_cache[idx] = state
        feat_idx[0] += 1

        return hidden_states

    def _handle_empty_output(
        self, hidden_states: torch.Tensor, feat_cache: list | None, feat_idx: list | None
    ) -> torch.Tensor:
        """Handle empty output when temporal dimension is 0."""
        if feat_cache is not None and feat_idx is not None:
            # Reserve cache slot for conv
            idx = feat_idx[0]
            while len(feat_cache) <= idx:
                feat_cache.append(None)
            feat_idx[0] += 1

        out_channels = self.conv.out_channels * self.stride[0] * self.stride[1] * self.stride[2]
        out_h = hidden_states.shape[3] // self.stride[1]
        out_w = hidden_states.shape[4] // self.stride[2]
        return hidden_states.new_empty(
            hidden_states.shape[0],
            out_channels,
            0,
            out_h,
            out_w,
        )

    def _compute_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute residual connection for pixel unshuffle."""
        residual = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        residual = residual.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        residual = residual.unflatten(1, (-1, self.group_size))
        return residual.mean(dim=2)

    def _unshuffle_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply pixel unshuffle to output."""
        hidden_states = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        return hidden_states


class LTX2VideoDownBlock3D(nn.Module):
    """Down block used in the LTXVideo model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels. If None, defaults to `in_channels`.
        num_layers: Number of resnet layers.
        dropout: Dropout rate.
        resnet_eps: Epsilon value for normalization layers.
        resnet_act_fn: Activation function to use.
        spatio_temporal_scale: Whether to use downsampling layer.
        downsample_type: Type of downsampling ("conv", "spatial", "temporal", "spatiotemporal")
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        downsample_type: str = "conv",
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList()

            if downsample_type == "conv":
                self.downsamplers.append(
                    LTX2VideoCausalConv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=(2, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "spatial":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(1, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "temporal":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(2, 1, 1),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "spatiotemporal":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(2, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        causal: bool = True,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        for _i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None:
                hidden_states = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb, generator, causal
                )
            else:
                hidden_states = resnet(
                    hidden_states, temb, generator, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx
                )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(
                    hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx
                )

        return hidden_states


class LTX2VideoMidBlock3d(nn.Module):
    """Middle block used in the LTXVideo model.

    Args:
        in_channels: Number of input channels.
        num_layers: Number of resnet layers.
        dropout: Dropout rate.
        resnet_eps: Epsilon value for normalization layers.
        resnet_act_fn: Activation function to use.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        causal: bool = True,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)

        for _i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None:
                hidden_states = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb, generator, causal
                )
            else:
                hidden_states = resnet(
                    hidden_states, temb, generator, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx
                )

        return hidden_states


class LTX2VideoEncoder3d(nn.Module):
    """Encoder layer for encoding video samples to latent representation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of latent channels.
        block_out_channels: Number of output channels for each block.
        spatio_temporal_scaling: Whether each block should contain downscaling.
        layers_per_block: Number of layers per block.
        downsample_type: Downsampling pattern per block.
        patch_size: Size of spatial patches.
        patch_size_t: Size of temporal patches.
        resnet_norm_eps: Epsilon for ResNet normalization.
        is_causal: Whether to use causal behavior.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        block_out_channels: tuple[int, ...] = (256, 512, 1024, 2048),
        down_block_types: tuple[str, ...] = (
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
        ),
        spatio_temporal_scaling: tuple[bool, ...] = (True, True, True, True),
        layers_per_block: tuple[int, ...] = (4, 6, 6, 2, 2),
        downsample_type: tuple[str, ...] = ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels * patch_size**2 * patch_size_t
        self.is_causal = is_causal

        output_channel = out_channels

        self.conv_in = LTX2VideoCausalConv3d(
            in_channels=self.in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        # down blocks
        num_block_out_channels = len(block_out_channels)
        self.down_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if down_block_types[i] == "LTX2VideoDownBlock3D":
                down_block = LTX2VideoDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    resnet_eps=resnet_norm_eps,
                    spatio_temporal_scale=spatio_temporal_scaling[i],
                    downsample_type=downsample_type[i],
                    spatial_padding_mode=spatial_padding_mode,
                )
            else:
                raise ValueError(f"Unknown down block type: {down_block_types[i]}")

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = LTX2VideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[-1],
            resnet_eps=resnet_norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )

        # out
        self.norm_out = PerChannelRMSNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = LTX2VideoCausalConv3d(
            in_channels=output_channel,
            out_channels=out_channels + 1,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool | None = None,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        causal = causal or self.is_causal
        if feat_cache is not None and feat_idx is None:
            feat_idx = [0]

        hidden_states = hidden_states.reshape(
            batch_size, num_channels, post_patch_num_frames, p_t, post_patch_height, p, post_patch_width, p
        )
        hidden_states = hidden_states.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        hidden_states = self.conv_in(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        if torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None:
            for down_block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    down_block, hidden_states, None, None, causal
                )

            hidden_states = self._gradient_checkpointing_func(
                self.mid_block, hidden_states, None, None, causal
            )
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(
                    hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx
                )

            hidden_states = self.mid_block(
                hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx
            )

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        last_channel = hidden_states[:, -1:]
        last_channel = last_channel.repeat(1, hidden_states.size(1) - 2, 1, 1, 1)
        hidden_states = torch.cat([hidden_states, last_channel], dim=1)

        return hidden_states
