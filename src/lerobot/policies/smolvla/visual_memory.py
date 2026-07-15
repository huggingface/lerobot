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

"""Short-horizon visual memory from MEM (arXiv:2603.03596).

The encoder keeps SigLIP's pretrained parameters and alternates its ordinary
per-frame spatial attention with causal attention across time for matching
spatial patches. Only the current frame's patch tokens leave the vision tower,
so enabling memory does not increase the VLM prefix length.
"""

import math

import torch
from torch import Tensor


def sample_visual_history(
    history: list[Tensor], *, num_frames: int, stride: int, steps_seen: int
) -> tuple[Tensor, Tensor]:
    """Subsample an inference queue and mark pre-episode padding frames."""
    sampled = history[::stride][-num_frames:]
    if len(sampled) != num_frames:
        raise ValueError(f"visual history has {len(sampled)} samples, expected {num_frames}")
    video = torch.stack(sampled, dim=1)
    required_ages = range((num_frames - 1) * stride, -1, -stride)
    valid = torch.tensor([steps_seen > age for age in required_ages], dtype=torch.bool, device=video.device)
    padding_mask = (~valid)[None, :].expand(video.shape[0], -1)
    return video, padding_mask


def temporal_sinusoidal_embedding(
    num_frames: int, hidden_size: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Return fixed temporal embeddings with an exactly-zero current position."""
    if hidden_size % 2:
        raise ValueError(f"hidden_size must be even, got {hidden_size}")

    # History is ordered oldest -> current, matching t in [-K, 0] in MEM.
    positions = torch.arange(1 - num_frames, 1, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, hidden_size, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / hidden_size)
    )[None, :]
    angles = positions * frequencies
    embedding = torch.zeros(num_frames, hidden_size, device=device, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(angles)
    embedding[:, 1::2] = torch.cos(angles) - 1.0
    return embedding.to(dtype=dtype)


def causal_temporal_mask(frame_mask: Tensor, *, dtype: torch.dtype, num_patches: int) -> Tensor:
    """Build an additive causal/key-padding mask for per-patch temporal attention."""
    if frame_mask.ndim != 2:
        raise ValueError(f"frame_mask must have shape (batch, frames), got {tuple(frame_mask.shape)}")

    batch_size, num_frames = frame_mask.shape
    allowed = torch.ones(num_frames, num_frames, dtype=torch.bool, device=frame_mask.device).tril()
    allowed = allowed[None, :, :] & frame_mask[:, None, :].bool()
    # The current frame is always present in normal use. Keeping the diagonal
    # valid also prevents NaNs for fully padded historical query rows.
    diagonal = torch.eye(num_frames, dtype=torch.bool, device=frame_mask.device)[None, :, :]
    allowed = allowed | diagonal
    mask = torch.zeros(batch_size, 1, num_frames, num_frames, dtype=dtype, device=frame_mask.device)
    mask.masked_fill_(~allowed[:, None, :, :], torch.finfo(dtype).min)
    return mask.repeat_interleave(num_patches, dim=0)


def encode_video_with_mem(
    vision_model,
    pixel_values: Tensor,
    frame_mask: Tensor,
    *,
    temporal_attention_every: int,
) -> Tensor:
    """Encode ``(B,T,C,H,W)`` video with MEM's space-time separable attention.

    No modules or learnable parameters are added. At temporal layers, the same
    layer-normalization and Q/K/V/out projections are reused for a second,
    causal attention operation along time. For a one-frame input the temporal
    branch is skipped, making this exactly equivalent to the original image
    encoder.
    """
    if pixel_values.ndim != 5:
        raise ValueError(f"pixel_values must have shape (B,T,C,H,W), got {tuple(pixel_values.shape)}")
    if temporal_attention_every < 1:
        raise ValueError("temporal_attention_every must be at least 1")

    batch_size, num_frames, channels, height, width = pixel_values.shape
    if frame_mask.shape != (batch_size, num_frames):
        raise ValueError(
            f"frame_mask must have shape {(batch_size, num_frames)}, got {tuple(frame_mask.shape)}"
        )

    required = ("embeddings", "encoder", "post_layernorm")
    if any(not hasattr(vision_model, name) for name in required):
        raise TypeError("MEM visual memory currently requires a SigLIP-compatible vision tower")

    flat_pixels = pixel_values.reshape(batch_size * num_frames, channels, height, width)
    if hasattr(vision_model, "patch_size"):
        patch_size = vision_model.patch_size
        patch_attention_mask = torch.ones(
            batch_size * num_frames,
            height // patch_size,
            width // patch_size,
            dtype=torch.bool,
            device=pixel_values.device,
        )
        hidden_states = vision_model.embeddings(
            pixel_values=flat_pixels,
            patch_attention_mask=patch_attention_mask,
        )
    else:
        hidden_states = vision_model.embeddings(flat_pixels)
    num_patches, hidden_size = hidden_states.shape[1:]
    hidden_states = hidden_states.reshape(batch_size, num_frames, num_patches, hidden_size)

    temporal_positions = temporal_sinusoidal_embedding(
        num_frames, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype
    )[None, :, None, :]
    temporal_mask = causal_temporal_mask(frame_mask, dtype=hidden_states.dtype, num_patches=num_patches)

    for layer_index, layer in enumerate(vision_model.encoder.layers):
        spatial_input = hidden_states.reshape(batch_size * num_frames, num_patches, hidden_size)
        if num_frames == 1 or (layer_index + 1) % temporal_attention_every:
            hidden_states = layer(spatial_input, attention_mask=None).reshape(
                batch_size, num_frames, num_patches, hidden_size
            )
            continue

        residual = hidden_states
        spatial_norm = layer.layer_norm1(spatial_input)
        spatial_output, _ = layer.self_attn(hidden_states=spatial_norm, attention_mask=None)
        spatial_output = spatial_output.reshape(batch_size, num_frames, num_patches, hidden_size)

        temporal_input = (hidden_states + temporal_positions).permute(0, 2, 1, 3)
        temporal_input = temporal_input.reshape(batch_size * num_patches, num_frames, hidden_size)
        temporal_norm = layer.layer_norm1(temporal_input)
        temporal_output, _ = layer.self_attn(
            hidden_states=temporal_norm,
            attention_mask=temporal_mask,
        )
        temporal_output = temporal_output.reshape(batch_size, num_patches, num_frames, hidden_size)
        temporal_output = temporal_output.permute(0, 2, 1, 3)

        hidden_states = residual + spatial_output + temporal_output
        hidden_states = hidden_states + layer.mlp(layer.layer_norm2(hidden_states))

    # MEM drops all historical tokens before the VLA backbone.
    return vision_model.post_layernorm(hidden_states[:, -1])
