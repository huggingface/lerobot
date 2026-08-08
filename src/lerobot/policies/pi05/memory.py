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

"""Short-horizon visual and proprioceptive memory from MEM (arXiv:2603.03596)."""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

# Attributes MEM reads off a `SiglipEncoderLayer` and its `SiglipAttention`. MEM
# re-implements the attention sub-block to compose the spatial and temporal
# attention weights (MEM appendix C, eq. 3), so it depends on these internals
# rather than on `SiglipAttention.forward`. Validated per layer so a transformers
# bump fails loudly instead of silently changing the architecture.
_REQUIRED_LAYER_ATTRS = ("layer_norm1", "self_attn", "layer_norm2", "mlp")
_REQUIRED_ATTENTION_ATTRS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "num_heads",
    "head_dim",
    "scale",
    "dropout",
)


def _hidden_tensor(output, *, source: str) -> Tensor:
    """Normalize supported Transformers layer outputs and fail clearly on API drift."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(
        f"{source} must return a Tensor or a tuple whose first item is a Tensor, got {type(output).__name__}"
    )


def _validate_siglip_layer(layer, *, layer_index: int) -> None:
    """Validate the private SigLIP encoder-layer contract used by MEM."""
    missing = [name for name in _REQUIRED_LAYER_ATTRS if not hasattr(layer, name)]
    if missing:
        raise TypeError(
            "MEM visual memory requires SigLIP encoder layers exposing "
            f"{_REQUIRED_LAYER_ATTRS}; layer {layer_index} is missing {tuple(missing)}"
        )
    missing = [name for name in _REQUIRED_ATTENTION_ATTRS if not hasattr(layer.self_attn, name)]
    if missing:
        raise TypeError(
            "MEM visual memory requires SigLIP self-attention exposing "
            f"{_REQUIRED_ATTENTION_ATTRS}; layer {layer_index} is missing {tuple(missing)}"
        )


def sample_observation_history(
    history: list[Tensor], *, num_frames: int, stride: int, steps_seen: int
) -> tuple[Tensor, Tensor]:
    """Subsample a homogeneous-batch inference queue and mark pre-episode padding.

    Frames are addressed by age relative to ``history[-1]``, the newest observation,
    so the current frame is always the last sampled frame regardless of queue length.

    ``steps_seen`` applies to every batch row. Pi05 clears the queue at each
    batched rollout boundary; independently resetting vector rows is unsupported.
    """
    # Descending ages, so the returned frames run oldest -> current.
    required_ages = list(range((num_frames - 1) * stride, -1, -stride))
    if len(history) <= required_ages[0]:
        raise ValueError(
            f"observation history holds {len(history)} frames, need at least "
            f"{required_ages[0] + 1} to sample {num_frames} frames at stride {stride}"
        )
    values = torch.stack([history[-1 - age] for age in required_ages], dim=1)
    valid = torch.tensor([steps_seen > age for age in required_ages], dtype=torch.bool, device=values.device)
    padding_mask = (~valid)[None, :].expand(values.shape[0], -1)
    return values, padding_mask


def temporal_sinusoidal_embedding(
    num_frames: int, hidden_size: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Return fixed temporal embeddings whose current position is exactly zero."""
    if hidden_size % 2:
        raise ValueError(f"hidden_size must be even, got {hidden_size}")
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
    """Build an additive causal and key-padding mask for temporal attention."""
    if frame_mask.ndim != 2:
        raise ValueError(f"frame_mask must have shape (batch, frames), got {tuple(frame_mask.shape)}")
    batch_size, num_frames = frame_mask.shape
    allowed = torch.ones(num_frames, num_frames, dtype=torch.bool, device=frame_mask.device).tril()
    allowed = allowed[None] & frame_mask[:, None, :].bool()
    # Keep padded query rows numerically safe; valid queries still cannot see padded keys.
    allowed |= torch.eye(num_frames, dtype=torch.bool, device=frame_mask.device)[None]
    mask = torch.zeros(batch_size, 1, num_frames, num_frames, dtype=dtype, device=frame_mask.device)
    mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    return mask.repeat_interleave(num_patches, dim=0)


def space_time_attention(attention, hidden_states: Tensor, temporal_mask: Tensor) -> Tensor:
    """Apply MEM's composed space-time attention (appendix C, eq. 3) to ``(B,T,P,D)``.

    ``hidden_states`` must already carry the temporal position embedding, so one set
    of the ViT's pretrained q/k/v projections serves both stages — MEM adds no
    learnable parameters to the vision tower.

    Eq. 3 composes the two attention operators, ``alpha_spatial[alpha_temporal[z]]``,
    and then "follow[s] the standard computation of a transformer layer". Composing
    the weights rather than stacking two attention sub-blocks matters twice over:
    ``out_proj`` is applied once, and a softmax over a single timestep is the
    identity, so ``T == 1`` reduces to stock SigLIP spatial attention by
    construction. Applying the composition is just spatial attention whose values
    are the temporally mixed ones, which keeps both stages on SDPA — no attention
    matrix is ever materialized.
    """
    batch_size, num_frames, num_patches, _ = hidden_states.shape
    heads, head_dim = attention.num_heads, attention.head_dim
    projected_shape = (batch_size, num_frames, num_patches, heads, head_dim)
    queries = attention.q_proj(hidden_states).view(projected_shape)
    keys = attention.k_proj(hidden_states).view(projected_shape)
    values = attention.v_proj(hidden_states).view(projected_shape)
    dropout = attention.dropout if attention.training else 0.0

    def as_temporal(tensor: Tensor) -> Tensor:
        # (B,T,P,h,d) -> (B*P,h,T,d): one sequence per patch, over frames.
        return tensor.permute(0, 2, 3, 1, 4).reshape(batch_size * num_patches, heads, num_frames, head_dim)

    def as_spatial(tensor: Tensor) -> Tensor:
        # (B,T,P,h,d) -> (B*T,h,P,d): one sequence per frame, over patches.
        return tensor.permute(0, 1, 3, 2, 4).reshape(batch_size * num_frames, heads, num_patches, head_dim)

    temporal = F.scaled_dot_product_attention(
        as_temporal(queries),
        as_temporal(keys),
        as_temporal(values),
        attn_mask=temporal_mask,
        dropout_p=dropout,
        scale=attention.scale,
    )
    temporal = temporal.reshape(batch_size, num_patches, heads, num_frames, head_dim).permute(0, 3, 1, 2, 4)

    attended = F.scaled_dot_product_attention(
        as_spatial(queries),
        as_spatial(keys),
        as_spatial(temporal),
        dropout_p=dropout,
        scale=attention.scale,
    )
    attended = attended.reshape(batch_size, num_frames, heads, num_patches, head_dim).permute(0, 1, 3, 2, 4)
    return attention.out_proj(attended.reshape(batch_size, num_frames, num_patches, heads * head_dim))


def encode_video_with_mem(
    vision_model,
    pixel_values: Tensor,
    frame_mask: Tensor,
    *,
    temporal_attention_every: int,
) -> Tensor:
    """Encode ``(B,T,C,H,W)`` using MEM space-time separable attention.

    Every Nth layer replaces its attention sub-block with the composed space-time
    attention of :func:`space_time_attention`, reusing the pretrained SigLIP
    projections. Past-frame tokens are dropped once the last such layer has run —
    no cross-frame mixing can happen above it — so the remaining layers and the
    downstream VLM prefix see exactly the single-frame token count.

    Temporal layers call SDPA directly and therefore ignore the vision tower's
    configured attention implementation; the spatial-only layers still use it.
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
    if any(not hasattr(vision_model, name) for name in ("embeddings", "encoder", "post_layernorm")):
        raise TypeError("MEM visual memory requires a SigLIP-compatible vision transformer")
    if not hasattr(vision_model.encoder, "layers"):
        raise TypeError("MEM visual memory requires a SigLIP encoder exposing a layers collection")

    layers = vision_model.encoder.layers
    temporal_layers = [i for i in range(len(layers)) if (i + 1) % temporal_attention_every == 0]
    last_temporal_index = temporal_layers[-1] if (temporal_layers and num_frames > 1) else -1

    flat_pixels = pixel_values.reshape(batch_size * num_frames, channels, height, width)
    hidden_states = vision_model.embeddings(flat_pixels)
    num_patches, hidden_size = hidden_states.shape[1:]
    hidden_states = hidden_states.reshape(batch_size, num_frames, num_patches, hidden_size)
    temporal_positions = temporal_sinusoidal_embedding(
        num_frames, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype
    )[None, :, None]
    temporal_mask = causal_temporal_mask(frame_mask, dtype=hidden_states.dtype, num_patches=num_patches)

    for layer_index, layer in enumerate(layers):
        _validate_siglip_layer(layer, layer_index=layer_index)
        active_frames = hidden_states.shape[1]
        if active_frames == 1 or (layer_index + 1) % temporal_attention_every:
            flat_hidden = hidden_states.reshape(batch_size * active_frames, num_patches, hidden_size)
            layer_output = _hidden_tensor(
                layer(flat_hidden, attention_mask=None),
                source=f"SigLIP encoder layer {layer_index}",
            )
            hidden_states = layer_output.reshape(batch_size, active_frames, num_patches, hidden_size)
            continue

        # eq. 1: both stages derive q/k/v from z + e(t). The residual stays the
        # layer input so the position signal does not accumulate across layers.
        attended = space_time_attention(
            layer.self_attn, layer.layer_norm1(hidden_states + temporal_positions), temporal_mask
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + layer.mlp(layer.layer_norm2(hidden_states))
        if layer_index == last_temporal_index:
            hidden_states = hidden_states[:, -1:]

    return vision_model.post_layernorm(hidden_states[:, -1])
