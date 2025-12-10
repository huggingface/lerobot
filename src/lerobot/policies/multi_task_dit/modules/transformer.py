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

"""Transformer backbone for noise prediction in Multi-Task DiT policy.

Adapted from DiT (Diffusion Transformer: https://github.com/facebookresearch/DiT) for 1D trajectory data.
"""

import math

import torch
from torch import Tensor, nn


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Modulate input with shift and scale for AdaLN-Zero.

    Args:
        x: Input tensor
        shift: Shift parameter
        scale: Scale parameter

    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    return x * (1 + scale) + shift


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps.

    Identical to the reference implementation - generates smooth embeddings
    for diffusion timestep values.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B,) tensor of timestep values

        Returns:
            (B, dim) positional embeddings
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformers.

    RoPE encodes position information by rotating query and key vectors,
    which naturally captures relative positions through the dot product.
    Applied at every attention layer rather than once at input.

    To do this, we need to reimplement the attention mechanism to apply RoPE
    to Q and K before computing the attention scores, so we cannot use the
    the built-in MultiheadAttention module.

    Original RoPE Paper: https://arxiv.org/abs/2104.09864 (RoFormer)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies: theta_i = 1 / (base^(2i/d))
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
        """Rotate half the hidden dims of the input.

        For x = [x1, x2], returns [-x2, x1]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        seq_len = q.shape[2]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}. "
                f"Increase max_seq_len in RoPE config."
            )

        # Slice precomputed cache to actual sequence length
        cos = self._cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self._sin_cached[:, :, :seq_len, :].to(q.dtype)

        # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated


class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE).

    Custom attention implementation that applies RoPE to Q and K before
    computing attention scores. This allows position information to be
    encoded at every attention layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        """
        Args:
            hidden_size: Total hidden dimension
            num_heads: Number of attention heads
            dropout: Attention dropout rate
            max_seq_len: Maximum sequence length for RoPE cache
            rope_base: Base for RoPE frequency computation
        """
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
        """
        Args:
            x: (B, T, hidden_size) input sequence

        Returns:
            (B, T, hidden_size) attention output
        """
        B, T, _ = x.shape  # noqa: N806

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3 * hidden_size)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, T, head_dim)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Scaled dot-product attention
        # Using PyTorch's efficient attention when available
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) and self.training else 0.0,
        )  # (B, num_heads, T, head_dim)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.hidden_size)  # (B, T, hidden_size)
        output = self.out_proj(attn_out)

        return output


class TransformerBlock(nn.Module):
    """DiT-style transformer block with AdaLN-Zero.

    Official DiT implementation with 6-parameter adaptive layer normalization:
    - shift_msa, scale_msa, gate_msa: for attention block
    - shift_mlp, scale_mlp, gate_mlp: for MLP block

    Supports both standard attention and RoPE attention.

    Reference: https://github.com/facebookresearch/DiT
    """

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
        """
        Args:
            hidden_size: Hidden dimension of transformer
            num_heads: Number of attention heads
            num_features: Size of conditioning features
            dropout: Dropout rate
            use_rope: Whether to use Rotary Position Embedding
            max_seq_len: Maximum sequence length (for RoPE cache)
            rope_base: Base frequency for RoPE
        """
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

        # Layer normalizations (no learnable affine parameters, all adaptation via conditioning)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # AdaLN-Zero modulation: produces 6 parameters (shift, scale, gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(num_features, 6 * hidden_size, bias=True))

    def forward(self, x: Tensor, features: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, hidden_size) input sequence
            features: (B, num_features) conditioning features

        Returns:
            (B, T, hidden_size) processed sequence
        """
        # Generate 6 modulation parameters from conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            features
        ).chunk(6, dim=1)

        # Attention block: norm → modulate → attn → gate × output → residual
        # modulate requires unsqueeze(1) to add sequence dimension for broadcasting
        attn_input = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))

        if self.use_rope:
            attn_out = self.attn(attn_input)
        else:
            attn_out, _ = self.multihead_attn(attn_input, attn_input, attn_input)

        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP block: norm → modulate → mlp → gate × output → residual
        mlp_input = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        mlp_out = self.mlp(mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class DiffusionTransformer(nn.Module):
    """Transformer-based diffusion noise prediction model."""

    def __init__(self, config, conditioning_dim: int):
        """Initialize transformer for noise prediction.

        Args:
            config: MultiTaskDiTConfig with transformer parameters
            conditioning_dim: Dimension of concatenated observation features
        """
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

        # Project action dimensions to hidden size
        self.input_proj = nn.Linear(self.action_dim, self.hidden_size)

        if config.use_positional_encoding:
            # Learnable positional embeddings for sequence positions (absolute encoding)
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

        # Project back to action dimensions
        self.output_proj = nn.Linear(self.hidden_size, self.action_dim)

        # Zero-initialize adaLN_modulation layers for AdaLN-Zero
        self._initialize_weights()

    def _initialize_weights(self):
        """Zero-initialize final linear layer of adaLN_modulation for training stability."""
        for block in self.transformer_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, timestep: Tensor, conditioning_vec: Tensor) -> Tensor:
        """Predict noise to remove from noisy actions.

        Args:
            x: (B, T, action_dim) noisy action sequences
            timestep: (B,) diffusion timesteps
            conditioning_vec: (B, conditioning_dim) observation features (required)

        Returns:
            (B, T, action_dim) predicted noise
        """
        _, seq_len, _ = x.shape

        timestep_features = self.time_mlp(timestep)  # (B, timestep_embed_dim)

        # conditioning_vec is now required
        cond_features = torch.cat([timestep_features, conditioning_vec], dim=-1)  # (B, cond_dim)

        # Project action sequence to hidden dimension
        hidden_seq = self.input_proj(x)  # (B, T, hidden_size)

        if self.pos_embedding is not None:
            # Add learned positional embeddings
            hidden_seq = hidden_seq + self.pos_embedding[:, :seq_len, :]  # (B, T, hidden_size)

        # Pass through transformer layers with conditioning
        for block in self.transformer_blocks:
            hidden_seq = block(hidden_seq, cond_features)  # (B, T, hidden_size)

        # Project back to action space
        output = self.output_proj(hidden_seq)  # (B, T, action_dim)

        return output
