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


class TransformerBlock(nn.Module):
    """DiT-style transformer block with AdaLN-Zero.

    Official DiT implementation with 6-parameter adaptive layer normalization:
    - shift_msa, scale_msa, gate_msa: for attention block
    - shift_mlp, scale_mlp, gate_mlp: for MLP block

    Reference: https://github.com/facebookresearch/DiT
    """

    def __init__(
        self, hidden_size: int = 128, num_heads: int = 4, num_features: int = 128, dropout: float = 0.0
    ):
        """
        Args:
            hidden_size: Hidden dimension of transformer
            num_heads: Number of attention heads
            num_features: Size of conditioning features
            dropout: Dropout rate
        """
        super().__init__()

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
        attn_out, _ = self.multihead_attn(attn_input, attn_input, attn_input)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP block: norm → modulate → mlp → gate × output → residual
        mlp_input = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        mlp_out = self.mlp(mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class DiffusionTransformer(nn.Module):
    """
    Transformer-based diffusion noise prediction model.
    """

    def __init__(self, config, conditioning_dim: int):
        """Initialize transformer for noise prediction.

        Args:
            config: Multi-Task DiTConfig with transformer parameters
            conditioning_dim: Dimension of concatenated observation features
        """
        super().__init__()

        self.config = config
        self.transformer_config = config.transformer
        self.conditioning_dim = conditioning_dim

        self.action_dim = config.action_feature.shape[0]
        self.horizon = config.horizon
        self.hidden_size = self.transformer_config.hidden_dim
        self.num_layers = self.transformer_config.num_layers
        self.num_heads = self.transformer_config.num_heads
        self.dropout = self.transformer_config.dropout

        self.timestep_embed_dim = self.transformer_config.diffusion_step_embed_dim
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

        if self.transformer_config.use_positional_encoding:
            # Learnable positional embeddings for sequence positions
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
                )
                for _ in range(self.num_layers)
            ]
        )

        # Project back to action dimensions
        self.output_proj = nn.Linear(self.hidden_size, self.action_dim)

        # Zero-initialize adaLN_modulation layers for AdaLN-Zero
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Zero-initializing the final linear layer of adaLN_modulation in each block improves training stability
        """
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
