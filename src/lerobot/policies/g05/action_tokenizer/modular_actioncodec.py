# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional


class BlockDCT(nn.Module):
    """Orthonormal block DCT-II over the action horizon."""

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.basis_cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}

    def basis(self, reference: Tensor) -> Tensor:
        key = (reference.device, reference.dtype)
        if key not in self.basis_cache:
            frequency = torch.arange(self.block_size, device=reference.device, dtype=torch.float32)
            time = torch.arange(self.block_size, device=reference.device, dtype=torch.float32)
            basis = torch.cos(math.pi / self.block_size * (time + 0.5)[None] * frequency[:, None])
            basis[0] *= math.sqrt(1 / self.block_size)
            basis[1:] *= math.sqrt(2 / self.block_size)
            self.basis_cache[key] = basis.to(reference.dtype)
        return self.basis_cache[key]

    def dct(self, values: Tensor) -> Tensor:
        batch_size, horizon, width = values.shape
        padding = (-horizon) % self.block_size
        values = functional.pad(values, (0, 0, 0, padding))
        values = rearrange(
            values, "batch (blocks time) width -> (batch blocks) time width", time=self.block_size
        )
        values = torch.einsum("kt,btw->bkw", self.basis(values), values)
        return rearrange(values, "(batch blocks) time width -> batch (blocks time) width", batch=batch_size)

    def idct(self, values: Tensor, horizon: int) -> Tensor:
        batch_size = values.shape[0]
        values = rearrange(
            values, "batch (blocks time) width -> (batch blocks) time width", time=self.block_size
        )
        values = torch.einsum("tk,bkw->btw", self.basis(values), values)
        values = rearrange(values, "(batch blocks) time width -> batch (blocks time) width", batch=batch_size)
        return values[:, :horizon]


class RotaryEmbedding(nn.Module):
    """Partial rotary embedding used by ActionCodec attention."""

    def __init__(self, dimension: int, base: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.base = base
        self.inverse_frequency_cache: dict[torch.device, Tensor] = {}

    def forward(self, sequence_length: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        positions = torch.arange(sequence_length, device=reference.device, dtype=torch.float32)
        if reference.device not in self.inverse_frequency_cache:
            dimensions = torch.arange(0, self.dimension, 2, device=reference.device).float()
            self.inverse_frequency_cache[reference.device] = 1 / (self.base ** (dimensions / self.dimension))
        frequencies = torch.einsum("t,f->tf", positions, self.inverse_frequency_cache[reference.device])
        angles = torch.cat((frequencies, frequencies), dim=-1).to(reference.dtype)
        return angles.cos()[None, None], angles.sin()[None, None]


def apply_rotary_embedding(query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor) -> tuple[Tensor, Tensor]:
    rotary_dim = cosine.shape[-1]

    def rotate_half(values: Tensor) -> Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    query_rotary = query[..., :rotary_dim]
    key_rotary = key[..., :rotary_dim]
    query = torch.cat((query_rotary * cosine + rotate_half(query_rotary) * sine, query[..., rotary_dim:]), -1)
    key = torch.cat((key_rotary * cosine + rotate_half(key_rotary) * sine, key[..., rotary_dim:]), -1)
    return query, key


class ActionCodecAttention(nn.Module):
    """Self-attention for flattened ActionCodec feature grids."""

    def __init__(
        self, dimension: int, num_heads: int, head_dim: int, use_qk_norm: bool, rope_base: int
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.to_qkv = nn.Linear(dimension, 3 * inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dimension, bias=False)
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = nn.Identity()
        self.rope = RotaryEmbedding(max(head_dim // 2, 32), rope_base)

    def forward(self, hidden_states: Tensor) -> Tensor:
        query, key, value = rearrange(
            self.to_qkv(hidden_states),
            "batch time (qkv heads dim) -> qkv batch heads time dim",
            qkv=3,
            heads=self.num_heads,
            dim=self.head_dim,
        )
        query, key = self.q_norm(query), self.k_norm(key)
        cosine, sine = self.rope(hidden_states.shape[1], hidden_states)
        query, key = apply_rotary_embedding(query, key, cosine, sine)
        attended = functional.scaled_dot_product_attention(query, key, value)
        attended = rearrange(attended, "batch heads time dim -> batch time (heads dim)")
        return self.to_out(attended)


class ActionCodecFeedForward(nn.Module):
    def __init__(self, dimension: int, multiplier: float) -> None:
        super().__init__()
        inner_dim = int(dimension * multiplier)
        self.w_up = nn.Linear(dimension, 2 * inner_dim, bias=False)
        self.w_down = nn.Linear(inner_dim, dimension, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        values, gates = self.w_up(hidden_states).chunk(2, dim=-1)
        return self.w_down(values * functional.gelu(gates))


class ActionCodecTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        head_dim: int,
        ffn_multiplier: float,
        use_layer_scale: bool,
        layer_scale_init: float,
        use_qk_norm: bool,
        rope_base: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, eps=1e-6)
        self.attn = ActionCodecAttention(dimension, num_heads, head_dim, use_qk_norm, rope_base)
        self.norm2 = nn.LayerNorm(dimension, eps=1e-6)
        self.ffn = ActionCodecFeedForward(dimension, ffn_multiplier)
        if use_layer_scale:
            self.ls1 = nn.Parameter(torch.full((dimension,), layer_scale_init))
            self.ls2 = nn.Parameter(torch.full((dimension,), layer_scale_init))
        else:
            self.register_parameter("ls1", None)
            self.register_parameter("ls2", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        attention = self.attn(self.norm1(hidden_states))
        feed_forward_scale = self.ls2 if self.ls2 is not None else 1
        hidden_states = hidden_states + attention * (self.ls1 if self.ls1 is not None else 1)
        return hidden_states + self.ffn(self.norm2(hidden_states)) * feed_forward_scale


def make_transformer_stack(
    depth: int,
    dimension: int,
    num_heads: int,
    head_dim: int,
    ffn_multiplier: float,
    use_layer_scale: bool,
    layer_scale_init: float,
    use_qk_norm: bool,
    rope_base: int,
) -> nn.ModuleList:
    return nn.ModuleList(
        ActionCodecTransformerBlock(
            dimension,
            num_heads,
            head_dim,
            ffn_multiplier,
            use_layer_scale,
            layer_scale_init,
            use_qk_norm,
            rope_base,
        )
        for _ in range(depth)
    )


class ActionCodecDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: list[int], depth: int, config) -> None:
        super().__init__()
        stride_height, stride_width = stride
        if stride_height > 1 or in_channels != out_channels:
            kernel_height = 2 * stride_height if stride_height > 1 else 1
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_height, 1),
                (stride_height, stride_width),
                (kernel_height // 2 - int(stride_height > 1), 0),
            )
        else:
            self.conv = nn.Identity()
        self.transformer_layers = make_transformer_stack(
            depth,
            out_channels,
            config.num_heads,
            config.dim_heads,
            config.ffn_mult,
            config.use_layer_scale,
            config.layer_scale_init,
            config.use_qk_norm,
            config.rope_base,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        height, width = hidden_states.shape[-2:]
        hidden_states = rearrange(
            hidden_states, "batch channels height width -> batch (height width) channels"
        )
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states)
        return rearrange(
            hidden_states,
            "batch (height width) channels -> batch channels height width",
            height=height,
            width=width,
        )


class ActionCodecUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: list[int], depth: int, config) -> None:
        super().__init__()
        self.transformer_layers = make_transformer_stack(
            depth,
            in_channels,
            config.num_heads,
            config.dim_heads,
            config.ffn_mult,
            config.use_layer_scale,
            config.layer_scale_init,
            config.use_qk_norm,
            config.rope_base,
        )
        stride_height, stride_width = stride
        if stride_height > 1 or in_channels != out_channels:
            kernel_height = 2 * stride_height if stride_height > 1 else 1
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (kernel_height, 1),
                (stride_height, stride_width),
                (kernel_height // 2 - int(stride_height > 1), 0),
            )
        else:
            self.conv = nn.Identity()

    def forward(self, hidden_states: Tensor) -> Tensor:
        height, width = hidden_states.shape[-2:]
        hidden_states = rearrange(
            hidden_states, "batch channels height width -> batch (height width) channels"
        )
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states)
        hidden_states = rearrange(
            hidden_states,
            "batch (height width) channels -> batch channels height width",
            height=height,
            width=width,
        )
        return self.conv(hidden_states)


class ActionCodecEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        channels = [config.encoder_channels] + [config.encoder_channels * value for value in config.c_mults]
        self.blocks = nn.ModuleList(
            ActionCodecDownBlock(in_dim, out_dim, stride, depth, config)
            for in_dim, out_dim, stride, depth in zip(
                channels[:-1],
                channels[1:],
                config.strides,
                config.transformer_depths,
                strict=True,
            )
        )
        self.out_proj = nn.Conv2d(channels[-1], config.latent_dim, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.out_proj(hidden_states)


class ActionCodecDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        channels = [config.encoder_channels] + [config.encoder_channels * value for value in config.c_mults]
        self.in_proj = nn.Conv2d(config.latent_dim, channels[-1], 1)
        self.blocks = nn.ModuleList(
            ActionCodecUpBlock(in_dim, out_dim, stride, depth, config)
            for in_dim, out_dim, stride, depth in zip(
                reversed(channels[1:]),
                reversed(channels[:-1]),
                reversed(config.strides),
                reversed(config.transformer_depths),
                strict=True,
            )
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.in_proj(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class EMAVectorQuantizer(nn.Module):
    """Released ActionCodec EMA codebook with inference-time nearest lookup."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_dim, codebook_dim, bias=False)
        self.out_proj = nn.Linear(codebook_dim, input_dim, bias=False)
        self.register_buffer("codebook", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("embed_avg", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("inited", torch.tensor(False))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.in_proj(
            rearrange(hidden_states.float(), "batch channels time -> batch time channels")
        )
        distances = (
            projected.square().sum(-1, keepdim=True)
            - 2 * torch.einsum("btc,vc->btv", projected, self.codebook.float())
            + self.codebook.float().square().sum(-1)
        )
        codes = distances.argmin(-1)
        quantized = functional.embedding(codes, self.codebook.float())
        quantized = self.out_proj(quantized)
        return rearrange(quantized, "batch time channels -> batch channels time").to(hidden_states), codes

    def decode(self, codes: Tensor) -> Tensor:
        quantized = self.out_proj(functional.embedding(codes, self.codebook.float()))
        return rearrange(quantized, "batch time channels -> batch channels time")


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList(
            EMAVectorQuantizer(config.latent_dim, config.codebook_size, config.codebook_dim)
            for _ in range(config.n_codebooks)
        )

    def encode(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        residual = hidden_states
        quantized = torch.zeros_like(hidden_states)
        codes = []
        for quantizer in self.quantizers:
            current, current_codes = quantizer(residual)
            quantized = quantized + current
            residual = residual - current
            codes.append(current_codes)
        return quantized, torch.stack(codes, dim=1)

    def decode(self, codes: Tensor) -> Tensor:
        if not 1 <= codes.shape[1] <= len(self.quantizers):
            raise ValueError("invalid number of residual codebooks")
        return torch.stack(
            [
                quantizer.decode(codes[:, level])
                for level, quantizer in enumerate(self.quantizers[: codes.shape[1]])
            ]
        ).sum(0)
