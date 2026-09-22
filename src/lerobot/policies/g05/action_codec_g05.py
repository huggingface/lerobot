#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

"""Native G0.5 ActionCodec: the discrete action tokenizer's encoder, quantizer, and decoder.

Split out of `modeling_g05.py`: the codec is a self-contained sub-model and the only
symbol the policy needs from it is `G05NativeActionCodec`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from torch import Tensor, distributed, nn


class _BlockDCT(nn.Module):
    """Blockwise DCT and its inverse over an action horizon."""

    def __init__(self, block_size: int) -> None:
        """Store the transform block size."""
        super().__init__()
        self.block_size = block_size
        frequency = torch.arange(block_size, dtype=torch.float32)
        time = torch.arange(block_size, dtype=torch.float32)
        basis = torch.cos(math.pi / block_size * (time + 0.5).unsqueeze(0) * frequency.unsqueeze(1))
        basis[0] *= math.sqrt(1 / block_size)
        basis[1:] *= math.sqrt(2 / block_size)
        self.register_buffer("basis", basis, persistent=False)

    def dct(self, values: Tensor) -> Tensor:
        """Apply the blockwise DCT, padding the horizon to a whole block."""
        batch, horizon, dimension = values.shape
        pad = (-horizon) % self.block_size
        if pad:
            values = functional.pad(values, (0, 0, 0, pad))
        blocks = values.shape[1] // self.block_size
        values = values.reshape(batch * blocks, self.block_size, dimension)
        transformed = torch.einsum("kn,bnd->bkd", self.basis.to(values), values)
        return transformed.reshape(batch, blocks * self.block_size, dimension)

    def idct(self, values: Tensor, horizon: int) -> Tensor:
        """Invert the blockwise DCT."""
        batch, padded_horizon, dimension = values.shape
        blocks = padded_horizon // self.block_size
        values = values.reshape(batch * blocks, self.block_size, dimension)
        restored = torch.einsum("nk,bkd->bnd", self.basis.to(values), values)
        return restored.reshape(batch, padded_horizon, dimension)[:, :horizon]


def _rotate_half(values: Tensor) -> Tensor:
    """Rotate the halves of the last dimension for rotary embeddings."""
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class _CodecAttention(nn.Module):
    """Self-attention with rotary position embeddings for the codec."""

    def __init__(self, dimension: int, num_heads: int, head_dim: int, rope_base: int) -> None:
        """Build the projections and the rotary cache."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.to_qkv = nn.Linear(dimension, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dimension, bias=False)
        self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        rope_dim = max(head_dim // 2, 32)
        inverse = 1 / (rope_base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
        self.register_buffer("_inverse_frequency", inverse, persistent=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Attend over the sequence with rotary positions."""
        batch, sequence_length, _ = hidden_states.shape
        query, key, value = self.to_qkv(hidden_states).chunk(3, dim=-1)

        def heads(values: Tensor) -> Tensor:
            """Split the last dimension into attention heads."""
            return values.view(batch, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = (heads(values) for values in (query, key, value))
        query, key = self.q_norm(query), self.k_norm(key)
        time = torch.arange(sequence_length, device=hidden_states.device, dtype=torch.float32)
        phase = torch.outer(time, self._inverse_frequency.to(hidden_states.device))
        phase = torch.cat((phase, phase), dim=-1).to(hidden_states.dtype)[None, None]
        cosine, sine = phase.cos(), phase.sin()
        rotary_dim = cosine.shape[-1]
        query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
        key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
        query = torch.cat((query_rotary * cosine + _rotate_half(query_rotary) * sine, query_pass), dim=-1)
        key = torch.cat((key_rotary * cosine + _rotate_half(key_rotary) * sine, key_pass), dim=-1)
        attended = functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, sequence_length, -1)
        return self.to_out(attended)


class _CodecFFN(nn.Module):
    """Gated GELU feed-forward block for the codec."""

    def __init__(self, dimension: int, multiplier: float) -> None:
        """Build the up and down projections."""
        super().__init__()
        inner_dim = int(dimension * multiplier)
        self.w_up = nn.Linear(dimension, inner_dim * 2, bias=False)
        self.w_down = nn.Linear(inner_dim, dimension, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the gated feed-forward."""
        value, gate = self.w_up(hidden_states).chunk(2, dim=-1)
        return self.w_down(value * functional.gelu(gate))


class _CodecTransformerLayer(nn.Module):
    """Pre-norm transformer layer with learned layer scales."""

    def __init__(self, dimension: int, config: Mapping[str, Any]) -> None:
        """Build the attention, feed-forward and layer scales."""
        super().__init__()
        layer_scale_init = float(config.get("layer_scale_init", 1.0))
        self.ls1 = nn.Parameter(torch.full((dimension,), layer_scale_init))
        self.ls2 = nn.Parameter(torch.full((dimension,), layer_scale_init))
        self.norm1 = nn.LayerNorm(dimension, eps=1e-6)
        self.attn = _CodecAttention(
            dimension,
            int(config["num_heads"]),
            int(config["dim_heads"]),
            int(config["rope_base"]),
        )
        self.norm2 = nn.LayerNorm(dimension, eps=1e-6)
        self.ffn = _CodecFFN(dimension, float(config["ffn_mult"]))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply attention and feed-forward with residual scaling."""
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states)) * self.ls1
        return hidden_states + self.ffn(self.norm2(hidden_states)) * self.ls2


class _CodecDownBlock(nn.Module):
    """Strided convolution and transformer layers that downsample the grid."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: tuple[int, int],
        depth: int,
        config: Mapping[str, Any],
    ) -> None:
        """Build the strided convolution and the transformer layers."""
        super().__init__()
        stride_h, stride_a = stride
        if stride_h > 1 or input_channels != output_channels:
            kernel_h = 2 * stride_h if stride_h > 1 else 1
            self.conv = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=(kernel_h, 1),
                stride=(stride_h, stride_a),
                padding=(kernel_h // 2 - int(stride_h > 1), 0),
            )
        else:
            self.conv = nn.Identity()
        self.transformer_layers = nn.ModuleList(
            [_CodecTransformerLayer(output_channels, config) for _ in range(depth)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Downsample the grid and mix it with the transformer layers."""
        hidden_states = self.conv(hidden_states)
        batch, channels, height, action_dim = hidden_states.shape
        sequence = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * action_dim, channels)
        for layer in self.transformer_layers:
            sequence = layer(sequence)
        return sequence.reshape(batch, height, action_dim, channels).permute(0, 3, 1, 2)


class _CodecUpBlock(nn.Module):
    """Transformer layers and a transposed convolution that upsample the grid."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: tuple[int, int],
        depth: int,
        config: Mapping[str, Any],
    ) -> None:
        """Build the transformer layers and the transposed convolution."""
        super().__init__()
        self.transformer_layers = nn.ModuleList(
            [_CodecTransformerLayer(input_channels, config) for _ in range(depth)]
        )
        stride_h, stride_a = stride
        if stride_h > 1 or input_channels != output_channels:
            kernel_h = 2 * stride_h if stride_h > 1 else 1
            self.conv = nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=(kernel_h, 1),
                stride=(stride_h, stride_a),
                padding=(kernel_h // 2 - int(stride_h > 1), 0),
            )
        else:
            self.conv = nn.Identity()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Mix the grid with the transformer layers and upsample it."""
        batch, channels, height, action_dim = hidden_states.shape
        sequence = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * action_dim, channels)
        for layer in self.transformer_layers:
            sequence = layer(sequence)
        hidden_states = sequence.reshape(batch, height, action_dim, channels).permute(0, 3, 1, 2)
        return self.conv(hidden_states)


class _CodecEncoder(nn.Module):
    """Encode an action grid down to codec latents."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Build the downsampling blocks."""
        super().__init__()
        base = int(config["encoder_channels"])
        channel_dims = [base * int(multiplier) for multiplier in config["c_mults"]]
        dims = [base] + channel_dims
        self.blocks = nn.ModuleList(
            [
                _CodecDownBlock(
                    dims[index],
                    dims[index + 1],
                    tuple(stride),
                    int(config["transformer_depths"][index]),
                    config,
                )
                for index, stride in enumerate(config["strides"])
            ]
        )
        self.out_proj = nn.Conv2d(dims[-1], int(config["latent_dim"]), kernel_size=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Run the downsampling blocks."""
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.out_proj(hidden_states)


class _CodecDecoder(nn.Module):
    """Decode codec latents back to an action grid."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Build the input projection and the upsampling blocks."""
        super().__init__()
        base = int(config["encoder_channels"])
        channel_dims = [base * int(multiplier) for multiplier in config["c_mults"]]
        dims = [base] + channel_dims
        self.in_proj = nn.Conv2d(int(config["latent_dim"]), dims[-1], kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _CodecUpBlock(
                    input_channels,
                    output_channels,
                    tuple(stride),
                    int(depth),
                    config,
                )
                for stride, depth, input_channels, output_channels in zip(
                    reversed(config["strides"]),
                    reversed(config["transformer_depths"]),
                    reversed(dims[1:]),
                    reversed(dims[:-1]),
                    strict=True,
                )
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Run the input projection and the upsampling blocks."""
        hidden_states = self.in_proj(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def _sample_codec_vectors(samples: Tensor, count: int) -> Tensor:
    """Sample or resample vectors to seed a codebook."""
    if samples.shape[0] >= count:
        indices = torch.randperm(samples.shape[0], device=samples.device)[:count]
    else:
        indices = torch.randint(0, samples.shape[0], (count,), device=samples.device)
    return samples[indices].float()


def _codec_kmeans(samples: Tensor, num_clusters: int, num_iterations: int = 10) -> tuple[Tensor, Tensor]:
    """Fit codebook centroids with k-means."""
    dimension = samples.shape[-1]
    means = _sample_codec_vectors(samples, num_clusters)
    for _ in range(num_iterations):
        distances = (
            samples.float().square().sum(1, keepdim=True)
            - 2 * samples.float() @ means.t()
            + means.float().square().sum(1, keepdim=True).t()
        )
        buckets = distances.argmin(-1)
        counts = torch.bincount(buckets, minlength=num_clusters)
        safe_counts = counts.masked_fill(counts == 0, 1)
        new_means = torch.zeros(num_clusters, dimension, device=samples.device)
        new_means.scatter_add_(0, buckets[:, None].expand(-1, dimension), samples.float())
        new_means = new_means / safe_counts.float()[:, None]
        means = torch.where((counts == 0)[:, None], means, new_means)
    distances = (
        samples.float().square().sum(1, keepdim=True)
        - 2 * samples.float() @ means.t()
        + means.float().square().sum(1, keepdim=True).t()
    )
    counts = torch.bincount(distances.argmin(-1), minlength=num_clusters).float()
    return means, counts


def _codec_ema_inplace(moving_average: Tensor, value: Tensor, decay: float) -> None:
    """Update a moving average in place."""
    moving_average.data.mul_(decay).add_(value.float(), alpha=1 - decay)


def _codec_rotation_trick(encoded: Tensor, quantized: Tensor) -> Tensor:
    """Pass gradients through quantization with the rotation trick."""
    encoded_float = encoded.float()
    quantized_float = quantized.float()
    encoded_norm = encoded_float.norm(dim=1, keepdim=True).clamp(min=1e-8)
    quantized_norm = quantized_float.norm(dim=1, keepdim=True).clamp(min=1e-8)
    rotated = encoded_float / encoded_norm * quantized_norm
    return (quantized_float - rotated).detach() + rotated


def _time_shift_positive(actions: Tensor) -> Tensor:
    """Shift actions forward one step, repeating the first."""
    shifted = torch.zeros_like(actions)
    shifted[:, 0] = actions[:, 0]
    shifted[:, 1:] = actions[:, :-1]
    return shifted


class _ActionTimeContrastiveLoss(nn.Module):
    """Contrastive loss between action and time embeddings."""

    def __init__(self, mode: str, temperature_init: float, bias_init: float) -> None:
        """Store the loss mode and the learnable temperature."""
        super().__init__()
        if mode not in {"siglip", "infonce"}:
            raise ValueError(f"unsupported action-time contrastive mode: {mode!r}")
        self.mode = mode
        if mode == "siglip":
            self.logit_scale = nn.Parameter(torch.tensor(float(temperature_init)).log())
            self.logit_bias = nn.Parameter(torch.tensor(float(bias_init)))
        else:
            self.register_buffer("temperature", torch.tensor(float(temperature_init)))

    @staticmethod
    def _flatten(hidden_states: Tensor) -> Tensor:
        """Flatten and L2-normalize the hidden states."""
        return functional.normalize(hidden_states.flatten(1), dim=-1)

    def forward(self, anchor_states: Tensor, positive_states: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Score anchors against their positives."""
        anchors = self._flatten(anchor_states)
        positives = self._flatten(positive_states)
        batch_size = anchors.shape[0]
        if positives.shape[0] % batch_size:
            raise ValueError("positive batch must be an integer multiple of anchor batch")
        if self.mode == "siglip":
            logits = anchors @ positives.t() * self.logit_scale.exp() + self.logit_bias
            labels = torch.zeros_like(logits)
            row_indices = torch.arange(batch_size, device=logits.device)
            for positive_index in range(positives.shape[0] // batch_size):
                labels[row_indices, row_indices + positive_index * batch_size] = 1
            signed_labels = 2 * labels - 1
            loss = -functional.logsigmoid(signed_labels * logits).mean()
            positive_logits = logits[labels == 1]
            negative_logits = logits[labels == 0]
            average_negative = negative_logits.mean() if negative_logits.numel() else logits.new_zeros(())
            return loss, {
                "consist/loss": loss.detach(),
                "contrastive/loss": loss.detach(),
                "contrastive/temperature": self.logit_scale.exp().detach(),
                "contrastive/logit_bias": self.logit_bias.detach(),
                "contrastive/avg_pos_sim": positive_logits.mean().detach(),
                "contrastive/avg_neg_sim": average_negative.detach(),
            }

        if batch_size < 2:
            raise ValueError("action-time 'infonce' mode requires batch_size >= 2")
        shift = torch.randint(1, batch_size, (1,), device=anchors.device).item()
        negatives = anchors[(torch.arange(batch_size, device=anchors.device) + shift) % batch_size]
        losses = []
        metrics: dict[str, Tensor] = {}
        for positive_index in range(positives.shape[0] // batch_size):
            positive = positives[positive_index * batch_size : (positive_index + 1) * batch_size]
            positive_similarity = (anchors * positive).sum(-1)
            negative_similarity = (anchors * negatives).sum(-1)
            losses.append(
                -functional.logsigmoid(self.temperature * (positive_similarity - negative_similarity)).mean()
            )
            metrics[f"contrastive/pos_sim_{positive_index}"] = positive_similarity.mean().detach()
            metrics[f"contrastive/neg_sim_{positive_index}"] = negative_similarity.mean().detach()
        loss = torch.stack(losses).mean()
        metrics.update(
            {
                "consist/loss": loss.detach(),
                "contrastive/loss": loss.detach(),
                "contrastive/temperature": self.temperature.detach(),
            }
        )
        return loss, metrics


def _codec_consistency_loss(
    residuals: list[Tensor],
    level_codes: list[Tensor],
    original_batch_size: int,
    layer_weights: list[float],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted consistency loss across the residual quantizer levels."""
    if not residuals or len(residuals) != len(level_codes) or len(level_codes) != len(layer_weights):
        raise ValueError("consistency residuals, codes, and layer weights must have equal nonzero lengths")
    device = residuals[0].device
    sequence_length = residuals[0].shape[-1]
    prefix_match = torch.ones(original_batch_size, sequence_length, device=device)
    total_loss = torch.tensor(0.0, device=device)
    hamming = 0.0
    metrics: dict[str, Tensor] = {}
    for level, (level_residuals, codes, weight) in enumerate(
        zip(residuals, level_codes, layer_weights, strict=True)
    ):
        original_residuals = level_residuals[:original_batch_size]
        positive_residuals = level_residuals[original_batch_size:]
        original_codes = codes[:original_batch_size]
        positive_codes = codes[original_batch_size:]
        diverged = (original_codes != positive_codes).float().detach()
        token_change_rate = diverged.mean()
        hamming += float(token_change_rate.item())
        residual_difference = (positive_residuals - original_residuals.detach()).norm(dim=1)
        active = prefix_match * diverged
        layer_loss = (active * residual_difference).mean()
        total_loss = total_loss + float(weight) * layer_loss
        metrics[f"consist/tcr_layer_{level}"] = token_change_rate.detach()
        metrics[f"consist/active_frac_{level}"] = active.mean().detach()
        metrics[f"consist/loss_layer_{level}"] = layer_loss.detach()
        prefix_match = prefix_match * (original_codes == positive_codes).float().detach()
    metrics["consist/loss"] = total_loss.detach()
    metrics["consist/hamming_dist"] = torch.tensor(hamming * sequence_length, device=device)
    return total_loss, metrics


class _CodecQuantizer(nn.Module):
    """Single vector-quantizer codebook with EMA updates."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Build the projections and the codebook buffers."""
        super().__init__()
        input_dim = int(config["latent_dim"])
        codebook_dim = int(config["codebook_dim"])
        codebook_size = int(config["codebook_size"])
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = float(config.get("ema_decay", 0.95))
        self.threshold_ema_dead = float(config.get("threshold_ema_dead", 2.0))
        self.use_rotation_trick = bool(config.get("use_rotation_trick", False))
        self.epsilon = 1e-5
        self.in_proj = nn.Linear(input_dim, codebook_dim, bias=False)
        self.out_proj = nn.Linear(codebook_dim, input_dim, bias=False)
        self.register_buffer("codebook", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("embed_avg", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("inited", torch.tensor(False))

    def _initialize_codebook(self, encodings: Tensor) -> None:
        """Seed the codebook from the first batch."""
        if self.inited.item():
            return
        if not distributed.is_initialized() or distributed.get_rank() == 0:
            means, counts = _codec_kmeans(encodings.float(), self.codebook_size)
        else:
            means = torch.zeros(self.codebook_size, self.codebook_dim, device=encodings.device)
            counts = torch.zeros(self.codebook_size, device=encodings.device)
        if distributed.is_initialized():
            distributed.broadcast(means, src=0)
            distributed.broadcast(counts, src=0)
        self.codebook.copy_(means)
        self.embed_avg.copy_(means)
        self.cluster_size.copy_(counts)
        self.inited.fill_(True)

    def _update_codebook(self, encodings: Tensor, one_hot_codes: Tensor) -> None:
        """Apply the EMA codebook update."""
        cluster_size = one_hot_codes.sum(0)
        embed_sum = encodings.t() @ one_hot_codes
        if distributed.is_initialized():
            distributed.all_reduce(cluster_size, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(embed_sum, op=distributed.ReduceOp.SUM)
        _codec_ema_inplace(self.cluster_size, cluster_size, self.decay)
        _codec_ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.epsilon) / (total + self.codebook_size * self.epsilon) * total
        self.codebook.copy_((self.embed_avg / smoothed[:, None]).float())

    def _replace_dead_codes(self, encodings: Tensor) -> None:
        """Resample codebook entries that stopped being used."""
        if self.threshold_ema_dead <= 0:
            return
        dead = self.cluster_size < self.threshold_ema_dead
        if not dead.any():
            return
        count = int(dead.sum().item())
        if not distributed.is_initialized() or distributed.get_rank() == 0:
            replacements = _sample_codec_vectors(encodings.float(), count)
        else:
            replacements = torch.zeros(count, self.codebook_dim, device=encodings.device)
        if distributed.is_initialized():
            distributed.broadcast(replacements, src=0)
        self.codebook[dead] = replacements.to(self.codebook.dtype)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize the input and return its codes and losses."""
        original_dtype = values.dtype
        projected = self.in_proj(values.float().transpose(1, 2)).transpose(1, 2)
        encodings = projected.transpose(1, 2).reshape(-1, self.codebook_dim)
        if not torch.compiler.is_compiling() and not self.inited.item():
            self._initialize_codebook(encodings.detach())
        codebook = self.codebook.float()
        distances = (
            encodings.square().sum(dim=1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.square().sum(dim=1)[None]
        )
        flat_codes = distances.argmin(dim=-1)
        codes = flat_codes.reshape(values.shape[0], values.shape[2])
        quantized = (
            functional.embedding(flat_codes, codebook)
            .reshape(values.shape[0], values.shape[2], self.codebook_dim)
            .transpose(1, 2)
        )
        inference_fast_path = not self.training and not torch.is_grad_enabled()
        commitment_loss = (
            torch.zeros(values.shape[0], device=values.device)
            if inference_fast_path
            else functional.mse_loss(projected, quantized.detach(), reduction="none").mean((1, 2))
        )
        if self.training and torch.is_grad_enabled():
            one_hot_codes = functional.one_hot(flat_codes, self.codebook_size).float()
            self._update_codebook(encodings.detach(), one_hot_codes)
            self._replace_dead_codes(encodings.detach())
        if inference_fast_path:
            straight_through = quantized
        elif self.use_rotation_trick:
            straight_through = _codec_rotation_trick(projected, quantized)
        else:
            straight_through = (quantized - projected).detach() + projected
        output = self.out_proj(straight_through.transpose(1, 2)).transpose(1, 2)
        return output.to(original_dtype), commitment_loss, codes

    def encode(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Return the quantized values and their codes."""
        quantized, _, codes = self(values)
        return quantized, codes

    def decode_codes(self, codes: Tensor) -> Tensor:
        """Look codes back up in the codebook."""
        return self.out_proj(functional.embedding(codes, self.codebook.float())).transpose(1, 2)


class _ResidualCodecQuantizer(nn.Module):
    """Stack of codebooks quantizing successive residuals."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Build the per-level quantizers."""
        super().__init__()
        self.n_codebooks = int(config["n_codebooks"])
        self.quantizer_dropout = float(config.get("quantizer_dropout", 0.5))
        self.quantizers = nn.ModuleList([_CodecQuantizer(config) for _ in range(self.n_codebooks)])

    def forward(
        self, values: Tensor, *, return_level_data: bool = False
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, list[Tensor], list[Tensor]]:
        """Quantize the residual at each level in turn."""
        batch_size = values.shape[0]
        if self.training:
            levels_per_sample = torch.full((batch_size,), float(self.n_codebooks + 1), device=values.device)
            dropout_mask = torch.rand(batch_size, device=values.device) < self.quantizer_dropout
            sampled_levels = torch.randint(1, self.n_codebooks + 1, (batch_size,), device=values.device)
            levels_per_sample[dropout_mask] = sampled_levels[dropout_mask].float()
        else:
            levels_per_sample = torch.full((batch_size,), self.n_codebooks + 0.5, device=values.device)
        residual = values
        quantized = torch.zeros_like(values)
        codes = []
        commitment_loss = values.new_zeros(())
        consistency_residual = values
        consistency_residuals = []
        level_codes = []
        for level, quantizer in enumerate(self.quantizers):
            active = (level < levels_per_sample).float()
            if return_level_data:
                consistency_residuals.append(
                    quantizer.in_proj(consistency_residual.float().transpose(1, 2)).transpose(1, 2)
                )
            current, current_commitment, current_codes = quantizer(residual)
            quantized = quantized + current * active[:, None, None]
            residual = residual - current
            commitment_loss = commitment_loss + (current_commitment * active).mean()
            codes.append(current_codes)
            if return_level_data:
                level_codes.append(current_codes)
                consistency_residual = consistency_residual - current.detach()
        stacked_codes = torch.stack(codes, dim=1)
        if return_level_data:
            return quantized, stacked_codes, commitment_loss, consistency_residuals, level_codes
        return quantized, stacked_codes, commitment_loss

    def encode(self, values: Tensor) -> Tensor:
        """Return the per-level codes."""
        _, codes, _ = self(values)
        return codes

    def from_codes(self, codes: Tensor) -> Tensor:
        """Rebuild values from their per-level codes."""
        if not 1 <= codes.shape[1] <= len(self.quantizers):
            raise ValueError("invalid number of residual codebooks")
        quantized = torch.zeros(
            codes.shape[0],
            self.quantizers[0].input_dim,
            codes.shape[-1],
            dtype=self.quantizers[0].codebook.dtype,
            device=codes.device,
        )
        for level, quantizer in enumerate(self.quantizers[: codes.shape[1]]):
            quantized = quantized + quantizer.decode_codes(codes[:, level])
        return quantized


class _ActionCodecModel(nn.Module):
    """Encoder, quantizer and decoder for the action codec."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Build the encoder, quantizer and decoder."""
        super().__init__()
        self.config = dict(config)
        self.block_dct = (
            _BlockDCT(int(config["block_dct_block_size"]))
            if bool(config.get("use_block_dct", False))
            else None
        )
        self.conv_in = nn.Conv2d(
            int(config["horizon_patch_size"]),
            int(config["encoder_channels"]),
            kernel_size=(1, int(config["conv_in_action_kernel"])),
        )
        self.encoder = _CodecEncoder(config)
        self.rvq = _ResidualCodecQuantizer(config)
        self.action_time_contrastive_loss: _ActionTimeContrastiveLoss | None = None
        if (
            float(config.get("consistency_loss_weight", 0.0)) > 0
            and str(config.get("consistency_loss_type", "action_time_contrastive"))
            == "action_time_contrastive"
        ):
            self.action_time_contrastive_loss = _ActionTimeContrastiveLoss(
                mode=str(config.get("action_time_contrastive_mode", "siglip")),
                temperature_init=float(config.get("action_time_contrastive_temperature_init", 0.07)),
                bias_init=float(config.get("action_time_contrastive_bias_init", -10.0)),
            )
        self.decoder = _CodecDecoder(config)
        self.conv_out = nn.ConvTranspose2d(
            int(config["encoder_channels"]),
            int(config["horizon_patch_size"]),
            kernel_size=(1, int(config["conv_in_action_kernel"])),
        )

    @property
    def code_h(self) -> int:
        """Code grid height after the configured strides."""
        height = int(self.config["horizon"]) // int(self.config["horizon_patch_size"])
        for stride_h, _ in self.config["strides"]:
            height //= int(stride_h)
        return height

    @property
    def code_a(self) -> int:
        """Code grid width along the action dimension."""
        return int(self.config["max_component_dim"]) - int(self.config["conv_in_action_kernel"]) + 1

    def _normalize_components(self, components: Mapping[str, Tensor]) -> tuple[list[str], Tensor, int]:
        """Stack the named action components into one padded tensor."""
        if not components:
            raise ValueError("ActionCodec requires at least one action component")
        names = list(components)
        batch_size = components[names[0]].shape[0]
        horizon = int(self.config["horizon"])
        maximum = int(self.config["max_component_dim"])
        normalized = []
        for name in names:
            values = components[name].float()
            if values.ndim != 3 or values.shape[0] != batch_size:
                raise ValueError(f"component {name!r} must have shape [B,T,D] with a shared batch size")
            values = values[:, :horizon, :maximum]
            values = functional.pad(
                values,
                (0, maximum - values.shape[-1], 0, horizon - values.shape[-2]),
            )
            normalized.append(values)
        return names, torch.cat(normalized), batch_size

    def _encode_tensor(
        self,
        values: Tensor,
        *,
        return_level_data: bool = False,
        return_encoder_hidden: bool = False,
    ) -> tuple[Any, ...]:
        """Encode a stacked action tensor to codes."""
        if self.block_dct is not None:
            values = self.block_dct.dct(values)
        patch = int(self.config["horizon_patch_size"])
        hidden_states = values.reshape(values.shape[0], -1, patch, values.shape[-1]).transpose(1, 2)
        hidden_states = self.encoder(self.conv_in(hidden_states)).flatten(2)
        quantized = self.rvq(hidden_states, return_level_data=return_level_data)
        if return_encoder_hidden:
            return (*quantized, hidden_states)
        return quantized

    def _decode_tensor(self, hidden_states: Tensor) -> Tensor:
        """Decode latents back to a stacked action tensor."""
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.code_h,
            self.code_a,
        )
        values = self.conv_out(self.decoder(hidden_states))
        values = values.transpose(1, 2).reshape(values.shape[0], -1, values.shape[-1])
        if self.block_dct is not None:
            values = self.block_dct.idct(values, int(self.config["horizon"]))
        return values[:, : int(self.config["horizon"])]

    def encode(self, components: dict[str, Tensor]) -> dict[str, Tensor]:
        """Encode named action components to codes."""
        names, values, batch_size = self._normalize_components(components)
        _, codes, _ = self._encode_tensor(values)
        return {
            name: codes[index * batch_size : (index + 1) * batch_size] for index, name in enumerate(names)
        }

    def decode(self, components: dict[str, Tensor], dimensions: Mapping[str, int]) -> dict[str, Tensor]:
        """Decode codes back to named action components."""
        names = list(components)
        batch_size = next(iter(components.values())).shape[0]
        codes = torch.cat([components[name] for name in names], dim=0)
        decoded = self._decode_tensor(self.rvq.from_codes(codes))
        return {
            name: decoded[index * batch_size : (index + 1) * batch_size, :, : dimensions[name]]
            for index, name in enumerate(names)
        }

    def forward(
        self,
        components: dict[str, Tensor],
        d_original: dict[str, int] | None = None,
        x_pos_dict: dict[str, Tensor] | None = None,
        layer_weights: list[float] | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Run reconstruction, RVQ commitment, and optional consistency training losses."""

        names, values, batch_size = self._normalize_components(components)
        target = values.clone()
        original_dims = d_original or {name: components[name].shape[-1] for name in names}
        packed_batch_size = batch_size * len(names)
        consistency_type = str(self.config.get("consistency_loss_type", "action_time_contrastive"))
        consistency_weight = float(self.config.get("consistency_loss_weight", 0.0))
        use_consistency = consistency_weight > 0
        if x_pos_dict is None and use_consistency and consistency_type == "action_time_contrastive":
            x_pos_dict = {name: _time_shift_positive(components[name]) for name in names}

        if x_pos_dict is None:
            quantized, packed_codes, commitment_loss = self._encode_tensor(values)
            consistency_residuals = level_codes = encoder_hidden = None
        else:
            if set(x_pos_dict) != set(names):
                raise ValueError("x_pos_dict must contain exactly the same keys as components")
            _, positive_values, positive_batch_size = self._normalize_components(x_pos_dict)
            if positive_batch_size != batch_size:
                raise ValueError("x_pos_dict must use the same batch size as components")
            return_level_data = use_consistency and consistency_type == "token_residual"
            return_encoder_hidden = use_consistency and consistency_type == "action_time_contrastive"
            encoded = self._encode_tensor(
                torch.cat((values, positive_values)),
                return_level_data=return_level_data,
                return_encoder_hidden=return_encoder_hidden,
            )
            if return_level_data and return_encoder_hidden:
                (
                    all_quantized,
                    all_codes,
                    commitment_loss,
                    consistency_residuals,
                    level_codes,
                    encoder_hidden,
                ) = encoded
            elif return_level_data:
                all_quantized, all_codes, commitment_loss, consistency_residuals, level_codes = encoded
                encoder_hidden = None
            elif return_encoder_hidden:
                all_quantized, all_codes, commitment_loss, encoder_hidden = encoded
                consistency_residuals = level_codes = None
            else:
                all_quantized, all_codes, commitment_loss = encoded
                consistency_residuals = level_codes = encoder_hidden = None
            quantized = all_quantized[:packed_batch_size]
            packed_codes = all_codes[:packed_batch_size]

        reconstructed = self._decode_tensor(quantized)
        reconstruction_loss = functional.mse_loss(reconstructed, target)
        loss = float(self.config.get("reconstruction_loss_weight", 1.0)) * reconstruction_loss
        loss = loss + float(self.config.get("commitment_loss_weight", 0.25)) * commitment_loss
        loss_dict: dict[str, Tensor] = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
        }
        for index, name in enumerate(names):
            dimension = original_dims.get(name, int(self.config["max_component_dim"]))
            component_slice = slice(index * batch_size, (index + 1) * batch_size)
            loss_dict[f"recon/{name}"] = functional.mse_loss(
                reconstructed[component_slice, :, :dimension],
                target[component_slice, :, :dimension],
            ).detach()

        for level, quantizer in enumerate(self.rvq.quantizers):
            cluster_size = quantizer.cluster_size.float()
            total = cluster_size.sum()
            if total > 0:
                probabilities = cluster_size / total
                perplexity = torch.exp(-(probabilities * torch.log(probabilities + 1e-10)).sum())
                utilization = (cluster_size >= quantizer.threshold_ema_dead).float().mean()
            else:
                perplexity = cluster_size.new_tensor(1.0)
                utilization = cluster_size.new_tensor(0.0)
            loss_dict[f"codebook/perplexity_l{level}"] = perplexity.detach()
            loss_dict[f"codebook/utilization_l{level}"] = utilization.detach()

        if x_pos_dict is not None and use_consistency and consistency_type == "token_residual":
            if consistency_residuals is None or level_codes is None:
                raise RuntimeError("token-residual consistency state was not returned")
            effective_layer_weights = layer_weights or [1.0] * int(self.config["n_codebooks"])
            consistency_loss, consistency_metrics = _codec_consistency_loss(
                consistency_residuals,
                level_codes,
                packed_batch_size,
                effective_layer_weights,
            )
            loss = loss + consistency_weight * consistency_loss
            loss_dict["loss"] = loss
            loss_dict.update(consistency_metrics)
        elif x_pos_dict is not None and use_consistency and consistency_type == "action_time_contrastive":
            if self.action_time_contrastive_loss is None or encoder_hidden is None:
                raise RuntimeError("action-time contrastive loss was not initialized")
            consistency_loss, consistency_metrics = self.action_time_contrastive_loss(
                encoder_hidden[:packed_batch_size], encoder_hidden[packed_batch_size:]
            )
            loss = loss + consistency_weight * consistency_loss
            loss_dict["loss"] = loss
            loss_dict.update(consistency_metrics)

        return {
            "loss": loss,
            "reconstructions": {
                name: reconstructed[
                    index * batch_size : (index + 1) * batch_size,
                    :,
                    : original_dims.get(name, int(self.config["max_component_dim"])),
                ]
                for index, name in enumerate(names)
            },
            "codes": {
                name: packed_codes[index * batch_size : (index + 1) * batch_size]
                for index, name in enumerate(names)
            },
            "loss_dict": loss_dict,
        }


class _NativeCodecModule(nn.Module):
    """Module hierarchy matching ``action_tokenizer.pt`` exactly."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Wrap the action codec model as a module."""
        super().__init__()
        self.model = _ActionCodecModel(config)


class _BinarySequenceCodec:
    """Encode run-length-constrained bit sequences as tokens."""

    def __init__(self, sequence_length: int, min_block_length: int, vocab_size: int) -> None:
        """Store the sequence, block and vocabulary sizes."""
        self.sequence_length = sequence_length
        self.min_block_length = min_block_length
        self.vocab_size = vocab_size
        self._count_cache: dict[tuple[int, int, int, bool], int] = {}
        self.num_sequences = self._count(sequence_length, -1, 0, True)
        self.num_tokens = max(1, math.ceil(math.log(self.num_sequences, vocab_size)))

    def _count(self, remaining: int, last: int, run_length: int, first: bool) -> int:
        """Count valid completions of a partial sequence, memoized."""
        cache_key = (remaining, last, run_length, first)
        if cache_key in self._count_cache:
            return self._count_cache[cache_key]
        if remaining == 0:
            return 1
        total = 0
        for bit in (0, 1):
            if last == -1 or bit == last:
                total += self._count(
                    remaining - 1,
                    bit,
                    min(run_length + 1, self.min_block_length + 1),
                    first,
                )
            elif first or run_length > self.min_block_length:
                total += self._count(remaining - 1, bit, 1, False)
        self._count_cache[cache_key] = total
        return total

    def _repair(self, bits: list[int]) -> list[int]:
        """Fix runs shorter than the minimum block length."""
        bits = bits.copy()
        while True:
            runs = []
            start = 0
            for index in range(1, len(bits)):
                if bits[index] != bits[index - 1]:
                    runs.append((bits[start], start, index))
                    start = index
            runs.append((bits[start], start, len(bits)))
            invalid = next(
                (
                    (start, stop, runs[index - 1][0])
                    for index, (_, start, stop) in enumerate(runs[1:-1], start=1)
                    if stop - start <= self.min_block_length
                ),
                None,
            )
            if invalid is None:
                return bits
            start, stop, value = invalid
            bits[start:stop] = [value] * (stop - start)

    def _zero_completions(self, remaining: int, last: int, run: int, first: bool) -> int:
        """Count the completions that start with a zero."""
        if last in (-1, 0):
            return self._count(
                remaining,
                0,
                1 if last == -1 else min(run + 1, self.min_block_length + 1),
                first,
            )
        return self._count(remaining, 0, 1, False) if first or run > self.min_block_length else 0

    def encode(self, values: Tensor, threshold: float) -> Tensor:
        """Encode bit rows to tokens."""
        output = []
        for row in values:
            bits = self._repair([int(value >= threshold) for value in row.tolist()])
            rank, last, run, first = 0, -1, 0, True
            for position, bit in enumerate(bits):
                remaining = len(bits) - position - 1
                if bit:
                    rank += self._zero_completions(remaining, last, run, first)
                if last == -1:
                    last, run = bit, 1
                elif bit == last:
                    run = min(run + 1, self.min_block_length + 1)
                else:
                    last, run, first = bit, 1, False
            tokens = []
            for _ in range(self.num_tokens):
                tokens.append(rank % self.vocab_size)
                rank //= self.vocab_size
            output.append(list(reversed(tokens)))
        return torch.tensor(output, dtype=torch.long, device=values.device)

    def decode(self, tokens: Tensor) -> Tensor:
        """Decode tokens back to bit rows."""
        rows = []
        for row in tokens.tolist():
            rank = 0
            for token in row:
                rank = rank * self.vocab_size + max(0, min(int(token), self.vocab_size - 1))
            rank = min(rank, self.num_sequences - 1)
            bits, last, run, first = [], -1, 0, True
            for position in range(self.sequence_length):
                remaining = self.sequence_length - position - 1
                zeros = self._zero_completions(remaining, last, run, first)
                if rank < zeros:
                    bit = 0
                else:
                    rank -= zeros
                    bit = 1
                bits.append(bit)
                if last == -1:
                    last, run = bit, 1
                elif bit == last:
                    run = min(run + 1, self.min_block_length + 1)
                else:
                    last, run, first = bit, 1, False
            rows.append(bits)
        return torch.tensor(rows, dtype=torch.float32, device=tokens.device)


class G05NativeActionCodec:
    """Non-registered sidecar wrapper for native ActionCodec encode/decode."""

    def __init__(self, config: Mapping[str, Any], *, action_token_begin: int) -> None:
        """Build the codec from its checkpoint configuration."""
        self.config = dict(config)
        architecture = dict(self.config["model_arch"])
        for key in (
            "action_time_contrastive_bias_init",
            "action_time_contrastive_mode",
            "action_time_contrastive_temperature_init",
            "commitment_loss_weight",
            "consistency_loss_type",
            "consistency_loss_weight",
            "ema_decay",
            "quantizer_dropout",
            "reconstruction_loss_weight",
            "threshold_ema_dead",
            "use_rotation_trick",
        ):
            if key in self.config:
                architecture[key] = self.config[key]
        self.module = _NativeCodecModule(architecture)
        self.model = self.module.model
        self.action_token_begin = action_token_begin
        self.parts = {
            key: int(value) for key, value in self.config["parts_meta"].items() if value is not None
        }
        patterns = tuple(self.config.get("rule_based_key_patterns") or ())
        self.rule_parts = [key for key in self.parts if any(pattern in key for pattern in patterns)]
        self.neural_parts = [key for key in self.parts if key not in self.rule_parts]
        self.codebook_size = int(architecture["codebook_size"])
        self.max_residuals = int(architecture["n_codebooks"])
        self.num_residuals = int(self.config.get("num_residuals") or self.max_residuals)
        self.code_length = self.model.code_h * self.model.code_a
        marker_names = [
            f"<{part}_{level}>" for level in range(self.max_residuals) for part in self.neural_parts
        ] + [f"<{part}>" for part in self.rule_parts]
        self.marker_indices = {name: self.codebook_size + index for index, name in enumerate(marker_names)}
        self.rule_codec = _BinarySequenceCodec(
            int(architecture["horizon"]),
            int(self.config.get("rule_based_min_block_len", 1)),
            self.codebook_size,
        )

    @property
    def action_token_length(self) -> int:
        """Number of tokens one action chunk occupies."""
        neural = len(self.neural_parts) * self.num_residuals * (self.code_length + 1)
        rules = len(self.rule_parts) * (self.rule_codec.num_tokens + 1)
        return neural + rules

    @classmethod
    def load(
        cls,
        config: Mapping[str, Any],
        *,
        action_token_begin: int,
        ckpt_path: str | Path,
    ) -> G05NativeActionCodec:
        """Load the codec weights from the checkpoint."""
        codec = cls(config, action_token_begin=action_token_begin)
        state_dict = load_file(Path(str(ckpt_path)), device="cpu")
        codec.module.load_state_dict(state_dict, strict=True)
        codec.module.eval()
        return codec

    def to(self, device: torch.device | str) -> G05NativeActionCodec:
        """Move the codec to a device."""
        self.module.to(device=device, dtype=torch.float32)
        return self

    def train(self, mode: bool = True) -> G05NativeActionCodec:
        """Set training mode."""
        self.module.train(mode)
        return self

    def eval(self) -> G05NativeActionCodec:
        """Set evaluation mode."""
        return self.train(False)

    def training_objective(
        self,
        components: dict[str, Tensor],
        d_original: dict[str, int] | None = None,
        x_pos_dict: dict[str, Tensor] | None = None,
        layer_weights: list[float] | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Run the codec's own training objective."""
        return self.model(
            components,
            d_original=d_original,
            x_pos_dict=x_pos_dict,
            layer_weights=layer_weights,
        )

    def _split(self, actions: Tensor) -> dict[str, Tensor]:
        """Split a flat action into its named parts."""
        splits = torch.split(actions[..., : sum(self.parts.values())], list(self.parts.values()), dim=-1)
        return dict(zip(self.parts, splits, strict=True))

    @torch.no_grad()
    def encode_for_language(self, payload: Mapping[str, Any]) -> list[int]:
        """Encode an action chunk into language token ids."""
        actions = torch.as_tensor(payload["value"])
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        components = self._split(actions)
        neural = {key: components[key] for key in self.neural_parts}
        codes = self.model.encode(neural)
        rule_codes = {
            key: self.rule_codec.encode(
                components[key][..., 0],
                float(self.config.get("rule_based_binarize_threshold", 0)),
            )
            for key in self.rule_parts
        }
        indices = []
        for level in range(self.num_residuals):
            for key in self.neural_parts:
                indices.append(self.marker_indices[f"<{key}_{level}>"])
                indices.extend(codes[key][0, level].tolist())
        for key in self.rule_parts:
            indices.append(self.marker_indices[f"<{key}>"])
            indices.extend(rule_codes[key][0].tolist())
        return [self.action_token_begin + int(index) for index in indices]

    @torch.no_grad()
    def decode_language_tokens(
        self,
        token_ids: Tensor,
        *,
        horizon: int,
        action_dim: int,
    ) -> tuple[Tensor, set[str]]:
        """Decode language token ids back into an action chunk."""
        indices = (token_ids.long() - self.action_token_begin).tolist()
        marker_to_name = {value: name for name, value in self.marker_indices.items()}
        neural: dict[str, list[list[int] | None]] = {
            key: [None] * self.num_residuals for key in self.neural_parts
        }
        rules: dict[str, list[int]] = {}
        cursor = 0
        while cursor < len(indices):
            marker = marker_to_name.get(indices[cursor])
            if marker is None:
                cursor += 1
                continue
            marker = marker[1:-1]
            if marker in self.rule_parts:
                length = self.rule_codec.num_tokens
                values = indices[cursor + 1 : cursor + 1 + length]
                if len(values) == length and all(0 <= value < self.codebook_size for value in values):
                    rules[marker] = values
                cursor += length + 1
                continue
            part, level_text = marker.rsplit("_", 1)
            level = int(level_text)
            if part in neural and level < self.num_residuals:
                values = indices[cursor + 1 : cursor + 1 + self.code_length]
                if len(values) == self.code_length and all(
                    0 <= value < self.codebook_size for value in values
                ):
                    neural[part][level] = values
            cursor += self.code_length + 1

        absent = {
            key
            for key in self.parts
            if (key in neural and not any(level is not None for level in neural[key]))
            or (key in self.rule_parts and key not in rules)
        }
        device = next(self.module.parameters()).device
        code_tensors = {}
        for key, levels in neural.items():
            if not any(level is not None for level in levels):
                continue
            filled = [level if level is not None else [0] * self.code_length for level in levels]
            code_tensors[key] = torch.tensor([filled], dtype=torch.long, device=device)
        decoded = (
            self.model.decode(code_tensors, {key: self.parts[key] for key in code_tensors})
            if code_tensors
            else {}
        )
        for key in self.rule_parts:
            if key in rules:
                tokens = torch.tensor([rules[key]], dtype=torch.long, device=device)
                binary = self.rule_codec.decode(tokens)
                decoded[key] = binary[:, :, None] * 2 - 1
        # ``absent_key_fill_value`` is an internal partitioner sentinel. The
        # released marker-aware final decoder converts absent/no-op body parts
        # to zero motion before returning an action.
        batch = torch.zeros((1, horizon, action_dim), dtype=torch.float32, device=device)
        offset = 0
        for key, dimension in self.parts.items():
            if key in decoded:
                batch[..., offset : offset + dimension] = decoded[key][..., :dimension]
            offset += dimension
        return batch[0], absent
