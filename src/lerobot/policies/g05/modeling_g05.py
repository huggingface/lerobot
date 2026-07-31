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

"""Native G0.5 policy, model, and ActionCodec implementation."""

from __future__ import annotations

import itertools
import json
import math
import shutil
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
    Qwen3_5VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import OptimizerParams
from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import resolve_safetensors_device

from .configuration_g05 import (
    G05_POLICY_PARTS,
    G05Config,
    make_g05_cot_prompt_template,
    make_g05_prompt_template,
)
from .processor_g05 import IGNORE_INDEX, G05SequenceBatch, G05Tokenizer, G05TokenType


class G05GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Qwen3.5 linear attention with the numerical contract used by G0.5."""

    def forward(
        self,
        hidden_states: Tensor,
        cache_params: DynamicCache | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None]

        batch_size, sequence_length, _ = hidden_states.shape
        use_cached_state = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_cached_state:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        gate = self.in_proj_z(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_v_heads,
            self.head_v_dim,
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = self.in_proj_a(hidden_states)

        if use_cached_state:
            if sequence_length == 1:
                mixed_qkv = self.causal_conv1d_update(
                    mixed_qkv,
                    conv_state,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.activation,
                )
            else:
                conv_input = torch.cat((conv_state[..., 1:], mixed_qkv), dim=-1)
                mixed_qkv = functional.silu(
                    functional.conv1d(
                        conv_input,
                        self.conv1d.weight,
                        self.conv1d.bias,
                        groups=self.conv1d.groups,
                    )
                )
                cache_params.layers[self.layer_idx].conv_states.copy_(
                    conv_input[..., -self.conv_kernel_size :]
                )
        else:
            if cache_params is not None:
                conv_state = functional.pad(
                    mixed_qkv,
                    (self.conv_kernel_size - mixed_qkv.shape[-1], 0),
                )
                cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = functional.silu(self.conv1d(mixed_qkv)[:, :, :sequence_length])

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2),
            (self.key_dim, self.key_dim, self.value_dim),
            dim=-1,
        )
        query = query.reshape(batch_size, sequence_length, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, sequence_length, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, sequence_length, self.num_v_heads, self.head_v_dim)

        with torch.autocast(hidden_states.device.type, enabled=False):
            decay = -self.A_log.float().exp() * functional.softplus(decay.float() + self.dt_bias.float())
        head_repeats = self.num_v_heads // self.num_k_heads
        if head_repeats > 1:
            query = query.repeat_interleave(head_repeats, dim=2)
            key = key.repeat_interleave(head_repeats, dim=2)

        if use_cached_state and sequence_length == 1:
            attended, recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            initial_state = recurrent_state.clone() if use_cached_state else None
            attended, recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                chunk_size=32,
                initial_state=initial_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(recurrent_state, self.layer_idx)

        attended = attended.reshape(-1, self.head_v_dim)
        gate = gate.reshape(-1, self.head_v_dim)
        with torch.autocast(hidden_states.device.type, enabled=False):
            attended = self.norm(attended.float(), gate.float())
        attended = attended.reshape(batch_size, sequence_length, self.value_dim)
        return self.out_proj(attended)


class _BlockDCT(nn.Module):
    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        frequency = torch.arange(block_size, dtype=torch.float32)
        time = torch.arange(block_size, dtype=torch.float32)
        basis = torch.cos(math.pi / block_size * (time + 0.5).unsqueeze(0) * frequency.unsqueeze(1))
        basis[0] *= math.sqrt(1 / block_size)
        basis[1:] *= math.sqrt(2 / block_size)
        self.register_buffer("basis", basis, persistent=False)

    def dct(self, values: Tensor) -> Tensor:
        batch, horizon, dimension = values.shape
        pad = (-horizon) % self.block_size
        if pad:
            values = functional.pad(values, (0, 0, 0, pad))
        blocks = values.shape[1] // self.block_size
        values = values.reshape(batch * blocks, self.block_size, dimension)
        transformed = torch.einsum("kn,bnd->bkd", self.basis.to(values), values)
        return transformed.reshape(batch, blocks * self.block_size, dimension)

    def idct(self, values: Tensor, horizon: int) -> Tensor:
        batch, padded_horizon, dimension = values.shape
        blocks = padded_horizon // self.block_size
        values = values.reshape(batch * blocks, self.block_size, dimension)
        restored = torch.einsum("nk,bkd->bnd", self.basis.to(values), values)
        return restored.reshape(batch, padded_horizon, dimension)[:, :horizon]


def _rotate_half(values: Tensor) -> Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class _CodecAttention(nn.Module):
    def __init__(self, dimension: int, num_heads: int, head_dim: int, rope_base: int) -> None:
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
        batch, sequence_length, _ = hidden_states.shape
        query, key, value = self.to_qkv(hidden_states).chunk(3, dim=-1)

        def heads(values: Tensor) -> Tensor:
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
    def __init__(self, dimension: int, multiplier: float) -> None:
        super().__init__()
        inner_dim = int(dimension * multiplier)
        self.w_up = nn.Linear(dimension, inner_dim * 2, bias=False)
        self.w_down = nn.Linear(inner_dim, dimension, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        value, gate = self.w_up(hidden_states).chunk(2, dim=-1)
        return self.w_down(value * functional.gelu(gate))


class _CodecTransformerLayer(nn.Module):
    def __init__(self, dimension: int, config: Mapping[str, Any]) -> None:
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
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states)) * self.ls1
        return hidden_states + self.ffn(self.norm2(hidden_states)) * self.ls2


class _CodecDownBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: tuple[int, int],
        depth: int,
        config: Mapping[str, Any],
    ) -> None:
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
        hidden_states = self.conv(hidden_states)
        batch, channels, height, action_dim = hidden_states.shape
        sequence = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * action_dim, channels)
        for layer in self.transformer_layers:
            sequence = layer(sequence)
        return sequence.reshape(batch, height, action_dim, channels).permute(0, 3, 1, 2)


class _CodecUpBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: tuple[int, int],
        depth: int,
        config: Mapping[str, Any],
    ) -> None:
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
        batch, channels, height, action_dim = hidden_states.shape
        sequence = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * action_dim, channels)
        for layer in self.transformer_layers:
            sequence = layer(sequence)
        hidden_states = sequence.reshape(batch, height, action_dim, channels).permute(0, 3, 1, 2)
        return self.conv(hidden_states)


class _CodecEncoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
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
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.out_proj(hidden_states)


class _CodecDecoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
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
        hidden_states = self.in_proj(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class _CodecQuantizer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        input_dim = int(config["latent_dim"])
        codebook_dim = int(config["codebook_dim"])
        codebook_size = int(config["codebook_size"])
        self.input_dim = input_dim
        self.in_proj = nn.Linear(input_dim, codebook_dim, bias=False)
        self.out_proj = nn.Linear(codebook_dim, input_dim, bias=False)
        self.register_buffer("codebook", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("embed_avg", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("inited", torch.tensor(False))

    def encode(self, values: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.in_proj(values.transpose(1, 2))
        flat = projected.reshape(-1, projected.shape[-1]).float()
        codebook = self.codebook.float()
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.square().sum(dim=1)[None]
        )
        codes = distances.argmin(dim=-1).reshape(values.shape[0], values.shape[2])
        quantized = functional.embedding(codes, self.codebook)
        quantized = self.out_proj(quantized).transpose(1, 2)
        return quantized.to(values.dtype), codes

    def decode_codes(self, codes: Tensor) -> Tensor:
        return self.out_proj(functional.embedding(codes, self.codebook)).transpose(1, 2)


class _ResidualCodecQuantizer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList([_CodecQuantizer(config) for _ in range(int(config["n_codebooks"]))])

    def encode(self, values: Tensor) -> Tensor:
        residual = values
        codes = []
        for quantizer in self.quantizers:
            quantized, level_codes = quantizer.encode(residual)
            residual = residual - quantized
            codes.append(level_codes)
        return torch.stack(codes, dim=1)

    def from_codes(self, codes: Tensor) -> Tensor:
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
    def __init__(self, config: Mapping[str, Any]) -> None:
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
        self.decoder = _CodecDecoder(config)
        self.conv_out = nn.ConvTranspose2d(
            int(config["encoder_channels"]),
            int(config["horizon_patch_size"]),
            kernel_size=(1, int(config["conv_in_action_kernel"])),
        )

    @property
    def code_h(self) -> int:
        height = int(self.config["horizon"]) // int(self.config["horizon_patch_size"])
        for stride_h, _ in self.config["strides"]:
            height //= int(stride_h)
        return height

    @property
    def code_a(self) -> int:
        return int(self.config["max_component_dim"]) - int(self.config["conv_in_action_kernel"]) + 1

    def _pad(self, values: Tensor) -> Tensor:
        maximum = int(self.config["max_component_dim"])
        if values.shape[-1] < maximum:
            return functional.pad(values, (0, maximum - values.shape[-1]))
        return values[..., :maximum]

    def encode(self, components: dict[str, Tensor]) -> dict[str, Tensor]:
        names = list(components)
        batch_size = next(iter(components.values())).shape[0]
        values = torch.cat([self._pad(components[name].float()) for name in names], dim=0)
        if self.block_dct is not None:
            values = self.block_dct.dct(values)
        patch = int(self.config["horizon_patch_size"])
        values = values.reshape(values.shape[0], -1, patch, values.shape[-1]).transpose(1, 2)
        latent = self.encoder(self.conv_in(values)).flatten(2)
        codes = self.rvq.encode(latent)
        return {
            name: codes[index * batch_size : (index + 1) * batch_size] for index, name in enumerate(names)
        }

    def decode(self, components: dict[str, Tensor], dimensions: Mapping[str, int]) -> dict[str, Tensor]:
        names = list(components)
        batch_size = next(iter(components.values())).shape[0]
        codes = torch.cat([components[name] for name in names], dim=0)
        quantized = self.rvq.from_codes(codes)
        quantized = quantized.reshape(
            quantized.shape[0],
            quantized.shape[1],
            self.code_h,
            self.code_a,
        )
        decoded = self.conv_out(self.decoder(quantized))
        decoded = decoded.transpose(1, 2).reshape(decoded.shape[0], -1, decoded.shape[-1])
        if self.block_dct is not None:
            decoded = self.block_dct.idct(decoded, int(self.config["horizon"]))
        return {
            name: decoded[index * batch_size : (index + 1) * batch_size, :, : dimensions[name]]
            for index, name in enumerate(names)
        }


class _NativeCodecModule(nn.Module):
    """Module hierarchy matching ``action_tokenizer.pt`` exactly."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = _ActionCodecModel(config)


class _BinarySequenceCodec:
    def __init__(self, sequence_length: int, min_block_length: int, vocab_size: int) -> None:
        self.sequence_length = sequence_length
        self.min_block_length = min_block_length
        self.vocab_size = vocab_size
        self._count_cache: dict[tuple[int, int, int, bool], int] = {}
        self.num_sequences = self._count(sequence_length, -1, 0, True)
        self.num_tokens = max(1, math.ceil(math.log(self.num_sequences, vocab_size)))

    def _count(self, remaining: int, last: int, run_length: int, first: bool) -> int:
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
        if last in (-1, 0):
            return self._count(
                remaining,
                0,
                1 if last == -1 else min(run + 1, self.min_block_length + 1),
                first,
            )
        return self._count(remaining, 0, 1, False) if first or run > self.min_block_length else 0

    def encode(self, values: Tensor, threshold: float) -> Tensor:
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
        self.config = dict(config)
        architecture = self.config["model_arch"]
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
        neural = len(self.neural_parts) * self.num_residuals * (self.code_length + 1)
        rules = len(self.rule_parts) * (self.rule_codec.num_tokens + 1)
        return neural + rules

    @classmethod
    def load(
        cls,
        config: Mapping[str, Any],
        *,
        action_token_begin: int,
    ) -> G05NativeActionCodec:
        codec = cls(config, action_token_begin=action_token_begin)
        checkpoint = torch.load(
            Path(str(config["ckpt_dir"])),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        codec.module.load_state_dict(state_dict, strict=True)
        codec.module.eval()
        return codec

    def to(self, device: torch.device | str) -> G05NativeActionCodec:
        self.module.to(device=device, dtype=torch.float32)
        return self

    def _split(self, actions: Tensor) -> dict[str, Tensor]:
        splits = torch.split(actions[..., : sum(self.parts.values())], list(self.parts.values()), dim=-1)
        return dict(zip(self.parts, splits, strict=True))

    @torch.no_grad()
    def encode_for_language(self, payload: Mapping[str, Any]) -> list[int]:
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


G05_RUNTIME_PREDICT_COT = "g05_runtime_predict_cot"


def _autoregressive_ce_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    ce_weight: float,
    z_loss_scale: float,
) -> Tensor:
    """Apply G0.5's checkpoint-configured autoregressive objective."""

    if ce_weight < 0:
        raise ValueError("G0.5 ar.ce_weight must be non-negative.")
    if z_loss_scale < 0:
        raise ValueError("G0.5 ar.ce_z_loss_scale must be non-negative.")
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "G0.5 autoregressive CE expects logits [N,V] and labels [N], "
            f"got {tuple(logits.shape)} and {tuple(labels.shape)}."
        )
    if logits.shape[0] == 0 or ce_weight == 0:
        return logits.sum() * 0

    token_loss = functional.cross_entropy(logits, labels, reduction="none")
    if z_loss_scale:
        log_z = torch.logsumexp(logits.float(), dim=-1)
        token_loss = token_loss.float() + z_loss_scale * log_z.square()
    return token_loss.mean() * ce_weight


def _qwen_text_config(values: Mapping[str, Any], *, vocab_size: int | None = None):
    """Translate the serialized G0.5 Qwen config into a Transformers config."""

    return Qwen3_5TextConfig(
        vocab_size=int(vocab_size if vocab_size is not None else values.get("vocab_size", 1)),
        hidden_size=int(values["hidden_size"]),
        intermediate_size=int(values["intermediate_size"]),
        num_hidden_layers=int(values["num_hidden_layers"]),
        num_attention_heads=int(values["num_attention_heads"]),
        num_key_value_heads=int(values["num_key_value_heads"]),
        head_dim=int(values["head_dim"]),
        rms_norm_eps=float(values["rms_norm_eps"]),
        max_position_embeddings=int(values["max_position_embeddings"]),
        attention_bias=bool(values.get("attention_bias", False)),
        hidden_act=str(values.get("hidden_act", "silu")),
        rope_parameters=dict(values["rope_parameters"]),
        linear_conv_kernel_dim=int(values.get("linear_conv_kernel_dim", 4)),
        linear_key_head_dim=int(values.get("linear_key_head_dim", 128)),
        linear_value_head_dim=int(values.get("linear_value_head_dim", 128)),
        linear_num_key_heads=int(values.get("linear_num_key_heads", 16)),
        linear_num_value_heads=int(values.get("linear_num_value_heads", 16)),
        layer_types=list(values["layer_types"]),
        pad_token_id=values.get("pad_token_id"),
        tie_word_embeddings=True,
    )


def _qwen_vision_config(values: Mapping[str, Any]):
    """Translate the serialized G0.5 vision config into Transformers."""

    config = Qwen3_5VisionConfig(
        depth=int(values["depth"]),
        hidden_size=int(values["hidden_size"]),
        num_heads=int(values["num_heads"]),
        patch_size=int(values["patch_size"]),
        temporal_patch_size=int(values["temporal_patch_size"]),
        spatial_merge_size=int(values["spatial_merge_size"]),
        in_channels=int(values.get("in_channels", 3)),
        intermediate_size=int(values["intermediate_size"]),
        out_hidden_size=int(values["out_hidden_size"]),
        num_position_embeddings=int(values["num_position_embeddings"]),
        hidden_act=str(values.get("hidden_act", "gelu_pytorch_tanh")),
    )
    config.temporal_freq = int(values.get("temporal_freq", 0))
    config.spacetime_mode = str(values.get("spacetime_mode", "factorized"))
    config.token_drop_layer = values.get("token_drop_layer")
    config.temporal_pe_pretrain_frames = values.get("temporal_pe_pretrain_frames")
    config.batch_all_cameras = bool(values.get("batch_all_cameras", False))
    return config


class G05ProprioEmbedder(nn.Module):
    """Project the padded G0.5 proprioception vector into the VLM hidden size."""

    def __init__(self, proprio_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, proprio: Tensor) -> Tensor:
        with torch.autocast(proprio.device.type, enabled=False):
            return self.mlp(proprio.float())


class G05QwenTextModel(nn.Module):
    """Qwen3.5 text stack with G0.5's checkpoint-compatible module names."""

    def __init__(self, values: Mapping[str, Any], *, vocab_size: int) -> None:
        super().__init__()

        self.config = _qwen_text_config(values, vocab_size=vocab_size)
        self.input_proj = nn.Embedding(vocab_size, self.config.hidden_size, self.config.pad_token_id)
        layers = []
        for layer_idx in range(self.config.num_hidden_layers):
            layer = Qwen3_5DecoderLayer(self.config, layer_idx)
            if self.config.layer_types[layer_idx] == "linear_attention":
                layer.linear_attn = G05GatedDeltaNet(self.config, layer_idx)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.norm = Qwen3_5RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.config)

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.input_proj(input_ids)

    def logits(self, hidden_states: Tensor) -> Tensor:
        return torch.nn.functional.linear(hidden_states, self.input_proj.weight)

    def forward(
        self,
        inputs_embeds: Tensor,
        *,
        full_attention_mask: Tensor,
        linear_attention_mask: Tensor,
        position_ids: Tensor,
        cache=None,
    ) -> tuple[Tensor, Any]:
        if cache is None:
            cache = DynamicCache(config=self.config)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer_index, layer in enumerate(self.layers):
            attention_mask = (
                linear_attention_mask
                if self.config.layer_types[layer_index] == "linear_attention"
                else full_attention_mask
            )
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids[0],
                past_key_values=cache,
                use_cache=True,
            )
        return self.norm(hidden_states), cache


class G05ActionDecoderLayer(nn.Module):
    """Qwen3.5 decoder layer with G0.5 adaptive RMSNorm conditioning."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()

        if config.layer_types[layer_idx] != "full_attention":
            raise ValueError("The released G0.5 action expert requires full-attention layers.")
        self.layer_idx = layer_idx
        self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = PiGemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cond_dim=config.hidden_size,
        )
        self.post_attention_layernorm = PiGemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cond_dim=config.hidden_size,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_ids: Tensor,
        cache,
        time_cond: Tensor,
    ) -> Tensor:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond=time_cond)
        key_length = cache.layers[self.layer_idx].get_seq_length() + hidden_states.shape[1]
        layer_attention_mask = (
            attention_mask[..., -key_length:] if attention_mask.shape[-1] != key_length else attention_mask
        )
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=layer_attention_mask,
            position_ids=position_ids[0],
            past_key_values=cache,
            position_embeddings=position_embeddings,
            use_cache=True,
        )
        hidden_states = residual + hidden_states if gate is None else residual + hidden_states * gate

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=time_cond)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states if gate is None else residual + hidden_states * gate


class G05ActionExpert(nn.Module):
    """Continuous G0.5 action expert with checkpoint-compatible parameter names."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        super().__init__()

        self.config = _qwen_text_config(values)
        input_dim = int(values["input_dim"])
        output_dim = int(values["output_dim"])
        hidden_size = int(values["hidden_size"])
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.layers = nn.ModuleList(
            [
                G05ActionDecoderLayer(self.config, layer_idx)
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = PiGemmaRMSNorm(
            hidden_size,
            eps=self.config.rms_norm_eps,
            cond_dim=hidden_size,
        )
        self.output_proj = nn.Linear(hidden_size, output_dim)
        self.time_mlp_in = nn.Linear(hidden_size, hidden_size)
        self.time_mlp_out = nn.Linear(hidden_size, hidden_size)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.config)

    def embed(self, actions: Tensor) -> Tensor:
        return self.input_proj(actions)

    def encode_time(self, timesteps: Tensor) -> Tensor:
        half = self.config.hidden_size // 2
        fraction = torch.linspace(0.0, 1.0, half, device=timesteps.device, dtype=torch.float32)
        periods = 4e-3 * (4.0 / 4e-3) ** fraction
        phase = timesteps.float().unsqueeze(-1) * (2 * math.pi / periods)
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        with torch.autocast(timesteps.device.type, enabled=False):
            return torch.nn.functional.silu(
                self.time_mlp_out(torch.nn.functional.silu(self.time_mlp_in(embedding)))
            )

    def forward(
        self,
        inputs_embeds: Tensor,
        *,
        attention_mask: Tensor,
        position_ids: Tensor,
        cache,
        time_cond: Tensor,
    ) -> Tensor:
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache=cache,
                time_cond=time_cond,
            )
        hidden_states, _ = self.norm(hidden_states, cond=time_cond)
        return hidden_states

    def decode(self, hidden_states: Tensor) -> Tensor:
        with torch.autocast(hidden_states.device.type, enabled=False):
            return self.output_proj(hidden_states.float())


class G05NativeModel(nn.Module):
    """Weight-owning native G0.5 model assembled from serialized checkpoint config."""

    def __init__(self, model_config: Mapping[str, Any], *, vocab_size: int) -> None:
        super().__init__()

        self.vision_tower = Qwen3_5VisionModel(_qwen_vision_config(model_config["vision"]))
        self.vlm = G05QwenTextModel(model_config["vlm"], vocab_size=vocab_size)
        self.action_expert = G05ActionExpert(model_config["action_expert"])
        self.proprio_embedder = G05ProprioEmbedder(
            int(model_config["proprio_dim"]),
            int(model_config["vlm"]["hidden_size"]),
        )


def _temporal_embedding(timesteps: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    phase = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.stack((phase.sin(), phase.cos() - 1), dim=-1).reshape(len(timesteps), dimension)


class G05NativeBackend(nn.Module):
    """Native LeRobot backend for G0.5.

    Inference and training routing are added around this checkpoint-compatible
    model core; no OpenGalaxea Python package is imported.
    """

    def __init__(
        self,
        model_config: Mapping[str, Any],
        *,
        vocab_size: int,
        processor_path: str | Path,
    ) -> None:
        super().__init__()
        self.model_config = dict(model_config)
        self.model = G05NativeModel(self.model_config, vocab_size=vocab_size)
        attention_implementation = str(self.model_config.get("attn_implementation", "eager"))
        self.model.vlm.config._attn_implementation = attention_implementation
        self.model.action_expert.config._attn_implementation = attention_implementation
        self.model.vision_tower.config._attn_implementation = attention_implementation
        self.processor = G05Tokenizer(processor_path, self.model_config)
        if len(self.processor) != vocab_size:
            raise ValueError(
                f"G0.5 tokenizer has {len(self.processor)} rows, but model expects {vocab_size}."
            )
        self.action_tokenizer = None
        action_config = self.model_config.get("AT_CONFIG")
        if isinstance(action_config, Mapping):
            checkpoint = Path(str(action_config.get("ckpt_dir", "")))
            if checkpoint.is_file() and not next(self.model.parameters()).is_meta:
                self.action_tokenizer = G05NativeActionCodec.load(
                    action_config,
                    action_token_begin=self.processor.action_token_begin,
                )
        self._last_vision_grids: list[tuple[int, int, int]] = []

    def materialize_runtime_buffers(self, device: torch.device | str) -> None:
        """Rebuild non-persistent Transformers buffers after meta construction."""

        for module in self.modules():
            if isinstance(module, Qwen3_5TextRotaryEmbedding) and module.inv_freq.is_meta:
                # The Transformers initializer creates its arange before moving it
                # to ``device``. Override the outer meta-device construction context
                # so that intermediate is materialized on the final device as well.
                with torch.device(device):
                    inverse_frequency, attention_scaling = module.compute_default_rope_parameters(
                        module.config,
                        device,
                    )
                module.inv_freq = inverse_frequency
                module.original_inv_freq = inverse_frequency.clone()
                module.attention_scaling = attention_scaling
            elif isinstance(module, Qwen3_5VisionRotaryEmbedding) and module.inv_freq.is_meta:
                frequency = torch.arange(0, module.dim, 2, dtype=torch.float32, device=device)
                module.inv_freq = 1.0 / (module.theta ** (frequency / module.dim))

        remaining = [name for name, buffer in self.named_buffers() if buffer.is_meta]
        if remaining:
            raise RuntimeError(f"G0.5 meta loading left runtime buffers unmaterialized: {remaining}")

    def apply_fp32_params(self) -> None:
        """Restore the FP32 islands used by the released mixed-precision runtime."""

        patterns = (
            "vision_tower.patch_embed",
            "vision_tower.pos_embed",
            "vision_tower.merger",
            "norm1",
            "norm2",
            "input_layernorm",
            "post_attention_layernorm",
            "q_norm",
            "k_norm",
            "linear_attn.norm",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "vlm.norm",
            "action_expert.norm",
            "action_expert.input_proj",
            "action_expert.output_proj",
            "action_expert.time_mlp",
            "proprio_embedder",
        )
        for name, parameter in self.named_parameters():
            if any(pattern in name for pattern in patterns):
                parameter.data = parameter.data.float()

    @staticmethod
    def _should_apply_weight_decay(
        owner_module: nn.Module | None,
        leaf_name: str,
        parameter: nn.Parameter,
    ) -> bool:
        return leaf_name != "bias" and parameter.ndim > 1 and not isinstance(owner_module, nn.Embedding)

    def get_optim_param_groups(
        self,
        *,
        lr: float,
        weight_decay: float,
        apply_decay_on_norm_and_bias: bool = False,
        backbone_lr_multiplier: float = 1.0,
        vision_lr_multiplier: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Build the released six native backbone/action/vision parameter groups."""

        action_parameters = {id(parameter) for parameter in self.model.action_expert.parameters()}
        vision_parameters = {id(parameter) for parameter in self.model.vision_tower.parameters()}
        modules = dict(self.model.named_modules())
        grouped: dict[str, list[nn.Parameter]] = {
            "backbone_decay": [],
            "action_decay": [],
            "vision_decay": [],
            "backbone_no_decay": [],
            "action_no_decay": [],
            "vision_no_decay": [],
        }
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            owner_name, _, leaf_name = name.rpartition(".")
            decay = apply_decay_on_norm_and_bias or self._should_apply_weight_decay(
                modules.get(owner_name),
                leaf_name,
                parameter,
            )
            if id(parameter) in action_parameters:
                family = "action"
            elif id(parameter) in vision_parameters:
                family = "vision"
            else:
                family = "backbone"
            grouped[f"{family}_{'decay' if decay else 'no_decay'}"].append(parameter)

        learning_rates = {
            "backbone": lr * backbone_lr_multiplier,
            "action": lr,
            "vision": lr * backbone_lr_multiplier * vision_lr_multiplier,
        }
        parameter_groups = [
            {
                "params": grouped[name],
                "lr": learning_rates[name.split("_", 1)[0]],
                "weight_decay": weight_decay if not name.endswith("_no_decay") else 0.0,
                "name": name,
            }
            for name in (
                "backbone_decay",
                "action_decay",
                "vision_decay",
                "backbone_no_decay",
                "action_no_decay",
                "vision_no_decay",
            )
        ]
        expected = sum(parameter.requires_grad for parameter in self.model.parameters())
        actual = sum(len(group["params"]) for group in parameter_groups)
        if actual != expected:
            raise RuntimeError(
                f"G0.5 optimizer grouping lost parameters: grouped {actual}, expected {expected}."
            )
        return parameter_groups

    @classmethod
    def from_config(cls, model_config: Mapping[str, Any]) -> G05NativeBackend:
        processor_path = Path(str(model_config["hf_processor_path"]))
        tokenizer_config = processor_path / "tokenizer_config.json"
        if not tokenizer_config.is_file():
            raise FileNotFoundError(f"G0.5 tokenizer config not found: {tokenizer_config}")

        tokenizer_metadata = json.loads(tokenizer_config.read_text())
        added = tokenizer_metadata.get("added_tokens_decoder") or {}
        base_vocab_size = max((int(token_id) for token_id in added), default=-1) + 1
        at_config = model_config["AT_CONFIG"]
        codebook_size = int(at_config["model_arch"]["codebook_size"])
        parts = at_config["parts_meta"]
        rule_patterns = tuple(at_config.get("rule_based_key_patterns") or ())
        rule_parts = [name for name in parts if any(pattern in name for pattern in rule_patterns)]
        neural_parts = [name for name in parts if name not in rule_parts]
        residuals = int(at_config["model_arch"]["n_codebooks"])
        marker_count = len(neural_parts) * residuals + len(rule_parts)
        # Action-code tokens, group markers, <EOV>, and the MLP <state> token.
        vocab_size = base_vocab_size + codebook_size + marker_count + 2
        return cls(model_config, vocab_size=vocab_size, processor_path=processor_path)

    @staticmethod
    def _patchify(images: Tensor, patch_size: int, temporal_patch_size: int, merge_size: int) -> Tensor:
        batch_frames, channels, height, width = images.shape
        grid_h, grid_w = height // patch_size, width // patch_size
        temporal = images.unsqueeze(2).expand(-1, -1, temporal_patch_size, -1, -1)
        return (
            temporal.reshape(
                batch_frames,
                temporal_patch_size,
                channels,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshape(batch_frames * grid_h * grid_w, -1)
        )

    def _vision_temporal_block(
        self,
        block,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor],
        batch_size: int,
        num_frames: int,
        patches_per_frame: int,
        temporal_pe: Tensor,
        temporal_mask: Tensor,
    ) -> Tensor:
        total, hidden_size = hidden_states.shape
        num_heads = block.attn.num_heads
        head_dim = block.attn.head_dim
        residual = hidden_states
        conditioned = (
            hidden_states.view(batch_size, num_frames, patches_per_frame, hidden_size)
            + temporal_pe[None, :, None, :]
        ).reshape(total, hidden_size)
        normed = block.norm1(conditioned)
        query, key, value = (
            block.attn.qkv(normed).reshape(total, 3, num_heads, head_dim).permute(1, 0, 2, 3).unbind(0)
        )

        def temporal_view(tensor: Tensor) -> Tensor:
            return (
                tensor.view(batch_size, num_frames, patches_per_frame, num_heads, head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(batch_size * patches_per_frame, num_heads, num_frames, head_dim)
            )

        query_t, key_t, value_t = (temporal_view(tensor) for tensor in (query, key, value))
        weights = torch.matmul(query_t, key_t.transpose(-2, -1)) * block.attn.scaling
        weights = functional.softmax(weights + temporal_mask[None, None], dim=-1, dtype=torch.float32).to(
            query_t.dtype
        )
        mixed_value = torch.matmul(weights, value_t)
        mixed_value = (
            mixed_value.view(batch_size, patches_per_frame, num_heads, num_frames, head_dim)
            .permute(0, 3, 1, 2, 4)
            .reshape(total, num_heads, head_dim)
        )

        cosine, sine = position_embeddings
        query, key = apply_rotary_pos_emb_vision(query, key, cosine, sine)
        spatial_outputs = []
        for start in range(0, total, patches_per_frame):
            stop = start + patches_per_frame
            spatial_outputs.append(
                functional.scaled_dot_product_attention(
                    query[start:stop].transpose(0, 1).unsqueeze(0),
                    key[start:stop].transpose(0, 1).unsqueeze(0),
                    mixed_value[start:stop].transpose(0, 1).unsqueeze(0),
                    scale=block.attn.scaling,
                )
            )
        spatial = torch.cat(spatial_outputs, dim=2).squeeze(0).transpose(0, 1).reshape(total, hidden_size)
        hidden_states = residual + block.attn.proj(spatial)
        return hidden_states + block.mlp(block.norm2(hidden_states))

    def _encode_camera(self, frames: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        """Encode one camera, including G0.5's causal temporal-memory mixing."""

        tower = self.model.vision_tower
        batch_size, num_frames, _, height, width = frames.shape
        patch_size = int(tower.config.patch_size)
        merge_size = int(tower.config.spatial_merge_size)
        temporal_patch_size = int(tower.config.temporal_patch_size)
        grid_h, grid_w = height // patch_size, width // patch_size
        patches_per_frame = grid_h * grid_w
        flattened = frames.reshape(batch_size * num_frames, *frames.shape[2:])
        patches = self._patchify(flattened, patch_size, temporal_patch_size, merge_size)
        grid = torch.tensor(
            [[1, grid_h, grid_w]] * (batch_size * num_frames),
            dtype=torch.long,
            device=frames.device,
        )

        with torch.autocast(frames.device.type, enabled=False):
            hidden_states = tower.patch_embed(patches)
            hidden_states = hidden_states + tower.fast_pos_embed_interpolate(grid)
        rotary = tower.rot_pos_emb(grid).reshape(hidden_states.shape[0], -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        cu_seqlens = torch.arange(
            0,
            (batch_size * num_frames + 1) * patches_per_frame,
            patches_per_frame,
            dtype=torch.int32,
            device=frames.device,
        )

        temporal_frequency = int(getattr(tower.config, "temporal_freq", 0))
        if num_frames > 1 and temporal_frequency > 0:
            timesteps = torch.arange(-(num_frames - 1), 1, device=frames.device)
            temporal_pe = _temporal_embedding(timesteps, hidden_states.shape[-1]).to(hidden_states.dtype)
            temporal_mask = torch.triu(
                torch.full(
                    (num_frames, num_frames),
                    float("-inf"),
                    device=frames.device,
                    dtype=hidden_states.dtype,
                ),
                diagonal=1,
            )
            drop_layer = int(getattr(tower.config, "token_drop_layer", None) or len(tower.blocks)) - 1
        else:
            temporal_pe = temporal_mask = None
            drop_layer = -1

        for layer_index, block in enumerate(tower.blocks):
            use_temporal = (
                temporal_pe is not None
                and layer_index <= drop_layer
                and (drop_layer - layer_index) % temporal_frequency == 0
            )
            if use_temporal:
                hidden_states = self._vision_temporal_block(
                    block,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    patches_per_frame=patches_per_frame,
                    temporal_pe=temporal_pe,
                    temporal_mask=temporal_mask,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )
            if layer_index == drop_layer and num_frames > 1:
                hidden_states = hidden_states.view(batch_size, num_frames, patches_per_frame, -1)[
                    :, -1
                ].reshape(batch_size * patches_per_frame, -1)
                num_frames = 1
                cu_seqlens = torch.arange(
                    0,
                    (batch_size + 1) * patches_per_frame,
                    patches_per_frame,
                    dtype=torch.int32,
                    device=frames.device,
                )
                cosine, sine = position_embeddings
                position_embeddings = (
                    cosine[: batch_size * patches_per_frame],
                    sine[: batch_size * patches_per_frame],
                )

        with torch.autocast(frames.device.type, enabled=False):
            merged = tower.merger(hidden_states)
        tokens_per_frame = (grid_h // merge_size) * (grid_w // merge_size)
        return merged.reshape(batch_size, tokens_per_frame, -1), (1, grid_h, grid_w)

    def _encode_vision(self, pixel_values: Mapping[str, Tensor]) -> Tensor:
        features = []
        grids = []
        for frames in pixel_values.values():
            feature, grid = self._encode_camera(frames)
            features.append(feature)
            grids.append(grid)
        self._last_vision_grids = grids
        return torch.cat(features, dim=1)

    def _embed(
        self,
        sequence: G05SequenceBatch,
        pixel_values: Mapping[str, Tensor],
        proprio: Tensor,
    ) -> Tensor:
        image_features = self._encode_vision(pixel_values)
        text_features = self.model.vlm.embed(sequence.input_ids).to(image_features.dtype)
        embeddings = text_features.clone()

        image_mask = sequence.token_types == G05TokenType.IMAGE
        image_indices = (image_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        if image_mask.any() and int(image_indices[image_mask].max()) >= image_features.shape[1]:
            raise ValueError("G0.5 prompt image-token count does not match the native vision encoder output.")
        gathered_images = torch.gather(
            image_features,
            1,
            image_indices.unsqueeze(-1).expand(-1, -1, image_features.shape[-1]),
        )
        embeddings[image_mask] = gathered_images[image_mask]

        state_mask = sequence.token_types == G05TokenType.PROPRIO
        state_features = self.model.proprio_embedder(proprio).to(embeddings.dtype)
        state_indices = (state_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        gathered_state = torch.gather(
            state_features,
            1,
            state_indices.unsqueeze(-1).expand(-1, -1, state_features.shape[-1]),
        )
        embeddings[state_mask] = gathered_state[state_mask]
        return embeddings

    def _mrope_positions(self, token_types: Tensor) -> Tensor:
        batch_size, sequence_length = token_types.shape
        positions = torch.zeros(
            3,
            batch_size,
            sequence_length,
            dtype=torch.long,
            device=token_types.device,
        )
        position_mode = str(self.model_config.get("position_ids_type", "pi0fast"))
        for batch_index in range(batch_size):
            cursor = 0
            grid_index = 0
            values = token_types[batch_index].detach().cpu().tolist()
            for token_type, entries in itertools.groupby(enumerate(values), key=lambda item: item[1]):
                entries = list(entries)
                start, stop = entries[0][0], entries[-1][0] + 1
                if int(token_type) == G05TokenType.PADDING:
                    continue
                length = stop - start
                if int(token_type) == G05TokenType.IMAGE:
                    if grid_index >= len(self._last_vision_grids):
                        raise ValueError("G0.5 MRoPE received more image segments than vision grids.")
                    _, raw_h, raw_w = self._last_vision_grids[grid_index]
                    grid_index += 1
                    merge = int(self.model_config["vision"]["spatial_merge_size"])
                    grid_h, grid_w = raw_h // merge, raw_w // merge
                    height = (
                        torch.arange(grid_h, device=token_types.device).repeat_interleave(grid_w)[:length]
                        + cursor
                    )
                    width = torch.arange(grid_w, device=token_types.device).repeat(grid_h)[:length] + cursor
                    positions[0, batch_index, start:stop] = cursor
                    positions[1, batch_index, start:stop] = height
                    positions[2, batch_index, start:stop] = width
                    cursor += max(grid_h, grid_w)
                    continue

                if position_mode == "gaussian":
                    if self.training:
                        steps = (
                            torch.normal(
                                mean=2.0,
                                std=0.5,
                                size=(length,),
                                device=token_types.device,
                            )
                            .round()
                            .clamp(1, 3)
                            .long()
                        )
                    else:
                        steps = torch.full((length,), 2, dtype=torch.long, device=token_types.device)
                else:
                    steps = torch.ones(length, dtype=torch.long, device=token_types.device)
                text_positions = cursor + steps.cumsum(0) - steps[0]
                positions[:, batch_index, start:stop] = text_positions
                cursor = int(text_positions[-1]) + int(steps[-1])
        return positions

    @staticmethod
    def _causal_mask(token_types: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        valid = token_types != G05TokenType.PADDING
        sequence_length = token_types.shape[1]
        causal = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=token_types.device,
        ).tril()
        allowed = causal[None] & valid[:, None, :] & valid[:, :, None]
        full = torch.zeros(
            token_types.shape[0],
            1,
            sequence_length,
            sequence_length,
            dtype=dtype,
            device=token_types.device,
        )
        full.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
        return full, valid.to(dtype)

    @staticmethod
    def _proprio(samples: list[dict[str, Any]], device: torch.device) -> Tensor:
        rows = []
        for sample in samples:
            value = sample["proprio"]
            value = value["value"] if isinstance(value, Mapping) else value
            value = torch.as_tensor(value, dtype=torch.float32, device=device)
            rows.append(value.unsqueeze(0) if value.ndim == 1 else value)
        return torch.stack(rows)

    def _prefill(
        self,
        sequence: G05SequenceBatch,
        pixel_values: Mapping[str, Tensor],
        proprio: Tensor,
    ) -> tuple[Tensor, Any, Tensor]:
        embeddings = self._embed(sequence, pixel_values, proprio)
        positions = self._mrope_positions(sequence.token_types)
        full_mask, linear_mask = self._causal_mask(sequence.token_types, embeddings.dtype)
        hidden_states, cache = self.model.vlm(
            embeddings,
            full_attention_mask=full_mask,
            linear_attention_mask=linear_mask,
            position_ids=positions,
        )
        return hidden_states, cache, positions

    def _decode_token(
        self,
        token_ids: Tensor,
        *,
        token_types: Tensor,
        positions: Tensor,
        cache,
    ) -> tuple[Tensor, Tensor, Tensor]:
        embeddings = self.model.vlm.embed(token_ids[:, None])
        batch_size = token_ids.shape[0]
        next_positions = positions.amax(dim=-1, keepdim=True) + 1
        prefix_length = token_types.shape[1]
        prefix_mask = (token_types == G05TokenType.PADDING).to(embeddings.dtype)
        full_mask = torch.zeros(
            batch_size,
            1,
            1,
            prefix_length + 1,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        full_mask[..., :prefix_length].masked_fill_(
            prefix_mask[:, None, None].bool(), torch.finfo(embeddings.dtype).min
        )
        hidden_states, cache = self.model.vlm(
            embeddings,
            full_attention_mask=full_mask,
            linear_attention_mask=torch.ones(batch_size, 1, dtype=embeddings.dtype, device=embeddings.device),
            position_ids=next_positions,
            cache=cache,
        )
        next_types = torch.full(
            (batch_size, 1),
            float(G05TokenType.PRED_TEXT),
            dtype=token_types.dtype,
            device=token_types.device,
        )
        return (
            hidden_states[:, -1],
            torch.cat((token_types, next_types), dim=1),
            torch.cat((positions, next_positions), dim=-1),
        )

    def _generate_text(
        self,
        last_hidden: Tensor,
        *,
        token_types: Tensor,
        positions: Tensor,
        cache,
        max_new_tokens: int,
        stop_token_id: int,
    ) -> tuple[Tensor, Any, Tensor, Tensor, Tensor]:
        generated = []
        finished = torch.zeros(last_hidden.shape[0], dtype=torch.bool, device=last_hidden.device)
        for _ in range(max_new_tokens):
            logits = self.model.vlm.logits(last_hidden)
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, stop_token_id),
                next_token,
            )
            generated.append(next_token)
            last_hidden, token_types, positions = self._decode_token(
                next_token,
                token_types=token_types,
                positions=positions,
                cache=cache,
            )
            finished |= next_token == stop_token_id
            if bool(finished.all()):
                break
        generated_ids = (
            torch.stack(generated, dim=1)
            if generated
            else torch.empty(last_hidden.shape[0], 0, dtype=torch.long, device=last_hidden.device)
        )
        return generated_ids, cache, last_hidden, token_types, positions

    def _action_cache(self, vlm_cache, prefix_length: int, *, repeats: int = 1):
        cache = DynamicCache(config=self.model.action_expert.config)
        layer_types = self.model.vlm.config.layer_types
        for layer_index, layer_type in enumerate(layer_types):
            if layer_type != "full_attention":
                continue
            source = vlm_cache.layers[layer_index]
            if not source.is_initialized:
                continue
            key = source.keys[..., :prefix_length, :].detach()
            value = source.values[..., :prefix_length, :].detach()
            if repeats > 1:
                key = key.repeat_interleave(repeats, dim=0)
                value = value.repeat_interleave(repeats, dim=0)
            cache.layers[layer_index].update(key, value)
        return cache

    def _action_mask_and_positions(
        self,
        token_types: Tensor,
        positions: Tensor,
        horizon: int,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        batch_size, prefix_length = token_types.shape
        prefix_mask = (token_types == G05TokenType.PADDING).to(dtype) * torch.finfo(dtype).min
        action_mask = torch.zeros(
            batch_size,
            horizon,
            horizon,
            dtype=dtype,
            device=token_types.device,
        )
        if bool(self.model_config["fm"].get("action_causal", False)):
            action_mask = torch.triu(torch.full_like(action_mask, torch.finfo(dtype).min), diagonal=1)
        mask = torch.cat((prefix_mask[:, None].expand(-1, horizon, -1), action_mask), dim=-1).unsqueeze(1)
        offset = positions.amax(dim=-1, keepdim=True)
        action_positions = torch.arange(1, horizon + 1, device=token_types.device)[None, None] + offset
        return mask, action_positions

    def _velocity(
        self,
        actions: Tensor,
        timesteps: Tensor,
        *,
        vlm_cache,
        token_types: Tensor,
        positions: Tensor,
    ) -> Tensor:
        action_embeddings = self.model.action_expert.embed(actions)
        time_cond = self.model.action_expert.encode_time(timesteps)
        mask, action_positions = self._action_mask_and_positions(
            token_types, positions, actions.shape[1], action_embeddings.dtype
        )
        cache = self._action_cache(vlm_cache, token_types.shape[1])
        hidden_states = self.model.action_expert(
            action_embeddings,
            attention_mask=mask,
            position_ids=action_positions,
            cache=cache,
            time_cond=time_cond,
        )
        return self.model.action_expert.decode(hidden_states)

    def _infer_flow(
        self,
        *,
        vlm_cache,
        token_types: Tensor,
        positions: Tensor,
        action_dim_is_pad: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        fm = self.model_config["fm"]
        batch_size = token_types.shape[0]
        horizon = int(fm["horizon_steps"])
        action_dim = int(fm["action_dim"])
        action = torch.randn(
            batch_size,
            horizon,
            action_dim,
            device=token_types.device,
            dtype=dtype,
        )
        dim_mask = (
            action_dim_is_pad.bool().unsqueeze(1)
            if action_dim_is_pad is not None and not bool(fm["zero_pad_action_target"])
            else None
        )
        if dim_mask is not None:
            action.masked_fill_(dim_mask, 0)
        steps = int(fm["num_inference_steps"])
        delta = 1.0 / steps
        pi_convention = fm["time_convention"] == "pi_convention"
        time_value = 1.0 if pi_convention else 0.0
        timesteps = torch.full((batch_size,), time_value, dtype=dtype, device=token_types.device)
        for _ in range(steps):
            velocity = self._velocity(
                action,
                timesteps,
                vlm_cache=vlm_cache,
                token_types=token_types,
                positions=positions,
            )
            action = action - delta * velocity if pi_convention else action + delta * velocity
            timesteps = timesteps - delta if pi_convention else timesteps + delta
            if dim_mask is not None:
                action.masked_fill_(dim_mask, 0)
        clip = fm.get("final_action_clip_value")
        return action.clamp(-float(clip), float(clip)) if clip is not None else action

    def _flow_loss(
        self,
        actions: Tensor,
        *,
        action_is_pad: Tensor,
        action_dim_is_pad: Tensor | None,
        vlm_cache,
        token_types: Tensor,
        positions: Tensor,
    ) -> Tensor:
        fm = self.model_config["fm"]
        samples = int(fm.get("num_flow_samples", 1))
        batch_size = actions.shape[0]
        beta = torch.distributions.Beta(1.5, 1.0)
        z = beta.sample((samples, batch_size)).to(actions.device, actions.dtype)
        if fm["time_convention"] == "pi_convention":
            timesteps = 1 - (1 - float(fm["flow_sig_min"])) * (1 - z)
        else:
            timesteps = (1 - float(fm["flow_sig_min"])) * (1 - z)
        timesteps = timesteps.reshape(-1)
        noise = torch.randn(
            samples,
            *actions.shape,
            device=actions.device,
            dtype=actions.dtype,
        ).flatten(0, 1)
        target_actions = actions.repeat(samples, 1, 1)
        t = timesteps[:, None, None]
        if fm["time_convention"] == "pi_convention":
            interpolated = (1 - t) * target_actions + t * noise
            target_velocity = noise - target_actions
        else:
            interpolated = t * target_actions + (1 - t) * noise
            target_velocity = target_actions - noise

        repeated_dim_mask = None
        if action_dim_is_pad is not None:
            repeated_dim_mask = action_dim_is_pad.repeat(samples, 1)
            if not bool(fm["zero_pad_action_target"]):
                interpolated = interpolated.masked_fill(repeated_dim_mask[:, None], 0)

        repeated_types = token_types.repeat(samples, 1)
        repeated_positions = positions.repeat(1, samples, 1)
        action_embeddings = self.model.action_expert.embed(interpolated)
        time_cond = self.model.action_expert.encode_time(timesteps)
        mask, action_positions = self._action_mask_and_positions(
            repeated_types, repeated_positions, actions.shape[1], action_embeddings.dtype
        )
        cache = self._action_cache(vlm_cache, token_types.shape[1], repeats=samples)
        predicted = self.model.action_expert.decode(
            self.model.action_expert(
                action_embeddings,
                attention_mask=mask,
                position_ids=action_positions,
                cache=cache,
                time_cond=time_cond,
            )
        )
        weights = torch.ones_like(predicted)
        weights[action_is_pad.repeat(samples, 1)] = float(fm["padding_action_weight"])
        if repeated_dim_mask is not None and not bool(fm["zero_pad_action_target"]):
            weights.masked_fill_(
                repeated_dim_mask[:, None],
                float(fm["padding_action_weight"]),
            )
        loss = (weights * (predicted - target_velocity).square()).sum() / weights.sum().clamp_min(1)
        return loss * float(fm["fm_weight"])

    def predict_action(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        samples = list(batch["samples"])
        pixel_values = batch["pixel_values"]
        first_image = next(iter(pixel_values.values()))
        sequence = self.processor.encode_inference(samples, device=first_image.device)
        proprio = self._proprio(samples, first_image.device)
        hidden_states, cache, positions = self._prefill(sequence, pixel_values, proprio)
        result: dict[str, Any] = {}
        last_hidden = hidden_states[:, -1]
        token_types = sequence.token_types
        predict_cot = bool(batch.get(G05_RUNTIME_PREDICT_COT, self.model_config.get("predict_cot", False)))
        if predict_cot:
            generated, cache, last_hidden, token_types, positions = self._generate_text(
                last_hidden,
                token_types=sequence.token_types,
                positions=positions,
                cache=cache,
                max_new_tokens=int(self.model_config["ar"].get("max_new_tokens", 300)),
                stop_token_id=self.processor.eov_token_id,
            )
            sequence.token_types = token_types
            result["generated_ids"] = generated
            result["cot_text"] = [
                self.processor.decode(
                    ids[
                        : next(
                            (
                                index
                                for index, token_id in enumerate(ids.tolist())
                                if token_id == self.processor.eov_token_id
                            ),
                            len(ids),
                        )
                    ]
                )
                for ids in generated
            ]
        sequence.token_types = token_types
        if bool(self.model_config.get("continuous_action", False)):
            dim_mask = batch.get("action_dim_is_pad")
            if not isinstance(dim_mask, Tensor):
                dim_mask = None
            result[ACTION] = self._infer_flow(
                vlm_cache=cache,
                token_types=sequence.token_types,
                positions=positions,
                action_dim_is_pad=dim_mask,
                dtype=first_image.dtype,
            )
        if bool(self.model_config.get("discrete_action", False)):
            if self.action_tokenizer is None:
                if ACTION not in result:
                    raise RuntimeError(
                        "The native G0.5 ActionCodec checkpoint has not been loaded; "
                        "select the continuous flow head for this checkpoint."
                    )
            else:
                generated_action, _, _, _, _ = self._generate_text(
                    last_hidden,
                    token_types=sequence.token_types,
                    positions=positions,
                    cache=cache,
                    max_new_tokens=self.action_tokenizer.action_token_length + 32,
                    stop_token_id=self.processor.eos_token_id,
                )
                decoded_actions = []
                decoded_tokens = []
                absent_keys = []
                for token_row in generated_action:
                    is_action = (token_row >= self.processor.action_token_begin) & (
                        token_row < self.processor.action_token_end_with_markers
                    )
                    action_tokens = token_row[is_action]
                    decoded, absent = self.action_tokenizer.decode_language_tokens(
                        action_tokens,
                        horizon=int(self.model_config["fm"]["horizon_steps"]),
                        action_dim=int(self.model_config["fm"]["action_dim"]),
                    )
                    decoded_actions.append(decoded)
                    decoded_tokens.append(action_tokens)
                    absent_keys.append(absent)
                result["ar_action"] = torch.stack(decoded_actions)
                result["decoded_action_tokens"] = decoded_tokens
                result["ar_absent_keys"] = absent_keys
                if ACTION not in result:
                    result[ACTION] = result["ar_action"]
        result["_timing"] = {"forward_inference_total_ms": (time.monotonic() - start) * 1000}
        return result

    def forward(self, batch: Mapping[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        samples = list(batch["samples"])
        pixel_values = batch["pixel_values"]
        first_image = next(iter(pixel_values.values()))
        sequence = self.processor.encode_train(
            samples,
            device=first_image.device,
            action_codec=self.action_tokenizer,
        )
        proprio = self._proprio(samples, first_image.device)
        hidden_states, cache, positions = self._prefill(sequence, pixel_values, proprio)
        loss_dict: dict[str, Tensor] = {}

        ar_config = self.model_config.get("ar") or {}
        ce_weight = float(ar_config.get("ce_weight", 1.0))
        z_loss_scale = float(ar_config.get("ce_z_loss_scale", 0.0))
        if ce_weight < 0:
            raise ValueError("G0.5 ar.ce_weight must be non-negative.")
        if z_loss_scale < 0:
            raise ValueError("G0.5 ar.ce_z_loss_scale must be non-negative.")
        skip_ce = (
            bool(self.model_config.get("continuous_action", False))
            and not bool(self.model_config.get("discrete_action", False))
            and not bool(self.model_config.get("predict_cot", False))
        )
        if not skip_ce:
            shift_labels = sequence.labels[:, 1:]
            valid = shift_labels != IGNORE_INDEX
            if valid.any() and ce_weight:
                shift_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
                valid_hidden = shift_hidden[valid.reshape(-1)]
                valid_labels = shift_labels.reshape(-1)[valid.reshape(-1)]
                loss_dict["ce_loss"] = _autoregressive_ce_loss(
                    self.model.vlm.logits(valid_hidden),
                    valid_labels,
                    ce_weight=ce_weight,
                    z_loss_scale=z_loss_scale,
                )
            else:
                loss_dict["ce_loss"] = hidden_states.sum() * 0

        if bool(self.model_config.get("continuous_action", False)):
            actions = batch.get(ACTION)
            if not isinstance(actions, Tensor):
                raise ValueError("G0.5 flow training requires an action tensor.")
            action_is_pad = batch.get("action_is_pad")
            if not isinstance(action_is_pad, Tensor):
                action_is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
            action_dim_is_pad = batch.get("action_dim_is_pad")
            if not isinstance(action_dim_is_pad, Tensor):
                action_dim_is_pad = None
            prefix = int(sequence.split_index)
            loss_dict["fm_loss"] = self._flow_loss(
                actions,
                action_is_pad=action_is_pad,
                action_dim_is_pad=action_dim_is_pad,
                vlm_cache=cache,
                token_types=sequence.token_types[:, :prefix],
                positions=positions[..., :prefix],
            )
        loss = sum(loss_dict.values())
        return loss, loss_dict


def _native_backend(config: G05Config) -> nn.Module:
    if not config.author_model_config:
        raise ValueError(
            "G0.5 author_model_config is empty. Load a packaged checkpoint, or "
            "inject a backend explicitly for testing."
        )
    model_config = dict(config.author_model_config)
    model_config.update(
        {
            "predict_cot": config.predict_cot,
            "discrete_action": config.discrete_action,
            "continuous_action": config.continuous_action,
            "return_continuous_action": config.return_continuous_action,
        }
    )
    return G05NativeBackend.from_config(model_config)


class G05Policy(PreTrainedPolicy):
    """LeRobot policy surface for G0.5's unified CoT and action stream."""

    config_class = G05Config
    name = "g05"

    def __init__(self, config: G05Config, backend: nn.Module | None = None):
        super().__init__(config)
        config.validate_features()
        self.backend = backend if backend is not None else _native_backend(config)
        if not isinstance(self.backend, nn.Module):
            raise TypeError(f"G0.5 backend must be an nn.Module, got {type(self.backend)}.")
        self._action_queue: deque[Tensor] = deque()

    @classmethod
    def _load_as_safetensor(
        cls,
        model: G05Policy,
        model_file: str,
        map_location: str,
        strict: bool,
    ) -> G05Policy:
        device = resolve_safetensors_device(map_location)
        state_dict = load_file(model_file, device=device, backend="pread")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
        materialize = getattr(model.backend, "materialize_runtime_buffers", None)
        if callable(materialize):
            materialize(device)
        remaining_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        if remaining_meta:
            raise RuntimeError(f"G0.5 checkpoint did not materialize model parameters: {remaining_meta}")
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Error(s) loading G0.5 safetensors: missing={missing_keys}, unexpected={unexpected_keys}"
            )
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: G05Config | None = None,
        **kwargs,
    ) -> G05Policy:
        resolved_path = Path(pretrained_name_or_path)
        if not resolved_path.is_dir():
            resolved_path = Path(
                snapshot_download(
                    repo_id=str(pretrained_name_or_path),
                    token=kwargs.get("token"),
                    cache_dir=kwargs.get("cache_dir"),
                    local_files_only=kwargs.get("local_files_only", False),
                    revision=kwargs.get("revision"),
                )
            )
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                resolved_path,
                token=kwargs.get("token"),
                cache_dir=kwargs.get("cache_dir"),
                local_files_only=kwargs.get("local_files_only", False),
                revision=kwargs.get("revision"),
            )
        if not isinstance(config, G05Config):
            raise TypeError(f"Expected a G05Config, got {type(config).__name__}.")
        author_config = dict(config.author_model_config)
        author_config["hf_processor_path"] = str(resolved_path / "hf_processor")
        at_config = dict(author_config.get("AT_CONFIG") or {})
        at_config["ckpt_dir"] = str(resolved_path / "action_tokenizer.pt")
        author_config["AT_CONFIG"] = at_config
        author_config["pretrained_model_path"] = None
        config.author_model_config = author_config
        with torch.device("meta"):
            policy = super().from_pretrained(
                resolved_path,
                config=config,
                **kwargs,
            )
        if isinstance(policy.backend, G05NativeBackend) and policy.backend.action_tokenizer is None:
            action_config = policy.backend.model_config.get("AT_CONFIG")
            if isinstance(action_config, Mapping) and Path(str(action_config.get("ckpt_dir", ""))).is_file():
                policy.backend.action_tokenizer = G05NativeActionCodec.load(
                    action_config,
                    action_token_begin=policy.backend.processor.action_token_begin,
                ).to(next(policy.backend.parameters()).device)
        return policy

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        super()._save_pretrained(save_directory, state_dict=state_dict)
        author_config = dict(self.config.author_model_config)
        processor_value = author_config.get("hf_processor_path")
        processor_path = Path(str(processor_value)) if processor_value else None
        at_config = dict(author_config.get("AT_CONFIG") or {})
        tokenizer_value = at_config.get("ckpt_dir")
        tokenizer_path = Path(str(tokenizer_value)) if tokenizer_value else None
        roots = [
            path.parent for path in (processor_path, tokenizer_path) if path is not None and path.exists()
        ]

        if (
            processor_path is not None
            and processor_path.is_dir()
            and processor_path.resolve() != (save_directory / "hf_processor").resolve()
        ):
            shutil.copytree(processor_path, save_directory / "hf_processor", dirs_exist_ok=True)
        if (
            tokenizer_path is not None
            and tokenizer_path.is_file()
            and tokenizer_path.resolve() != (save_directory / "action_tokenizer.pt").resolve()
        ):
            shutil.copy2(tokenizer_path, save_directory / "action_tokenizer.pt")
        for name in (
            "g05_dataset_stats.json",
            "author_config.yaml",
            "LICENSE-G0.5",
            "LICENSE_QWEN3_5.txt",
            "THIRD_PARTY_NOTICES.md",
            "NOTICE",
            "README.md",
        ):
            source = next((root / name for root in roots if (root / name).is_file()), None)
            if source is not None and source.resolve() != (save_directory / name).resolve():
                shutil.copy2(source, save_directory / name)

        # Serialized paths are portable sidecar names. Local/Hub loading resolves them
        # against the downloaded checkpoint directory before constructing the native model.
        if (processor_path is not None and processor_path.exists()) or (
            tokenizer_path is not None and tokenizer_path.exists()
        ):
            portable = dict(author_config)
            portable["hf_processor_path"] = "hf_processor"
            portable_at = dict(portable.get("AT_CONFIG") or {})
            portable_at["ckpt_dir"] = "action_tokenizer.pt"
            portable["AT_CONFIG"] = portable_at
            portable["pretrained_model_path"] = None
            runtime_config = self.config.author_model_config
            self.config.author_model_config = portable
            self.config._save_pretrained(save_directory)
            self.config.author_model_config = runtime_config

    def reset(self) -> None:
        self._action_queue.clear()
        reset = getattr(self.backend, "reset", None)
        if callable(reset):
            reset()

    def _apply_author_inference_precision(self) -> None:
        """Match the released serving path's BF16 weights with declared FP32 islands."""

        self.backend.to(dtype=torch.bfloat16)
        apply_fp32_params = getattr(self.backend, "apply_fp32_params", None)
        if callable(apply_fp32_params):
            apply_fp32_params()
        if self.config.predict_cot:
            # The author Qwen3.5 final norm is an FP32 island and its fused CE
            # kernel disables autocast, so the tied output projection must be
            # FP32 as well. Otherwise CoT training reaches FLCE with FP32 hidden
            # states and a BF16 weight and fails before computing text loss.
            model = getattr(self.backend, "model", None)
            vlm = getattr(model, "vlm", None)
            output_proj = getattr(vlm, "output_proj", None)
            weight = getattr(output_proj, "weight", None)
            if isinstance(weight, nn.Parameter):
                weight.data = weight.data.float()

    def to(self, *args, **kwargs) -> G05Policy:
        """Apply the released inference precision and move the ActionCodec sidecar."""

        result = super().to(*args, **kwargs)
        explicit_dtype = "dtype" in kwargs or any(isinstance(arg, torch.dtype | Tensor) for arg in args)
        if (
            self.config.model_weights_to_bf16
            and not explicit_dtype
            and next(self.backend.parameters()).device.type == "cuda"
        ):
            self._apply_author_inference_precision()
        action_tokenizer = getattr(self.backend, "action_tokenizer", None)
        move_tokenizer = getattr(action_tokenizer, "to", None)
        if callable(move_tokenizer):
            device = next(self.backend.parameters()).device
            move_tokenizer(device)
        return result

    def get_optim_params(self) -> OptimizerParams:
        get_param_groups = getattr(self.backend, "get_optim_param_groups", None)
        if callable(get_param_groups):
            return get_param_groups(
                lr=self.config.optimizer_lr,
                weight_decay=self.config.optimizer_weight_decay,
                apply_decay_on_norm_and_bias=self.config.optimizer_apply_decay_on_norm_and_bias,
                backbone_lr_multiplier=self.config.optimizer_backbone_lr_multiplier,
                vision_lr_multiplier=self.config.optimizer_vision_lr_multiplier,
            )
        get_params = getattr(self.backend, "get_optim_params", None)
        if callable(get_params):
            params = get_params()
            return [params] if isinstance(params, dict) and "params" in params else params
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @staticmethod
    def _task_values(batch: Mapping[str, Any], task: str | None, batch_size: int) -> list[str]:
        if task is not None:
            return [task] * batch_size
        value = batch.get("task")
        if isinstance(value, str):
            return [value] * batch_size
        if isinstance(value, list | tuple) and len(value) == batch_size:
            return [str(item) for item in value]
        raise ValueError(
            "G0.5 requires the already-selected LeRobot task string; no task augmentation "
            "or model-local sampling is performed."
        )

    @staticmethod
    def _batch_item(value: Any, index: int, batch_size: int) -> Any:
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            return value[index]
        if isinstance(value, list | tuple) and len(value) == batch_size:
            return value[index]
        return value

    def _recipe_cot_targets(
        self,
        batch: Mapping[str, Any],
        index: int,
        batch_size: int,
    ) -> tuple[str | None, str | None]:
        """Read the selected recipe's supervised Subtask/BBox messages."""

        messages = batch.get("messages")
        target_indices = batch.get("target_message_indices")
        if messages is None or target_indices is None:
            return None, None

        sample_messages = messages
        if (
            isinstance(messages, list | tuple)
            and len(messages) == batch_size
            and (not messages or isinstance(messages[0], list | tuple))
        ):
            sample_messages = messages[index]
        sample_target_indices = target_indices
        has_batched_target_indices = (isinstance(target_indices, Tensor) and target_indices.ndim > 1) or (
            isinstance(target_indices, list | tuple)
            and len(target_indices) == batch_size
            and (not target_indices or isinstance(target_indices[0], list | tuple | Tensor))
        )
        if has_batched_target_indices:
            sample_target_indices = target_indices[index]
        if isinstance(sample_messages, Mapping):
            sample_messages = [sample_messages]
        if isinstance(sample_target_indices, Tensor):
            sample_target_indices = sample_target_indices.detach().cpu().tolist()
        if not isinstance(sample_messages, list | tuple) or not isinstance(
            sample_target_indices, list | tuple
        ):
            return None, None

        subtask: str | None = None
        bbox_json: str | None = None
        for target_index in sample_target_indices:
            message = sample_messages[int(target_index)]
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str):
                continue
            if content.startswith("Subtask:"):
                value = content.removeprefix("Subtask:").strip()
                if value:
                    subtask = value
            elif content.startswith("BBoxJSON:"):
                value = content.removeprefix("BBoxJSON:").strip()
                if value:
                    bbox_json = value
        return subtask, bbox_json

    @staticmethod
    def _format_bbox_target(bbox_json: str | None, image_size: tuple[int, int]) -> str | None:
        """Convert LeRobot grounded-VQA JSON into G0.5's location-token format."""

        if not bbox_json:
            return None
        try:
            payload = json.loads(bbox_json)
            if isinstance(payload, str):
                payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        if isinstance(payload.get("answer"), Mapping):
            payload = payload["answer"]

        height, width = image_size
        boxes: list[tuple[str, list[float]]] = []
        detections = payload.get("detections")
        if isinstance(detections, list):
            for detection in detections:
                if not isinstance(detection, Mapping) or detection.get("bbox_format", "xyxy") != "xyxy":
                    continue
                coords = detection.get("bbox")
                if not isinstance(coords, list | tuple) or len(coords) != 4:
                    continue
                label = str(detection.get("label") or "object")
                boxes.append((label, [float(value) for value in coords]))
        else:
            for label, coords in payload.items():
                if isinstance(coords, list | tuple) and len(coords) == 4:
                    boxes.append((str(label), [float(value) for value in coords]))
        if not boxes:
            return None

        def normalize(coords: list[float]) -> list[float]:
            if max(abs(value) for value in coords) <= 1.0:
                return coords
            x1, y1, x2, y2 = coords
            return [x1 / width, y1 / height, x2 / width, y2 / height]

        def location_token(value: float) -> str:
            location = max(0, min(1023, round(value * 1024)))
            return f"<loc{location:04d}>"

        formatted = []
        for label, raw_coords in boxes:
            x1, y1, x2, y2 = normalize(raw_coords)
            locations = "".join(location_token(value) for value in (y1, x1, y2, x2))
            formatted.append(f"{label} {locations}")
        return "BBox: " + "; ".join(formatted)

    def _apply_recipe_cot(
        self,
        sample: dict[str, Any],
        batch: Mapping[str, Any],
        index: int,
        batch_size: int,
    ) -> bool:
        """Populate one author sample from recipe-rendered CoT targets."""

        subtask, bbox_json = self._recipe_cot_targets(batch, index, batch_size)
        image_size = batch.get("g05_bbox_image_size")
        if (
            isinstance(image_size, list | tuple)
            and len(image_size) == batch_size
            and image_size
            and isinstance(image_size[0], list | tuple | Tensor)
        ):
            image_size = image_size[index]
        if isinstance(image_size, Tensor):
            image_size = image_size.detach().cpu().tolist()
        if not isinstance(image_size, list | tuple) or len(image_size) != 2:
            camera = self.config.cot_bbox_camera or self.config.camera_order[0]
            image_size = self.config.camera_sizes[camera]
        bbox = self._format_bbox_target(bbox_json, (int(image_size[0]), int(image_size[1])))

        fields = tuple(field for field, value in (("bbox", bbox), ("subtask", subtask)) if value)
        if not fields:
            return False
        flow_only = "<action_action" not in self.config.prompt_template
        sample["template"] = make_g05_cot_prompt_template(
            self.config.num_prompt_images,
            fields=fields,
            flow_only=flow_only,
        )
        if bbox is not None:
            sample["bbox"] = bbox
        if subtask is not None:
            sample["atomic_task"] = f"Subtask: {subtask}"
        sample["prompt"] = {
            ("bbox",): "predict bbox",
            ("subtask",): "predict subtask",
            ("bbox", "subtask"): "predict bbox, subtask and action",
        }[fields]
        return True

    def _prepare_author_batch(
        self,
        batch: Mapping[str, Any],
        task: str | None = None,
        *,
        predict_cot: bool | None = None,
    ) -> dict[str, Any]:
        run_predict_cot = self.config.predict_cot if predict_cot is None else predict_cot
        prepare = getattr(self.backend, "prepare_lerobot_batch", None)
        if callable(prepare):
            prepared = prepare(batch, task=task, config=self.config)
            prepared[G05_RUNTIME_PREDICT_COT] = run_predict_cot
            return prepared

        state = batch.get(OBS_STATE)
        if not isinstance(state, Tensor):
            raise ValueError(f"G0.5 requires tensor {OBS_STATE!r}.")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        tasks = self._task_values(batch, task, batch_size)
        state_mask = batch.get("proprio_dim_is_pad")
        if state_mask is None:
            state_mask = torch.zeros(
                batch_size, self.config.policy_state_dim, dtype=torch.bool, device=state.device
            )
        elif isinstance(state_mask, Tensor) and state_mask.ndim == 1:
            state_mask = state_mask.unsqueeze(0).expand(batch_size, -1)

        pixel_values: dict[str, Tensor] = {}
        for key in self.config.camera_order:
            image = batch.get(key)
            if not isinstance(image, Tensor):
                raise ValueError(f"G0.5 requires camera {key!r}; camera order is checkpoint state.")
            if image.ndim == 4:
                image = image.unsqueeze(1)
            pixel_values[key] = image
        image_count = sum(image.shape[1] for image in pixel_values.values())
        if image_count != self.config.num_input_images:
            raise ValueError(
                f"G0.5 received {image_count} camera/history frames, but the checkpoint "
                f"template requires {self.config.num_input_images}."
            )

        samples = []
        flow_only = "<action_action" not in self.config.prompt_template
        inference_template = (
            self.config.prompt_template
            if run_predict_cot
            else make_g05_prompt_template(
                self.config.num_prompt_images,
                predict_cot=False,
                flow_only=flow_only,
            )
        )
        for index, raw_task in enumerate(tasks):
            proprio = state[index]
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            sample = {
                "template": inference_template,
                # This is the author InputPreprocessor command slot. Keep it byte-for-byte
                # unchanged; checkpoint-specific chat formatting occurs downstream.
                "command": raw_task,
                "embodiment": self.config.embodiment,
                "proprio": {
                    "value": proprio,
                    "proprio_dim_is_pad": state_mask[index],
                },
            }
            frequency = self.config.processor_metadata.get("frequency")
            if frequency is not None:
                sample["frequency"] = frequency
            if run_predict_cot:
                rendered_recipe = "messages" in batch
                applied_recipe_cot = rendered_recipe and self._apply_recipe_cot(
                    sample, batch, index, batch_size
                )
                if not applied_recipe_cot:
                    # During mixed-recipe training an applicable no-CoT branch is a
                    # genuine target format. At inference, where actions are absent,
                    # retain the checkpoint's configured System 2 prompt.
                    if rendered_recipe and isinstance(batch.get(ACTION), Tensor):
                        sample["template"] = make_g05_prompt_template(
                            self.config.num_prompt_images,
                            predict_cot=False,
                            flow_only="<action_action" not in self.config.prompt_template,
                        )
                    else:
                        sample["prompt"] = "predict subtask"
                        atomic_task = batch.get("atomic_task")
                        if atomic_task is not None:
                            atomic_task = str(self._batch_item(atomic_task, index, batch_size))
                            sample["atomic_task"] = (
                                atomic_task
                                if atomic_task.startswith("Subtask:")
                                else f"Subtask: {atomic_task}"
                            )
            for image_index in range(self.config.num_prompt_images):
                camera = self.config.camera_order[image_index % len(self.config.camera_order)]
                sample[f"image{image_index}"] = self.config.camera_sizes[camera]
            action = batch.get(ACTION)
            if "<action_action" in self.config.prompt_template:
                if not isinstance(action, Tensor):
                    action = state.new_zeros(
                        batch_size, self.config.chunk_size, self.config.policy_action_dim
                    )
                action_dim_is_pad = batch.get("action_dim_is_pad")
                if action_dim_is_pad is None:
                    action_dim_is_pad = torch.zeros(
                        batch_size,
                        self.config.policy_action_dim,
                        dtype=torch.bool,
                        device=action.device,
                    )
                elif action_dim_is_pad.ndim == 1:
                    action_dim_is_pad = action_dim_is_pad.unsqueeze(0).expand(batch_size, -1)
                action_payload = {
                    "value": action[index],
                    "action_dim_is_pad": action_dim_is_pad[index],
                }
                action_op_mask = batch.get("action_op_mask")
                if isinstance(action_op_mask, Tensor):
                    action_payload["action_op_mask"] = (
                        action_op_mask[index] if action_op_mask.ndim > 1 else action_op_mask
                    )
                else:
                    action_payload["action_op_mask"] = ~action_dim_is_pad[index]
                action_payload["parts_meta"] = batch.get(
                    "action_parts_meta", G05_POLICY_PARTS[self.config.policy_action_dim]
                )
                sample["action"] = action_payload
            samples.append(sample)
        prepared = dict(batch)
        prepared["samples"] = samples
        prepared["pixel_values"] = pixel_values
        prepared[G05_RUNTIME_PREDICT_COT] = run_predict_cot
        return prepared

    def _run_inference(
        self,
        batch: Mapping[str, Any],
        *,
        task: str | None = None,
        system_mode: str | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        if system_mode is None:
            system_mode = self.config.runtime_system
        if system_mode not in {"system1", "system2"}:
            raise ValueError("G0.5 system_mode must be 'system1' or 'system2'.")
        if system_mode == "system2" and not self.config.predict_cot:
            raise ValueError("G0.5 System 2 requires predict_cot=True in the packaged checkpoint.")
        prepared = self._prepare_author_batch(
            batch,
            task=task,
            predict_cot=system_mode == "system2",
        )
        predict = getattr(self.backend, "predict_action", None)
        device = next(self.backend.parameters()).device
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.model_weights_to_bf16 and device.type == "cuda",
        ):
            result = predict(prepared) if callable(predict) else self.backend(prepared)
        if isinstance(result, Tensor):
            result = {ACTION: result}
        if not isinstance(result, Mapping):
            raise TypeError("G0.5 backend inference must return a tensor or mapping.")

        if self.config.action_head == "actioncodec":
            action = result.get("ar_action", result.get(ACTION))
        else:
            action = result.get(ACTION)
        if not isinstance(action, Tensor):
            raise ValueError(f"G0.5 {self.config.action_head} output is missing its action tensor.")
        metadata_keys = ("decoded_action_tokens", "ar_absent_keys", "_timing")
        if system_mode == "system2":
            metadata_keys = ("cot_text", "generated_ids", *metadata_keys)
        metadata = {key: result[key] for key in metadata_keys if key in result}
        return action, metadata

    def predict_action_chunk_with_runtime(
        self,
        batch: dict[str, Any],
        *,
        task: str,
        system_mode: str | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Run the selected system and return its action plus same-pass telemetry."""

        return self._run_inference(batch, task=task, system_mode=system_mode)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> Tensor:
        action, _ = self._run_inference(batch)
        return action

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            if chunk.ndim != 3:
                raise ValueError(f"G0.5 action chunk must be [B,T,D], got {tuple(chunk.shape)}.")
            # LeRobot's synchronous select_action queue is intentionally batch-size one.
            if chunk.shape[0] != 1:
                raise ValueError(
                    "G0.5 select_action requires batch size 1; use predict_action_chunk for B>1."
                )
            self._action_queue.extend(chunk[0, : self.config.n_action_steps])
        return self._action_queue.popleft().unsqueeze(0)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any] | None]:
        prepared = self._prepare_author_batch(batch)
        device = next(self.backend.parameters()).device
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.model_weights_to_bf16 and device.type == "cuda",
        ):
            result = self.backend(prepared)
        if isinstance(result, tuple) and len(result) == 2:
            loss, loss_dict = result
        elif isinstance(result, Mapping) and "loss" in result:
            loss = result["loss"]
            loss_dict = {key: value for key, value in result.items() if key != "loss"}
        else:
            raise TypeError("G0.5 training backend must return (loss, loss_dict) or {'loss': ...}.")
        if not isinstance(loss, Tensor):
            raise TypeError("G0.5 training loss must be a torch.Tensor.")
        logging_values = {
            key: value.detach().item() if isinstance(value, Tensor) and value.numel() == 1 else value
            for key, value in (loss_dict or {}).items()
        }
        return loss, logging_values
