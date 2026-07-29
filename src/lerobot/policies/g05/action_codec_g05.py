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

"""Native inference implementation of G0.5's ActionCodec sidecar."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


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
        self.ls1 = nn.Parameter(torch.empty(dimension))
        self.ls2 = nn.Parameter(torch.empty(dimension))
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
