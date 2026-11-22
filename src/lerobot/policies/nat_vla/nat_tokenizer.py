# Copyright 2026 Ahad Jawaid and The HuggingFace Inc. team. All rights reserved.
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

"""NAT tokenizer: action chunks to time-ordered tokens and back."""

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional


class NATTokenizer(nn.Module):
    """Action chunks to time-ordered tokens and back. Actions wider than group_width columns are split into
    groups of columns, each tokenized on its own, and their tokens concatenated. prefix_token_counts maps a
    number of steps to the tokens of one group that decode them."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        encoder, quantizer = config["encoder"], config["quantizer"]
        self.time_horizon, self.group_width = config["time_horizon"], config["group_width"]
        values_per_latent = math.prod(encoder["downsampling_ratios"])
        if values_per_latent % self.group_width or self.time_horizon * self.group_width % values_per_latent:
            raise ValueError("each latent must cover whole steps and the chunk whole latents")
        self.steps_per_latent = steps_per_latent = values_per_latent // self.group_width
        latents = self.time_horizon // steps_per_latent

        self.encoder = ActionEncoder(encoder)
        self.quantizer = MultiScaleFSQ(encoder["latent_dim"], latents=latents, **quantizer)
        self.decoder = ActionDecoder(config["decoder"], encoder["latent_dim"], values_per_latent, latents)

        self.vocab_size = len(quantizer["scales"]) * self.quantizer.codebook_size
        self.tokens_per_chunk = len(self.quantizer.token_order)
        self.token_latent_spans = [
            (start, min(start + quantizer["scales"][scale], latents))
            for start, scale, _ in self.quantizer.token_order
        ]
        self.prefix_token_counts = {
            (latent + 1) * steps_per_latent: sum(
                start <= latent for start, _, _ in self.quantizer.token_order
            )
            for latent in range(latents)
        }

    def encode(self, actions: Tensor) -> Tensor:
        """(batch, time_horizon, action_dim) actions in [-1, 1] to (batch, tokens) token ids, computed in float32
        under any autocast."""
        groups = -(-actions.shape[-1] // self.group_width)
        packed = functional.pad(
            actions.float().clamp(-1, 1), (0, groups * self.group_width - actions.shape[-1])
        )
        packed = packed.unflatten(-1, (groups, self.group_width)).transpose(1, 2).flatten(0, 1)

        with torch.autocast(actions.device.type, enabled=False):
            tokens = self.quantizer.tokens(self.encoder(packed.reshape(len(packed), 1, -1)))
        return tokens.reshape(len(actions), -1)

    def decode(self, tokens: Tensor, action_dim: int) -> Tensor:
        """(batch, tokens) token ids, whole chunks or one group's first latents, to (batch, time_horizon,
        action_dim) actions, computed in float32 under any autocast."""
        groups = -(-action_dim // self.group_width)
        tokens = functional.pad(tokens, (0, groups * self.tokens_per_chunk - tokens.shape[1]))
        tokens = tokens.reshape(-1, self.tokens_per_chunk)

        with torch.autocast(tokens.device.type, enabled=False):
            actions = self.decoder(self.quantizer.features(tokens)).reshape(
                len(tokens), self.time_horizon, -1
            )
        actions = actions.clamp(-1, 1).unflatten(0, (-1, groups)).transpose(1, 2).flatten(2)
        return actions[..., :action_dim]

    def token_count(self, steps: int, action_dim: int) -> tuple[int, int]:
        """The tokens that decode the first `steps` steps, rounded up to whole latents, and the steps they
        decode."""
        groups = -(-action_dim // self.group_width)
        if groups > 1:
            return groups * self.tokens_per_chunk, self.time_horizon
        return next((count, prefix) for prefix, count in self.prefix_token_counts.items() if prefix >= steps)

    def tokens_with_actions(self, real_steps: Tensor, action_dim: int) -> Tensor:
        """(batch, time_horizon) whether each step is a real action, to (batch, tokens): whether each token
        describes at least one real step."""
        real_latents = real_steps.unflatten(1, (-1, self.steps_per_latent)).any(-1)
        has_actions = torch.stack(
            [real_latents[:, start:end].any(1) for start, end in self.token_latent_spans], 1
        )
        return has_actions.repeat(1, -(-action_dim // self.group_width))

    def tokens_starting_with(self, actions: Tensor) -> Tensor:
        """The first tokens of a chunk that starts with these (batch, steps, action_dim) actions, for real-
        time chunking: the actions are padded to the chunk by repeating the last one, and the tokens of their
        latents kept."""
        steps = actions.shape[1]
        repeated = actions[:, -1:].expand(-1, self.time_horizon - steps, -1)
        return self.encode(torch.cat([actions, repeated], 1))[:, : self.prefix_token_counts[steps]]

    def stream(self, action_dim: int, first_tokens: Tensor | None = None) -> "StreamingDecoder":
        return StreamingDecoder(self, action_dim, first_tokens)


class StreamingDecoder:
    """Decodes a chunk while its tokens are generated: push() each new token (one per sample) and, when the
    tokens complete a latent, it returns that latent's actions. A chunk can start from first_tokens, whose
    actions are not returned."""

    def __init__(self, tokenizer: NATTokenizer, action_dim: int, first_tokens: Tensor | None = None) -> None:
        self.tokenizer, self.action_dim, self.tokens = tokenizer, action_dim, first_tokens
        counts = (tokenizer.token_count(steps, action_dim) for steps in tokenizer.prefix_token_counts)
        self.steps_after_tokens = dict(counts)
        self.decoded_steps = 0 if first_tokens is None else self.steps_after_tokens[first_tokens.shape[1]]

    def push(self, token: Tensor) -> Tensor | None:
        """The (batch, steps, action_dim) actions the token completes, or None."""
        self.tokens = token[:, None] if self.tokens is None else torch.cat([self.tokens, token[:, None]], 1)
        steps = self.steps_after_tokens.get(self.tokens.shape[1])
        if steps is None:
            return None

        actions = self.tokenizer.decode(self.tokens, self.action_dim)[:, self.decoded_steps : steps]
        self.decoded_steps = steps
        return actions


class ActionEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        filters = config["filters"]
        self.input_conv = CausalConv1d(config["channels"], filters, config["kernel_size"])
        self.stages = nn.ModuleList()
        for ratio in config["downsampling_ratios"]:
            self.stages.append(
                EncoderStage(filters, config["compress"], config["residual_kernel_size"], ratio)
            )
            filters *= 2
        self.lstm = nn.LSTM(filters, filters, config["lstm_layers"])
        self.output_conv = CausalConv1d(filters, config["latent_dim"], config["last_kernel_size"])

    def forward(self, actions: Tensor) -> Tensor:
        hidden = self.input_conv(actions)
        for stage in self.stages:
            hidden = stage(hidden)
        sequence = hidden.permute(2, 0, 1)
        return self.output_conv(functional.elu((self.lstm(sequence)[0] + sequence).permute(1, 2, 0)))


class EncoderStage(nn.Module):
    def __init__(self, channels: int, compress: int, kernel_size: int, ratio: int) -> None:
        super().__init__()
        self.residual = nn.ModuleDict(
            {
                "conv": CausalConv1d(channels, channels // compress, kernel_size),
                "projection": CausalConv1d(channels // compress, channels, 1),
                "shortcut": CausalConv1d(channels, channels, 1),
            }
        )
        self.downsample = CausalConv1d(channels, 2 * channels, 2 * ratio, stride=ratio)

    def forward(self, hidden: Tensor) -> Tensor:
        unit = self.residual
        hidden = unit.shortcut(hidden) + unit.projection(functional.elu(unit.conv(functional.elu(hidden))))
        return self.downsample(functional.elu(hidden))


class CausalConv1d(nn.Conv1d):
    """Zero-pads the input on the left, so each output sees only current and earlier steps."""

    def forward(self, hidden: Tensor) -> Tensor:
        kernel, stride = (self.kernel_size[0] - 1) * self.dilation[0] + 1, self.stride[0]
        frames = math.ceil((hidden.shape[-1] - stride) / stride + 1)
        right = (frames - 1) * stride + stride - hidden.shape[-1]
        return super().forward(functional.pad(hidden, (kernel - stride, right)))


class MultiScaleFSQ(nn.Module):
    """Residual FSQ at several time scales. Tokens are ordered by the first latent they cover, so a prefix
    of whole latents decodes on its own."""

    def __init__(self, latent_dim: int, levels: list[int], scales: list[int], latents: int) -> None:
        super().__init__()
        self.scales, self.latents, self.codebook_size = scales, latents, math.prod(levels)
        self.project_in = nn.ModuleList([nn.Linear(latent_dim, len(levels)) for _ in scales])
        self.project_out = nn.ModuleList([nn.Linear(len(levels), latent_dim) for _ in scales])
        self.register_buffer("levels", torch.tensor(levels), persistent=False)
        self.register_buffer("basis", torch.tensor([1, *levels[:-1]]).cumprod(0), persistent=False)
        self.token_order = sorted(
            (position * span, scale, position)
            for scale, span in enumerate(scales)
            for position in range(-(-latents // span))
        )
        self.columns = [
            [index for index, (_, owner, _) in enumerate(self.token_order) if owner == s]
            for s in range(len(scales))
        ]

    def tokens(self, latents: Tensor) -> Tensor:
        residual, codes = latents, []
        for scale, span in enumerate(self.scales):
            pooled = functional.avg_pool1d(residual, span) if span > 1 else residual
            indices, quantized = self.quantize(self.project_in[scale](pooled.transpose(1, 2)).float())
            residual = residual - self.upsampled(self.project_out[scale](quantized).transpose(1, 2), span)
            codes.append(indices + scale * self.codebook_size)
        return torch.stack([codes[scale][:, position] for _, scale, position in self.token_order], dim=1)

    def features(self, tokens: Tensor) -> Tensor:
        features = torch.zeros((), device=tokens.device)
        for scale, span in enumerate(self.scales):
            indices = (tokens[:, self.columns[scale]] - scale * self.codebook_size).clamp(
                0, self.codebook_size - 1
            )
            digits = (indices.unsqueeze(-1) // self.basis) % self.levels
            codes = (digits - self.levels // 2) / (self.levels // 2)
            features = features + self.upsampled(self.project_out[scale](codes).transpose(1, 2), span)
        return features

    def quantize(self, projected: Tensor) -> tuple[Tensor, Tensor]:
        half_level = (self.levels - 1) * (1 + 1e-3) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        bounded = (projected + (offset / half_level).atanh()).tanh() * half_level - offset
        codes = (bounded + (bounded.round() - bounded)) / (self.levels // 2)
        return ((codes * (self.levels // 2) + self.levels // 2) * self.basis).sum(-1).long(), codes

    def upsampled(self, quantized: Tensor, span: int) -> Tensor:
        return torch.repeat_interleave(quantized, span, dim=-1)[..., : self.latents]


class ActionDecoder(nn.Module):
    def __init__(self, config: dict[str, Any], latent_dim: int, values_per_latent: int, latents: int) -> None:
        super().__init__()
        dim, groups, conditions = config["dim"], config["group_norm_groups"], config["condition_embeddings"]
        self.embed = CausalConv1d(latent_dim, dim, 7)
        blocks = [ResnetBlock(dim, groups) for _ in range(4)]
        blocks.insert(2, CausalSelfAttention(dim, groups, config["attention_context"], latents))
        self.temporal_blocks = nn.Sequential(*blocks)
        self.temporal_norm = PerTimeGroupNorm(groups, dim, eps=1e-6)
        self.input_norm = ConditionedLayerNorm(conditions, dim)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(dim, config["intermediate_dim"], conditions) for _ in range(config["num_blocks"])]
        )
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, values_per_latent)
        self.register_buffer("condition_id", torch.tensor([config["condition_id"]]), persistent=False)

    def forward(self, features: Tensor) -> Tensor:
        cudnn = torch.backends.cudnn
        with cudnn.flags(cudnn.enabled, cudnn.benchmark, deterministic=cudnn.deterministic, allow_tf32=False):
            hidden = self.temporal_norm(self.temporal_blocks(self.embed(features))).transpose(1, 2)
            hidden = self.input_norm(hidden, self.condition_id).transpose(1, 2)
            for block in self.blocks:
                hidden = block(hidden, self.condition_id)
            return self.head(self.output_norm(hidden.transpose(1, 2))).reshape(len(features), -1)


class ResnetBlock(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = (PerTimeGroupNorm(groups, channels, eps=1e-6) for _ in range(2))
        self.conv1, self.conv2 = (CausalConv1d(channels, channels, 3) for _ in range(2))

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.conv2(swish(self.norm2(self.conv1(swish(self.norm1(hidden))))))


class CausalSelfAttention(nn.Module):
    """Attention over the current and a window of earlier latents."""

    def __init__(self, channels: int, groups: int, context: int, latents: int) -> None:
        super().__init__()
        self.norm = PerTimeGroupNorm(groups, channels, eps=1e-6)
        self.query, self.key, self.value, self.output = (nn.Conv1d(channels, channels, 1) for _ in range(4))
        distances = torch.arange(latents)[:, None] - torch.arange(latents)[None, :]
        self.register_buffer("allowed", (distances >= 0) & (distances < context), persistent=False)

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = self.norm(hidden)
        queries, keys, values = self.query(normalized), self.key(normalized), self.value(normalized)
        weights = torch.bmm(queries.transpose(1, 2), keys) * queries.shape[1] ** -0.5
        allowed = self.allowed[: queries.shape[-1], : keys.shape[-1]]
        weights = weights.masked_fill(~allowed, -torch.inf).softmax(-1)
        return hidden + self.output(torch.bmm(values, weights.transpose(1, 2)))


class PerTimeGroupNorm(nn.GroupNorm):
    """Group normalization of each latent on its own, so no latent sees a later one."""

    def forward(self, hidden: Tensor) -> Tensor:
        batch, channels, length = hidden.shape
        normalized = super().forward(hidden.transpose(1, 2).reshape(batch * length, channels, 1))
        return normalized.reshape(batch, length, channels).transpose(1, 2)


class ConditionedLayerNorm(nn.Module):
    def __init__(self, conditions: int, channels: int) -> None:
        super().__init__()
        self.scale, self.shift = nn.Embedding(conditions, channels), nn.Embedding(conditions, channels)

    def forward(self, hidden: Tensor, condition_id: Tensor) -> Tensor:
        normalized = functional.layer_norm(hidden, (hidden.shape[-1],), eps=1e-6)
        return normalized * self.scale(condition_id) + self.shift(condition_id)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, intermediate_channels: int, conditions: int) -> None:
        super().__init__()
        self.depthwise_conv = CausalConv1d(channels, channels, 7, groups=channels)
        self.norm = ConditionedLayerNorm(conditions, channels)
        self.expand, self.contract = (
            nn.Linear(channels, intermediate_channels),
            nn.Linear(intermediate_channels, channels),
        )
        self.layer_scale = nn.Parameter(torch.ones(channels))

    def forward(self, hidden: Tensor, condition_id: Tensor) -> Tensor:
        update = self.norm(self.depthwise_conv(hidden).transpose(1, 2), condition_id)
        return hidden + (self.layer_scale * self.contract(functional.gelu(self.expand(update)))).transpose(
            1, 2
        )


def swish(hidden: Tensor) -> Tensor:
    return hidden * torch.sigmoid(hidden)
