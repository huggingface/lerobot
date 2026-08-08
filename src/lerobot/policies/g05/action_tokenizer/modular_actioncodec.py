# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0

from __future__ import annotations

import math

import torch
import torch.distributed as distributed
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


def _sample_vectors(samples: Tensor, count: int) -> Tensor:
    if samples.shape[0] >= count:
        indices = torch.randperm(samples.shape[0], device=samples.device)[:count]
    else:
        indices = torch.randint(0, samples.shape[0], (count,), device=samples.device)
    return samples[indices].float()


def _kmeans(samples: Tensor, num_clusters: int, num_iterations: int = 10) -> tuple[Tensor, Tensor]:
    dimension = samples.shape[-1]
    means = _sample_vectors(samples, num_clusters)
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


def _ema_inplace(moving_average: Tensor, value: Tensor, decay: float) -> None:
    moving_average.data.mul_(decay).add_(value.float(), alpha=1 - decay)


def _rotation_trick_ste(encoded: Tensor, quantized: Tensor) -> Tensor:
    encoded_float = encoded.float()
    quantized_float = quantized.float()
    encoded_norm = encoded_float.norm(dim=1, keepdim=True).clamp(min=1e-8)
    quantized_norm = quantized_float.norm(dim=1, keepdim=True).clamp(min=1e-8)
    rotated = encoded_float / encoded_norm * quantized_norm
    return (quantized_float - rotated).detach() + rotated


class EMAVectorQuantizer(nn.Module):
    """EMA vector quantizer with trainable projections."""

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float,
        decay: float,
        threshold_ema_dead: float,
        use_rotation_trick: bool,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead = threshold_ema_dead
        self.use_rotation_trick = use_rotation_trick
        self.in_proj = nn.Linear(input_dim, codebook_dim, bias=False)
        self.out_proj = nn.Linear(codebook_dim, input_dim, bias=False)
        self.register_buffer("codebook", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("embed_avg", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("inited", torch.tensor(False))

    def _initialize_codebook(self, encodings: Tensor) -> None:
        if self.inited.item():
            return
        if not distributed.is_initialized() or distributed.get_rank() == 0:
            means, counts = _kmeans(encodings.float(), self.codebook_size)
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
        cluster_size = one_hot_codes.sum(0)
        embed_sum = encodings.t() @ one_hot_codes
        if distributed.is_initialized():
            distributed.all_reduce(cluster_size, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(embed_sum, op=distributed.ReduceOp.SUM)
        _ema_inplace(self.cluster_size, cluster_size, self.decay)
        _ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.epsilon) / (total + self.codebook_size * self.epsilon) * total
        self.codebook.copy_((self.embed_avg / smoothed[:, None]).float())

    def _replace_dead_codes(self, encodings: Tensor) -> None:
        if self.threshold_ema_dead <= 0:
            return
        dead = self.cluster_size < self.threshold_ema_dead
        if not dead.any():
            return
        count = int(dead.sum().item())
        if not distributed.is_initialized() or distributed.get_rank() == 0:
            replacements = _sample_vectors(encodings.float(), count)
        else:
            replacements = torch.zeros(count, self.codebook_dim, device=encodings.device)
        if distributed.is_initialized():
            distributed.broadcast(replacements, src=0)
        self.codebook[dead] = replacements.to(self.codebook.dtype)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        original_dtype = hidden_states.dtype
        projected = self.in_proj(
            rearrange(hidden_states.float(), "batch channels time -> batch time channels")
        )
        projected = rearrange(projected, "batch time channels -> batch channels time")
        encodings = rearrange(projected, "batch channels time -> (batch time) channels")

        if not torch.compiler.is_compiling() and not self.inited.item():
            self._initialize_codebook(encodings.detach())

        distances = (
            encodings.square().sum(1, keepdim=True)
            - 2 * encodings @ self.codebook.float().t()
            + self.codebook.float().square().sum(1, keepdim=True).t()
        )
        flat_codes = distances.argmin(-1)
        codes = rearrange(flat_codes, "(batch time) -> batch time", batch=hidden_states.shape[0])
        quantized = functional.embedding(flat_codes, self.codebook.float())
        quantized = rearrange(
            quantized, "(batch time) channels -> batch channels time", batch=hidden_states.shape[0]
        )

        inference_fast_path = not self.training and not torch.is_grad_enabled()
        if inference_fast_path:
            commitment_loss = torch.zeros(hidden_states.shape[0], device=hidden_states.device)
        else:
            commitment_loss = functional.mse_loss(projected, quantized.detach(), reduction="none").mean(
                (1, 2)
            )
            commitment_loss = commitment_loss * self.commitment

        if self.training and torch.is_grad_enabled():
            one_hot_codes = functional.one_hot(flat_codes, self.codebook_size).float()
            self._update_codebook(encodings.detach(), one_hot_codes)
            self._replace_dead_codes(encodings.detach())

        if inference_fast_path:
            straight_through = quantized
        elif self.use_rotation_trick:
            straight_through = _rotation_trick_ste(projected, quantized)
        else:
            straight_through = (quantized - projected).detach() + projected
        output = self.out_proj(rearrange(straight_through, "batch channels time -> batch time channels"))
        output = rearrange(output, "batch time channels -> batch channels time")
        return output.to(original_dtype), commitment_loss, codes

    def decode(self, codes: Tensor) -> Tensor:
        quantized = self.out_proj(functional.embedding(codes, self.codebook.float()))
        return rearrange(quantized, "batch time channels -> batch channels time")


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_codebooks = config.n_codebooks
        self.quantizer_dropout = config.quantizer_dropout
        self.quantizers = nn.ModuleList(
            EMAVectorQuantizer(
                config.latent_dim,
                config.codebook_size,
                config.codebook_dim,
                1.0,
                config.ema_decay,
                config.threshold_ema_dead,
                config.use_rotation_trick,
            )
            for _ in range(config.n_codebooks)
        )

    def forward(
        self, hidden_states: Tensor, return_level_data: bool = False
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, list[Tensor], list[Tensor]]:
        batch_size = hidden_states.shape[0]
        if self.training:
            levels_per_sample = torch.full(
                (batch_size,),
                float(self.n_codebooks + 1),
                device=hidden_states.device,
            )
            dropout_mask = torch.rand(batch_size, device=hidden_states.device) < self.quantizer_dropout
            sampled_levels = torch.randint(
                1, self.n_codebooks + 1, (batch_size,), device=hidden_states.device
            )
            levels_per_sample[dropout_mask] = sampled_levels[dropout_mask].float()
        else:
            levels_per_sample = torch.full((batch_size,), self.n_codebooks + 0.5, device=hidden_states.device)

        residual = hidden_states
        quantized = torch.zeros_like(hidden_states)
        codes: list[Tensor] = []
        commitment_loss = hidden_states.new_tensor(0.0)
        consistency_residual = hidden_states
        consistency_residuals: list[Tensor] = []
        level_codes: list[Tensor] = []

        for level, quantizer in enumerate(self.quantizers):
            active = (level < levels_per_sample).float()
            if return_level_data:
                projected_residual = quantizer.in_proj(
                    rearrange(consistency_residual.float(), "batch channels time -> batch time channels")
                )
                consistency_residuals.append(
                    rearrange(projected_residual, "batch time channels -> batch channels time")
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

    def encode(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        quantized, codes, _ = self(hidden_states)
        return quantized, codes

    def decode(self, codes: Tensor) -> Tensor:
        if not 1 <= codes.shape[1] <= len(self.quantizers):
            raise ValueError("invalid number of residual codebooks")
        return torch.stack(
            [
                quantizer.decode(codes[:, level])
                for level, quantizer in enumerate(self.quantizers[: codes.shape[1]])
            ]
        ).sum(0)


def time_shift_positive(actions: Tensor) -> Tensor:
    """Build a one-step delayed positive view."""
    shifted = torch.zeros_like(actions)
    shifted[:, 0] = actions[:, 0]
    shifted[:, 1:] = actions[:, :-1]
    return shifted


class ActionTimeContrastiveLoss(nn.Module):
    """Contrast original and time-shifted pre-RVQ encoder representations."""

    def __init__(
        self,
        mode: str = "siglip",
        temperature_init: float = 0.07,
        bias_init: float = -10.0,
    ) -> None:
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
        return functional.normalize(hidden_states.flatten(1), dim=-1)

    def forward(self, anchor_states: Tensor, positive_states: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        anchors = self._flatten(anchor_states)
        positives = self._flatten(positive_states)
        if self.mode == "siglip":
            return self._siglip_loss(anchors, positives)
        return self._infonce_loss(anchors, positives)

    def _siglip_loss(self, anchors: Tensor, positives: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size = anchors.shape[0]
        if positives.shape[0] % batch_size:
            raise ValueError("positive batch must be an integer multiple of anchor batch")
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
        metrics = {
            "consist/loss": loss.detach(),
            "contrastive/loss": loss.detach(),
            "contrastive/temperature": self.logit_scale.exp().detach(),
            "contrastive/logit_bias": self.logit_bias.detach(),
            "contrastive/avg_pos_sim": positive_logits.mean().detach(),
            "contrastive/avg_neg_sim": average_negative.detach(),
        }
        return loss, metrics

    def _infonce_loss(self, anchors: Tensor, positives: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size = anchors.shape[0]
        if batch_size < 2:
            raise ValueError("action-time 'infonce' mode requires batch_size >= 2")
        if positives.shape[0] % batch_size:
            raise ValueError("positive batch must be an integer multiple of anchor batch")
        shift = torch.randint(1, batch_size, (1,), device=anchors.device).item()
        negative_indices = (torch.arange(batch_size, device=anchors.device) + shift) % batch_size
        negatives = anchors[negative_indices]
        losses: list[Tensor] = []
        metrics: dict[str, Tensor] = {}
        for positive_index in range(positives.shape[0] // batch_size):
            positive = positives[positive_index * batch_size : (positive_index + 1) * batch_size]
            positive_similarity = (anchors * positive).sum(-1)
            negative_similarity = (anchors * negatives).sum(-1)
            margin = positive_similarity - negative_similarity
            losses.append(-functional.logsigmoid(self.temperature * margin).mean())
            metrics[f"contrastive/pos_sim_{positive_index}"] = positive_similarity.mean().detach()
            metrics[f"contrastive/neg_sim_{positive_index}"] = negative_similarity.mean().detach()
        loss = torch.stack(losses).mean()
        metrics["consist/loss"] = loss.detach()
        metrics["contrastive/loss"] = loss.detach()
        metrics["contrastive/temperature"] = self.temperature.detach()
        return loss, metrics


def compute_consistency_loss(
    consistency_residuals: list[Tensor],
    level_codes: list[Tensor],
    original_batch_size: int,
    layer_weights: list[float],
) -> tuple[Tensor, dict[str, Tensor]]:
    """First-divergence-masked residual consistency loss from ActionCodecV2."""
    if not consistency_residuals:
        raise ValueError("consistency loss requires per-level RVQ residuals")
    if len(consistency_residuals) != len(level_codes) or len(level_codes) != len(layer_weights):
        raise ValueError("consistency residuals, codes, and layer weights must have equal lengths")

    device = consistency_residuals[0].device
    sequence_length = consistency_residuals[0].shape[-1]
    prefix_match = torch.ones(original_batch_size, sequence_length, device=device)
    total_loss = torch.tensor(0.0, device=device)
    hamming = 0.0
    metrics: dict[str, Tensor] = {}

    for level, (residuals, codes, weight) in enumerate(
        zip(consistency_residuals, level_codes, layer_weights, strict=True)
    ):
        original_residuals = residuals[:original_batch_size]
        positive_residuals = residuals[original_batch_size:]
        original_codes = codes[:original_batch_size]
        positive_codes = codes[original_batch_size:]
        diverged = (original_codes != positive_codes).float().detach()
        token_change_rate = float(diverged.mean().item())
        hamming += token_change_rate
        residual_difference = (positive_residuals - original_residuals.detach()).norm(dim=1)
        active = prefix_match * diverged
        layer_loss = (active * residual_difference).mean()
        total_loss = total_loss + float(weight) * layer_loss
        metrics[f"consist/tcr_layer_{level}"] = torch.tensor(token_change_rate)
        metrics[f"consist/active_frac_{level}"] = torch.tensor(float(active.mean().item()))
        metrics[f"consist/loss_layer_{level}"] = layer_loss.detach()
        prefix_match = prefix_match * (original_codes == positive_codes).float().detach()

    metrics["consist/loss"] = total_loss.detach()
    metrics["consist/hamming_dist"] = torch.tensor(hamming * sequence_length)
    return total_loss, metrics
