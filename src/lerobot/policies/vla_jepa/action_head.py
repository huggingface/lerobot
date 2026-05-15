from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributions import Beta

from lerobot.utils.import_utils import _diffusers_available

if TYPE_CHECKING or _diffusers_available:
    from diffusers import ConfigMixin, ModelMixin
    from diffusers.configuration_utils import register_to_config
    from diffusers.models.attention import Attention, FeedForward
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
else:

    class ModelMixin:  # type: ignore[no-redef]
        pass

    class ConfigMixin:  # type: ignore[no-redef]
        pass

    register_to_config = lambda f: f  # noqa: E731
    Attention = FeedForward = TimestepEmbedding = Timesteps = None

from .configuration_vla_jepa import VLAJEPAConfig


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class _MLP2(nn.Module):
    """Two-layer GELU MLP with layer1/layer2 attribute names matching the original checkpoint."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.gelu(self.layer1(x)))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        batch_size, seq_len = timesteps.shape
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device)
        exponent = exponent * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / max(half_dim, 1))
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).view(batch_size, seq_len, -1)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [batch_size].")
        timesteps = timesteps.unsqueeze(1).expand(-1, seq_len)
        action_emb = self.layer1(actions)
        time_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        return self.layer3(swish(self.layer2(torch.cat([action_emb, time_emb], dim=-1))))


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        projected = self.time_proj(timesteps).to(dtype=next(self.parameters()).dtype)
        return self.timestep_embedder(projected)


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float,
        cross_attention_dim: int | None,
    ) -> None:
        super().__init__()
        self.is_cross_attention = cross_attention_dim is not None
        self.norm1 = AdaLayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=True,
            cross_attention_dim=cross_attention_dim,
            out_bias=True,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn="gelu-approximate", final_dropout=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.norm1(hidden_states, temb)
        hidden_states = hidden_states + self.attn1(attn_input, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        cross_attention_dim: int,
    ) -> None:
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = TimestepEncoder(self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    # Even blocks attend to context (cross-attention), odd blocks are self-attention.
                    cross_attention_dim=cross_attention_dim if i % 2 == 0 else None,
                )
                for i in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(self.inner_dim, self.inner_dim * 2)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        temb = self.timestep_encoder(timestep)
        x = hidden_states
        for block in self.transformer_blocks:
            es = encoder_hidden_states if block.is_cross_attention else None
            x = block(x, encoder_hidden_states=es, temb=temb)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(x)


@dataclass
class ActionModelPreset:
    hidden_size: int
    attention_head_dim: int
    num_attention_heads: int


DIT_PRESETS = {
    "DiT-B": ActionModelPreset(hidden_size=768, attention_head_dim=64, num_attention_heads=12),
    "DiT-L": ActionModelPreset(hidden_size=1536, attention_head_dim=48, num_attention_heads=32),
    "DiT-test": ActionModelPreset(hidden_size=16, attention_head_dim=8, num_attention_heads=2),
}


class VLAJEPAActionHead(nn.Module):
    def __init__(self, config: VLAJEPAConfig, cross_attention_dim: int) -> None:
        super().__init__()
        preset = DIT_PRESETS[config.action_model_type]
        self.config = config
        num_heads = preset.num_attention_heads
        head_dim = preset.attention_head_dim
        inner_dim = num_heads * head_dim  # e.g. DiT-B: 12 × 64 = 768

        self.input_embedding_dim = inner_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.model = DiT(
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            output_dim=config.action_hidden_size,
            num_layers=config.action_num_layers,
            dropout=config.action_dropout,
            cross_attention_dim=cross_attention_dim,
        )
        # action_encoder/decoder and state_encoder use action_hidden_size (DiT output dim).
        # action_encoder and state_encoder produce inner_dim-sized tokens (DiT input width).
        # action_decoder takes DiT output (action_hidden_size) and produces action_dim predictions.
        self.action_encoder = ActionEncoder(config.action_dim, inner_dim)
        self.action_decoder = _MLP2(config.action_hidden_size, config.action_hidden_size, config.action_dim)
        self.state_encoder = (
            _MLP2(config.state_dim, config.action_hidden_size, inner_dim) if config.state_dim > 0 else None
        )
        # future_tokens and position_embedding operate at inner_dim (DiT input width),
        # not at action_hidden_size (DiT output width).
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, inner_dim)
        self.position_embedding = nn.Embedding(config.action_max_seq_len, inner_dim)
        self.beta_dist = Beta(config.action_noise_beta_alpha, config.action_noise_beta_beta)

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return (self.config.action_noise_s - sample) / self.config.action_noise_s

    def _build_inputs(
        self,
        conditioning_tokens: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        action_features = self.action_encoder(actions, timesteps)
        pos_ids = torch.arange(action_features.shape[1], device=actions.device)
        action_features = action_features + self.position_embedding(pos_ids)[None]

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(actions.shape[0], -1, -1)
        seq = [future_tokens, action_features]
        if state is not None and self.state_encoder is not None:
            if state.ndim == 2:
                state = state.unsqueeze(1)
            seq.insert(0, self.state_encoder(state))
        return torch.cat(seq, dim=1)

    def forward(
        self,
        conditioning_tokens: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        noisy_actions = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        velocity = actions - noise
        t_discretized = (t * self.config.action_num_timestep_buckets).long()

        hidden_states = self._build_inputs(conditioning_tokens, noisy_actions, state, t_discretized)
        pred = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=conditioning_tokens,
            timestep=t_discretized,
        )
        pred_actions = self.action_decoder(pred[:, -actions.shape[1] :])

        if action_is_pad is None:
            action_is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)

        loss = F.mse_loss(pred_actions, velocity, reduction="none")  # [B, T, action_dim]
        valid_mask = ~action_is_pad.unsqueeze(-1)  # [B, T, 1]
        num_valid = valid_mask.sum() * loss.shape[-1]
        return (loss * valid_mask).sum() / num_valid.clamp_min(1)

    @torch.no_grad()
    def predict_action(
        self,
        conditioning_tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = conditioning_tokens.shape[0]
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.config.action_dim,
            dtype=conditioning_tokens.dtype,
            device=conditioning_tokens.device,
        )
        dt = 1.0 / max(self.num_inference_timesteps, 1)
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(max(self.num_inference_timesteps, 1))
            t_value = int(t_cont * self.config.action_num_timestep_buckets)
            timesteps = torch.full(
                (batch_size,), t_value, device=conditioning_tokens.device, dtype=torch.long
            )
            hidden_states = self._build_inputs(conditioning_tokens, actions, state, timesteps)
            pred = self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=conditioning_tokens,
                timestep=timesteps,
            )
            pred_velocity = self.action_decoder(pred[:, -self.action_horizon :])
            actions = actions + dt * pred_velocity
        return actions
