from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn
from torch.distributions import Beta

from .configuration_vla_jepa import VLAJEPAConfig


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


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
        self.w1 = nn.Linear(action_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size * 2, hidden_size)
        self.w3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [batch_size].")
        timesteps = timesteps.unsqueeze(1).expand(-1, seq_len)
        action_emb = self.w1(actions)
        time_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        return self.w3(swish(self.w2(torch.cat([action_emb, time_emb], dim=-1))))


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
        cross_attention_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn = Attention(
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
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.norm1(hidden_states, temb)
        hidden_states = hidden_states + self.attn(attn_input, encoder_hidden_states=encoder_hidden_states)
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
        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
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
        for block in self.blocks:
            x = block(x, encoder_hidden_states=encoder_hidden_states, temb=temb)
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
}


class VLAJEPAActionHead(nn.Module):
    def __init__(self, config: VLAJEPAConfig, cross_attention_dim: int) -> None:
        super().__init__()
        preset = DIT_PRESETS[config.action_model_type]
        self.config = config
        self.input_embedding_dim = preset.hidden_size
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.model = DiT(
            num_attention_heads=config.action_num_heads or preset.num_attention_heads,
            attention_head_dim=config.action_attention_head_dim or preset.attention_head_dim,
            output_dim=config.action_hidden_size,
            num_layers=config.action_num_layers,
            dropout=config.action_dropout,
            cross_attention_dim=cross_attention_dim,
        )
        self.action_encoder = ActionEncoder(config.action_dim, config.action_hidden_size)
        self.action_decoder = nn.Sequential(
            nn.Linear(config.action_hidden_size, config.action_hidden_size),
            nn.GELU(),
            nn.Linear(config.action_hidden_size, config.action_dim),
        )
        self.state_encoder = (
            nn.Sequential(
                nn.Linear(config.state_dim, config.action_hidden_size),
                nn.GELU(),
                nn.Linear(config.action_hidden_size, config.action_hidden_size),
            )
            if config.state_dim > 0
            else None
        )
        self.future_tokens = nn.Embedding(config.num_action_tokens_per_timestep, config.action_hidden_size)
        self.position_embedding = nn.Embedding(config.chunk_size + config.num_action_tokens_per_timestep + 4, config.action_hidden_size)
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
        return F.mse_loss(pred_actions, velocity, reduction="mean")

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
            timesteps = torch.full((batch_size,), t_value, device=conditioning_tokens.device, dtype=torch.long)
            hidden_states = self._build_inputs(conditioning_tokens, actions, state, timesteps)
            pred = self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=conditioning_tokens,
                timestep=timesteps,
            )
            pred_velocity = self.action_decoder(pred[:, -self.action_horizon :])
            actions = actions + dt * pred_velocity
        return actions
