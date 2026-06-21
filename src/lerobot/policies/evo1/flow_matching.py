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

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _cfgget(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int):
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :]

    def _extend_pe(self, new_max_len):
        old_max_len, dim = self.pe.size(1), self.pe.size(2)
        if new_max_len <= old_max_len:
            return
        extra_positions = torch.arange(old_max_len, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        extra_pe = torch.zeros(new_max_len - old_max_len, dim)
        extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
        extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
        extra_pe = extra_pe.unsqueeze(0)
        new_pe = torch.cat([self.pe, extra_pe.to(self.pe.device)], dim=1)
        self.pe = new_pe


class CategorySpecificLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1):
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.weight = nn.Parameter(torch.empty(num_categories, in_dim, out_dim))
            self.bias = nn.Parameter(torch.zeros(num_categories, out_dim))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        if self.num_categories <= 1:
            if x.dtype != self.linear.weight.dtype:
                x = x.to(dtype=self.linear.weight.dtype)
            return self.linear(x)

        if x.dtype != self.weight.dtype:
            x = x.to(dtype=self.weight.dtype)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        if category_id.dim() == 0:
            cid = category_id.item()
            out = x_flat @ self.weight[cid] + self.bias[cid]
        else:
            category_id = category_id.reshape(-1)
            if category_id.numel() != x_flat.size(0):
                raise ValueError(
                    f"category_id length {category_id.numel()} does not match flattened batch {x_flat.size(0)}"
                )
            weight_selected = self.weight[category_id]
            bias_selected = self.bias[category_id]
            out = torch.bmm(x_flat.unsqueeze(1), weight_selected).squeeze(1) + bias_selected
        out_shape = orig_shape[:-1] + (out.shape[-1],)
        return out.view(out_shape)


class CategorySpecificMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_categories: int = 1):
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        out = self.activation(self.fc1(x, category_id))
        out = self.fc2(out, category_id)
        return out


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(
        self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1
    ):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories

        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)

        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):
        batch_size, horizon, action_dim = action_seq.shape
        if self.horizon != horizon:
            raise ValueError(
                f"Action sequence length must match horizon: got {horizon}, expected {self.horizon}."
            )

        x = action_seq.reshape(batch_size * horizon, action_dim)
        if category_id.dim() == 0:
            cat_ids = category_id.expand(horizon * batch_size)
        else:
            cat_ids = category_id.unsqueeze(1).expand(batch_size, horizon).reshape(batch_size * horizon)

        out = self.activation(self.W1(x, cat_ids))
        pos_enc = self.pos_encoding(horizon).to(device=out.device, dtype=out.dtype)
        out = out.view(batch_size, horizon, -1) + pos_enc
        out = out.view(batch_size * horizon, -1)
        out = self.activation(self.W2(out, cat_ids))
        out = self.W3(out, cat_ids)
        return out.view(batch_size, horizon, self.embed_dim)


class BasicTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, action_tokens: torch.Tensor, context_tokens: torch.Tensor, time_emb: torch.Tensor):
        x = self.norm1(action_tokens)
        attn_out, _ = self.attn(x, context_tokens, context_tokens)
        x = action_tokens + attn_out
        x2 = self.norm2(x)
        if time_emb is not None:
            x2 = x2 + time_emb.unsqueeze(1)
        ff_out = self.ff(x2)
        return x + ff_out


class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        config=None,
        embed_dim: int = 896,
        hidden_dim: int = 1024,
        action_dim: int = 16 * 7,
        horizon: int = 16,
        per_action_dim: int = 7,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.0,
        num_inference_timesteps: int = 20,
        num_categories: int = 1,
    ):
        super().__init__()

        if config is not None:
            embed_dim = _cfgget(config, "embed_dim", embed_dim)
            hidden_dim = _cfgget(config, "hidden_dim", hidden_dim)
            action_dim = _cfgget(config, "action_dim", action_dim)
            horizon = _cfgget(config, "horizon", horizon)
            per_action_dim = _cfgget(config, "per_action_dim", per_action_dim)
            num_heads = _cfgget(config, "num_heads", num_heads)
            num_layers = _cfgget(config, "num_layers", num_layers)
            dropout = _cfgget(config, "dropout", dropout)
            num_inference_timesteps = _cfgget(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = _cfgget(config, "num_categories", num_categories)
            self.config = config
        else:
            self.config = SimpleNamespace(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                horizon=horizon,
                per_action_dim=per_action_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                num_inference_timesteps=num_inference_timesteps,
                num_categories=num_categories,
            )

        logger.info("FlowmatchingActionHead num_inference_timesteps=%s", num_inference_timesteps)
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = _cfgget(self.config, "per_action_dim", per_action_dim)
        self.action_dim = _cfgget(self.config, "action_dim", action_dim)

        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=embed_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(embed_dim)
        self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim)
        self.mlp_head = CategorySpecificMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_categories=num_categories,
        )

        self.state_encoder = None
        state_dim = _cfgget(self.config, "state_dim")
        if state_dim is not None:
            state_hidden = _cfgget(self.config, "state_hidden_dim", embed_dim)
            self.state_encoder = CategorySpecificMLP(
                input_dim=state_dim,
                hidden_dim=state_hidden,
                output_dim=embed_dim,
                num_categories=num_categories,
            )

        if horizon > 1:
            self.action_encoder = MultiEmbodimentActionEncoder(
                action_dim=self.per_action_dim,
                embed_dim=embed_dim,
                hidden_dim=embed_dim,
                horizon=horizon,
                num_categories=num_categories,
            )
            self.single_action_proj = None
        else:
            self.action_encoder = None
            self.single_action_proj = nn.Linear(self.per_action_dim, self.embed_dim)

    def _project_actions(self, action_seq: torch.Tensor, embodiment_id: torch.LongTensor) -> torch.Tensor:
        if self.horizon > 1 and self.action_encoder is not None:
            return self.action_encoder(action_seq, embodiment_id)
        if self.single_action_proj is None:
            raise RuntimeError("single_action_proj is not initialized for horizon <= 1.")
        return self.single_action_proj(action_seq)

    def _expand_action_mask(
        self,
        action_mask: torch.Tensor,
        batch_size: int,
        per_action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if action_mask is None:
            raise ValueError("action_mask must be provided for flow matching inference.")

        if action_mask.dim() == 2:
            expected_last_dim = self.horizon * per_action_dim
            if action_mask.shape == (batch_size, expected_last_dim):
                expanded_mask = action_mask.reshape(batch_size, self.horizon, per_action_dim)
            elif action_mask.shape == (batch_size, per_action_dim):
                expanded_mask = action_mask.unsqueeze(1).expand(batch_size, self.horizon, per_action_dim)
            else:
                raise ValueError(
                    f"Expected action_mask shape {(batch_size, expected_last_dim)} or "
                    f"{(batch_size, per_action_dim)}, got {tuple(action_mask.shape)}"
                )
        elif action_mask.dim() == 3:
            expected_shape = (batch_size, self.horizon, per_action_dim)
            if tuple(action_mask.shape) != expected_shape:
                raise ValueError(
                    f"Expected action_mask shape {expected_shape}, got {tuple(action_mask.shape)}"
                )
            expanded_mask = action_mask
        else:
            raise ValueError(f"Unsupported action_mask rank: {action_mask.dim()}")

        return expanded_mask.to(device=device, dtype=dtype)

    def forward(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor = None,
        actions_gt: torch.Tensor = None,
        embodiment_id: torch.LongTensor = None,
        state_mask: torch.Tensor = None,
        action_mask: torch.Tensor = None,
    ):
        if actions_gt is None:
            return self.get_action(
                fused_tokens, state=state, embodiment_id=embodiment_id, action_mask=action_mask
            )

        batch_size = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        context_tokens = fused_tokens
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1)
            context_tokens = torch.cat([context_tokens, state_emb], dim=1)

        t = (
            torch.distributions.Beta(2, 2)
            .sample((batch_size,))
            .clamp(0.02, 0.98)
            .to(device)
            .to(dtype=self.dtype)
        )
        time_index = (t * 999).long().clamp_(0, 999)
        time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0).to(dtype=context_tokens.dtype)

        actions_gt_seq = actions_gt
        noise = torch.rand_like(actions_gt) * 2 - 1
        if action_mask is not None:
            action_mask = action_mask.to(dtype=noise.dtype, device=noise.device)
            if action_mask.shape != noise.shape:
                raise ValueError(f"action_mask shape {action_mask.shape} != noise shape {noise.shape}")
            actions_gt_seq = actions_gt_seq * action_mask
            noise = noise * action_mask

        if self.horizon > 1:
            noise_seq = noise.view(batch_size, self.horizon, self.per_action_dim)
        else:
            noise_seq = noise if noise.dim() == 3 else noise.unsqueeze(1)
        t_broadcast = t.view(batch_size, 1, 1)
        action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq

        action_tokens = self._project_actions(action_intermediate_seq, embodiment_id)
        target_dtype = self.dtype
        action_tokens = action_tokens.to(dtype=target_dtype)
        context_tokens = context_tokens.to(dtype=target_dtype)
        time_emb = time_emb.to(dtype=target_dtype)

        x = action_tokens
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)
        x = self.norm_out(x)

        if self.horizon > 1:
            x_flat = x.reshape(batch_size, -1)
            x_pooled = self.seq_pool_proj(x_flat)
        else:
            x_pooled = x.squeeze(1)

        pred_velocity = self.mlp_head(x_pooled, embodiment_id)
        return pred_velocity, noise

    def get_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor = None,
        embodiment_id: torch.LongTensor = None,
        action_mask: torch.Tensor = None,
    ):
        batch_size = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        context_tokens = fused_tokens
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1)
            context_tokens = torch.cat([context_tokens, state_emb], dim=1)

        action_dim_total = _cfgget(self.config, "action_dim", self.action_dim)
        per_action_dim = _cfgget(self.config, "per_action_dim", action_dim_total // max(self.horizon, 1))

        action = torch.rand(batch_size, action_dim_total, device=device, dtype=context_tokens.dtype) * 2 - 1
        action_seq = (
            action.view(batch_size, self.horizon, per_action_dim)
            if self.horizon > 1
            else action.view(batch_size, 1, per_action_dim)
        )
        action_mask = self._expand_action_mask(
            action_mask,
            batch_size=batch_size,
            per_action_dim=per_action_dim,
            device=action_seq.device,
            dtype=action_seq.dtype,
        )
        action_seq = action_seq * action_mask

        target_dtype = self.dtype
        context_tokens = context_tokens.to(dtype=target_dtype)

        num_steps = int(_cfgget(self.config, "num_inference_timesteps", 32))
        if num_steps <= 0:
            raise ValueError(f"num_inference_timesteps must be positive, got {num_steps}")
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = i / num_steps
            time_index = min(int(t * 999), 999)
            time_emb = (
                self.time_pos_enc(1000)[:, time_index, :].to(device).squeeze(0).to(dtype=context_tokens.dtype)
            )
            time_emb = time_emb.unsqueeze(0).repeat(batch_size, 1)

            action_seq = action_seq * action_mask
            action_tokens = self._project_actions(action_seq, embodiment_id).to(dtype=target_dtype)
            time_emb = time_emb.to(dtype=target_dtype)

            x = action_tokens
            for block in self.transformer_blocks:
                x = block(x, context_tokens, time_emb)
            x = self.norm_out(x)

            if self.horizon > 1:
                x_flat = x.reshape(batch_size, -1)
                x_pooled = self.seq_pool_proj(x_flat)
            else:
                x_pooled = x.squeeze(1)

            pred = self.mlp_head(x_pooled, embodiment_id)
            action = action + dt * pred
            action_seq = (
                action.view(batch_size, self.horizon, per_action_dim)
                if self.horizon > 1
                else action.view(batch_size, 1, per_action_dim)
            )

        action_seq = action_seq * action_mask
        return action_seq.reshape(batch_size, -1)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
