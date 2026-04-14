from __future__ import annotations

import torch
from torch import nn


def build_block_causal_attention_mask(num_steps: int, tokens_per_step: int, cond_tokens: int) -> torch.Tensor:
    total_tokens = num_steps * (tokens_per_step + cond_tokens)
    mask = torch.full((total_tokens, total_tokens), float("-inf"))
    for current_step in range(num_steps):
        row_start = current_step * (tokens_per_step + cond_tokens)
        row_end = row_start + tokens_per_step + cond_tokens
        allowed_end = row_end
        mask[row_start:row_end, :allowed_end] = 0
    return mask


class ActionConditionedVideoPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        action_embed_dim: int,
        predictor_embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        num_action_tokens_per_step: int,
    ) -> None:
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.proj = nn.Linear(predictor_embed_dim, embed_dim)
        self.num_action_tokens_per_step = num_action_tokens_per_step

    def forward(self, frame_tokens: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps, tokens_per_frame, _ = frame_tokens.shape
        _, action_steps, _, _ = action_tokens.shape
        if action_steps != num_steps:
            raise ValueError(f"Expected {num_steps} action steps, got {action_steps}.")

        frame_tokens = self.predictor_embed(frame_tokens)
        action_tokens = self.action_encoder(action_tokens)
        fused_steps = []
        for step in range(num_steps):
            fused_steps.append(torch.cat([action_tokens[:, step], frame_tokens[:, step]], dim=1))
        fused = torch.cat(fused_steps, dim=1)

        attn_mask = build_block_causal_attention_mask(
            num_steps=num_steps,
            tokens_per_step=tokens_per_frame,
            cond_tokens=self.num_action_tokens_per_step,
        ).to(device=fused.device, dtype=fused.dtype)
        encoded = self.encoder(fused, mask=attn_mask)
        encoded = encoded.view(batch_size, num_steps, self.num_action_tokens_per_step + tokens_per_frame, -1)
        predicted_frame_tokens = encoded[:, :, self.num_action_tokens_per_step :, :]
        return self.proj(self.norm(predicted_frame_tokens))
