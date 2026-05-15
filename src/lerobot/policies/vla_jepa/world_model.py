from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def build_block_causal_attention_mask(num_steps: int, tokens_per_step: int, cond_tokens: int) -> torch.Tensor:
    total_tokens = num_steps * (tokens_per_step + cond_tokens)
    mask = torch.full((total_tokens, total_tokens), float("-inf"))
    for current_step in range(num_steps):
        row_start = current_step * (tokens_per_step + cond_tokens)
        row_end = row_start + tokens_per_step + cond_tokens
        mask[row_start:row_end, :row_end] = 0
    return mask


class _Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class _MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _PredictorBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(embed_dim, mlp_ratio)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        return x + self.mlp(self.norm2(x))


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
        self.predictor_blocks = nn.ModuleList(
            [_PredictorBlock(predictor_embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)
        self.num_action_tokens_per_step = num_action_tokens_per_step

    def forward(self, frame_tokens: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps, tokens_per_frame, _ = frame_tokens.shape
        _, action_steps, _, _ = action_tokens.shape
        if action_steps != num_steps:
            raise ValueError(f"Expected {num_steps} action steps, got {action_steps}.")

        frame_tokens = self.predictor_embed(frame_tokens)
        action_tokens = self.action_encoder(action_tokens)
        fused_steps = [
            torch.cat([action_tokens[:, step], frame_tokens[:, step]], dim=1) for step in range(num_steps)
        ]
        fused = torch.cat(fused_steps, dim=1)

        attn_mask = build_block_causal_attention_mask(
            num_steps=num_steps,
            tokens_per_step=tokens_per_frame,
            cond_tokens=self.num_action_tokens_per_step,
        ).to(device=fused.device, dtype=fused.dtype)

        for block in self.predictor_blocks:
            fused = block(fused, attn_mask=attn_mask)

        fused = self.predictor_norm(fused)
        fused = fused.view(batch_size, num_steps, self.num_action_tokens_per_step + tokens_per_frame, -1)
        predicted_frame_tokens = fused[:, :, self.num_action_tokens_per_step :, :]
        return self.predictor_proj(predicted_frame_tokens)
