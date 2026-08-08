# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as functional


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = int(num_categories)
        self.W = nn.Parameter(0.02 * torch.randn(self.num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(self.num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(cat_ids, torch.Tensor):
            raise TypeError(
                f"CategorySpecificLinear expects `cat_ids` as torch.Tensor, got {type(cat_ids).__name__}."
            )
        if x.dim() != 3:
            raise ValueError(f"CategorySpecificLinear expects `x` with shape [B,T,D], got {tuple(x.shape)}")
        if cat_ids.ndim == 2 and cat_ids.size(1) == 1:
            cat_ids = cat_ids.squeeze(1)
        elif cat_ids.ndim != 1:
            raise ValueError(
                f"CategorySpecificLinear expects `cat_ids` with shape [B] or [B,1], got {tuple(cat_ids.shape)}"
            )
        if cat_ids.shape[0] != x.shape[0]:
            raise ValueError(
                f"CategorySpecificLinear batch mismatch: x B={x.shape[0]} vs cat_ids B={cat_ids.shape[0]}"
            )
        cat_ids = cat_ids.to(device=x.device, dtype=torch.long)
        selected_weight = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_weight) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(cat_ids, torch.Tensor):
            raise TypeError(
                f"CategorySpecificMLP expects `cat_ids` as torch.Tensor, got {type(cat_ids).__name__}."
            )
        squeeze_time = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_time = True
        elif x.dim() != 3:
            raise ValueError(
                f"CategorySpecificMLP expects `x` with shape [B,D] or [B,T,D], got {tuple(x.shape)}"
            )
        hidden = functional.relu(self.layer1(x, cat_ids))
        out = self.layer2(hidden, cat_ids)
        return out.squeeze(1) if squeeze_time else out


def build_modal_block_attention_mask(
    num_frames: int,
    grid_height: int,
    grid_width: int,
    add_tokens: int = 1,
    num_queries: int = 1,
) -> torch.Tensor:
    if add_tokens < 0:
        raise ValueError("add_tokens must be non-negative")
    tokens_per_frame = grid_height * grid_width + add_tokens
    frame_modalities = torch.ones(tokens_per_frame, dtype=torch.long)
    if add_tokens:
        frame_modalities[grid_height * grid_width :] = 0
    modalities = torch.cat(
        [
            frame_modalities.repeat(num_frames),
            frame_modalities.new_full((num_queries,), 2),
        ]
    )
    row = modalities.unsqueeze(1)
    col = modalities.unsqueeze(0)
    same_modality = row == col
    query_rows = (row == 2).expand(-1, modalities.numel())
    return same_modality | query_rows


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 16):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = self.norm1(context)
        attn_out, _ = self.attn(queries, context, context)
        queries = self.norm2(queries + attn_out)
        return queries + self.ff(queries)


class QFormerAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_frames: int,
        num_queries: int,
        grid_hw: tuple[int, int],
        add_tokens: int = 1,
        num_layers: int = 6,
        num_heads: int = 16,
        ffn_expansion_factor: float = 2,
        dropout: float = 0.1,
        use_mask: bool = False,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.grid_height = int(grid_hw[0])
        self.grid_width = int(grid_hw[1])
        self.num_frames = int(num_frames)
        self.add_tokens = int(add_tokens)
        self.use_mask = bool(use_mask)
        self.queries = nn.Parameter(torch.randn(1, num_queries, context_dim))
        self.q_cross_attn = CrossAttentionBlock(context_dim, num_heads)
        self.num_queries = num_queries
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=context_dim,
                    nhead=num_heads,
                    dim_feedforward=int(context_dim * ffn_expansion_factor),
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def _build_src_mask(self, num_frames: int, device: torch.device) -> torch.Tensor | None:
        if not self.use_mask:
            return None
        allowed = build_modal_block_attention_mask(
            num_frames,
            self.grid_height,
            self.grid_width,
            add_tokens=self.add_tokens,
            num_queries=self.num_queries,
        )
        return (~allowed).to(device=device)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _, hidden_dim = context.shape
        queries = self.queries.expand(batch_size, -1, -1)
        context = context.reshape(batch_size, -1, hidden_dim)
        queries = self.q_cross_attn(queries, context)
        context = torch.cat([context, queries], dim=1)
        src_mask = self._build_src_mask(num_frames=num_frames, device=context.device)
        for layer in self.layers:
            context = layer(context, src_mask=src_mask)
        return context[:, -self.num_queries :, :]
