# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pe.unsqueeze(0))  # [1, max_len, model_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, model_dim]
        return x + self.pos_enc[:, : x.size(1), :].to(x.device)


class Fixed3DPositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_frames: int,
        height: int,
        width: int,
        uniform_power: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.uniform_power = uniform_power

        if not uniform_power:
            t_dim = embed_dim // 2
            h_dim = embed_dim // 4
            w_dim = embed_dim - t_dim - h_dim
        else:
            t_dim = h_dim = w_dim = int(math.ceil(embed_dim / 6) * 2)

        self.register_buffer(
            "pe_t", self._build_1d_pos_embed(t_dim, num_frames), persistent=False
        )  # [T, t_dim]
        self.register_buffer("pe_h", self._build_1d_pos_embed(h_dim, height), persistent=False)  # [H, h_dim]
        self.register_buffer("pe_w", self._build_1d_pos_embed(w_dim, width), persistent=False)  # [W, w_dim]

        pe_t_expand = self.pe_t[:, None, None, :]  # [T,1,1,t_dim]
        pe_h_expand = self.pe_h[None, :, None, :]  # [1,H,1,h_dim]
        pe_w_expand = self.pe_w[None, None, :, :]  # [1,1,W,w_dim]

        pe_t_expand = nn.functional.pad(pe_t_expand, (0, embed_dim - t_dim))
        pe_h_expand = nn.functional.pad(pe_h_expand, (0, embed_dim - h_dim))
        pe_w_expand = nn.functional.pad(pe_w_expand, (0, embed_dim - w_dim))

        self.register_buffer(
            "pos_embed", pe_t_expand + pe_h_expand + pe_w_expand, persistent=False
        )  # [T,H,W,D]

    def _build_1d_pos_embed(self, dim: int, length: int):
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [length, dim]

    def forward(self, x: torch.Tensor):
        if x.ndim != 5:
            # Flattened input [B, T*H*W, D]
            x = x.view(x.shape[0], self.num_frames, self.height, self.width, self.embed_dim)

        x = x + self.pos_embed.to(x.device)
        return x.reshape(x.shape[0], self.num_frames, -1, self.embed_dim)


class Fixed2DPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, height: int, width: int, uniform_power: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        self.uniform_power = uniform_power

        if not uniform_power:
            h_dim = embed_dim // 2
            w_dim = embed_dim - h_dim
        else:
            h_dim = w_dim = int(math.ceil(embed_dim / 4) * 2)

        self.register_buffer("pe_h", self._build_1d_pos_embed(h_dim, height), persistent=False)  # [H, h_dim]
        self.register_buffer("pe_w", self._build_1d_pos_embed(w_dim, width), persistent=False)  # [W, w_dim]

        pe_h_expand = self.pe_h[:, None, :]  # [H,1,h_dim]
        pe_w_expand = self.pe_w[None, :, :]  # [1,W,w_dim]

        pe_h_expand = nn.functional.pad(pe_h_expand, (0, embed_dim - h_dim))
        pe_w_expand = nn.functional.pad(pe_w_expand, (0, embed_dim - w_dim))

        self.register_buffer("pos_embed", pe_h_expand + pe_w_expand, persistent=False)  # [H,W,D]

    def _build_1d_pos_embed(self, dim: int, length: int):
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [length, dim]

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            # Flattened input [B, H*W, D]
            x = x.view(x.shape[0], self.height, self.width, self.embed_dim)

        x = x + self.pos_embed.to(x.device)
        return x.reshape(x.shape[0], -1, self.embed_dim)
