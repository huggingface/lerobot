# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pe.unsqueeze(0))  # [1, max_len, model_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, model_dim]
        return x + self.pos_enc[:, :x.size(1), :].to(x.device)

class Fixed3DPositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, T: int, H: int, W: int, uniform_power: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.T, self.H, self.W = T, H, W
        self.uniform_power = uniform_power

        if not uniform_power:
            t_dim = embed_dim // 2
            h_dim = embed_dim // 4
            w_dim = embed_dim - t_dim - h_dim
        else:
            t_dim = h_dim = w_dim = int(math.ceil(embed_dim / 6) * 2)

        self.register_buffer("pe_t", self._build_1d_pos_embed(t_dim, T), persistent=False)  # [T, t_dim]
        self.register_buffer("pe_h", self._build_1d_pos_embed(h_dim, H), persistent=False)  # [H, h_dim]
        self.register_buffer("pe_w", self._build_1d_pos_embed(w_dim, W), persistent=False)  # [W, w_dim]

        pe_t_expand = self.pe_t[:, None, None, :]          # [T,1,1,t_dim]
        pe_h_expand = self.pe_h[None, :, None, :]          # [1,H,1,h_dim]
        pe_w_expand = self.pe_w[None, None, :, :]          # [1,1,W,w_dim]

        pe_t_expand = nn.functional.pad(pe_t_expand, (0, embed_dim - t_dim))
        pe_h_expand = nn.functional.pad(pe_h_expand, (0, embed_dim - h_dim))
        pe_w_expand = nn.functional.pad(pe_w_expand, (0, embed_dim - w_dim))

        self.register_buffer("pos_embed", pe_t_expand + pe_h_expand + pe_w_expand, persistent=False)  # [T,H,W,D]

    def _build_1d_pos_embed(self, dim: int, length: int):
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [length, dim]

    def forward(self, x: torch.Tensor):
        if x.ndim !=5:
            # Flattened input [B, T*H*W, D]
            x = x.view(x.shape[0], self.T, self.H, self.W, self.embed_dim)

        x = x + self.pos_embed.to(x.device)
        return x.reshape(x.shape[0], self.T, -1, self.embed_dim)


class Fixed2DPositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, H: int, W: int, uniform_power: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.H, self.W = H, W
        self.uniform_power = uniform_power

        if not uniform_power:
            h_dim = embed_dim // 2
            w_dim = embed_dim - h_dim
        else:
            h_dim = w_dim = int(math.ceil(embed_dim / 4) * 2)

        self.register_buffer("pe_h", self._build_1d_pos_embed(h_dim, H), persistent=False)  # [H, h_dim]
        self.register_buffer("pe_w", self._build_1d_pos_embed(w_dim, W), persistent=False)  # [W, w_dim]

        pe_h_expand = self.pe_h[:, None, :]          # [H,1,h_dim]
        pe_w_expand = self.pe_w[None, :, :]          # [1,W,w_dim]

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
            x = x.view(x.shape[0], self.H, self.W, self.embed_dim)

        x = x + self.pos_embed.to(x.device)
        return x.reshape(x.shape[0], -1, self.embed_dim)

def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(
        grid_h, grid_d, grid_w
    )  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
