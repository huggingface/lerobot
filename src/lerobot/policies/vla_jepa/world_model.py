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

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def build_action_block_causal_attention_mask(
    num_frames: int, grid_height: int, grid_width: int, add_tokens: int = 1
) -> torch.Tensor:
    tokens_per_frame = add_tokens + grid_height * grid_width
    num_tokens = num_frames * tokens_per_frame
    mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
    mask_block = torch.ones(tokens_per_frame, tokens_per_frame, dtype=torch.bool)
    local_window_time = num_frames

    for current_frame in range(num_frames):
        first_context_frame = max(0, current_frame - local_window_time + 1)
        for context_frame in range(first_context_frame, current_frame + 1):
            row = slice(current_frame * tokens_per_frame, (current_frame + 1) * tokens_per_frame)
            col = slice(context_frame * tokens_per_frame, (context_frame + 1) * tokens_per_frame)
            mask[row, col] = mask_block
    return mask


def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    _, _, _, dim = x.size()
    if dim % 2 != 0:
        raise ValueError("Embedding dimension must be even for rotary position encoding.")

    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freqs = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freqs.sin().squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = freqs.cos().squeeze(-1).repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return x * emb_cos + y * emb_sin


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ACRoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
        is_causal: bool = False,
        grid_size: int = 16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * ((self.head_dim // 3) // 2))
        self.h_dim = int(2 * ((self.head_dim // 3) // 2))
        self.w_dim = int(2 * ((self.head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    @staticmethod
    def _get_frame_pos(ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return ids // int(height * width)

    def _get_height_pos(self, ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
        frame_ids = self._get_frame_pos(ids, height, width)
        ids = ids - int(height * width) * frame_ids
        return ids // width

    def separate_positions(
        self, ids: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_ids = self._get_frame_pos(ids, height, width)
        height_ids = self._get_height_pos(ids, height, width)
        width_ids = ids - int(height * width) * frame_ids - width * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        num_frames: int | None = None,
        grid_height: int | None = None,
        grid_width: int | None = None,
        action_tokens: int = 0,
    ) -> torch.Tensor:
        batch_size, num_tokens, channels = x.size()
        if num_frames is None or grid_height is None or grid_width is None:
            raise ValueError("num_frames, grid_height and grid_width are required.")

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, grid_height, grid_width)
        else:
            mask = torch.arange(int(num_frames * grid_height * grid_width), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, grid_height, grid_width)

        h_mask *= self.grid_size / grid_height
        w_mask *= self.grid_size / grid_width

        if action_tokens > 0:
            x = x.view(batch_size, -1, action_tokens + grid_height * grid_width, channels)
            action_q, action_k, action_v = [], [], []
            for idx in range(action_tokens):
                action_token = x[:, :, idx : idx + 1, :].flatten(1, 2)
                qkv = self.qkv(action_token).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                qd = rotate_queries_or_keys(
                    q[..., : self.d_dim], pos=torch.arange(num_frames, device=x.device)
                )
                kd = rotate_queries_or_keys(
                    k[..., : self.d_dim], pos=torch.arange(num_frames, device=x.device)
                )
                qr = q[..., self.d_dim :]
                kr = k[..., self.d_dim :]
                action_q.append(
                    torch.cat([qd, qr], dim=-1).view(batch_size, self.num_heads, num_frames, 1, -1)
                )
                action_k.append(
                    torch.cat([kd, kr], dim=-1).view(batch_size, self.num_heads, num_frames, 1, -1)
                )
                action_v.append(v.view(batch_size, self.num_heads, num_frames, 1, -1))

            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        offset = 0
        qd = rotate_queries_or_keys(q[..., offset : offset + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., offset : offset + self.d_dim], pos=d_mask)
        offset += self.d_dim
        qh = rotate_queries_or_keys(q[..., offset : offset + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., offset : offset + self.h_dim], pos=h_mask)
        offset += self.h_dim
        qw = rotate_queries_or_keys(q[..., offset : offset + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., offset : offset + self.w_dim], pos=w_mask)
        offset += self.w_dim

        if offset < self.head_dim:
            q = torch.cat([qd, qh, qw, q[..., offset:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., offset:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens > 0:

            def merge(frame_tokens: torch.Tensor, action_token_values: torch.Tensor) -> torch.Tensor:
                frame_tokens = frame_tokens.view(
                    batch_size, self.num_heads, num_frames, grid_height * grid_width, -1
                )
                action_token_values = action_token_values.view(
                    batch_size, self.num_heads, num_frames, action_tokens, -1
                )
                return torch.cat([action_token_values, frame_tokens], dim=3).flatten(2, 3)

            q = merge(q, action_q)
            k = merge(k, action_k)
            v = merge(v, action_v)

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class ACBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_sdpa: bool = True,
        is_causal: bool = False,
        grid_size: int = 16,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not use_rope:
            raise ValueError("JEVLA1 world predictor uses AC RoPE attention.")
        self.attn = ACRoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            grid_size=grid_size,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        num_frames: int | None = None,
        grid_height: int | None = None,
        grid_width: int | None = None,
        action_tokens: int = 0,
    ) -> torch.Tensor:
        y = self.norm1(x)
        y = self.attn(
            y,
            mask=None,
            attn_mask=attn_mask,
            num_frames=num_frames,
            grid_height=grid_height,
            grid_width=grid_width,
            action_tokens=action_tokens,
        )
        x = x + self.drop_path(y)
        y = self.norm2(x)
        return x + self.drop_path(self.mlp(y))


class ActionConditionedVideoPredictor(nn.Module):
    """JEVLA1-compatible action-conditioned V-JEPA predictor."""

    def __init__(
        self,
        num_frames: int,
        img_size: tuple[int, int],
        patch_size: int,
        tubelet_size: int,
        embed_dim: int,
        action_embed_dim: int,
        predictor_embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        num_action_tokens_per_step: int,
        use_extrinsics: bool = False,
    ) -> None:
        super().__init__()
        self.is_frame_causal = True
        self.use_extrinsics = use_extrinsics
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)

        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_height = self.img_height // self.patch_size
        self.grid_width = self.img_width // self.patch_size

        self.predictor_blocks = nn.ModuleList(
            [
                ACBlock(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
                    grid_size=self.grid_height,
                    use_rope=True,
                )
                for _ in range(depth)
            ]
        )
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.num_action_tokens_per_step = num_action_tokens_per_step

    @property
    def norm(self) -> nn.LayerNorm:
        return self.predictor_norm

    @property
    def proj(self) -> nn.Linear:
        return self.predictor_proj

    def forward(
        self,
        frame_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # starVLA input convention: frame_tokens [B, T*H*W, D], actions [B, T*A, D].
        x = self.predictor_embed(frame_tokens)
        batch_size, num_context_tokens, hidden_dim = x.size()
        num_frames = num_context_tokens // (self.grid_height * self.grid_width)

        actions = self.action_encoder(action_tokens)
        actions = actions.view(batch_size, num_frames, -1, hidden_dim)
        cond_tokens = actions.shape[2]

        x = x.view(batch_size, num_frames, self.grid_height * self.grid_width, hidden_dim)
        if self.use_extrinsics:
            if extrinsics is None:
                raise ValueError("extrinsics are required when use_extrinsics=True.")
            cond_tokens += 1
            extrinsic_tokens = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([actions, extrinsic_tokens, x], dim=2).flatten(1, 2)
        else:
            x = torch.cat([actions, x], dim=2).flatten(1, 2)

        attn_mask = build_action_block_causal_attention_mask(
            num_frames, self.grid_height, self.grid_width, add_tokens=cond_tokens
        )
        attn_mask = attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        for block in self.predictor_blocks:
            x = block(
                x,
                attn_mask=attn_mask,
                num_frames=num_frames,
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                action_tokens=cond_tokens,
            )

        x = x.view(batch_size, num_frames, cond_tokens + self.grid_height * self.grid_width, hidden_dim)
        x = x[:, :, cond_tokens:, :].flatten(1, 2)
        x = self.predictor_norm(x)
        return self.predictor_proj(x)
