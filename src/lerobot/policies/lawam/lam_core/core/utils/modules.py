# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path


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
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


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
        hidden = F.relu(self.layer1(x, cat_ids))
        out = self.layer2(hidden, cat_ids)
        if squeeze_time:
            out = out.squeeze(1)
        return out


def build_modal_block_attention_mask(T, H, W, add_tokens=1, num_queries: int = 1):
    assert add_tokens >= 0, (
        "add_tokens is the number of extra non-query frame-level tokens per frame, such as state; it must be >= 0"
    )
    N_T = (H * W) + add_tokens
    frame_modality_ids = torch.full((N_T,), 1, dtype=torch.long)
    if add_tokens > 0:
        frame_modality_ids[H * W :] = 0  # state-like
    modality_ids = frame_modality_ids.repeat(T)  # [T*N_T]
    query_modality_ids = modality_ids.new_full((num_queries,), 2)
    modality_ids = torch.cat([modality_ids, query_modality_ids], dim=0)
    N = modality_ids.numel()
    row = modality_ids.unsqueeze(1)  # [N,1]
    col = modality_ids.unsqueeze(0)  # [1,N]
    same_modality = row == col
    row_is_query = (row == 2).expand(-1, N)
    allowed = same_modality | row_is_query
    return allowed


def rotate_queries_or_keys(x, pos, omega: torch.Tensor = None):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position (allow passing precomputed omega to avoid recompute)
    if omega is None:
        omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
        omega /= D / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)
    else:
        # ensure dtype/device alignment
        omega = omega.to(dtype=x.dtype, device=x.device, non_blocking=True)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)
    # -- NOTE: This expansion has a subtle bug where frequencies are duplicated across the vector pair.
    # -- Fixing the bug would break compatibility with the pretrained model, but the fix can be applied by commenting
    # -- out the two lines below, and uncommenting the following two lines.
    # -- Thanks to @echosprint, original PR: https://github.com/facebookresearch/vjepa2/pull/15
    # emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    # emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
    emb_sin = emb_sin.repeat_interleave(2, dim=-1)  # (..., N, D)
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)  # (..., N, D)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        drop=0.0,
        wide_silu=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class ACRoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

        # Precompute omega frequencies for d/h/w sub-dimensions to avoid per-forward construction
        def _build_omega(dim_half: int) -> torch.Tensor:
            base = torch.arange(dim_half, dtype=torch.float32)
            base /= float(dim_half)
            return 1.0 / (10000.0**base)  # (D/2,)

        self.register_buffer("omega_d_base", _build_omega(self.d_dim // 2), persistent=False)
        self.register_buffer("omega_h_base", _build_omega(self.h_dim // 2), persistent=False)
        self.register_buffer("omega_w_base", _build_omega(self.w_dim // 2), persistent=False)

    def _get_frame_pos(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches, W_patches):
        # Remove frame component from ids
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        B, N, C = x.size()

        # -- compute position of each frame token
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)
        else:
            mask = torch.arange(int(T * H * W), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)

        # -- snap spatial positions to grid size
        h_mask *= self.grid_size / H
        w_mask *= self.grid_size / W

        # -- split out action tokens from sequence
        if action_tokens > 0:
            x = x.view(B, -1, action_tokens + H * W, C)  # [B, T, 1+H*W, D]

            action_q, action_k, action_v = [], [], []
            for i in range(action_tokens):
                a = x[:, :, i : i + 1, :].flatten(1, 2)
                # Note action tokens do not work with masking
                # -- compute qkv for action tokens and rotate
                qkv = self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
                # --
                pos_T = torch.arange(T, device=x.device, dtype=torch.float32)
                qd = rotate_queries_or_keys(q[..., : self.d_dim], pos=pos_T, omega=self.omega_d_base)
                kd = rotate_queries_or_keys(k[..., : self.d_dim], pos=pos_T, omega=self.omega_d_base)
                qr = q[..., self.d_dim :]
                kr = k[..., self.d_dim :]
                action_q += [torch.cat([qd, qr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_k += [torch.cat([kd, kr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_v += [v.view(B, self.num_heads, T, 1, -1)]

            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)

        # -- compute qkv for frame tokens and rotate
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask, omega=self.omega_d_base)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask, omega=self.omega_d_base)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask, omega=self.omega_h_base)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask, omega=self.omega_h_base)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask, omega=self.omega_w_base)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask, omega=self.omega_w_base)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens > 0:

            def merge_(tx, ta):
                """tx, tx in [B, num_heads, N, D]"""
                tx = tx.view(B, self.num_heads, T, H * W, -1)  # [B, T, H*W, D]
                ta = ta.view(B, self.num_heads, T, action_tokens, -1)  # [B, T, A, D]
                return torch.cat([ta, tx], dim=3).flatten(2, 3)

            q = merge_(q, action_q)
            k = merge_(k, action_k)
            v = merge_(v, action_v)

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

        # Precompute omega frequencies for d/h/w sub-dimensions to avoid per-forward construction
        def _build_omega(dim_half: int) -> torch.Tensor:
            base = torch.arange(dim_half, dtype=torch.float32)
            base /= float(dim_half)
            return 1.0 / (10000.0**base)  # (D/2,)

        self.register_buffer("omega_d_base", _build_omega(self.d_dim // 2), persistent=False)
        self.register_buffer("omega_h_base", _build_omega(self.h_dim // 2), persistent=False)
        self.register_buffer("omega_w_base", _build_omega(self.w_dim // 2), persistent=False)

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        # Remove frame component from ids
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask, omega=self.omega_d_base)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask, omega=self.omega_d_base)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask, omega=self.omega_h_base)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask, omega=self.omega_h_base)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask, omega=self.omega_w_base)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask, omega=self.omega_w_base)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ACBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=True,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
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
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        y = self.norm1(x)
        if isinstance(self.attn, ACRoPEAttention):
            y = self.attn(y, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=action_tokens)
        else:
            y = self.attn(y, mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        y = self.norm2(x)
        x = x + self.drop_path(self.mlp(y))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=True,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(
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
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(
                self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches
            )
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class QFormer(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim,
        num_queries=4,
        num_layers=6,
        num_heads=16,
        ffn_expansion_factor=2,
        dropout=0.1,
    ):
        super().__init__()

        self.queries = nn.Parameter(torch.randn(1, num_queries, query_dim))

        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=query_dim,
                    kdim=context_dim,
                    vdim=context_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_qs = nn.ModuleList([nn.LayerNorm(query_dim) for _ in range(num_layers)])
        self.norm_kvs = nn.ModuleList([nn.LayerNorm(context_dim) for _ in range(num_layers)])
        hidden_dim = int(query_dim * ffn_expansion_factor)
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(query_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, query_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(query_dim)

    def forward(self, context):
        B, N, Dc = context.shape
        queries = self.queries.expand(B, -1, -1)  # [B, num_queries, query_dim]
        for norm_q, norm_kv, xattn, ffn in zip(self.norm_qs, self.norm_kvs, self.cross_attns, self.ffns):
            q = norm_q(queries)
            kv = norm_kv(context)
            attn_out, _ = xattn(q, kv, kv)
            queries = queries + attn_out
            queries = queries + ffn(norm_q(queries))
        queries = self.final_norm(queries)

        return queries


class QFormer_att(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim,
        num_frames,
        num_queries,
        grid_hw: tuple[int, int],
        add_tokens=1,
        num_layers=6,
        num_heads=16,
        ffn_expansion_factor=2,
        dropout=0.1,
        use_mask: bool = False,
    ):
        super().__init__()

        self.query_dim = query_dim
        self.grid_height = int(grid_hw[0])
        self.grid_width = int(grid_hw[1])
        self.num_frames = int(num_frames)
        self.add_tokens = int(add_tokens)
        self.use_mask = bool(use_mask)
        self.queries = nn.Parameter(torch.randn(1, num_queries, context_dim))  # [1, n, context_dim]
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

        self.src_mask = None

    def _build_src_mask(self, T: int, device: torch.device) -> torch.Tensor | None:
        if not self.use_mask:
            return None
        modal_allowed = build_modal_block_attention_mask(
            T,
            self.grid_height,
            self.grid_width,
            add_tokens=self.add_tokens,
            num_queries=self.num_queries,
        )
        return (~modal_allowed).to(device=device)

    def forward(self, context):
        B, T, _, D = context.shape
        queries = self.queries.expand(B, -1, -1)  # [B, n, D]
        ctx = context.reshape(B, -1, D)
        queries = self.q_cross_attn(queries, ctx)
        ctx = torch.cat([ctx, queries], dim=1)
        src_mask = self._build_src_mask(T=T, device=ctx.device)
        for layer in self.layers:
            ctx = layer(ctx, src_mask=src_mask)
        return ctx[:, -self.num_queries :, :]  # [B, n, D]


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def _encode_btchw(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(images).flatten(2).transpose(1, 2)  # [B*T, K, D]
        return tokens

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            B, T, C, H, W = images.shape
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            B, C, H, W = images.shape
            T = 1
        tokens = self._encode_btchw(images)
        return tokens.reshape(B, T, tokens.shape[-2], tokens.shape[-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


class Attn_Crossn_Block(nn.Module):
    def __init__(self, feature_dim, num_heads=8, ffn_expansion_factor=4, dropout=0.1):
        super().__init__()

        self.norm_sa = nn.LayerNorm(feature_dim)

        self.attn_sa = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm_ca_q = nn.LayerNorm(feature_dim)
        self.norm_ca_kv = nn.LayerNorm(feature_dim)

        self.attn_ca = nn.MultiheadAttention(
            embed_dim=feature_dim,
            kdim=feature_dim,
            vdim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, int(feature_dim * ffn_expansion_factor)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(feature_dim * ffn_expansion_factor), feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, sa_features, ca_features):
        # print("state_features shape:", state_features.shape)
        # print("LAM_features shape:", LAM_features.shape)
        sa_output, _ = self.attn_sa(
            self.norm_sa(sa_features), self.norm_sa(sa_features), self.norm_sa(sa_features)
        )
        sa_features = sa_features + sa_output

        ca_output, _ = self.attn_ca(
            query=self.norm_ca_q(sa_features),
            key=self.norm_ca_kv(ca_features),
            value=self.norm_ca_kv(ca_features),
        )
        sa_features = sa_features + ca_output

        ffn_output = self.ffn(self.norm_ffn(sa_features))
        sa_features = sa_features + ffn_output

        return sa_features


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        kv = self.norm1(kv)
        attn_out, _ = self.attn(q, kv, kv)
        q = q + attn_out
        q = self.norm2(q)
        q = q + self.ff(q)
        return q
