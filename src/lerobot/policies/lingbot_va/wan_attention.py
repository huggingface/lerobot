# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
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

"""Attention and rotary-position-embedding modules for the LingBot-VA Wan transformer.

Vendored and lightly adapted from the upstream LingBot-VA repository
(https://github.com/Robbyant/lingbot-va, ``wan_va/modules/model.py``). The ``torch``
SDPA backend is the default and is always available; the ``flashattn`` and ``flex``
backends are imported lazily and only required when the corresponding ``attn_mode`` is
selected. State-dict parameter names are preserved verbatim so that conversion from the
original diffusers-style checkpoint is near-identity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ``flash_attn`` and the flex-attention APIs are optional. We import them lazily inside the
# backends that need them so that the (default) ``torch`` SDPA path works on any platform,
# including CPU-only and macOS where neither package is available.


def custom_sdpa(q, k, v):
    """Scaled-dot-product attention operating on ``(B, S, H, D)`` tensors."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)


def _load_flash_attn_func():
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError:
        try:
            from flash_attn import flash_attn_func
        except ImportError as e:
            raise ImportError(
                "attn_mode='flashattn' requires the `flash_attn` package, which is not installed. "
                "Install it, or use attn_mode='torch' (the default)."
            ) from e
    return flash_attn_func


class WanRotaryPosEmbed(nn.Module):
    """Rotary position embedding with separate frequency bases for frame / height / width."""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (
            self.theta ** (torch.arange(0, self.f_dim, 2)[: (self.f_dim // 2)].double() / self.f_dim)
        )
        h_freqs_base = 1.0 / (
            self.theta ** (torch.arange(0, self.h_dim, 2)[: (self.h_dim // 2)].double() / self.h_dim)
        )
        w_freqs_base = 1.0 / (
            self.theta ** (torch.arange(0, self.w_dim, 2)[: (self.w_dim // 2)].double() / self.w_dim)
        )
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(nn.Module):
    """Self/cross attention with KV-caching for autoregressive streaming inference.

    Backends:
      * ``torch`` (default): standard SDPA, available everywhere.
      * ``flashattn``: FlashAttention kernels (optional dependency).
      * ``flex``: PyTorch flex-attention (optional, used for block-causal training masks).
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode="torch",
    ):
        super().__init__()
        if attn_mode == "torch":
            self.attn_op = custom_sdpa
        elif attn_mode == "flashattn":
            self.attn_op = _load_flash_attn_func()
        elif attn_mode == "flex":
            # Imported lazily to avoid a hard dependency on torch flex-attention at import time.
            from .wan_flex_attention import FlexAttnFunc

            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support 'torch', 'flashattn' and 'flex'"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads
        )

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, dim, bias=True),
                nn.Dropout(dropout),
            ]
        )
        self.norm_q = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        # KV cache only lives on self-attention modules (cross_attention_dim_head is None).
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache["is_pred"]
        cache["mask"][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim, device, dtype, batch_size):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            "k": torch.empty([batch_size, total_tolen, num_head, head_dim], device=device, dtype=dtype),
            "v": torch.empty([batch_size, total_tolen, num_head, head_dim], device=device, dtype=dtype),
            "id": torch.full((total_tolen,), -1, device=device),
            "mask": torch.zeros((total_tolen,), dtype=torch.bool, device=device),
            "is_pred": torch.zeros((total_tolen,), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]["id"]
        mask = self.attn_caches[cache_name]["mask"]

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache["k"][:, slots] = key
        cache["v"][:, slots] = value
        cache["mask"][slots] = True
        cache["id"][slots] = new_id
        cache["is_pred"][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]["mask"][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name="pos",
    ):
        kv_cache = (
            self.attn_caches[cache_name]
            if (self.attn_caches is not None) and (cache_name in self.attn_caches)
            else None
        )

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
                )
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        slots = None
        if kv_cache is not None and kv_cache["k"] is not None:
            slots = self.update_cache(cache_name, key, value, is_pred=(update_cache == 1))
            key_pool = self.attn_caches[cache_name]["k"]
            value_pool = self.attn_caches[cache_name]["v"]
            mask = self.attn_caches[cache_name]["mask"]
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        hidden_states = self.attn_op(query, key, value)

        if update_cache == 0:
            if kv_cache is not None and kv_cache["k"] is not None:
                self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


__all__ = ["WanAttention", "WanRotaryPosEmbed", "custom_sdpa"]
