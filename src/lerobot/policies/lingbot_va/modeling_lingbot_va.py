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

"""LingBot-VA policy: an autoregressive video-action world model on the Wan2.2 stack.

The sampling loop is a faithful re-implementation of the upstream streaming server
(``wan_va/wan_va_server.py``) and LIBERO client (``evaluation/libero/client.py``), adapted
to LeRobot's ``select_action`` interface:

  * the trainable dual-stream transformer is owned as a sub-module and round-trips in the
    single ``model.safetensors`` checkpoint;
  * the frozen Wan VAE + UMT5 text encoder + tokenizer are *lazily pulled* from
    ``config.wan_pretrained_path`` (not bundled), so the LeRobot checkpoint stays small;
  * ``predict_action_chunk`` runs one autoregressive chunk (video stream then action
    stream, each with CFG and its own flow-matching scheduler) and updates the KV cache;
  * ``select_action`` drains a per-step action queue and records the real observed
    keyframes that are fed back into the KV cache when the queue is refilled.

NOTE: matching the upstream LIBERO success rate is the Phase-5 correctness gate and must be
validated on a CUDA GPU with the converted checkpoint (tensor-diff against upstream on
identical inputs). The streaming path is written for single-environment eval
(``--eval.batch_size=1``).
"""

import math
from collections import deque
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from .configuration_lingbot_va import LingBotVAConfig


# Grid-id / patch utilities
def data_seq_to_patch(patch_size, data_seq, latent_num_frames, latent_height, latent_width, batch_size=1):
    """Reshape a flattened patch sequence back into a ``(B, C, F, H, W)`` latent grid."""
    p_t, p_h, p_w = patch_size
    post_patch_num_frames = latent_num_frames // p_t
    post_patch_height = latent_height // p_h
    post_patch_width = latent_width // p_w

    data_patch = data_seq.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return data_patch


def get_mesh_id(f, h, w, t, f_w=1, f_shift=0, action=False):
    """Build the (frame, height, width, stream) grid ids used to index the rotary embedding."""
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)
    return grid_id


# Flow-matching scheduler
# LingBot-VA uses two independent instances at inference (one for the video-latent stream,
# one for the action stream), each with its own ``shift`` and number of denoising steps.
class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        training=False,
        shift=None,
        dynamic_shift_len=None,
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = (
                self.calculate_shift(dynamic_shift_len)
                if dynamic_shift_len is not None
                else self.exponential_shift_mu
            )
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stabilized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stabilized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep, t_dim=2):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep = timestep[None]
        timestep_id = torch.argmin((self.timesteps[:, None] - timestep).abs(), dim=0)
        shape = [1] * noise.ndim
        shape[t_dim] = timestep_id.shape[0]
        sigma = self.sigmas[timestep_id].to(original_samples).view(shape)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0
        )
        weights = self.linear_timesteps_weights.to(timestep.device)[timestep_id].to(timestep.device)
        return weights

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu


# Attention backends
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


class FlexAttnFunc(nn.Module):
    """Flex-attention backend (training only; ``attn_mode='flex'``).

    Builds the block-causal / window / noise-vs-clean masks used by the dual-stream
    flow-matching training. Inference uses the ``torch`` SDPA backend. The flex-attention
    APIs and their ``torch.compile`` wrappers are imported/initialised lazily so importing
    this module never requires a flex-attention-capable PyTorch build.
    """

    flex_attn = None
    compiled_create_block_mask = None
    attention_mask = None
    cross_attention_mask = None

    def __init__(self, is_cross=False) -> None:
        super().__init__()
        self.is_cross = is_cross

    @classmethod
    def _ensure_compiled(cls):
        if cls.flex_attn is None:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention

            cls.flex_attn = torch.compile(flex_attention, dynamic=True)
            cls.compiled_create_block_mask = torch.compile(create_block_mask)

    def forward(self, query, key, value, dtype=torch.bfloat16):
        self._ensure_compiled()
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask

        x_out = FlexAttnFunc.flex_attn(
            q_varlen,
            k_varlen,
            v_varlen,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        )

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(latent_shape, action_shape, padded_length, chunk_size, window_size, patch_size, device):
        FlexAttnFunc._ensure_compiled()
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        latent_seq_id = (
            torch.arange(B)[:, None, None, None]
            .expand(-1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2])
            .flatten()
        )
        action_seq_id = torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

        latent_frame_id = (
            torch.arange(L_F)[None, :, None, None]
            .expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None]
            .flatten()
        )
        action_frame_id = torch.arange(A_F)[None, :, None, None].expand(B, -1, A_H, A_W)[None].flatten()
        frame_ids = torch.cat(
            [latent_frame_id // chunk_size * 2] * 2 + [action_frame_id // chunk_size * 2 + 1] * 2
        )

        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),
                torch.ones_like(latent_frame_id),
                torch.zeros_like(action_frame_id),
                torch.ones_like(action_frame_id),
            ]
        )

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        mask_mod = FlexAttnFunc._get_mask_mod(
            seq_ids.long().to(device), frame_ids.long().to(device), noise_ids.long().to(device), window_size
        )
        block_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device, _compile=True
        )
        FlexAttnFunc.attention_mask = block_mask

        text_seq_ids = torch.arange(B)[:, None].expand(-1, 512).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(
            seq_ids.long().to(device), text_seq_ids.long().to(device)
        )
        block_mask_cross = FlexAttnFunc.compiled_create_block_mask(
            mask_mod_cross, 1, 1, len(seq_ids), len(text_seq_ids), device=device, _compile=True
        )
        FlexAttnFunc.cross_attention_mask = block_mask_cross

    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        def seq_mask(b, h, q_idx, kv_idx):
            return (
                (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >= 0) & (text_seq_ids[kv_idx] >= 0)
            )

        return seq_mask

    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        from torch.nn.attention.flex_attention import and_masks, or_masks

        def seq_mask(b, h, q_idx, kv_idx):
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >= 0) & (seq_ids[kv_idx] >= 0)

        def block_causal_mask(b, h, q_idx, kv_idx):
            return frame_ids[kv_idx] <= frame_ids[q_idx]

        def block_causal_mask_exclude_self(b, h, q_idx, kv_idx):
            return frame_ids[kv_idx] < frame_ids[q_idx]

        def block_self_mask(b, h, q_idx, kv_idx):
            return frame_ids[kv_idx] == frame_ids[q_idx]

        def clean2clean_mask(b, h, q_idx, kv_idx):
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)

        def noise2clean_mask(b, h, q_idx, kv_idx):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)

        def noise2noise_mask(b, h, q_idx, kv_idx):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)

        def block_window_mask(b, h, q_idx, kv_idx, window_size: int):
            return (frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size

        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask


class WanRotaryPosEmbed(nn.Module):
    """Rotary position embedding with separate frequency bases for frame / height / width."""

    def __init__(self, attention_head_dim: int, patch_size, max_seq_len: int, theta: float = 10000.0):
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

    Backends: ``torch`` (default SDPA), ``flashattn`` (optional), ``flex`` (training masks).
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
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=True), nn.Dropout(dropout)])
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

    def init_kv_cache(self, cache_name, total_token_len, num_head, head_dim, device, dtype, batch_size):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            "k": torch.empty([batch_size, total_token_len, num_head, head_dim], device=device, dtype=dtype),
            "v": torch.empty([batch_size, total_token_len, num_head, head_dim], device=device, dtype=dtype),
            "id": torch.full((total_token_len,), -1, device=device),
            "mask": torch.zeros((total_token_len,), dtype=torch.bool, device=device),
            "is_pred": torch.zeros((total_token_len,), dtype=torch.bool, device=device),
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

    def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
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


# Dual-stream Wan2.2 transformer
class WanTimeTextImageEmbedding(nn.Module):
    def __init__(self, dim, time_freq_dim, time_proj_dim, text_embed_dim, pos_embed_seq_len):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(self, timestep: torch.Tensor, dtype=None):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanTransformerBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, cross_attn_norm=False, eps=1e-6, attn_mode: str = "torch"):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self, hidden_states, encoder_hidden_states, temb, rotary_emb, update_cache=0, cache_name="pos"
    ) -> torch.Tensor:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = rearrange(
            temb_scale_shift_table, "b l n c -> b n l c"
        ).chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(
            hidden_states
        )
        attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb,
            update_cache=update_cache,
            cache_name=cache_name,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            None,
            update_cache=0,
            cache_name=cache_name,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    """Dual-stream (video + action) Wan2.2 DiT backbone with autoregressive KV caching."""

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding_mlp",
        "condition_embedder",
        "condition_embedder_action",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "scale_shift_table_action",
        "norm1",
        "action_norm1",
        "text_norm1",
        "norm2",
        "action_norm2",
        "text_norm2",
        "norm3",
        "action_norm3",
        "text_norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2], inner_dim
        )
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, cross_attn_norm, eps, attn_mode=attn_mode
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    # KV-cache management for autoregressive streaming inference
    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(
        self,
        cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        device,
        dtype,
        batch_size,
    ):
        total_token_len = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2
        ) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(
                cache_name,
                total_token_len,
                self.num_attention_heads,
                self.attention_head_dim,
                device,
                dtype,
                batch_size,
            )

    # Embedding helpers (shared by train + inference paths)
    def _input_embed(self, latents, input_type="latent"):
        if input_type == "latent":
            hidden_states = rearrange(
                latents,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == "action":
            hidden_states = rearrange(latents, "b c f h w -> b (f h w) c")
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == "text":
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        patch_scale_h, patch_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps, (H // patch_scale_h) * (W // patch_scale_w), dim=1
        )
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, timestep_proj = current_condition_embedder(latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    # Dual-stream training forward (flow matching). Requires attn_mode='flex'.
    def forward_train(self, input_dict):
        input_dict["latent_dict"]["noisy_latents"] = input_dict["latent_dict"]["noisy_latents"].to(
            torch.bfloat16
        )
        input_dict["latent_dict"]["latent"] = input_dict["latent_dict"]["latent"].to(torch.bfloat16)
        input_dict["action_dict"]["noisy_latents"] = input_dict["action_dict"]["noisy_latents"].to(
            torch.bfloat16
        )
        input_dict["action_dict"]["latent"] = input_dict["action_dict"]["latent"].to(torch.bfloat16)

        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        batch_size = latent_dict["noisy_latents"].shape[0]

        latent_hidden_states = self._input_embed(latent_dict["noisy_latents"], input_type="latent").flatten(
            0, 1
        )[None]
        action_hidden_states = self._input_embed(action_dict["noisy_latents"], input_type="action").flatten(
            0, 1
        )[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type="text")

        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

        condition_latent_hidden_states = self._input_embed(
            latent_dict["latent"], input_type="latent"
        ).flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(
            action_dict["latent"], input_type="action"
        ).flatten(0, 1)[None]

        hidden_states = torch.cat(
            [
                latent_hidden_states,
                condition_latent_hidden_states,
                action_hidden_states,
                condition_action_hidden_states,
            ],
            dim=1,
        )

        latent_grid_id = latent_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)

        rotary_emb = self.rope(full_grid_id)[:, :, None]

        latent_time_steps = torch.cat(
            [latent_dict["timesteps"].flatten(0, 1), latent_dict["cond_timesteps"].flatten(0, 1)]
        )[None]
        action_time_steps = torch.cat(
            [action_dict["timesteps"].flatten(0, 1), action_dict["cond_timesteps"].flatten(0, 1)]
        )[None]
        latent_temb, latent_timestep_proj = self._time_embed(
            latent_time_steps,
            latent_dict["noisy_latents"].shape[-2],
            latent_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=False,
        )
        action_temb, action_timestep_proj = self._time_embed(
            action_time_steps,
            action_dict["noisy_latents"].shape[-2],
            action_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=True,
        )
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [
            latent_hidden_states.shape[1],
            condition_latent_hidden_states.shape[1],
            action_hidden_states.shape[1],
            condition_action_hidden_states.shape[1],
            padded_length,
        ]

        FlexAttnFunc.init_mask(
            latent_dict["noisy_latents"].shape,
            action_dict["noisy_latents"].shape,
            padded_length,
            input_dict["chunk_size"],
            window_size=input_dict["window_size"],
            patch_size=self.patch_size,
            device=hidden_states.device,
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states, text_hidden_states, timestep_proj, rotary_emb, update_cache=False
            )
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(
            latent_hidden_states, "1 (b l) (n c) -> b (l n) c", n=math.prod(self.patch_size), b=batch_size
        )
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states, "1 (b l) c -> b l c", b=batch_size)

        return latent_hidden_states, action_hidden_states

    # Single-stream inference forward (one denoising step for one stream)
    def forward(self, input_dict, update_cache=0, cache_name="pos", action_mode=False, train_mode=False):
        if train_mode:
            return self.forward_train(input_dict)
        if action_mode:  # action input emb
            latent_hidden_states = rearrange(input_dict["noisy_latents"], "b c f h w -> b (f h w) c")
            latent_hidden_states = self.action_embedder(latent_hidden_states)  # B L1 C
        else:  # latent input emb
            latent_hidden_states = rearrange(
                input_dict["noisy_latents"],
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            latent_hidden_states = self.patch_embedding_mlp(latent_hidden_states)
        text_hidden_states = self.condition_embedder.text_embedder(input_dict["text_emb"])  # B L2 C

        latent_grid_id = input_dict["grid_id"]
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # 1 L 1 C
        patch_scale_h, patch_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])

        latent_time_steps = torch.repeat_interleave(
            input_dict["timesteps"],
            (input_dict["noisy_latents"].shape[-2] // patch_scale_h)
            * (input_dict["noisy_latents"].shape[-1] // patch_scale_w),
            dim=1,
        )  # L
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, timestep_proj = current_condition_embedder(latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        for block in self.blocks:
            latent_hidden_states = block(
                latent_hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (self.norm_out(latent_hidden_states.float()) * (1.0 + scale) + shift).type_as(
            latent_hidden_states
        )

        if action_mode:
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(
                latent_hidden_states, "b l (n c) -> b (l n) c", n=math.prod(self.patch_size)
            )

        return latent_hidden_states


# Wan2.2 VAE helpers (stock diffusers ``AutoencoderKLWan``)
def _vae_patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(
        batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(
        batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size
    )
    return x


def denormalize_latents(latents: torch.Tensor, latents_mean, latents_std, z_dim) -> torch.Tensor:
    """Inverse of the encode-time latent normalization, for VAE-decoding predicted latents."""
    mean = torch.tensor(latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    inv_std = 1.0 / torch.tensor(latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    return latents / inv_std + mean


def load_vae(vae_path, torch_dtype, torch_device, subfolder=None):
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(vae_path, subfolder=subfolder, torch_dtype=torch_dtype)
    return vae.to(torch_device)


def load_text_encoder(text_encoder_path, torch_dtype, torch_device, subfolder=None):
    from transformers import UMT5EncoderModel

    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path, subfolder=subfolder, torch_dtype=torch_dtype
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, subfolder=None):
    from transformers import T5TokenizerFast

    return T5TokenizerFast.from_pretrained(tokenizer_path, subfolder=subfolder)


class WanVAEStreamingWrapper:
    """Wraps an ``AutoencoderKLWan`` encoder to support causal streaming encoding across chunks."""

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = _vae_patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _sample_timestep_id(
    batch_size: int = 1,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Sample per-frame flow-matching timestep ids (upstream ``utils.sample_timestep_id``)."""
    u = torch.rand(size=[batch_size]) * (max_timestep_bd - min_timestep_bd) + min_timestep_bd
    return (u * num_train_timesteps).clamp(min=0, max=num_train_timesteps - 1).to(torch.int64)


class LingBotVAPolicy(PreTrainedPolicy):
    """LeRobot wrapper for the LingBot-VA autoregressive video-action world model."""

    config_class = LingBotVAConfig
    name = "lingbot_va"

    def __init__(self, config: LingBotVAConfig, **kwargs):
        require_package("diffusers", extra="lingbot_va")
        require_package("transformers", extra="lingbot_va")
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.dtype = _torch_dtype(config.dtype)

        # Trainable dual-stream transformer (the only sub-module saved in the LeRobot checkpoint).
        self.transformer = WanTransformer3DModel(
            patch_size=tuple(config.patch_size),
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            action_dim=config.action_dim,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            rope_max_seq_len=config.rope_max_seq_len,
            attn_mode=config.attn_mode,
        )
        # Run the transformer in config.dtype (bf16); norm/modulation paths upcast to fp32 internally.
        self.transformer = self.transformer.to(self.dtype)

        # Frozen modules are stored OUTSIDE the nn.Module registry (plain dict) so they are
        # neither saved into model.safetensors nor moved by ``.to()``. They are lazily loaded
        # from ``config.wan_pretrained_path`` the first time inference runs.
        self._frozen: dict = {}

        self.last_predicted_frames: Tensor | None = None
        self.last_predicted_latents: Tensor | None = None
        self.reset()

    # Frozen-module lazy loading (VAE + UMT5 + tokenizer)
    def _ensure_frozen_modules(self):
        if self._frozen:
            return
        path = self.config.wan_pretrained_path
        device = self.config.device

        # The frozen modules always live in ``vae/``, ``text_encoder/`` and ``tokenizer/``
        # sub-folders -- both in the released diffusers-style HF repos and in the local
        # ``--bundle-frozen`` output dir. ``from_pretrained(path, subfolder=...)`` resolves
        # them for either a HF repo id or a local directory.
        vae = load_vae(path, torch_dtype=self.dtype, torch_device=device, subfolder="vae")
        # The UMT5-XXL text encoder (~11 GB) runs once per episode; keep it on its own
        # (CPU by default) device so the 5B transformer + VAE fit on a single GPU.
        text_encoder = load_text_encoder(
            path,
            torch_dtype=self.dtype,
            torch_device=self.config.text_encoder_device,
            subfolder="text_encoder",
        )
        tokenizer = load_tokenizer(path, subfolder="tokenizer")
        self._frozen = {
            "vae": vae.eval(),
            "streaming_vae": WanVAEStreamingWrapper(vae),
            "text_encoder": text_encoder.eval(),
            "tokenizer": tokenizer,
        }
        # RoboTwin's T-shape layout encodes the half-resolution wrist cameras through a second
        # streaming VAE (separate causal cache) alongside the full-res head camera.
        if self.config.camera_layout == "robotwin_tshape":
            vae_half = load_vae(path, torch_dtype=self.dtype, torch_device=device, subfolder="vae")
            self._frozen["streaming_vae_half"] = WanVAEStreamingWrapper(vae_half.eval())

    @property
    def _vae(self):
        return self._frozen["vae"]

    @property
    def _streaming_vae(self):
        return self._frozen["streaming_vae"]

    # PreTrainedPolicy API
    def get_optim_params(self) -> dict:
        # Only the transformer is trainable; the VAE / text encoder stay frozen (kept outside the
        # nn.Module registry). With PEFT/LoRA this naturally returns just the adapter params.
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def reset(self):
        """Reset all per-episode streaming state (KV cache, queues, frame counter)."""
        cfg = self.config
        self._action_queue: deque = deque(maxlen=cfg.n_action_steps)
        self._obs_buffer: list = []  # raw keyframe obs (one per env substep) observed this chunk
        self._executed_actions: Tensor | None = (
            None  # last chunk's actions (model-normalized) for KV feedback
        )
        self._started = False  # first select_action call uses the obs as the conditioning frame
        self._exec_step = 0  # index of the action being executed within the current chunk
        self._prev_j = 0  # sub-step index (within a predicted frame) of the last executed action
        # Sample one keyframe every ``action_per_frame / temporal_downsample`` executed sub-steps so
        # that exactly ``frame_chunk_size * temporal_downsample`` frames are VAE-encoded per chunk
        # (the Wan2.2 VAE temporal downsample is 4 -> ``frame_chunk_size`` latent frames).
        self._keyframe_stride = max(1, cfg.action_per_frame // 4)
        self._frame_st_id = 0
        self._first_chunk = True
        self._prompt: str | None = None
        self._prompt_embeds = None
        self._negative_prompt_embeds = None
        self.last_predicted_frames = None
        self.last_predicted_latents = None
        self._use_cfg = (cfg.guidance_scale > 1) or (cfg.action_guidance_scale > 1)
        # Two independent flow-matching schedulers (video latent + action streams).
        self._scheduler = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
        self._action_scheduler = FlowMatchScheduler(
            shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self._scheduler.set_timesteps(1000, training=True)
        self._action_scheduler.set_timesteps(1000, training=True)
        self._cache_initialised = False
        # Clear KV cache on the (already-built) transformer, if present.
        if hasattr(self, "transformer"):
            self.transformer.clear_cache("pos")
        # Reset the causal streaming-VAE feat cache between episodes (mirrors upstream ``_reset``).
        # Without this the encoder carries over the previous episode's temporal state, corrupting the
        # latent frame counts on the next episode's first encode.
        if self._frozen:
            self._frozen["streaming_vae"].clear_cache()
            if "streaming_vae_half" in self._frozen:
                self._frozen["streaming_vae_half"].clear_cache()

    # Training (flow-matching dual-stream loss). Requires attn_mode="flex".
    def _ensure_train_schedulers(self):
        if getattr(self, "_train_sched_latent", None) is None:
            cfg = self.config
            self._train_sched_latent = FlowMatchScheduler(
                shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True
            )
            self._train_sched_latent.set_timesteps(1000, training=True)
            self._train_sched_action = FlowMatchScheduler(
                shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True
            )
            self._train_sched_action.set_timesteps(1000, training=True)

    @torch.no_grad()
    def _add_noise_stream(self, latent, scheduler, action_mask, action_mode, noisy_cond_prob):
        """Flow-matching noising of one stream (port of upstream ``Trainer._add_noise``)."""
        device = latent.device
        B, _C, F, _H, _W = latent.shape
        p = self.config.patch_size
        patch_f, patch_h, patch_w = (1, 1, 1) if action_mode else (p[0], p[1], p[2])

        ts_ids = _sample_timestep_id(F, num_train_timesteps=scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = scheduler.timesteps[ts_ids].to(device)
        noisy_latents = scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = scheduler.training_target(latent, noise, timesteps)

        grid_id = (
            get_mesh_id(
                latent.shape[-3] // patch_f,
                latent.shape[-2] // patch_h,
                latent.shape[-1] // patch_w,
                t=1 if action_mode else 0,
                f_w=1,
                f_shift=0,
                action=action_mode,
            )
            .to(device)[None]
            .repeat(B, 1, 1)
        )

        if torch.rand(1).item() < noisy_cond_prob:
            cond_ids = _sample_timestep_id(
                F, min_timestep_bd=0.5, max_timestep_bd=1.0, num_train_timesteps=scheduler.num_train_timesteps
            )
            cond_noise = torch.zeros_like(latent).normal_()
            cond_timesteps = scheduler.timesteps[cond_ids].to(device)
            latent = scheduler.add_noise(latent, cond_noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents = noisy_latents * action_mask.float()
            targets = targets * action_mask.float()
            latent = latent * action_mask.float()

        return {
            "timesteps": timesteps[None].repeat(B, 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps[None].repeat(B, 1),
            "grid_id": grid_id,
        }

    def _flow_matching_loss(self, input_dict, pred):
        """Dual-stream flow-matching loss (port of upstream ``Trainer.compute_loss``)."""
        latent_pred, action_pred = pred
        ld, ad = input_dict["latent_dict"], input_dict["action_dict"]
        action_pred = rearrange(action_pred, "b (f n) c -> b c f n 1", f=ad["targets"].shape[-3])
        latent_pred = data_seq_to_patch(
            self.config.patch_size,
            latent_pred,
            ld["targets"].shape[-3],
            ld["targets"].shape[-2],
            ld["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )
        Bn, Fn = ld["timesteps"].shape
        lw = self._train_sched_latent.training_weight(ld["timesteps"].flatten()).reshape(Bn, Fn)
        aw = self._train_sched_action.training_weight(ad["timesteps"].flatten()).reshape(Bn, Fn)

        latent_loss = F.mse_loss(latent_pred.float(), ld["targets"].float().detach(), reduction="none")
        latent_loss = (
            (latent_loss * lw[:, None, :, None, None]).permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        )
        latent_loss = (latent_loss.sum(dim=1) / (torch.ones_like(latent_loss).sum(dim=1) + 1e-6)).mean()

        amask = ad["actions_mask"].float()
        action_loss = F.mse_loss(action_pred.float(), ad["targets"].float().detach(), reduction="none")
        action_loss = (
            (action_loss * aw[:, None, :, None, None] * amask).permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        )
        amask_f = amask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_loss = (action_loss.sum(dim=1) / (amask_f.sum(dim=1) + 1e-6)).mean()
        return latent_loss, action_loss

    def training_loss_from_streams(self, latents, actions, actions_mask, text_emb):
        """Core dual-stream training loss given prepared latents / actions / text embeddings.

        ``latents``: ``[B, in_channels, F, h, w]`` (normalized video latents).
        ``actions`` / ``actions_mask``: ``[B, action_dim, F, action_per_frame, 1]``.
        ``text_emb``: ``[B, seq_len, text_dim]``. Returns ``(loss, {latent_loss, action_loss})``.
        """
        if self.config.attn_mode != "flex":
            raise ValueError(
                "LingBot-VA training requires attn_mode='flex' (block-causal flow-matching masks). "
                "Load/convert the policy with --policy.attn_mode=flex for training/fine-tuning."
            )
        self._ensure_train_schedulers()
        latent_dict = self._add_noise_stream(
            latents, self._train_sched_latent, action_mask=None, action_mode=False, noisy_cond_prob=0.5
        )
        action_dict = self._add_noise_stream(
            actions, self._train_sched_action, action_mask=actions_mask, action_mode=True, noisy_cond_prob=0.0
        )
        latent_dict["text_emb"] = text_emb
        action_dict["text_emb"] = text_emb
        action_dict["actions_mask"] = actions_mask
        input_dict = {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": int(torch.randint(1, 5, (1,)).item()),
            "window_size": int(torch.randint(4, 65, (1,)).item()),
        }
        pred = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self._flow_matching_loss(input_dict, pred)
        loss = latent_loss + action_loss
        return loss, {"latent_loss": latent_loss.detach(), "action_loss": action_loss.detach()}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Training forward: dual-stream flow-matching loss.

        Builds the (video-latent, action, text) training streams from a LeRobot batch
        (VAE-encoding the camera frames and UMT5-encoding the task), then runs the flow-matching
        dual-stream loss. Requires the policy to be built with ``attn_mode='flex'``.
        """
        self._ensure_frozen_modules()
        latents, actions, actions_mask, text_emb = self._build_training_streams(batch)
        return self.training_loss_from_streams(latents, actions, actions_mask, text_emb)

    @torch.no_grad()
    def _build_training_streams(self, batch):
        """Build (latents, actions, actions_mask, text_emb) from a LeRobot training batch.

        Camera frames per ``obs_cam_keys`` are expected as a temporal clip ``[B, C, T, H, W]`` (or
        ``[B, T, C, H, W]``); they are VAE-encoded into ``F = T / temporal_downsample`` latent frames.
        Actions ``[B, F*action_per_frame, n_used]`` are scattered into the model's ``action_dim`` space.
        """
        cfg = self.config
        device = cfg.device
        # text embeddings
        task = batch.get("task")
        if isinstance(task, str):
            task = [task]
        text_emb = self._get_t5_prompt_embeds(list(task), cfg.max_sequence_length)

        # video latents (VAE-encode the camera clips)
        latents = self._encode_training_latents(batch)

        # actions -> [B, action_dim, F, action_per_frame, 1]
        act = batch[ACTION].to(device)  # [B, F*apf, n_used]
        B = act.shape[0]
        used = cfg.used_action_channel_ids
        apf, Fc = cfg.action_per_frame, cfg.frame_chunk_size
        act = act[:, : Fc * apf].reshape(B, Fc, apf, len(used)).permute(0, 3, 1, 2)  # [B, n_used, F, apf]
        full = act.new_zeros(B, cfg.action_dim, Fc, apf)
        idx = torch.as_tensor(used, device=device)
        full[:, idx] = act
        actions = full.unsqueeze(-1).to(self.dtype)  # [B, action_dim, F, apf, 1]
        mask = torch.zeros(cfg.action_dim, device=device, dtype=self.dtype)
        mask[idx] = 1.0
        actions_mask = mask.view(1, -1, 1, 1, 1).expand_as(actions)
        return latents, actions, actions_mask, text_emb

    @torch.no_grad()
    def _encode_training_latents(self, batch) -> Tensor:
        """VAE-encode the per-camera training clips into normalized video latents [B, C, F, h, w]."""
        vae_device = next(self._vae.parameters()).device

        def _clip(key):
            x = batch[key].to(vae_device)
            if x.dim() == 4:  # [B, C, H, W] -> single frame clip
                x = x.unsqueeze(2)
            elif x.shape[1] not in (1, 3) and x.shape[2] in (1, 3):  # [B, T, C, H, W] -> [B, C, T, H, W]
                x = x.permute(0, 2, 1, 3, 4)
            return x.contiguous()

        def _encode(x, size):
            b, c, t = x.shape[:3]
            x = F.interpolate(x.flatten(0, 1).float(), size=size, mode="bilinear", align_corners=False)
            x = (x.view(b, c, t, *size) * 2.0 - 1.0).to(self.dtype)
            mu = self._vae.encode(x).latent_dist.mode()  # [B, z_dim, F, h, w]
            mean = torch.tensor(self._vae.config.latents_mean).view(1, -1, 1, 1, 1).to(mu.device)
            inv_std = (1.0 / torch.tensor(self._vae.config.latents_std)).view(1, -1, 1, 1, 1).to(mu.device)
            return ((mu.float() - mean) * inv_std).to(mu)

        keys = self.config.obs_cam_keys
        if self.config.camera_layout == "robotwin_tshape":
            h, w = self.config.height, self.config.width
            head = _encode(_clip(keys[0]), (h, w))
            left = _encode(_clip(keys[1]), (h // 2, w // 2))
            right = _encode(_clip(keys[2]), (h // 2, w // 2))
            return torch.cat([torch.cat([left, right], dim=-1), head], dim=-2).to(self.config.device)
        per_cam = [_encode(_clip(k), (self.config.height, self.config.width)) for k in keys]
        return torch.cat(per_cam, dim=-1).to(self.config.device)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action, refilling the chunk (and feeding back observed keyframes) as needed.

        Mirrors the upstream LIBERO client loop (``evaluation/libero/client.py``): the first obs is
        the conditioning frame; every observation produced afterwards is buffered as a keyframe and,
        once the chunk's actions are exhausted, the buffered frames + executed actions are fed back
        into the KV cache before the next chunk is predicted.
        """
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)

        if not self._started:
            # First call: this observation conditions the first chunk (it is *not* a keyframe).
            self._started = True
            actions = self.predict_action_chunk(batch)  # [B, chunk_size, n_used]
            self._action_queue.extend(actions.transpose(0, 1))  # [chunk_size, B, n_used]
            self._obs_buffer = []
            self._exec_step = 0
        else:
            # This observation is the result of the previously executed action -> a candidate
            # keyframe. Buffer it on the sub-step boundary the upstream client samples on.
            if (self._prev_j + 1) % self._keyframe_stride == 0:
                self._obs_buffer.append(self._extract_raw_obs(batch))
            if len(self._action_queue) == 0:
                # All actions for the current chunk have been executed; feed the observed
                # keyframes + executed actions back and predict the next chunk.
                actions = self.predict_action_chunk(None)
                self._action_queue.extend(actions.transpose(0, 1))
                self._exec_step = 0

        self._prev_j = self._exec_step % self.config.action_per_frame
        self._exec_step += 1
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Run one autoregressive chunk and return actions ``[B, chunk_size, n_used]`` (normalized)."""
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)

        is_first = self._first_chunk
        if is_first:
            init_latent = self._encode_frames([self._extract_raw_obs(batch)])
            self._init_latent = init_latent
            self._init_streaming_cache(init_latent)
            self._obs_buffer = []  # frame 0 (the init obs) conditions the chunk; it is not fed back
            actions, latents = self._infer(init_latent, frame_st_id=0)
            self._first_chunk = False
        else:
            # Feed the real observed keyframes + the executed actions back into the KV cache.
            self._compute_kv_cache(self._obs_buffer, self._executed_actions)
            self._obs_buffer = []
            actions, latents = self._infer(None, frame_st_id=self._frame_st_id)

        # actions: [B, action_dim, F, action_per_frame, 1] (model-normalized). Keep for KV feedback.
        self._executed_actions = actions

        if self.config.save_predicted_video:
            # Match upstream LingBot-VA visualization: collect chunk latents and decode the
            # concatenated latent sequence once after the rollout finishes.
            self.last_predicted_frames = None
            self.last_predicted_latents = latents.detach().to("cpu")

        # On the first chunk, frame 0 is the conditioning frame (already "known"): the upstream
        # LIBERO client skips it (start_idx=1), so we drop the first frame's actions here.
        used = self.config.used_action_channel_ids
        a = actions[:, used]  # [B, n_used, F, action_per_frame, 1]
        if is_first:
            a = a[:, :, 1:]  # drop frame 0 -> (F-1) frames of actions
        a = a.squeeze(-1).flatten(2)  # [B, n_used, n_steps]
        a = a.transpose(1, 2).contiguous()  # [B, n_steps, n_used]
        return a.to(torch.float32)

    # Prompt / text encoding
    def _maybe_init_prompt(self, batch):
        if self._prompt_embeds is not None or batch is None:
            return
        task = batch.get("task")
        prompt = task[0] if isinstance(task, list | tuple) else task
        self._prompt = prompt or ""
        self._prompt_embeds, self._negative_prompt_embeds = self._encode_prompt(self._prompt)

    def _get_t5_prompt_embeds(self, prompt, max_sequence_length):
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        tokenizer = self._frozen["tokenizer"]
        text_encoder = self._frozen["text_encoder"]
        device = self.config.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        te_device = next(text_encoder.parameters()).device
        prompt_embeds = text_encoder(text_input_ids.to(te_device), mask.to(te_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens, strict=False)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )
        return prompt_embeds.to(device)

    def _encode_prompt(self, prompt):
        max_len = self.config.max_sequence_length
        prompt_embeds = self._get_t5_prompt_embeds(prompt, max_len)
        negative_prompt_embeds = None
        if self._use_cfg:
            negative_prompt_embeds = self._get_t5_prompt_embeds("", max_len)
        return prompt_embeds, negative_prompt_embeds

    # Observation (image) encoding -> normalized video latents
    def _extract_raw_obs(self, batch) -> dict[str, Tensor]:
        """Snapshot the configured camera images from a batch (kept raw for later VAE encoding)."""
        return {k: batch[k].detach() for k in self.config.obs_cam_keys}

    def _camera_frame(self, raw_obs, key, size=None) -> Tensor:
        """Return a single-frame camera tensor [1, C, 1, H, W] resized + scaled to [-1, 1]."""
        img = raw_obs[key]
        if img.dim() == 3:  # [C, H, W]
            img = img.unsqueeze(0)
        # LeRobot images arrive as float in [0, 1], shape [B, C, H, W].
        img = img.to(self.config.device, torch.float32)
        if self.config.image_hflip:
            img = torch.flip(img, dims=[-1])  # undo the env processor's horizontal flip
        if size is None:
            size = (self.config.height, self.config.width)
        img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        img = img * 2.0 - 1.0
        return img.unsqueeze(2).to(self.dtype)  # [1, C, F=1, H, W]

    def _normalize_vae_latent(self, enc_out: Tensor) -> Tensor:
        """Take the mean of a VAE encoder output and channel-normalize it (matches upstream)."""
        mu, _logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self._vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self._vae.config.latents_std).to(mu.device)
        mean = latents_mean.view(1, -1, 1, 1, 1)
        inv_std = (1.0 / latents_std).view(1, -1, 1, 1, 1)
        return ((mu.float() - mean) * inv_std).to(mu)

    @torch.no_grad()
    def _encode_frames(self, raw_frames: list) -> Tensor:
        """VAE-encode a temporal clip of observed frames and concat the per-camera latents on width.

        ``raw_frames`` is a list of per-frame obs dicts (one per env sub-step). Each configured
        camera is stacked along the temporal axis into a ``[1, C, F, H, W]`` clip and encoded in a
        single streaming ``encode_chunk`` call so the VAE temporal downsample (x4) collapses the F
        input frames into ``F / 4`` latent frames, with the causal ``feat_cache`` carried across
        chunks (mirrors upstream ``_encode_obs``).
        """
        vae_device = next(self._vae.parameters()).device
        if self.config.camera_layout == "robotwin_tshape":
            return self._encode_frames_tshape(raw_frames, vae_device)
        per_cam_videos = []
        for k in self.config.obs_cam_keys:
            frames = [self._camera_frame(fb, k) for fb in raw_frames]
            per_cam_videos.append(torch.cat(frames, dim=2))  # [1, C, F, H, W]
        videos = torch.cat(per_cam_videos, dim=0)  # [num_cam, C, F, H, W]
        enc_out = self._streaming_vae.encode_chunk(videos.to(vae_device).to(self.dtype))
        mu_norm = self._normalize_vae_latent(enc_out)
        # Concatenate the per-camera latents along width.
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.config.device)

    @torch.no_grad()
    def _encode_frames_tshape(self, raw_frames: list, vae_device) -> Tensor:
        """RoboTwin T-shape latent assembly: full-res head + half-res wrists (second streaming VAE).

        The two wrist latents are concatenated on width and stacked (on the height axis) on top of
        the head latent, mirroring upstream ``_encode_obs`` for ``env_type='robotwin_tshape'``.
        """
        cfg = self.config
        h, w = cfg.height, cfg.width
        head_key, left_key, right_key = cfg.obs_cam_keys[0], cfg.obs_cam_keys[1], cfg.obs_cam_keys[2]
        head = torch.cat([self._camera_frame(fb, head_key, size=(h, w)) for fb in raw_frames], dim=2)
        left = torch.cat(
            [self._camera_frame(fb, left_key, size=(h // 2, w // 2)) for fb in raw_frames], dim=2
        )
        right = torch.cat(
            [self._camera_frame(fb, right_key, size=(h // 2, w // 2)) for fb in raw_frames], dim=2
        )
        wrists = torch.cat([left, right], dim=0)  # [2, C, F, H/2, W/2]
        enc_high = self._streaming_vae.encode_chunk(head.to(vae_device).to(self.dtype))
        enc_lr = self._frozen["streaming_vae_half"].encode_chunk(wrists.to(vae_device).to(self.dtype))
        # wrists side-by-side on width, then stacked on top of the head latent on the height axis.
        enc_out = torch.cat([torch.cat(enc_lr.split(1, dim=0), dim=-1), enc_high], dim=-2)
        video_latent = self._normalize_vae_latent(enc_out)
        return video_latent.to(self.config.device)

    # KV cache management
    @property
    def _latent_hw(self):
        if self.config.camera_layout == "robotwin_tshape":
            # head (full) on the bottom, two half-res wrists side-by-side on top -> 1.5x height.
            return ((self.config.height // 16) * 3) // 2, self.config.width // 16
        h = self.config.height // 16
        w = (self.config.width // 16) * len(self.config.obs_cam_keys)
        return h, w

    def _init_streaming_cache(self, init_latent):
        cfg = self.config
        latent_h, latent_w = self._latent_hw
        p = cfg.patch_size
        latent_token_per_chunk = (cfg.frame_chunk_size * latent_h * latent_w) // (p[0] * p[1] * p[2])
        action_token_per_chunk = cfg.frame_chunk_size * cfg.action_per_frame
        self.transformer.create_empty_cache(
            "pos",
            cfg.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            device=self.config.device,
            dtype=self.dtype,
            batch_size=2 if self._use_cfg else 1,
        )
        self._cache_initialised = True

    def _repeat_input_for_cfg(self, input_dict):
        if self._use_cfg:
            input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1)
            input_dict["text_emb"] = torch.cat(
                [
                    self._prompt_embeds.to(self.dtype).clone(),
                    self._negative_prompt_embeds.to(self.dtype).clone(),
                ],
                dim=0,
            )
            input_dict["grid_id"] = input_dict["grid_id"][None].repeat(2, 1, 1)
            input_dict["timesteps"] = input_dict["timesteps"][None].repeat(2, 1)
        else:
            input_dict["grid_id"] = input_dict["grid_id"][None]
            input_dict["timesteps"] = input_dict["timesteps"][None]
        return input_dict

    def _prepare_latent_input(
        self,
        latent_model_input,
        action_model_input,
        latent_t=0,
        action_t=0,
        latent_cond=None,
        action_cond=None,
        frame_st_id=0,
    ):
        cfg = self.config
        device = self.config.device
        p = cfg.patch_size
        out = {}
        if latent_model_input is not None:
            out["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones([latent_model_input.shape[2]], dtype=torch.float32, device=device)
                * latent_t,
                "grid_id": get_mesh_id(
                    latent_model_input.shape[-3] // p[0],
                    latent_model_input.shape[-2] // p[1],
                    latent_model_input.shape[-1] // p[2],
                    0,
                    1,
                    frame_st_id,
                ).to(device),
                "text_emb": self._prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                out["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
                out["latent_res_lst"]["timesteps"][0:1] *= 0
        if action_model_input is not None:
            out["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones([action_model_input.shape[2]], dtype=torch.float32, device=device)
                * action_t,
                "grid_id": get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                ).to(device),
                "text_emb": self._prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                out["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                out["action_res_lst"]["timesteps"][0:1] *= 0
            out["action_res_lst"]["noisy_latents"][:, ~self._action_mask] *= 0
        return out

    @property
    def _action_mask(self):
        mask = torch.zeros([self.config.action_dim], dtype=torch.bool)
        mask[self.config.used_action_channel_ids] = True
        return mask

    # Action conditioning (executed action history) (de)normalization
    def _preprocess_action_state(self, action_norm: Tensor) -> Tensor:
        """Build the action-conditioning tensor from the already-normalized executed actions.

        ``action_norm`` is the model-space action chunk ``[B, action_dim, F, action_per_frame, 1]``.
        Upstream re-derives the conditioning from the raw executed action via quantile norm; here
        the executed actions are already in the model-normalized space, so we pass them through.
        """
        return action_norm.to(self.config.device, self.dtype)

    def _compute_kv_cache(self, obs_buffer, executed_actions):
        """Feed real observed keyframes + executed actions back into the KV cache."""
        if not obs_buffer or executed_actions is None:
            return
        self.transformer.clear_pred_cache("pos")
        # Encode the buffered keyframe clip in one streaming call (carries the causal VAE cache).
        latent_model_input = self._encode_frames(obs_buffer)
        # On the first feedback, prepend the init latent so the latent/action frame counts align
        # (upstream prepends ``init_latent`` to the observed keyframes when frame_st_id == 0).
        if self._frame_st_id == 0 and getattr(self, "_init_latent", None) is not None:
            latent_model_input = torch.cat([self._init_latent, latent_model_input], dim=2)
        action_model_input = self._preprocess_action_state(executed_actions)
        action_model_input = action_model_input.to(latent_model_input)
        input_dict = self._prepare_latent_input(
            latent_model_input, action_model_input, frame_st_id=self._frame_st_id
        )
        with torch.no_grad():
            self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=2,
                cache_name="pos",
                action_mode=False,
            )
            self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=2,
                cache_name="pos",
                action_mode=True,
            )
        self._frame_st_id += latent_model_input.shape[2]

    # The core dual-stream denoising loop (one chunk)
    @torch.no_grad()
    def _infer(self, init_latent, frame_st_id=0):
        cfg = self.config
        device = self.config.device
        latent_h, latent_w = self._latent_hw
        frame_chunk_size = cfg.frame_chunk_size

        latents = torch.randn(1, 48, frame_chunk_size, latent_h, latent_w, device=device, dtype=self.dtype)
        actions = torch.randn(
            1, cfg.action_dim, frame_chunk_size, cfg.action_per_frame, 1, device=device, dtype=self.dtype
        )

        self._scheduler.set_timesteps(cfg.num_inference_steps)
        self._action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        timesteps = F.pad(self._scheduler.timesteps, (0, 1), mode="constant", value=0)
        if cfg.video_exec_step != -1:
            timesteps = timesteps[: cfg.video_exec_step]
        action_timesteps = F.pad(self._action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        # 1. Video-latent denoising loop
        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            latent_cond = (
                init_latent[:, :, 0:1].to(self.dtype)
                if frame_st_id == 0 and init_latent is not None
                else None
            )
            input_dict = self._prepare_latent_input(
                latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id
            )
            video_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name="pos",
                action_mode=False,
            )
            if not last_step or cfg.video_exec_step != -1:
                video_noise_pred = data_seq_to_patch(
                    cfg.patch_size,
                    video_noise_pred,
                    frame_chunk_size,
                    latent_h,
                    latent_w,
                    batch_size=2 if self._use_cfg else 1,
                )
                if cfg.guidance_scale > 1:
                    video_noise_pred = video_noise_pred[1:] + cfg.guidance_scale * (
                        video_noise_pred[:1] - video_noise_pred[1:]
                    )
                else:
                    video_noise_pred = video_noise_pred[:1]
                latents = self._scheduler.step(video_noise_pred, t, latents, return_dict=False)
            if frame_st_id == 0 and latent_cond is not None:
                latents[:, :, 0:1] = latent_cond

        # 2. Action denoising loop
        for i, t in enumerate(action_timesteps):
            last_step = i == len(action_timesteps) - 1
            action_cond = (
                torch.zeros([1, cfg.action_dim, 1, cfg.action_per_frame, 1], device=device, dtype=self.dtype)
                if frame_st_id == 0
                else None
            )
            input_dict = self._prepare_latent_input(
                None, actions, t, t, None, action_cond, frame_st_id=frame_st_id
            )
            action_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name="pos",
                action_mode=True,
            )
            if not last_step:
                from einops import rearrange

                action_noise_pred = rearrange(action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size)
                if cfg.action_guidance_scale > 1:
                    action_noise_pred = action_noise_pred[1:] + cfg.action_guidance_scale * (
                        action_noise_pred[:1] - action_noise_pred[1:]
                    )
                else:
                    action_noise_pred = action_noise_pred[:1]
                actions = self._action_scheduler.step(action_noise_pred, t, actions, return_dict=False)
            if frame_st_id == 0 and action_cond is not None:
                actions[:, :, 0:1] = action_cond

        actions[:, ~self._action_mask] *= 0
        return actions, latents

    # Predicted-video decoding (opt-in)
    @torch.no_grad()
    def decode_predicted_latents(self, latents) -> Tensor:
        """Decode a concatenated predicted-latent sequence into ``[T, H, W, 3]`` uint8 frames."""
        return self._decode_predicted_video(latents)

    @torch.no_grad()
    def _decode_predicted_video(self, latents) -> Tensor:
        """VAE-decode predicted latents into a uint8 frame stack ``[T, H, W, 3]`` on CPU."""
        vae = self._vae
        z_dim = vae.config.z_dim
        vae_device = next(vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=vae.dtype)
        latents = denormalize_latents(
            latents, vae.config.latents_mean, vae.config.latents_std, z_dim
        )
        video = vae.decode(latents, return_dict=False)[0]  # [B, C, F, H, W] in [-1, 1]
        video = (video.float().clamp(-1, 1) + 1.0) / 2.0
        video = (video[0].permute(1, 2, 3, 0) * 255.0).round().to(torch.uint8)  # [F, H, W, C]
        return video.cpu()
