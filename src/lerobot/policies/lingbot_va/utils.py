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

"""Vendored Wan2.2 model code and plumbing for the LingBot-VA policy.

Everything the policy builds on lives here: grid/patch reshaping, attention backends,
VAE (de)normalization + frozen-component loaders, the flow-matching scheduler, and the
dual-stream Wan transformer (``WanTransformer3DModel`` and its sub-modules). Only the
LeRobot-facing ``LingBotVAPolicy`` orchestrator stays in ``modeling_lingbot_va.py``; this
module imports nothing from it (one-directional dependency).
"""

import html
import math
import re
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange

from lerobot.utils.import_utils import _diffusers_available, _transformers_available

if TYPE_CHECKING or _diffusers_available:
    from diffusers import AutoencoderKLWan
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.attention import FeedForward
    from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.models.normalization import FP32LayerNorm
else:
    AutoencoderKLWan = FeedForward = PixArtAlphaTextProjection = None
    TimestepEmbedding = Timesteps = FP32LayerNorm = None

    class ModelMixin:
        pass

    class ConfigMixin:
        pass

    def register_to_config(func):
        return func


if TYPE_CHECKING or _transformers_available:
    from transformers import T5TokenizerFast, UMT5EncoderModel
else:
    T5TokenizerFast = UMT5EncoderModel = None


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
    vae = AutoencoderKLWan.from_pretrained(vae_path, subfolder=subfolder, torch_dtype=torch_dtype)
    return vae.to(torch_device)


def load_text_encoder(text_encoder_path, torch_dtype, torch_device, subfolder=None):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path, subfolder=subfolder, torch_dtype=torch_dtype
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, subfolder=None):
    return T5TokenizerFast.from_pretrained(tokenizer_path, subfolder=subfolder)


# Misc
def clean_prompt(text: str) -> str:
    """Normalize a task prompt (HTML-unescape + whitespace collapse).

    Mirrors diffusers' Wan ``prompt_clean`` minus ``ftfy.fix_text``,
    which is a no-op for the ASCII task strings used here, so we avoid the extra ``ftfy`` dep.
    """
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


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
        if dtype not in half_dtypes:
            raise ValueError(f"Flex attention requires a half-precision dtype, got {dtype}.")

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
        b, _, l_f, l_h, l_w = latent_shape
        _, _, a_f, a_h, a_w = action_shape

        latent_seq_id = (
            torch.arange(b)[:, None, None, None]
            .expand(-1, l_f // patch_size[0], l_h // patch_size[1], l_w // patch_size[2])
            .flatten()
        )
        action_seq_id = torch.arange(b)[:, None, None, None].expand(-1, a_f, a_h, a_w).flatten()
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

        latent_frame_id = (
            torch.arange(l_f)[None, :, None, None]
            .expand(b, -1, l_h // patch_size[1], l_w // patch_size[2])[None]
            .flatten()
        )
        action_frame_id = torch.arange(a_f)[None, :, None, None].expand(b, -1, a_h, a_w)[None].flatten()
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

        text_seq_ids = torch.arange(b)[:, None].expand(-1, 512).flatten()
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

        if free.numel() < key_size:
            raise RuntimeError(f"KV cache exhausted: need {key_size} free slots, have {free.numel()}.")
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

        if update_cache == 0 and kv_cache is not None and kv_cache["k"] is not None:
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
        b, seq_len = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(b, seq_len, -1), timestep_proj.reshape(b, seq_len, -1)


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

    def _time_embed(self, timesteps, h, w, dtype, action_mode=False):
        patch_scale_h, patch_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps, (h // patch_scale_h) * (w // patch_scale_w), dim=1
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
