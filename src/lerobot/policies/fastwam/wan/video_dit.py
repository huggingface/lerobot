# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange

from .model import (
    WanAttentionBlock,
    WanLayerNorm,
    WanModel,
    WanRMSNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)

logger = logging.getLogger(__name__)


def get_sampling_sigmas(sampling_steps, shift):
    # Vendored from Wan2.2 (formerly wan/utils/fm_solvers.py); computes the
    # noise-level (sigma) schedule for Wan-compatible flow-matching inference.
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)

    return custom_forward


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    *args,
    **kwargs,
):
    if use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output


def fastwam_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: torch.Tensor | None = None,
    fp32_attention: bool = True,
) -> torch.Tensor:
    """FastWAM masked attention wrapper for MoT masks and CPU test coverage.

    The official Wan attention implementation is still used as the source of
    the projection/norm modules. This wrapper only replaces the final attention
    kernel because FastWAM needs explicit boolean masks for video/action MoT
    routing, while the upstream FlashAttention path accepts sequence lengths
    but not arbitrary [query, key] masks.
    """

    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    if fp32_attention:
        q = q.float()
        k = k.float()
        v = v.float()
    else:
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)
    x = functional.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based Wan sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        sigma_steps = torch.as_tensor(
            get_sampling_sigmas(num_inference_steps, shift),
            device=device,
            dtype=torch.float32,
        )
        timesteps = sigma_steps * float(self.num_train_timesteps)
        sigma_next = torch.cat([sigma_steps[1:], sigma_steps.new_zeros(1)])
        deltas = sigma_next - sigma_steps
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    return rope_params(end, dim, theta)


def apply_dense_rope(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float32).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def _linear_input(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=linear.weight.dtype)


def _wan_layer_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(norm, WanLayerNorm) and norm.weight is not None:
        weight = norm.weight.float()
        bias = norm.bias.float() if norm.bias is not None else None
        return functional.layer_norm(x.float(), norm.normalized_shape, weight, bias, norm.eps).to(
            dtype=x.dtype
        )
    return norm(x)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    if mode not in ["causal", "group_diagonal"]:
        raise ValueError(f"`mode` must be 'causal' or 'group_diagonal', got {mode}.")
    if num_temporal_groups <= 0:
        raise ValueError(f"`num_temporal_groups` must be positive, got {num_temporal_groups}.")
    if num_query_per_group <= 0:
        raise ValueError(f"`num_query_per_group` must be positive, got {num_query_per_group}.")
    if num_key_per_group <= 0:
        raise ValueError(f"`num_key_per_group` must be positive, got {num_key_per_group}.")

    total_num_query_tokens = num_temporal_groups * num_query_per_group
    total_num_key_tokens = num_temporal_groups * num_key_per_group
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group).unsqueeze(1)
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group).unsqueeze(0)

    if mode == "causal":
        attn_mask = query_time_indices >= key_time_indices
    else:
        attn_mask = query_time_indices == key_time_indices

    if attn_mask.shape != (total_num_query_tokens, total_num_key_tokens):
        raise RuntimeError("Attention mask shape mismatch.")
    return attn_mask


class FastWAMAttentionBlock(WanAttentionBlock):
    """Wan attention block with FastWAM's arbitrary boolean mask support."""

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        fp32_attention: bool = True,
    ):
        attention_dim = attn_head_dim * num_heads
        if hidden_dim == attention_dim:
            super().__init__(
                dim=hidden_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                qk_norm=True,
                cross_attn_norm=True,
                eps=eps,
            )
        else:
            nn.Module.__init__(self)
            self.dim = hidden_dim
            self.ffn_dim = ffn_dim
            self.num_heads = num_heads
            self.qk_norm = True
            self.cross_attn_norm = True
            self.eps = eps
            self.norm1 = WanLayerNorm(hidden_dim, eps)
            self.self_attn = _FastWAMProjectedAttention(hidden_dim, attention_dim, num_heads, eps)
            self.norm3 = WanLayerNorm(hidden_dim, eps, elementwise_affine=True)
            self.cross_attn = _FastWAMProjectedAttention(hidden_dim, attention_dim, num_heads, eps)
            self.norm2 = WanLayerNorm(hidden_dim, eps)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(ffn_dim, hidden_dim),
            )
            self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.attn_head_dim = attn_head_dim
        self.fp32_attention = bool(fp32_attention)

    @staticmethod
    def split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(
            6, dim=chunk_dim
        )
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def project_self_attention(
        self, x: torch.Tensor, freqs: torch.Tensor | dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.self_attn.norm_q(self.self_attn.q(x))
        k = self.self_attn.norm_k(self.self_attn.k(x))
        v = self.self_attn.v(x)
        if isinstance(freqs, dict):
            b, s = x.shape[:2]
            q = rope_apply(
                q.view(b, s, self.num_heads, self.attn_head_dim),
                freqs["grid_sizes"],
                freqs["freqs"],
            ).flatten(2)
            k = rope_apply(
                k.view(b, s, self.num_heads, self.attn_head_dim),
                freqs["grid_sizes"],
                freqs["freqs"],
            ).flatten(2)
        else:
            q = apply_dense_rope(q, freqs, self.num_heads)
            k = apply_dense_rope(k, freqs, self.num_heads)
        return q, k, v

    def apply_cross_attention(
        self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        attn = self.cross_attn
        b, n, d = x.size(0), attn.num_heads, attn.head_dim
        q = attn.norm_q(attn.q(x)).view(b, -1, n * d)
        k = attn.norm_k(attn.k(context)).view(b, -1, n * d)
        v = attn.v(context).view(b, -1, n * d)
        x = fastwam_masked_attention(
            q=q,
            k=k,
            v=v,
            num_heads=n,
            ctx_mask=context_mask,
            fp32_attention=self.fp32_attention,
        )
        return attn.o(_linear_input(attn.o, x))

    def project_self_attention_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn.o(_linear_input(self.self_attn.o, x))

    def apply_norm1(self, x: torch.Tensor) -> torch.Tensor:
        return _wan_layer_norm(self.norm1, x)

    def apply_norm2(self, x: torch.Tensor) -> torch.Tensor:
        return _wan_layer_norm(self.norm2, x)

    def apply_norm3(self, x: torch.Tensor) -> torch.Tensor:
        return _wan_layer_norm(self.norm3, x)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.split_modulation(self, t_mod)
        residual_x = x
        attn_input = modulate(self.apply_norm1(x), shift_msa, scale_msa)
        q, k, v = self.project_self_attention(attn_input, freqs)
        y = fastwam_masked_attention(
            q=q,
            k=k,
            v=v,
            num_heads=self.num_heads,
            ctx_mask=self_attn_mask,
            fp32_attention=self.fp32_attention,
        )
        x = residual_x + gate_msa * self.project_self_attention_output(y)
        x = x + self.apply_cross_attention(self.apply_norm3(x), context, context_mask=context_mask)
        mlp_input = modulate(self.apply_norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp * self.ffn(mlp_input)


class _FastWAMProjectedAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int, num_heads: int, eps: float):
        super().__init__()
        self.dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.q = nn.Linear(hidden_dim, attention_dim)
        self.k = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(hidden_dim, attention_dim)
        self.o = nn.Linear(attention_dim, hidden_dim)
        self.norm_q = WanRMSNorm(attention_dim, eps=eps)
        self.norm_k = WanRMSNorm(attention_dim, eps=eps)


class WanVideoDiT(WanModel):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool = False,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode="causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
        fp32_attention: bool = True,
    ):
        del in_dim_control_adapter
        if has_image_input:
            raise ValueError("FastWAM currently expects Wan2.2 TI2V latents with fused image conditioning.")
        if has_image_pos_emb:
            raise ValueError("FastWAM does not support extra image positional embeddings in WanVideoDiT.")
        if has_ref_conv:
            raise ValueError("FastWAM does not support reference convolutions in WanVideoDiT.")
        if add_control_adapter:
            raise ValueError("FastWAM does not support control adapters in WanVideoDiT.")
        if require_clip_embedding:
            raise ValueError("FastWAM does not support CLIP embedding conditioning in WanVideoDiT.")
        if require_vae_embedding or not fuse_vae_embedding_in_latents:
            raise ValueError("FastWAM expects VAE conditioning to be fused in latents.")
        if attn_head_dim != hidden_dim // num_heads:
            raise ValueError(
                "`attn_head_dim` must match the upstream Wan head dimension `hidden_dim // num_heads`; "
                f"got {attn_head_dim} vs {hidden_dim // num_heads}."
            )

        super().__init__(
            model_type="ti2v",
            patch_size=patch_size,
            text_len=512,
            in_dim=in_dim,
            dim=hidden_dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            qk_norm=True,
            cross_attn_norm=True,
            eps=eps,
        )
        self.blocks = torch.nn.ModuleList(
            [
                FastWAMAttentionBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                    fp32_attention=fp32_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.init_weights()

        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        self.fp32_attention = bool(fp32_attention)

        if self.action_conditioned:
            self.action_embedding = torch.nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode

        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info(
                "Using gradient checkpointing for DiT blocks. This will save memory but use more computation."
            )

    def patchify(self, x: torch.Tensor):
        return self.patch_embedding(x)

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        action: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            allow_text_only_single_frame = num_latent_frames == 1 and action is None
            if not allow_text_only_single_frame:
                if action is None:
                    raise ValueError("Action input is required for action-conditioned model.")
                if action.ndim != 3:
                    raise ValueError(
                        f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}"
                    )
                if action.shape[2] != self.action_dim:
                    raise ValueError(
                        f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}"
                    )
                if num_latent_frames <= 1:
                    raise ValueError(
                        f"video length must be > 1 for action-conditioned model, got {num_latent_frames}"
                    )
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        "action horizon must be divisible by (num_latent_frames - 1), "
                        f"got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones(
                (context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    "`context_mask` shape must match `context` shape [B, L], "
                    f"got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        fuse_vae_embedding_in_latents: bool = False,
    ) -> dict[str, Any]:
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )
        model_dtype = self.patch_embedding.weight.dtype
        x = x.to(dtype=model_dtype)
        context = context.to(dtype=model_dtype)
        if action is not None:
            action = action.to(dtype=model_dtype)

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if not (self.seperated_timestep and fuse_vae_embedding_in_latents):
            raise NotImplementedError(
                "FastWAM currently requires separated timesteps with fused VAE latents."
            )

        token_timesteps = torch.ones(
            (batch_size, x.shape[2], tokens_per_frame),
            dtype=model_dtype,
            device=timestep.device,
        ) * timestep.to(dtype=model_dtype).view(batch_size, 1, 1)
        token_timesteps[:, 0, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        # Wan keeps the time embedding in fp32: the AdaLN modulation in the vendored
        # Head/Block asserts e.dtype == float32 (numerical stability of the scale/shift).
        # Upstream guarantees this via an fp32 autocast region, so it holds even when the
        # model runs in bf16. Mirror that here, then cast the per-block modulation back to
        # model_dtype so the bf16 attention blocks are not upcast to fp32.
        with torch.amp.autocast("cuda", dtype=torch.float32):
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1)).float()
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        t_mod = t_mod.to(dtype=model_dtype)

        x = self.patchify(x)
        f, h, w = x.shape[2:]

        context = self.text_embedding(context)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action)
            action_pos_embed = sinusoidal_embedding_1d(
                self.hidden_dim, torch.arange(action_len, device=action_emb.device)
            ).to(dtype=action_emb.dtype)
            action_emb = action_emb + action_pos_embed.unsqueeze(0)
            context = torch.cat([context, action_emb], dim=1)

            num_temporal_groups = f - 1
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned context mask requires at least 2 latent frames when `action` is provided."
                )
            if action_emb.shape[1] % num_temporal_groups != 0:
                raise ValueError(
                    f"Action embedding length {action_emb.shape[1]} must be divisible by "
                    f"number of temporal groups {num_temporal_groups}"
                )
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device)

            seq_len = f * h * w
            final_context_mask = torch.zeros(
                (batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device
            )
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            context_mask = final_context_mask
        elif self.action_conditioned and action is None:
            if f != 1:
                raise ValueError(
                    "Action-conditioned model requires `action` unless running single-frame text-only mode "
                    "with num_latent_frames=1."
                )
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1)
        else:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        grid_sizes = torch.tensor([[f, h, w]] * batch_size, dtype=torch.long, device=x_tokens.device)
        freqs = {"grid_sizes": grid_sizes, "freqs": self.freqs.to(x_tokens.device)}

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_sizes": grid_sizes,
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        x = self.head(x_tokens, pre_state["t"])
        return torch.stack(super().unpatchify(x, pre_state["meta"]["grid_sizes"]))

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = (
            self.build_video_to_video_mask(
                video_seq_len=x_tokens.shape[1],
                video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
                device=x_tokens.device,
            )
            if self.video_attention_mask_mode != "bidirectional"
            else None
        )

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                    self_attn_mask=self_attn_mask,
                )
            else:
                x_tokens = block(
                    x_tokens,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                    self_attn_mask=self_attn_mask,
                )

        return self.post_dit(x_tokens, pre_state)
