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

"""The dual-stream Wan2.2 video-action transformer backbone for LingBot-VA.

Vendored and lightly adapted from the upstream LingBot-VA repository
(https://github.com/Robbyant/lingbot-va, ``wan_va/modules/model.py``).

The model keeps the diffusers ``ModelMixin``/``ConfigMixin`` mixins so the original
sharded ``transformer/`` checkpoint can be loaded with ``from_pretrained`` during
conversion, but in LeRobot it is owned as a plain ``nn.Module`` sub-component of
:class:`~lerobot.policies.lingbot_va.modeling_lingbot_va.LingBotVAPolicy`. State-dict
parameter names are preserved verbatim so conversion is near-identity.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange

from .wan_attention import WanAttention, WanRotaryPosEmbed

__all__ = ["WanTransformer3DModel", "WanTransformerBlock", "WanTimeTextImageEmbedding"]


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
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
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "torch",
    ):
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
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name="pos",
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

    # ------------------------------------------------------------------
    # KV-cache management for autoregressive streaming inference
    # ------------------------------------------------------------------
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
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2
        ) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(
                cache_name,
                total_tolen,
                self.num_attention_heads,
                self.attention_head_dim,
                device,
                dtype,
                batch_size,
            )

    # ------------------------------------------------------------------
    # Embedding helpers (shared by train + inference paths)
    # ------------------------------------------------------------------
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
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps, (H // pach_scale_h) * (W // pach_scale_w), dim=1
        )
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, timestep_proj = current_condition_embedder(latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    # ------------------------------------------------------------------
    # Dual-stream training forward (flow matching). Requires attn_mode='flex'.
    # ------------------------------------------------------------------
    def forward_train(self, input_dict):
        from .wan_flex_attention import FlexAttnFunc

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

    # ------------------------------------------------------------------
    # Single-stream inference forward (one denoising step for one stream)
    # ------------------------------------------------------------------
    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
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
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])

        latent_time_steps = torch.repeat_interleave(
            input_dict["timesteps"],
            (input_dict["noisy_latents"].shape[-2] // pach_scale_h)
            * (input_dict["noisy_latents"].shape[-1] // pach_scale_w),
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
