# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timestep_shape = timesteps.shape
        timesteps_proj = self.time_proj(timesteps.reshape(-1)).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb.reshape(*timestep_shape, -1)


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=-1)
        if scale.ndim == x.ndim - 1:
            scale = scale[:, None]
            shift = shift[:, None]
        x = self.norm(x) * (1 + scale) + shift
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: str | None = None,
        num_positional_embeddings: int | None = None,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_position_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.LongTensor | None = None,
        hidden_positional_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.positional_embeddings == "continuous_time":
            if hidden_positional_embeddings is None:
                raise ValueError(
                    "`hidden_positional_embeddings` is required when "
                    "`positional_embeddings='continuous_time'`."
                )
            if hidden_positional_embeddings.shape != norm_hidden_states.shape:
                raise ValueError(
                    "`hidden_positional_embeddings` must match `hidden_states` shape, got "
                    f"{tuple(hidden_positional_embeddings.shape)} vs {tuple(norm_hidden_states.shape)}."
                )
            norm_hidden_states = norm_hidden_states + hidden_positional_embeddings.to(
                device=norm_hidden_states.device,
                dtype=norm_hidden_states.dtype,
            )
        elif hidden_positional_embeddings is not None:
            if hidden_positional_embeddings.shape != norm_hidden_states.shape:
                raise ValueError(
                    "`hidden_positional_embeddings` must match `hidden_states` shape, got "
                    f"{tuple(hidden_positional_embeddings.shape)} vs {tuple(norm_hidden_states.shape)}."
                )
            norm_hidden_states = norm_hidden_states + hidden_positional_embeddings.to(
                device=norm_hidden_states.device,
                dtype=norm_hidden_states.dtype,
            )
        elif self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=(encoder_attention_mask if encoder_hidden_states is not None else attention_mask),
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.2,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: str | None = None,
        interleave_self_attention=False,
        cross_attention_dim: int | None = None,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder
        # Older checkpoints may not carry compute_dtype in their saved config.
        compute_dtype = getattr(self.config, "compute_dtype", torch.float32)
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim, compute_dtype=compute_dtype)

        all_blocks = []
        for idx in range(self.config.num_layers):
            # In interleaved mode, odd blocks are self-attention and even blocks cross-attend.
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            if use_self_attn and curr_cross_attention_dim is not None:
                raise ValueError(
                    f"Layer {idx}: interleave_self_attention=True but cross_attention_dim={curr_cross_attention_dim} "
                    f"(expected None for self-attention layers)"
                )

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)
        # print(
        #     "Total number of DiT parameters: ",
        #     sum(p.numel() for p in self.parameters() if p.requires_grad),
        # )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: torch.LongTensor | None = None,
        hidden_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
        hidden_positional_embeddings: torch.Tensor | None = None,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                # Pass None explicitly so diffusers selects the self-attention path.
                hidden_states = block(
                    hidden_states,
                    attention_mask=hidden_attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                    hidden_positional_embeddings=hidden_positional_embeddings,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                    hidden_positional_embeddings=hidden_positional_embeddings,
                )
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=-1)
        if scale.ndim == hidden_states.ndim - 1:
            scale = scale[:, None]
            shift = shift[:, None]
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class AlternateVLDiT(DiT):
    """DiT variant that alternates cross-attention between image and VLM tokens."""

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        # AlternateVLDiT requires interleaved self-attention blocks.
        kwargs["interleave_self_attention"] = True
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: torch.LongTensor | None = None,
        hidden_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
        image_mask: torch.Tensor | None = None,
        vlm_mask: torch.Tensor | None = None,  # [B, S], True=VLM tokens
        hidden_positional_embeddings: torch.Tensor | None = None,
    ):
        assert image_mask is not None and vlm_mask is not None, (
            "AlternateVLDiT requires image_mask and vlm_mask"
        )

        temb = self.timestep_encoder(timestep)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        # encoder_attention_mask is a padding mask over the full image+VLM token sequence.
        if encoder_attention_mask is not None:
            backbone_attention_mask = encoder_attention_mask.bool()
        else:
            backbone_attention_mask = torch.ones(
                hidden_states.shape[0],
                encoder_hidden_states.shape[1],
                dtype=torch.bool,
                device=hidden_states.device,
            )

        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = vlm_mask & backbone_attention_mask

        # print(f"[AlternateVLDiT] image_mask: {image_mask.sum(dim=1).tolist()}, vlm_mask: {vlm_mask.sum(dim=1).tolist()}")
        # print(f"[AlternateVLDiT] backbone_mask: {backbone_attention_mask.sum(dim=1).tolist()}")
        # print(f"[AlternateVLDiT] image_attn_mask: {image_attention_mask.sum(dim=1).tolist()}, non_image: {non_image_attention_mask.sum(dim=1).tolist()}")

        all_hidden_states = [hidden_states]

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=hidden_attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                    hidden_positional_embeddings=hidden_positional_embeddings,
                )
            else:
                # Start with VLM/text tokens, then alternate with image tokens.
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    curr_encoder_attention_mask = non_image_attention_mask
                else:
                    curr_encoder_attention_mask = image_attention_mask

                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_encoder_attention_mask,
                    temb=temb,
                    hidden_positional_embeddings=hidden_positional_embeddings,
                )
            all_hidden_states.append(hidden_states)

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=-1)
        if scale.ndim == hidden_states.ndim - 1:
            scale = scale[:, None]
            shift = shift[:, None]
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift

        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)
