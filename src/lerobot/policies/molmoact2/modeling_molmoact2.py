# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""MolmoAct2: a Molmo2-4B vision-language model with a flow-matching action expert.

The VLM is transformers' `Molmo2ForConditionalGeneration`. Its decoder layers run in a loop owned here, so that the
action expert (one block per LLM layer) can cross-attend to each layer's keys and values.

Paper: https://allenai.org/blog/molmoact2
Code: https://github.com/allenai/molmoact2
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import _transformers_available, require_package

from ..checkpoint import SafetensorsCheckpoint
from ..common.flow_matching import (
    FlowConvention,
    euler_integrate,
    make_flow_matching_inputs,
    sample_noise,
    sample_time_beta,
)
from ..pretrained import PreTrainedPolicy, RTCActionSelectKwargs, T
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import log_model_loading_keys
from .configuration_molmoact2 import MolmoAct2Config
from .processor_molmoact2 import (
    EOS_TOKEN_ID,
    UniversalActionProcessor,
    decode_discrete_action_chunk,
    select_constrained_discrete_action_token,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import Molmo2Config, Molmo2ForConditionalGeneration
    from transformers.initialization import no_init_weights
    from transformers.models.molmo2.modeling_molmo2 import Molmo2RMSNorm, apply_rotary_pos_emb, repeat_kv
else:
    Molmo2Config = Molmo2ForConditionalGeneration = Molmo2RMSNorm = None
    no_init_weights = apply_rotary_pos_emb = repeat_kv = None

logger = logging.getLogger(__name__)

# Rows of the token embedding beyond this index embed the 128 extra image tokens (upstream `new_embedding`).
_BASE_VOCAB_SIZE = 154624
# Weight of the z-loss, the squared log-normalizer of the action-token logits (upstream
# `softmax_auxiliary_loss_scale`).
_SOFTMAX_AUXILIARY_LOSS_SCALE = 1e-4
# Learning rate of the LLM, vision and connector LoRA groups in upstream's LoRA recipe (experiments/README.md).
_LORA_LEARNING_RATE = 5e-5


# Molmo2 decoder: our forward loop over transformers' Molmo2 modules


def _get_molmo2_config() -> Molmo2Config:
    """Return the Molmo2-4B configuration shared by every released MolmoAct2 checkpoint."""
    return Molmo2Config(
        text_config={
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "vocab_size": _BASE_VOCAB_SIZE,
            "max_position_embeddings": 16384,
            "rope_parameters": {"rope_type": "default", "rope_theta": 5_000_000.0},
        },
        adapter_config={"intermediate_size": 9728, "text_hidden_size": 2560},
        image_start_token_id=154624,
        image_end_token_id=154625,
        image_patch_id=154626,
        low_res_image_start_token_id=154628,
        frame_start_token_id=154631,
        frame_end_token_id=154632,
    )


def _rms_norm(x: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    """RMSNorm in float32, scaled by `weight` before casting back.

    Upstream and transformers cast back first and then scale; scaling first saves one bf16 rounding.
    """
    dtype = x.dtype
    with torch.autocast(x.device.type, enabled=False):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            x = weight.float() * x
        return x.to(dtype)


def _rotary_embedding(
    position_ids: Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """RoPE cos/sin tables for explicit positions, computed in float32."""
    with torch.autocast(position_ids.device.type, enabled=False):
        exponents = torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.int64).float()
        inv_freq = 1.0 / (theta ** (exponents / head_dim))
        freqs = position_ids[..., None].float() * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _attention_bias(attention_mask: Tensor, image_token_mask: Tensor | None, dtype: torch.dtype) -> Tensor:
    """Additive `[B, 1, S, S]` bias: causal, padded keys masked, image tokens bidirectional when given."""
    valid = attention_mask.bool()
    seq_len = valid.shape[1]
    allowed = torch.ones(seq_len, seq_len, device=valid.device, dtype=torch.bool).tril()[None]
    if image_token_mask is not None:
        image = image_token_mask.bool()
        allowed = allowed | (image[:, :, None] & image[:, None, :])
    allowed = allowed & valid[:, None, :]
    return torch.where(allowed[:, None], 0.0, torch.finfo(dtype).min).to(dtype)


def _decoder_layer(
    layer: nn.Module,
    hidden: Tensor,
    cos: Tensor,
    sin: Tensor,
    bias: Tensor,
    dropout: float,
    past_key_value: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run a pre-norm `Molmo2DecoderLayer` with residual dropout.

    Returns the hidden states and the post-RoPE keys and values of every position so far (with
    `past_key_value`), each `[B, kv_heads, S, head_dim]`.
    """
    attn = layer.self_attn
    x = _rms_norm(hidden, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
    head_shape = (*x.shape[:-1], -1, attn.head_dim)
    query = _rms_norm(attn.q_proj(x).view(head_shape), attn.q_norm.weight, attn.q_norm.variance_epsilon)
    key = _rms_norm(attn.k_proj(x).view(head_shape), attn.k_norm.weight, attn.k_norm.variance_epsilon)
    value = attn.v_proj(x).view(head_shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query.transpose(1, 2), key.transpose(1, 2), cos, sin)
    if past_key_value is not None:
        key = torch.cat((past_key_value[0], key), dim=2)
        value = torch.cat((past_key_value[1], value), dim=2)
    out = F.scaled_dot_product_attention(
        query,
        repeat_kv(key, attn.num_key_value_groups),
        repeat_kv(value, attn.num_key_value_groups),
        attn_mask=bias,
    )
    out = attn.o_proj(out.transpose(1, 2).reshape(*x.shape[:-1], -1))
    hidden = hidden + F.dropout(out, dropout, layer.training)
    norm = layer.post_attention_layernorm
    hidden = hidden + F.dropout(
        layer.mlp(_rms_norm(hidden, norm.weight, norm.variance_epsilon)), dropout, layer.training
    )
    return hidden, key, value


def _to_sequence(states: Tensor) -> Tensor:
    """`[B, heads, S, head_dim]` -> `[B, S, heads * head_dim]`."""
    batch_size, _, seq_len, _ = states.shape
    return states.transpose(1, 2).reshape(batch_size, seq_len, -1)


# Action expert (ports experiments/olmo/nn/action_expert.py of allenai/molmoact2)


def _round_up_multiple(value: int, multiple_of: int) -> int:
    if multiple_of <= 0:
        return value
    return int(math.ceil(value / multiple_of) * multiple_of)


def _init_linear(linear: nn.Linear, *, zero: bool = False, scale: float = 1.0) -> None:
    if zero:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.xavier_uniform_(linear.weight)
        if scale != 1.0:
            with torch.no_grad():
                linear.weight.mul_(scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """adaLN modulation of `[B, T, D]` hidden states by per-sample `[B, D]` shift and scale."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _apply_rope(x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = rope
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _build_cross_attention_mask(encoder_attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Additive `[B, 1, 1, S]` mask over the LLM tokens from a `[B, S]` boolean mask."""
    return (1.0 - encoder_attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min


class SinusoidalTimeEmbedding(nn.Module):
    """Embed `[B]` timesteps as `[sin, cos]` features with geometrically spaced frequencies."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: Tensor) -> Tensor:
        half_dim = self.dim // 2
        exponents = torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
        args = timesteps[:, None] * torch.exp(exponents * (-math.log(10000.0) / (half_dim - 1)))[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ActionExpertSelfAttention(nn.Module):
    """Self-attention over the action chunk, with per-head RMSNorm on queries and keys, and RoPE."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        batch_size, seq_len, hidden_size = x.shape
        q, k, v = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).unbind(2)
        q = _apply_rope(_rms_norm(q.transpose(1, 2)), rope)
        k = _apply_rope(_rms_norm(k.transpose(1, 2)), rope)
        out = F.scaled_dot_product_attention(q, k, v.transpose(1, 2))
        return self.out_proj(out.transpose(1, 2).reshape(batch_size, seq_len, hidden_size))


class ActionExpertCrossAttention(nn.Module):
    """Cross-attention from the action chunk to the projected keys and values of one LLM layer."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor) -> Tensor:
        batch_size, seq_len, hidden_size = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Per-head q_norm and k_norm (upstream `_apply_qk_norm`); the released norms have no weights.
        q, key = _rms_norm(q), _rms_norm(key)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attn_mask
        )
        return self.out_proj(out.transpose(1, 2).reshape(batch_size, seq_len, hidden_size))


class ActionExpertMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, mlp_ratio: float, multiple_of: int) -> None:
        super().__init__()
        inner_dim = _round_up_multiple(int(hidden_size * mlp_ratio), multiple_of)
        self.up_proj = nn.Linear(hidden_size, inner_dim)
        self.gate_proj = nn.Linear(hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ActionExpertModulation(nn.Module):
    """Project the timestep conditioning to `num_chunks` adaLN vectors (shifts, scales and gates)."""

    def __init__(self, hidden_size: int, num_chunks: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_chunks * hidden_size)

    def forward(self, conditioning: Tensor) -> Tensor:
        return self.linear(self.act(conditioning))


class ActionExpertBlock(nn.Module):
    """DiT block: self-attention, cross-attention and MLP, each pre-normed and modulated by the timestep."""

    def __init__(self, hidden_size: int, num_heads: int, *, mlp_ratio: float, ffn_multiple_of: int) -> None:
        super().__init__()
        self.self_attn = ActionExpertSelfAttention(hidden_size, num_heads)
        self.cross_attn = ActionExpertCrossAttention(hidden_size, num_heads)
        self.mlp = ActionExpertMLP(hidden_size, mlp_ratio, ffn_multiple_of)
        # A shift, a scale and a gate for each of the three sublayers.
        self.modulation = ActionExpertModulation(hidden_size, 9)

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        key: Tensor,
        value: Tensor,
        cross_mask: Tensor,
        rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(conditioning).chunk(9, dim=1)
        )
        h = _modulate(_rms_norm(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(h, rope)
        h = _modulate(_rms_norm(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(h, key, value, cross_mask)
        h = _modulate(_rms_norm(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.mlp(h)


class ActionExpertFinalLayer(nn.Module):
    """Modulated RMSNorm followed by the projection to action velocities."""

    def __init__(self, hidden_size: int, action_dim: int) -> None:
        super().__init__()
        self.modulation = ActionExpertModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, action_dim)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=1)
        return self.linear(_modulate(_rms_norm(x), shift, scale))


class ActionExpert(nn.Module):
    """Flow-matching DiT over the action chunk, with one block per LLM layer.

    Block `i` cross-attends to the keys and values of LLM layer `i`, projected by `prepare_kv_context`. Training
    embeds the inputs, then runs `prepare_kv_context` and `run_block` after each LLM layer, then `final_layer`;
    sampling calls `forward`.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Width of the action expert.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_layers (`int`, *optional*, defaults to 36):
            Number of blocks, one per LLM layer.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Inner width of the MLPs relative to `hidden_size`.
        ffn_multiple_of (`int`, *optional*, defaults to 256):
            The inner width of the MLPs is rounded up to a multiple of this.
        timestep_embed_dim (`int`, *optional*, defaults to 256):
            Size of the sinusoidal timestep features.
        max_action_dim (`int`, *optional*, defaults to 32):
            Size of the padded action vectors.
        llm_kv_dim (`int`, *optional*, defaults to 1024):
            Size of the LLM keys and values with heads flattened.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 36,
        mlp_ratio: float = 4.0,
        ffn_multiple_of: int = 256,
        timestep_embed_dim: int = 256,
        max_action_dim: int = 32,
        llm_kv_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.action_embed = nn.Linear(max_action_dim, hidden_size)
        self.context_k_proj = nn.Linear(llm_kv_dim, hidden_size, bias=False)
        self.context_v_proj = nn.Linear(llm_kv_dim, hidden_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio, ffn_multiple_of=ffn_multiple_of
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = ActionExpertFinalLayer(hidden_size, max_action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.time_embed.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)
        _init_linear(self.action_embed)
        self.context_k_proj.reset_parameters()
        self.context_v_proj.reset_parameters()

        residual_scale = (2 * max(len(self.blocks), 1)) ** -0.5
        for block in self.blocks:
            _init_linear(block.self_attn.qkv)
            _init_linear(block.self_attn.out_proj, scale=residual_scale)
            _init_linear(block.cross_attn.q_proj)
            _init_linear(block.cross_attn.out_proj, scale=residual_scale)
            _init_linear(block.mlp.up_proj)
            _init_linear(block.mlp.gate_proj)
            _init_linear(block.mlp.down_proj, scale=residual_scale)
            _init_linear(block.modulation.linear, zero=True)

        _init_linear(self.final_layer.modulation.linear, zero=True)
        _init_linear(self.final_layer.linear, zero=True)

    def timestep_embedding(self, timesteps: Tensor) -> Tensor:
        return self.time_embed(timesteps)

    def _project_kv_tensor(self, x: Tensor, proj: nn.Linear) -> Tensor:
        """Project keys or values `[B, S, llm_kv_dim]` to `[B, S, num_heads, head_dim]` (weightless context norm)."""
        return _rms_norm(proj(x)).view(*x.shape[:-1], self.num_heads, self.head_dim)

    def prepare_kv_context(
        self, layer_kv_states: Sequence[tuple[Tensor, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        """Project the keys and values `[B, S, llm_kv_dim]` of each LLM layer for the cross-attention."""
        return [
            (
                self._project_kv_tensor(k_in, self.context_k_proj),
                self._project_kv_tensor(v_in, self.context_v_proj),
            )
            for k_in, v_in in layer_kv_states
        ]

    def build_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """RoPE tables of the action self-attention, computed in float32."""
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim))
        freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq)
        return freqs.cos().to(dtype)[None, None], freqs.sin().to(dtype)[None, None]

    def run_block(
        self,
        index: int,
        x: Tensor,
        conditioning: Tensor,
        key: Tensor,
        value: Tensor,
        cross_mask: Tensor,
        rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Run block `index` with the projected keys and values of LLM layer `index`."""
        return self.blocks[index](x, conditioning, key, value, cross_mask, rope)

    def forward(
        self,
        xt: Tensor,
        timesteps: Tensor,
        kv_contexts: Sequence[tuple[Tensor, Tensor]],
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        """Predict the velocity `[B, T, max_action_dim]`.

        Args:
            xt (`Tensor`):
                Noisy actions `[B, T, max_action_dim]`.
            timesteps (`Tensor`):
                Flow-matching times `[B]`.
            kv_contexts (`Sequence[tuple[Tensor, Tensor]]`):
                Keys and values of every LLM layer, projected by `prepare_kv_context`.
            encoder_attention_mask (`Tensor`):
                `[B, S]` mask of the LLM tokens to attend to.

        Returns:
            `Tensor`: The velocity.
        """
        x = self.action_embed(xt)
        conditioning = self.timestep_embedding(timesteps)
        cross_mask = _build_cross_attention_mask(encoder_attention_mask, x.dtype)
        rope = self.build_rope(x.shape[1], x.device, x.dtype)
        for index, (key, value) in enumerate(kv_contexts):
            x = self.run_block(index, x, conditioning, key, value, cross_mask, rope)
        return self.final_layer(x, conditioning)


# Model


def _weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    """Weighted mean whose denominator is averaged across DDP ranks, as upstream does."""
    if weights is None:
        numerator, denominator = values.sum(), values.new_tensor(float(values.numel()))
    else:
        weights = weights.to(values.dtype)
        numerator, denominator = torch.dot(values, weights), weights.sum()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        denominator = denominator.detach().clone()
        torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM)
        denominator = denominator / torch.distributed.get_world_size()
    return numerator / denominator.clamp_min(1.0)


def _weighted_per_example(
    values: Tensor, weights: Tensor, example_indices: Tensor, batch_size: int
) -> Tensor:
    """Per-example weighted sums, scaled so that their mean equals the batch's weighted mean."""
    values = values.float()
    loss_sum = torch.zeros(batch_size, device=values.device).scatter_add_(
        0, example_indices, values * weights
    )
    weight_sum = torch.zeros(batch_size, device=values.device).scatter_add_(0, example_indices, weights)
    return loss_sum * float(batch_size) / weight_sum.sum().clamp_min(1.0)


def _zero_base_vocabulary_grad(grad: Tensor) -> Tensor:
    grad = grad.clone()
    grad[:_BASE_VOCAB_SIZE] = 0
    return grad


class MolmoAct2Model(nn.Module):
    """The Molmo2 VLM and the action expert, with MolmoAct2's training losses and action sampling.

    Args:
        config (`MolmoAct2Config`):
            The policy config. With `pretrained_path` set, the VLM is built without initializing the weights that
            the checkpoint replaces.
        rtc_processor (`RTCProcessor`, *optional*):
            Real-time chunking of `sample_actions`.
    """

    def __init__(self, config: MolmoAct2Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor
        with no_init_weights() if config.pretrained_path else nullcontext():
            self.vlm = Molmo2ForConditionalGeneration(_get_molmo2_config())
        text_config = self.vlm.config.text_config
        self.action_expert = ActionExpert(
            num_layers=text_config.num_hidden_layers,
            llm_kv_dim=text_config.num_key_value_heads * text_config.head_dim,
            max_action_dim=config.max_action_dim,
        )
        self.gradient_checkpointing_enabled = False
        # bf16 on Apple MPS produces wrong actions (#5375).
        self.llm_dtype = (
            torch.float32 if str(config.device).startswith("mps") else cast(torch.dtype, config.dtype)
        )
        self.to_bfloat16_for_selected_params(self.llm_dtype)
        self._set_requires_grad()
        if config.compile_model:
            self.sample_actions = torch.compile(self.sample_actions)

    @property
    def layers(self) -> nn.ModuleList:
        return self.vlm.model.language_model.layers

    def to_bfloat16_for_selected_params(self, precision: torch.dtype) -> None:
        """Store the LLM in `precision`; keep the vision tower, connector, norms and action expert in float32."""
        self.vlm.to(precision)
        self.vlm.model.vision_tower.float()
        self.vlm.model.multi_modal_projector.float()
        for module in self.vlm.modules():
            if isinstance(module, (nn.LayerNorm, Molmo2RMSNorm)):
                module.float()
        self.action_expert.float()

    def _set_requires_grad(self) -> None:
        if self.config.train_expert_only:
            self.vlm.requires_grad_(False)
        elif self.config.freeze_vision_encoder:
            self.vlm.model.vision_tower.requires_grad_(False)
        embedding = self.vlm.model.language_model.embed_tokens.weight
        if self.config.freeze_embedding and embedding.requires_grad:
            # Freeze the base vocabulary; only the extra image-token rows train (upstream ft_embedding="lm_head").
            embedding.register_hook(_zero_base_vocabulary_grad)
        # Frozen modules stay in train mode, as upstream only sets requires_grad: the frozen LLM keeps its
        # residual dropout.

    def gradient_checkpointing_enable(self) -> None:
        """Checkpoint every LLM layer together with its action expert block, and the vision tower layers."""
        self.gradient_checkpointing_enabled = True
        self.vlm.model.vision_tower.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.vlm.model.vision_tower.gradient_checkpointing_disable()

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        return sample_noise(shape, device)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """Flow-matching times in [0, 1], where 0 is noise: `offset + scale * Beta(alpha, beta)`."""
        config = self.config
        return sample_time_beta(
            batch_size,
            device,
            alpha=config.flow_matching_beta_alpha,
            beta=config.flow_matching_beta_beta,
            scale=config.flow_matching_time_scale,
            offset=config.flow_matching_time_offset,
        )

    def embed_inputs(self, batch: dict[str, Tensor]) -> Tensor:
        """Embed the prompt tokens and add the pooled image features to the image patch tokens."""
        model = self.vlm.model
        input_ids = batch["input_ids"]
        embeds = model.language_model.embed_tokens(input_ids.clamp(min=0))
        if torch.is_autocast_enabled(embeds.device.type):
            embeds = embeds.to(torch.get_autocast_dtype(embeds.device.type))
        pixel_values = batch.get("pixel_values")
        if pixel_values is None:
            return embeds
        # The vision tower and connector run in float32: under bf16 they drift far more than the LLM does
        # (the released code computed vision attention in float32).
        with torch.autocast(pixel_values.device.type, enabled=False):
            features = model.get_image_features(
                pixel_values.float(), batch["image_token_pooling"], batch["image_grids"]
            ).pooler_output
        features = torch.cat(features).to(embeds.dtype)
        is_patch = input_ids == self.vlm.config.image_patch_id
        if int(is_patch.sum()) != features.shape[0]:
            raise ValueError(
                f"Got {int(is_patch.sum())} image patch tokens but {features.shape[0]} image features."
            )
        embeds = embeds.clone()
        embeds[is_patch] = embeds[is_patch] + features
        return embeds

    def _rope(self, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        text_config = self.vlm.config.text_config
        theta = text_config.rope_parameters["rope_theta"]
        return _rotary_embedding(position_ids, text_config.head_dim, theta, dtype)

    def _final_norm(self, hidden: Tensor) -> Tensor:
        norm = self.vlm.model.language_model.norm
        return _rms_norm(hidden, norm.weight, norm.variance_epsilon)

    def _residual_dropout(self) -> float:
        return self.config.residual_dropout if self.training else 0.0

    def prefill(self, batch: dict[str, Tensor]) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Run the prompt as at inference (image tokens bidirectional); return the final hidden states and K/V."""
        hidden = self.embed_inputs(batch)
        cos, sin = self._rope(batch["position_ids"], hidden.dtype)
        bias = _attention_bias(batch["attention_mask"], batch["mm_token_type_ids"], hidden.dtype)
        layer_kv_states = []
        for layer in self.layers:
            hidden, key, value = _decoder_layer(layer, hidden, cos, sin, bias, dropout=0.0)
            layer_kv_states.append((key, value))
        return self._final_norm(hidden), layer_kv_states

    def _cross_entropy_loss(self, hidden: Tensor, labels: Tensor, reduction: str) -> tuple[Tensor, Tensor]:
        """Token cross-entropy and z-loss on the action tokens.

        Each example is weighted by 2 / sqrt(its labeled tokens), upstream's "root_subsegments_root_tokens" loss
        weighting, and the denominator is the summed weight averaged across ranks.
        """
        shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
        valid = shift_labels != -100
        if not bool(valid.any()):
            raise ValueError("MolmoAct2 discrete training got labels with no action tokens.")
        logits = self.vlm.lm_head(hidden[valid]).float()
        log_z = logits.logsumexp(dim=-1)
        token_loss = log_z - logits.gather(-1, shift_labels[valid][:, None]).squeeze(-1)
        token_counts = valid.sum(dim=1).float()
        example_weights = torch.where(token_counts > 0, 2.0 / token_counts.clamp_min(1.0).sqrt(), 0.0)
        weights = example_weights[:, None].expand_as(valid)[valid]
        if reduction == "none":
            example_indices = valid.nonzero()[:, 0]
            ce = _weighted_per_example(token_loss, weights, example_indices, labels.shape[0])
            z = _weighted_per_example(log_z.pow(2), weights, example_indices, labels.shape[0])
        else:
            ce, z = _weighted_mean(token_loss, weights), _weighted_mean(log_z.pow(2), weights)
        return ce, _SOFTMAX_AUXILIARY_LOSS_SCALE * z

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the training losses of `action_mode`: `"flow_loss"`, and `"ce_loss"` with `"z_loss"`.

        Each LLM layer is followed by the action expert block that reads its keys and values, and gradient
        checkpointing recomputes the pair together.

        Args:
            batch (`dict[str, Tensor]`):
                Output of the preprocessor.
            reduction (`str`, *optional*, defaults to `"mean"`):
                `"mean"` over the batch, or `"none"` for per-example losses.
            noise (`Tensor`, *optional*):
                Flow-matching noise `[B, num_flow_timesteps, chunk_size, max_action_dim]`, sampled when `None`.
            time (`Tensor`, *optional*):
                Flow-matching times `[B, num_flow_timesteps]`, sampled when `None`.

        Returns:
            `dict[str, Tensor]`: The losses, scalars or `[B]` per example.
        """
        config = self.config
        hidden = self.embed_inputs(batch)
        cos, sin = self._rope(batch["position_ids"], hidden.dtype)
        # The official fine-tuning config sets `bi_directional_attn=None` (lerobot_utils/hf.py), so image tokens
        # attend causally during training. The processor's `mm_token_type_ids` make them bidirectional only at
        # inference (`prefill`), as the released model does.
        bias = _attention_bias(batch["attention_mask"], None, hidden.dtype)
        dropout = self._residual_dropout()

        action_hidden = None
        if config.action_mode != "discrete":
            # LeRobot clamps future actions at episode ends and marks them as padding (`action_is_pad`). The
            # original MolmoAct2 training objective keeps supervising those clamped actions for fixed-horizon
            # fine-tuning, so the horizon is intentionally not masked.
            batch_size, horizon, action_dim = batch[ACTION].shape
            num_flow_timesteps = config.num_flow_timesteps
            dim_mask = (~batch["action_dim_is_pad"].bool()).float()[:, None, None, :]
            if time is None:
                time = self.sample_time(batch_size * num_flow_timesteps, hidden.device)
                time = time.view(batch_size, num_flow_timesteps)
            if noise is None:
                noise = self.sample_noise(
                    (batch_size, num_flow_timesteps, horizon, action_dim), hidden.device
                )
            # Match the original MolmoAct2 fp32+AMP training path: normalized actions, sampled flow noise,
            # timesteps and the velocity target are kept in fp32. Every flow timestep of an example is one row of
            # the action expert's batch.
            timesteps = time.float().flatten()
            actions = (batch[ACTION].float()[:, None] * dim_mask).expand_as(noise).flatten(0, 1)
            xt, target_velocity, _ = make_flow_matching_inputs(
                actions,
                (noise.float() * dim_mask).flatten(0, 1),
                timesteps,
                convention=FlowConvention.NOISE_AT_ZERO,
            )
            action_hidden = self.action_expert.action_embed(xt)
            conditioning = self.action_expert.timestep_embedding(timesteps)
            # Masks and RoPE use the action activation dtype (bf16 under autocast), as in the original AMP path.
            cross_mask = _build_cross_attention_mask(
                batch["encoder_attention_mask"].repeat_interleave(num_flow_timesteps, dim=0),
                action_hidden.dtype,
            )
            rope = self.action_expert.build_rope(horizon, hidden.device, action_hidden.dtype)

        def layer_step(
            idx: int, hidden: Tensor, action_hidden: Tensor | None
        ) -> tuple[Tensor, Tensor | None]:
            hidden, key, value = _decoder_layer(self.layers[idx], hidden, cos, sin, bias, dropout)
            if action_hidden is not None:
                ((key, value),) = self.action_expert.prepare_kv_context(
                    [(_to_sequence(key), _to_sequence(value))]
                )
                action_hidden = self.action_expert.run_block(
                    idx,
                    action_hidden,
                    conditioning,
                    key.repeat_interleave(num_flow_timesteps, dim=0),
                    value.repeat_interleave(num_flow_timesteps, dim=0),
                    cross_mask,
                    rope,
                )
            return hidden, action_hidden

        for idx in range(len(self.layers)):
            if self.gradient_checkpointing_enabled and torch.is_grad_enabled():
                hidden, action_hidden = checkpoint(
                    partial(layer_step, idx), hidden, action_hidden, use_reentrant=False
                )
            else:
                hidden, action_hidden = layer_step(idx, hidden, action_hidden)

        losses: dict[str, Tensor] = {}
        if action_hidden is not None:
            prediction = self.action_expert.final_layer(action_hidden, conditioning).float()
            squared_error = (prediction - target_velocity).pow(2)
            squared_error = squared_error.view(batch_size, num_flow_timesteps, horizon, action_dim)
            # Mean over the real action dims, then over the chunk and the flow samples.
            per_step = (squared_error * dim_mask).sum(-1) / dim_mask.sum(-1).clamp_min(1.0)
            flow_loss = per_step.flatten(1).mean(1)
            losses["flow_loss"] = flow_loss if reduction == "none" else flow_loss.mean()
        if config.action_mode != "continuous":
            losses["ce_loss"], losses["z_loss"] = self._cross_entropy_loss(
                self._final_norm(hidden), batch["labels"], reduction
            )
        return losses

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        num_steps: int | None = None,
        **rtc_kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        """Sample a normalized action chunk `[B, chunk_size, max_action_dim]` with Euler steps from noise (t=0).

        Padded action dims stay zero.
        """
        config = self.config
        _, layer_kv_states = self.prefill(batch)
        if noise is None:
            shape = (batch["input_ids"].shape[0], config.chunk_size, config.max_action_dim)
            noise = self.sample_noise(shape, batch["input_ids"].device)
        dim_mask = (~batch["action_dim_is_pad"].bool()).float()[:, None, :]
        # The projected keys and values do not depend on the denoising step.
        kv_contexts = self.action_expert.prepare_kv_context(
            [(_to_sequence(key), _to_sequence(value)) for key, value in layer_kv_states]
        )
        encoder_attention_mask = batch["encoder_attention_mask"]

        def velocity(xt: Tensor, timesteps: Tensor) -> Tensor:
            return self.action_expert(xt, timesteps, kv_contexts, encoder_attention_mask).float() * dim_mask

        return euler_integrate(
            velocity,
            noise.float() * dim_mask,
            num_steps or config.num_inference_steps,
            convention=FlowConvention.NOISE_AT_ZERO,
            velocity_scale=dim_mask,
            rtc_processor=self.rtc_processor,
            rtc_enabled=config.rtc_config is not None and config.rtc_config.enabled,
            inference_delay=rtc_kwargs.get("inference_delay"),
            prev_chunk_left_over=rtc_kwargs.get("prev_chunk_left_over"),
            execution_horizon=rtc_kwargs.get("execution_horizon"),
        )

    @torch.no_grad()
    def generate_tokens(
        self,
        batch: dict[str, Tensor],
        max_new_tokens: int,
        select_token: Callable[[Tensor, list[list[int]]], Tensor],
        eos_token_id: int,
    ) -> list[list[int]]:
        """Generate tokens after the prompt with a KV cache until every row emits `eos_token_id`.

        `select_token` picks the next token `[B]` from the last logits `[B, vocab]` and the tokens generated so far.
        Returns the generated token ids of each row.
        """
        hidden, layer_kv_states = self.prefill(batch)
        attention_mask = batch["attention_mask"].bool()
        positions = batch["position_ids"][:, -1:]
        generated: list[list[int]] = [[] for _ in range(hidden.shape[0])]
        finished = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        logits = self.vlm.lm_head(hidden[:, -1]).float()
        for _ in range(max_new_tokens):
            next_token = select_token(logits, generated)
            for row, token in enumerate(next_token.tolist()):
                if not finished[row]:
                    generated[row].append(token)
            finished |= next_token == eos_token_id
            if bool(finished.all()):
                break
            positions = positions + 1
            attention_mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1)
            # Finished rows keep decoding on a pad token; their outputs are ignored.
            step_ids = next_token.masked_fill(finished, 0)[:, None]
            step_hidden = self.vlm.model.language_model.embed_tokens(step_ids)
            if torch.is_autocast_enabled(step_hidden.device.type):
                step_hidden = step_hidden.to(torch.get_autocast_dtype(step_hidden.device.type))
            cos, sin = self._rope(positions, step_hidden.dtype)
            bias = torch.where(attention_mask[:, None, None, :], 0.0, torch.finfo(step_hidden.dtype).min)
            bias = bias.to(step_hidden.dtype)
            for idx, layer in enumerate(self.layers):
                step_hidden, key, value = _decoder_layer(
                    layer, step_hidden, cos, sin, bias, dropout=0.0, past_key_value=layer_kv_states[idx]
                )
                layer_kv_states[idx] = (key, value)
            logits = self.vlm.lm_head(self._final_norm(step_hidden)[:, -1]).float()
        return generated


# Policy


class MolmoAct2Policy(PreTrainedPolicy):
    """MolmoAct2 policy: Molmo2-4B VLM with a flow-matching action expert."""

    config_class = MolmoAct2Config
    config: MolmoAct2Config
    name = "molmoact2"

    def __init__(self, config: MolmoAct2Config, **kwargs: Any):
        """
        Args:
            config (`MolmoAct2Config`):
                Policy config. The weights come from a LeRobot checkpoint (`from_pretrained`); a policy built
                without `pretrained_path` is randomly initialized.
        """
        require_package("transformers", extra="molmoact2")
        super().__init__(config)
        config.validate_features()
        self.config = config
        if config.pretrained_path is None:
            logger.warning(
                "MolmoAct2 is randomly initialized. Fine-tune it from a LeRobot checkpoint instead, e.g. "
                "--policy.path=lerobot/MolmoAct2-LeRobot."
            )
        self.init_rtc_processor()
        # Loaded on the first discrete prediction.
        self._fast_tokenizer: UniversalActionProcessor | None = None
        self.model = MolmoAct2Model(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: type[T], pretrained_name_or_path: str | Path, *, config=None, **kwargs: Any
    ) -> T:
        """Load a LeRobot MolmoAct2 checkpoint, reading its weights once.

        Loading is strict by default: the VLM is built without initializing its weights, so a key missing from
        the checkpoint must fail instead of leaving uninitialized values.
        """
        if config is not None and config.pretrained_path is None:
            config.pretrained_path = Path(pretrained_name_or_path)
        kwargs.setdefault("strict", True)
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    @classmethod
    def _load_checkpoint(
        cls, model: T, checkpoint: SafetensorsCheckpoint, map_location: str, strict: bool
    ) -> T:
        """Copy the checkpoint into the model one tensor at a time.

        Peak memory is the model plus its largest tensor, not the model plus the whole checkpoint.
        """
        state = model.state_dict()
        loaded: set[str] = set()
        unexpected: list[str] = []
        with torch.no_grad():
            for key, tensor in checkpoint.tensors():
                if key in state:
                    state[key].copy_(tensor)
                    loaded.add(key)
                else:
                    unexpected.append(key)
        missing = sorted(set(state) - loaded)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error(s) in loading state_dict for {type(model).__name__}: "
                f"missing keys {missing}, unexpected keys {unexpected}"
            )
        log_model_loading_keys(missing, unexpected)
        return model

    def init_rtc_processor(self) -> None:
        self.rtc_processor = (
            RTCProcessor(self.config.rtc_config) if self.config.rtc_config is not None else None
        )
        if hasattr(self, "model"):
            self.model.rtc_processor = self.rtc_processor

    def supports_rtc(self) -> bool:
        return self.config.inference_action_mode == "continuous"

    def _action_dim(self) -> int:
        """Number of real action dims; predictions are cropped to it."""
        return self.config.output_features[ACTION].shape[0]

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> list[dict[str, Any]]:
        """Return the upstream parameter groups: LLM, ViT, connector (with the embedding) and action expert.

        With LoRA, the LLM, ViT and connector groups train at 5e-5, as in upstream's LoRA recipe; the action expert
        is always fully trained with `optimizer_action_expert_lr`.
        """
        lora = self.config.use_peft
        lrs = {
            "llm": _LORA_LEARNING_RATE if lora else self.config.optimizer_lr,
            "vit": _LORA_LEARNING_RATE if lora else self.config.optimizer_vit_lr,
            "connector": _LORA_LEARNING_RATE if lora else self.config.optimizer_connector_lr,
            "action_expert": self.config.optimizer_action_expert_lr,
        }
        groups: dict[str, list[Tensor]] = {name: [] for name in lrs}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "action_expert" in name:
                groups["action_expert"].append(param)
            elif "vision_tower" in name:
                groups["vit"].append(param)
            # Native MolmoAct2 owns the checkpoint's additional vocabulary table (`new_embedding`) under
            # get_connector_parameters(). It stays trainable in the published ft_embedding="lm_head" recipe, so
            # route the embedding to the connector LR and clip group even though it sits in the text model.
            elif "multi_modal_projector" in name or "embed_tokens" in name:
                groups["connector"].append(param)
            else:
                groups["llm"].append(param)
        return [{"params": params, "lr": lrs[name]} for name, params in groups.items() if params]

    def _autocast(self) -> AbstractContextManager:
        """bf16 autocast for a bf16 LLM; otherwise autocast off, so that a float32 LLM runs in float32."""
        device_type = next(self.parameters()).device.type
        if self.model.llm_dtype == torch.bfloat16:
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if torch.amp.autocast_mode.is_autocast_available(device_type):
            return torch.autocast(device_type=device_type, enabled=False)
        return nullcontext()

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute the training loss: the sum of the flow-matching, FAST token cross-entropy and z-losses.

        Args:
            batch (`dict[str, Tensor]`):
                Output of the preprocessor.
            reduction (`str`, *optional*, defaults to `"mean"`):
                `"mean"` over the batch, or `"none"` for a per-example loss `[B]`.
            noise (`Tensor`, *optional*):
                Flow-matching noise `[B, num_flow_timesteps, chunk_size, max_action_dim]`, sampled when `None`.
            time (`Tensor`, *optional*):
                Flow-matching times `[B, num_flow_timesteps]`, sampled when `None`.

        Returns:
            `tuple[Tensor, dict[str, Any]]`: The loss and the value of each loss term for logging.
        """
        if reduction not in {"mean", "none"}:
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}.")
        with self._autocast():
            losses = self.model(batch, reduction=reduction, noise=noise, time=time)
        loss = sum(losses.values())
        log = {name: value.mean().item() for name, value in losses.items()}
        log["loss"] = loss.mean().item()
        return loss, log

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        noise: Tensor | None = None,
        **kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        """Predict a chunk of normalized actions.

        Args:
            batch (`dict[str, Tensor]`):
                Output of the preprocessor.
            noise (`Tensor`, *optional*):
                Starting sample of flow matching `[B, chunk_size, max_action_dim]`, drawn when `None`. Discrete
                decoding ignores it.
            **kwargs ([`RTCActionSelectKwargs`]):
                Real-time chunking arguments of continuous inference.

        Returns:
            `Tensor`: The actions `[B, chunk_size, action_dim]`.
        """
        self.eval()
        with self._autocast():
            if self.config.inference_action_mode == "discrete":
                actions = self._generate_discrete_actions(batch)
            else:
                actions = self.model.sample_actions(batch, noise=noise, **kwargs)
        return actions[:, :, : self._action_dim()].float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        """Return the next action, predicting a new chunk when the queue of `n_action_steps` is empty."""
        if self.config.rtc_config is not None and self.config.rtc_config.enabled:
            raise ValueError("RTC is not supported for select_action, use it with predict_action_chunk")
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _generate_discrete_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate FAST action tokens, constrained to one well-formed chunk, and decode them."""
        if self._fast_tokenizer is None:
            self._fast_tokenizer = UniversalActionProcessor.from_pretrained(self.config.action_tokenizer_name)
        action_tokenizer = self._fast_tokenizer
        action_horizon, action_dim = self.config.chunk_size, self._action_dim()
        select_token = partial(
            select_constrained_discrete_action_token,
            action_tokenizer=action_tokenizer,
            target_coefficients=action_horizon * action_dim,
        )
        # Upstream's action token budget; constrained decoding closes the chunk well before it.
        generated = self.model.generate_tokens(batch, action_horizon * 16, select_token, EOS_TOKEN_ID)
        if not all(ids and ids[-1] == EOS_TOKEN_ID for ids in generated):
            raise RuntimeError(
                f"Discrete action generation did not emit EOS within {action_horizon * 16} tokens."
            )
        actions = decode_discrete_action_chunk(generated, action_tokenizer, action_horizon, action_dim)
        return torch.as_tensor(actions, dtype=torch.float32, device=batch["input_ids"].device)

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """LoRA on every linear layer of the VLM, with the action expert fully trained (upstream settings).

        The native model keeps its vocabulary projection at `transformer.ff_out` and the image pooling and
        projection inside `vision_backbone`; transformers' conversion moves them to `lm_head` and
        `multi_modal_projector`, so both are targeted explicitly to cover the same linear layers as upstream.
        Adapters start as an identity update with A ~ N(0, 1/r) and B = 0 (upstream's "gaussian" LoRA init).
        """
        return {
            "target_modules": (
                r"model\.vlm\.(lm_head|model\.(language_model|vision_tower|multi_modal_projector)\.(?:.*\.)?"
                r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|fc1|fc2|patch_embedding))"
            ),
            "modules_to_save": ["model.action_expert"],
            "r": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "init_lora_weights": "gaussian",
        }
