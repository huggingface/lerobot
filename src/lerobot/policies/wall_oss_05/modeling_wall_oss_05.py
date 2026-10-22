#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import _transformers_available, require_package
from lerobot.utils.language import require_single_text_output

if TYPE_CHECKING or _transformers_available:
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLProcessor
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
        Qwen2_5_VLRotaryEmbedding,
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )

from .configuration_wall_oss_05 import WallOSS05Config
from .processor_wall_oss_05 import (
    WALL_OSS_ACTION_TOKEN,
    WALL_OSS_IMAGE_TOKEN,
    WALL_OSS_PREDICT_ACTIONS,
    WALL_OSS_PROPRIO_TOKEN,
)

_ACTION_TOKEN = WALL_OSS_ACTION_TOKEN
_IMAGE_TOKEN = WALL_OSS_IMAGE_TOKEN
_PROPRIO_TOKEN = WALL_OSS_PROPRIO_TOKEN
_TEXT_TARGET_START = "<|wall_text_target_start|>"  # nosec B105
_TEXT_TARGET_END = "<|wall_text_target_end|>"  # nosec B105
_CAMERA_LABELS = {
    "face_view": "front view",
    "right_wrist_view": "right wrist view",
    "left_wrist_view": "left wrist view",
}


@dataclass
class WallOSS05Output:
    loss: Tensor | None = None
    flow_loss: Tensor | None = None
    text_loss: Tensor | None = None


type WallOSS05KeyValue = tuple[Tensor, Tensor]


class WallOSS05RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return (self.weight * normalized).to(dtype)


class WallOSS05MLP(nn.Module):
    """Fused SwiGLU projection matching the released checkpoint layout."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(functional.silu(gate) * up)


class WallOSS05VisionMLP(nn.Module):
    """Fused form of the Transformers vision MLP used by the checkpoint."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(functional.silu(gate) * up)


class WallOSS05SparseMLP(nn.Module):
    def __init__(self, experts: list[dict[str, Any]], dim_inputs: list[int], hidden_size: int):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.hidden_size = hidden_size
        self.experts = nn.ModuleList(
            [
                WallOSS05MLP(
                    hidden_size=int(expert["hidden_size"]),
                    intermediate_size=int(expert["intermediate_size"]),
                )
                for expert in experts
            ]
        )

    def forward(self, hidden_states: Tensor, token_types: Tensor) -> Tensor:
        output = torch.zeros_like(hidden_states)
        for expert_index, (expert, width) in enumerate(zip(self.experts, self.dim_inputs, strict=True)):
            mask = token_types == expert_index
            if mask.any():
                output[mask, :width] = expert(hidden_states[mask, :width])
        return output


class WallOSS05JointAttention(nn.Module):
    """Qwen2.5 multimodal attention with one input/output projection per expert."""

    def __init__(self, config: Qwen2_5_VLConfig, layer_index: int):
        super().__init__()
        del layer_index  # Kept in the signature for FSDP-friendly layer construction.
        text_config = config.text_config
        self.hidden_size = int(text_config.hidden_size)
        self.num_heads = int(text_config.num_attention_heads)
        self.num_key_value_heads = int(text_config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = float(text_config.attention_dropout)
        self.dim_inputs = [int(width) for width in config.dim_inputs]
        qkv_width = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        self.qkv_proj_experts = nn.ModuleList(
            [nn.Linear(width, qkv_width, bias=True) for width in self.dim_inputs]
        )
        self.o_proj_experts = nn.ModuleList(
            [nn.Linear(self.hidden_size, width, bias=False) for width in self.dim_inputs]
        )
        rope = getattr(text_config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if rope is None:
            raise ValueError("Wall-OSS-0.5 requires multimodal rotary embedding parameters.")
        self.mrope_section = list(rope["mrope_section"])

    def forward(
        self,
        hidden_states: Tensor,
        token_types: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        past_key_value: WallOSS05KeyValue | None = None,
    ) -> tuple[Tensor, WallOSS05KeyValue]:
        batch_size, sequence_length, _ = hidden_states.shape
        query = hidden_states.new_zeros(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = hidden_states.new_zeros(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value = torch.zeros_like(key)

        for expert_index, (projection, width) in enumerate(
            zip(self.qkv_proj_experts, self.dim_inputs, strict=True)
        ):
            mask = token_types == expert_index
            if not mask.any():
                continue
            projected = projection(hidden_states[mask, :width].to(projection.weight.dtype))
            projected = projected.view(-1, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
            query[mask] = projected[:, : self.num_heads]
            key[mask] = projected[:, self.num_heads : self.num_heads + self.num_key_value_heads]
            value[mask] = projected[:, self.num_heads + self.num_key_value_heads :]

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        cosine, sine = position_embeddings
        query, key = apply_multimodal_rotary_pos_emb(
            query,
            key,
            cosine,
            sine,
            self.mrope_section,
            unsqueeze_dim=1,
        )
        projection_dtype = self.qkv_proj_experts[0].weight.dtype
        query = query.to(projection_dtype)
        key = key.to(projection_dtype)
        value = value.to(projection_dtype)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        present = (key, value)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask[:, None],
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, self.hidden_size)
        output = torch.zeros_like(hidden_states)
        for expert_index, (projection, width) in enumerate(
            zip(self.o_proj_experts, self.dim_inputs, strict=True)
        ):
            mask = token_types == expert_index
            if mask.any():
                output[mask, :width] = projection(attended[mask].to(projection.weight.dtype))
        return output, present


class WallOSS05DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_index: int):
        super().__init__()
        text_config = config.text_config
        widths = [int(width) for width in config.dim_inputs]
        self.input_layernorms = nn.ModuleList(
            [WallOSS05RMSNorm(width, float(text_config.rms_norm_eps)) for width in widths]
        )
        self.post_attention_layernorms = nn.ModuleList(
            [WallOSS05RMSNorm(width, float(text_config.rms_norm_eps)) for width in widths]
        )
        self.self_attn = WallOSS05JointAttention(config, layer_index)
        self.moe = WallOSS05SparseMLP(config.experts, widths, int(text_config.hidden_size))
        self.dim_inputs = widths

    def _normalize(self, hidden_states: Tensor, token_types: Tensor, norms: nn.ModuleList) -> Tensor:
        output = torch.zeros_like(hidden_states)
        for expert_index, (norm, width) in enumerate(zip(norms, self.dim_inputs, strict=True)):
            mask = token_types == expert_index
            if mask.any():
                output[mask, :width] = norm(hidden_states[mask, :width])
        return output

    def forward(
        self,
        hidden_states: Tensor,
        token_types: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        past_key_value: WallOSS05KeyValue | None = None,
    ) -> tuple[Tensor, WallOSS05KeyValue]:
        normalized = self._normalize(hidden_states, token_types, self.input_layernorms)
        attended, present = self.self_attn(
            normalized, token_types, attention_mask, position_embeddings, past_key_value
        )
        hidden_states = hidden_states + attended
        normalized = self._normalize(hidden_states, token_types, self.post_attention_layernorms)
        return hidden_states + self.moe(normalized, token_types), present


class WallOSS05Decoder(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__()
        text_config = config.text_config
        self.embed_tokens = nn.Embedding(
            int(text_config.vocab_size),
            int(text_config.hidden_size),
            text_config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                WallOSS05DecoderLayer(config, layer_index)
                for layer_index in range(int(text_config.num_hidden_layers))
            ]
        )
        self.norms = nn.ModuleList(
            [WallOSS05RMSNorm(int(width), float(text_config.rms_norm_eps)) for width in config.dim_inputs]
        )
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(text_config)
        self.dim_inputs = [int(width) for width in config.dim_inputs]

    def forward(
        self,
        inputs_embeds: Tensor,
        token_types: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        *,
        past_key_values: tuple[WallOSS05KeyValue, ...] | None = None,
        use_cache: bool = False,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[WallOSS05KeyValue, ...] | None]:
        if position_embeddings is None:
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        cache = []
        for index, layer in enumerate(self.layers):
            hidden_states, present = layer(
                hidden_states,
                token_types,
                attention_mask,
                position_embeddings,
                past_key_values[index] if past_key_values is not None else None,
            )
            if use_cache:
                cache.append(present)
        output = torch.zeros_like(hidden_states)
        for expert_index, (norm, width) in enumerate(zip(self.norms, self.dim_inputs, strict=True)):
            mask = token_types == expert_index
            if mask.any():
                output[mask, :width] = norm(hidden_states[mask, :width])
        return output, tuple(cache) if use_cache else None


class WallOSS05SinusoidalEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, timesteps: Tensor) -> Tensor:
        half_size = self.hidden_size // 2
        exponent = math.log(10000) / (half_size - 1)
        frequencies = torch.exp(
            torch.arange(half_size, device=timesteps.device, dtype=torch.float32) * -exponent
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class WallOSS05ActionProcessor(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__()
        self.action_dim = 26
        self.action_hidden_size = int(config.action_hidden_size)
        self.hidden_size = int(config.text_config.hidden_size)
        self.s = float(config.noise_scheduler.get("s", 0.999))
        self.time_shift = float(config.noise_scheduler.get("time_shift", 1.0))
        self.time_embed = WallOSS05SinusoidalEmbedding(self.action_hidden_size)
        self.w1 = nn.Linear(2 * self.action_dim, self.action_hidden_size, bias=False)
        self.w2 = nn.Linear(2 * self.action_hidden_size, self.action_hidden_size, bias=False)
        self.w3 = nn.Linear(self.action_hidden_size, self.action_hidden_size, bias=False)
        self.action_proj_back = nn.Linear(self.action_hidden_size, self.action_dim, bias=False)

    def inference_times(self, steps: int, device: torch.device) -> Tensor:
        times = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)
        if self.time_shift != 1:
            times = self.time_shift * times / (1 + (self.time_shift - 1) * times)
        return times * self.s

    def embed(self, timesteps: Tensor, actions: Tensor, dof_mask: Tensor) -> Tensor:
        action_input = torch.cat((actions.float(), dof_mask.float()), dim=-1)
        action_embedding = self.w1(action_input.to(self.w1.weight.dtype))
        time_embedding = self.time_embed(timesteps).to(self.w2.weight.dtype)
        time_embedding = time_embedding[:, None].expand(-1, actions.shape[1], -1)
        hidden = self.w2(torch.cat((action_embedding, time_embedding), dim=-1))
        hidden = self.w3(functional.silu(hidden))
        return functional.pad(hidden, (0, self.hidden_size - self.action_hidden_size))

    def noisy_training_input(
        self, actions: Tensor, sample_time: Tensor, dof_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        actions = actions.float()
        noise = torch.randn_like(actions)
        expanded_time = sample_time[:, None, None]
        noisy_actions = (1 - expanded_time) * noise + expanded_time * actions
        return self.embed(sample_time, noisy_actions, dof_mask), noisy_actions, noise


class WallOSS05Model(nn.Module):
    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        processor: Qwen2_5_VLProcessor | None = None,
    ):
        super().__init__()
        self.config = config
        self.visual = Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)
        # WALL-X scales normalized activations in FP32, including in vision.
        # Transformers' Qwen norm casts before scaling, changing BF16 outputs.
        vision_width = int(config.vision_config.hidden_size)
        for block in self.visual.blocks:
            block.norm1 = WallOSS05RMSNorm(vision_width, eps=1e-6)
            block.norm2 = WallOSS05RMSNorm(vision_width, eps=1e-6)
            block.mlp = WallOSS05VisionMLP(
                vision_width,
                int(config.vision_config.intermediate_size),
            )
        self.visual.merger.ln_q = WallOSS05RMSNorm(vision_width, eps=1e-6)
        self.model = WallOSS05Decoder(config)
        self.action_preprocessor = WallOSS05ActionProcessor(config)
        if processor is None and not hasattr(config, "action_token_id"):
            raise ValueError(
                "Wall-OSS-0.5 requires the native LeRobot checkpoint config. "
                "Load the current lerobot/wall-oss-0.5 checkpoint."
            )
        self.action_token_id = (
            processor.tokenizer.convert_tokens_to_ids(_ACTION_TOKEN)
            if processor is not None
            else int(config.action_token_id)
        )
        self.eos_token_id = (
            getattr(processor.tokenizer, "eos_token_id", config.text_config.eos_token_id)
            if processor is not None
            else config.text_config.eos_token_id
        )
        self.pad_token_id = (
            getattr(processor.tokenizer, "pad_token_id", config.text_config.pad_token_id)
            if processor is not None
            else config.text_config.pad_token_id
        )
        self.lm_head = nn.Linear(
            int(config.text_config.hidden_size),
            int(config.text_config.vocab_size),
            bias=False,
        )
        self.tie_weights()

    def tie_weights(self) -> None:
        """Tie the text head to the checkpoint's Qwen token embeddings."""
        self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _causal_mask(attention_mask: Tensor) -> Tensor:
        length = attention_mask.shape[-1]
        causal = torch.ones(length, length, device=attention_mask.device, dtype=torch.bool).tril()
        return causal[None] & attention_mask[:, :, None].bool() & attention_mask[:, None, :].bool()

    def _position_ids(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        image_grid_thw: Tensor | None,
    ) -> Tensor:
        if image_grid_thw is None:
            positions = attention_mask.long().cumsum(-1) - 1
            positions.masked_fill_(attention_mask == 0, 1)
            return positions[None].expand(3, -1, -1)

        merge_size = int(self.config.vision_config.spatial_merge_size)
        position_ids = torch.ones(3, *input_ids.shape, device=input_ids.device, dtype=input_ids.dtype)
        image_index = 0
        for batch_index, ids in enumerate(input_ids):
            active = attention_mask[batch_index].bool()
            active_ids = ids[active]
            vision_starts = torch.where(active_ids == int(self.config.vision_start_token_id))[0]
            image_count = int((active_ids[vision_starts + 1] == int(self.config.image_token_id)).sum())
            token_list = active_ids.tolist()
            start = 0
            chunks: list[Tensor] = []
            for _ in range(image_count):
                end = token_list.index(int(self.config.image_token_id), start)
                temporal, height, width = image_grid_thw[image_index].tolist()
                image_index += 1
                grid_height, grid_width = height // merge_size, width // merge_size
                text_length = end - start
                offset = chunks[-1].max() + 1 if chunks else 0
                chunks.append(torch.arange(text_length, device=input_ids.device)[None].expand(3, -1) + offset)
                temporal_index = torch.arange(temporal, device=input_ids.device)[:, None]
                temporal_index = temporal_index.expand(-1, grid_height * grid_width).flatten()
                height_index = (
                    torch.arange(grid_height, device=input_ids.device)[None, :, None]
                    .expand(temporal, -1, grid_width)
                    .flatten()
                )
                width_index = (
                    torch.arange(grid_width, device=input_ids.device)[None, None, :]
                    .expand(temporal, grid_height, -1)
                    .flatten()
                )
                chunks.append(torch.stack((temporal_index, height_index, width_index)) + text_length + offset)
                start = end + temporal * grid_height * grid_width
            if start < len(token_list):
                offset = chunks[-1].max() + 1 if chunks else 0
                chunks.append(
                    torch.arange(len(token_list) - start, device=input_ids.device)[None].expand(3, -1)
                    + offset
                )
            position_ids[:, batch_index, active] = torch.cat(chunks, dim=1)
        return position_ids

    def _input_embeddings(
        self,
        input_ids: Tensor,
        pixel_values: Tensor | None,
        image_grid_thw: Tensor | None,
    ) -> Tensor:
        embeddings = self.model.embed_tokens(input_ids)
        if pixel_values is None:
            return embeddings
        vision_output = self.visual(pixel_values.to(self.visual.dtype), grid_thw=image_grid_thw)
        image_embeddings = vision_output.pooler_output.to(embeddings)
        image_mask = (input_ids == int(self.config.image_token_id)).unsqueeze(-1)
        expected = int(image_mask.sum())
        if expected != image_embeddings.shape[0]:
            raise ValueError(
                f"Image token/features mismatch: {expected} tokens and {image_embeddings.shape[0]} features."
            )
        return embeddings.masked_scatter(image_mask.expand_as(embeddings), image_embeddings)

    def _hidden_states(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor | None,
        image_grid_thw: Tensor | None,
        moe_token_types: Tensor,
        action_embeddings: Tensor | None,
    ) -> Tensor:
        embeddings = self._input_embeddings(input_ids, pixel_values, image_grid_thw)
        action_mask = input_ids == self.action_token_id
        if action_mask.any():
            if action_embeddings is None:
                raise ValueError("Action-token inputs require action embeddings.")
            expected = int(action_mask.sum())
            available = action_embeddings.numel() // action_embeddings.shape[-1]
            if expected != available:
                raise ValueError(
                    f"Action token/embedding mismatch: {expected} tokens and {available} embeddings."
                )
            embeddings = embeddings.masked_scatter(
                action_mask.unsqueeze(-1).expand_as(embeddings),
                action_embeddings.to(embeddings.dtype),
            )
        return self._decode_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            moe_token_types=moe_token_types,
            embeddings=embeddings,
        )

    def _decode_embeddings(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        image_grid_thw: Tensor | None,
        moe_token_types: Tensor,
        embeddings: Tensor,
    ) -> Tensor:
        position_ids = self._position_ids(input_ids, attention_mask, image_grid_thw)
        hidden_states, _ = self.model(
            embeddings,
            moe_token_types.long(),
            self._causal_mask(attention_mask),
            position_ids,
        )
        return hidden_states

    @staticmethod
    def _sample_text_token(logits: Tensor, temperature: float, top_p: float) -> Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        probabilities = functional.softmax(logits / temperature, dim=-1)
        if top_p < 1:
            sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
            remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities > top_p
            sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
            sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probabilities, num_samples=1)
            return sorted_indices.gather(-1, sampled).squeeze(-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate_text(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        max_new_tokens: int = 100,
        min_new_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        **_: Any,
    ) -> Tensor:
        """Autoregressively generate text through Wall's vision-language expert."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
            raise ValueError("min_new_tokens must be in [0, max_new_tokens].")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")

        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        pad_token_id = self.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        token_types = torch.zeros_like(input_ids)
        embeddings = self._input_embeddings(input_ids, pixel_values, image_grid_thw)
        generated: list[Tensor] = []
        finished = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)

        for step in range(max_new_tokens):
            hidden_states = self._decode_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                moe_token_types=token_types,
                embeddings=embeddings,
            )
            logits = self.lm_head(hidden_states[:, -1, : self.lm_head.in_features]).float()
            if eos_token_id is not None and step < min_new_tokens:
                logits[:, eos_token_id] = -torch.inf
            next_token = self._sample_text_token(logits, temperature, top_p)
            if finished.any():
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, int(pad_token_id)),
                    next_token,
                )
            generated.append(next_token)

            if eos_token_id is not None:
                finished |= next_token == eos_token_id
                if finished.all():
                    break

            input_ids = torch.cat((input_ids, next_token[:, None]), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token[:, None])), dim=-1)
            token_types = torch.cat((token_types, torch.zeros_like(next_token[:, None])), dim=-1)
            next_embedding = self.model.embed_tokens(next_token[:, None])
            embeddings = torch.cat((embeddings, next_embedding), dim=1)

        return torch.stack(generated, dim=1)

    def _text_cross_entropy(self, hidden_states: Tensor, labels: Tensor) -> Tensor | None:
        """Compute next-token CE only for supervised positions to avoid full-vocabulary logits."""
        if labels.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"Text labels must match input sequence shape {tuple(hidden_states.shape[:2])}, "
                f"got {tuple(labels.shape)}."
            )
        shifted_labels = labels[:, 1:].contiguous()
        loss_mask = shifted_labels != -100
        if not loss_mask.any():
            return None
        shifted_hidden = hidden_states[:, :-1, : self.lm_head.in_features]
        logits = self.lm_head(shifted_hidden[loss_mask]).float()
        return functional.cross_entropy(logits, shifted_labels[loss_mask].to(logits.device))

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        moe_token_types: Tensor,
        action_chunk: Tensor | None = None,
        dof_mask: Tensor | None = None,
        sample_time: Tensor | None = None,
        flow_loss_mask: Tensor | None = None,
        text_labels: Tensor | None = None,
        **_: Any,
    ) -> WallOSS05Output:
        action_token_mask = input_ids == self.action_token_id
        action_rows = action_token_mask.any(dim=-1)
        action_embeddings = None
        flow_inputs = None
        if action_rows.any():
            if action_chunk is None or dof_mask is None or sample_time is None:
                raise ValueError("Action-token inputs require actions, dof_mask, and sample_time.")
            token_counts = action_token_mask.sum(dim=-1)
            expected_horizon = action_chunk.shape[1]
            if not torch.equal(
                token_counts[action_rows],
                torch.full_like(token_counts[action_rows], expected_horizon),
            ):
                raise ValueError(
                    "Each action-supervised row must contain exactly one action token per horizon step."
                )
            active_action_chunk = action_chunk[action_rows]
            active_dof_mask = dof_mask[action_rows]
            active_sample_time = sample_time[action_rows]
            action_embeddings, noisy_actions, _ = self.action_preprocessor.noisy_training_input(
                active_action_chunk,
                active_sample_time,
                active_dof_mask,
            )
            flow_inputs = (active_action_chunk, active_dof_mask, active_sample_time, noisy_actions)
        hidden_states = self._hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            moe_token_types=moe_token_types,
            action_embeddings=action_embeddings,
        )
        text_loss = self._text_cross_entropy(hidden_states, text_labels) if text_labels is not None else None

        flow_loss = None
        if flow_inputs is not None:
            active_action_chunk, active_dof_mask, active_sample_time, noisy_actions = flow_inputs
            action_hidden = hidden_states[action_token_mask, : self.config.action_hidden_size]
            velocity = self.action_preprocessor.action_proj_back(action_hidden.float())
            expanded_time = active_sample_time[:, None, None].expand_as(active_action_chunk).reshape(-1, 26)
            prediction = (1 - expanded_time) * velocity + noisy_actions.reshape(-1, 26)
            loss = (prediction - active_action_chunk.reshape(-1, 26).float()).square()
            valid_mask = active_dof_mask.reshape(-1, 26).float()
            if flow_loss_mask is not None:
                valid_mask = valid_mask * flow_loss_mask[action_rows].reshape(-1, 1).float()
            flow_loss = (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1)

        losses = [value for value in (flow_loss, text_loss) if value is not None]
        total_loss = sum(losses) if losses else None
        return WallOSS05Output(loss=total_loss, flow_loss=flow_loss, text_loss=text_loss)

    @torch.no_grad()
    def generate_flow_action(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        moe_token_types: Tensor,
        dof_mask: Tensor,
        action_horizon: int,
        action_dim: int,
        num_inference_timesteps: int,
        noise: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        if action_dim != 26:
            raise ValueError("Wall-OSS-0.5 has a fixed 26D action space.")
        batch_size = input_ids.shape[0]
        if noise is None:
            noisy_actions = torch.randn(
                batch_size,
                action_horizon,
                action_dim,
                device=input_ids.device,
                dtype=torch.float32,
            )
        else:
            noisy_actions = noise.to(input_ids.device, dtype=torch.float32).clone()
        times = self.action_preprocessor.inference_times(num_inference_timesteps, input_ids.device)
        prefix_length = input_ids.shape[1] - action_horizon
        if (
            prefix_length < 1
            or (input_ids[:, :prefix_length] == self.action_token_id).any()
            or not (input_ids[:, prefix_length:] == self.action_token_id).all()
        ):
            raise ValueError("Flow inputs must end with exactly action_horizon contiguous action tokens.")
        # The causal observation/text prefix is independent of the flow timestep.
        # Keep this cache local to a chunk so the next observation always gets a fresh prefill.
        embeddings = self._input_embeddings(input_ids, pixel_values, image_grid_thw)
        positions = self._position_ids(input_ids, attention_mask, image_grid_thw)
        causal_mask = self._causal_mask(attention_mask)
        position_embeddings = self.model.rotary_emb(embeddings, positions)
        _, prefix_cache = self.model(
            embeddings[:, :prefix_length],
            moe_token_types[:, :prefix_length].long(),
            causal_mask[:, :prefix_length, :prefix_length],
            positions[:, :, :prefix_length],
            use_cache=True,
            position_embeddings=tuple(value[..., :prefix_length, :] for value in position_embeddings),
        )
        for index in range(num_inference_timesteps):
            timestep = times[index].expand(batch_size)
            action_embeddings = self.action_preprocessor.embed(timestep, noisy_actions, dof_mask)
            hidden_states, _ = self.model(
                action_embeddings.to(embeddings.dtype),
                moe_token_types[:, prefix_length:].long(),
                causal_mask[:, prefix_length:],
                positions[:, :, prefix_length:],
                past_key_values=prefix_cache,
                position_embeddings=tuple(value[..., prefix_length:, :] for value in position_embeddings),
            )
            action_hidden = hidden_states[..., : self.config.action_hidden_size]
            velocity = self.action_preprocessor.action_proj_back(action_hidden.float())
            velocity = velocity.view(batch_size, action_horizon, action_dim)
            noisy_actions = noisy_actions + (times[index + 1] - times[index]) * velocity

        return {"predict_action": noisy_actions}


class WallOSS05Policy(PreTrainedPolicy):
    """Native LeRobot implementation of Wall-OSS-0.5."""

    config_class = WallOSS05Config
    name = "wall_oss_05"
    _fsdp_wrap_modules = ["WallOSS05DecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(
        self,
        config: WallOSS05Config,
        dataset_stats: dict | None = None,
        dataset_meta: Any | None = None,
        load_model: bool = True,
        **kwargs: Any,
    ):
        del dataset_stats, dataset_meta, kwargs
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues: dict[str, deque[Tensor]] = {}
        self.processor: Qwen2_5_VLProcessor | None = None
        if load_model:
            self._initialize_model()
        self.reset()

    def _initialize_model(self) -> None:
        """Build from config; the shared persistence layer owns all weight loading."""
        require_package("transformers", extra="wall_oss_05")
        model_config = self.config.vlm_backbone_config
        model_config.vision_config._attn_implementation = "sdpa"
        previous_dtype = torch.get_default_dtype()
        try:
            if self.config.dtype is not None:
                torch.set_default_dtype(self.config.dtype)
            self.model = WallOSS05Model(model_config)
        finally:
            torch.set_default_dtype(previous_dtype)
        # Match the released mixed-precision parameter layout.
        self.model.action_preprocessor.float()
        for module in self.model.model.modules():
            if isinstance(module, WallOSS05RMSNorm):
                module.float()

    def _get_processor(self) -> Qwen2_5_VLProcessor:
        if self.processor is None:
            require_package("transformers", extra="wall_oss_05")
            self.processor = Qwen2_5_VLProcessor.from_pretrained(
                self.config.pretrained_name_or_path,
                revision=self.config.tokenizer_revision,
                fix_mistral_regex=True,
            )
            self.processor.tokenizer.add_tokens([_PROPRIO_TOKEN, _ACTION_TOKEN])
            self.processor.tokenizer.padding_side = "left"
        return self.processor

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample the checkpoint's ``(1 - Beta(alpha, beta)) * s`` flow time."""
        scheduler = getattr(getattr(self.model, "config", None), "noise_scheduler", None) or {}
        alpha = float(scheduler.get("beta_alpha", 1.5))
        beta = float(scheduler.get("beta_beta", 1.0))
        scale = float(scheduler.get("s", 0.999))
        time_shift = float(scheduler.get("time_shift", 1.0))
        sample = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
        sample_time = 1 - sample
        if time_shift != 1:
            sample_time = (time_shift * sample_time) / (1 + (time_shift - 1) * sample_time)
        return sample_time * scale

    @staticmethod
    def _pretokenized_inputs(batch: dict[str, Any]) -> dict[str, Tensor]:
        names = (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "moe_token_types",
            "dof_mask",
            "text_labels",
        )
        inputs = {name: batch[name] for name in names if name in batch}
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "moe_token_types"}
        missing = sorted(required - inputs.keys())
        if missing:
            raise ValueError(
                f"Wall-OSS-0.5 requires tokenized inputs from its policy preprocessor; missing {missing}."
            )
        return inputs

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        inputs = self._pretokenized_inputs(batch)
        action_mask = inputs["dof_mask"]
        predict_actions = batch.get(
            WALL_OSS_PREDICT_ACTIONS,
            torch.ones(action_mask.shape[0], device=action_mask.device, dtype=torch.bool),
        )
        actions = batch[ACTION]
        if actions.ndim != 3 or actions.shape[-1] != self.config.max_action_dim:
            raise ValueError(f"Expected canonical actions shaped (B,H,26), got {tuple(actions.shape)}.")

        horizon = actions.shape[1]
        if horizon > self.config.chunk_size:
            raise ValueError(
                f"Action horizon {horizon} exceeds configured chunk_size {self.config.chunk_size}."
            )
        if horizon < self.config.chunk_size:
            pad = self.config.chunk_size - horizon
            actions = torch.nn.functional.pad(actions, (0, 0, 0, pad))
            action_mask[:, horizon:] = False

        inputs["dof_mask"] = action_mask.to(inputs["input_ids"].device)
        inputs["sample_time"] = self._sample_time(actions.shape[0], actions.device)

        flow_loss_mask = action_mask.any(dim=-1)
        flow_loss_mask &= predict_actions[:, None].to(flow_loss_mask.device)
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            valid = ~action_is_pad.to(flow_loss_mask.device, dtype=torch.bool)
            flow_loss_mask[:, : valid.shape[1]] &= valid
        outputs = self.model(
            **inputs,
            action_chunk=actions,
            flow_loss_mask=flow_loss_mask.to(actions.device),
            mode="train",
        )
        flow_loss = outputs.flow_loss
        text_loss = getattr(outputs, "text_loss", None)
        loss = None
        if flow_loss is not None:
            loss = self.config.flow_loss_weight * flow_loss
        if text_loss is not None:
            weighted_text_loss = self.config.text_loss_weight * text_loss
            loss = weighted_text_loss if loss is None else loss + weighted_text_loss
        if loss is None:
            raise RuntimeError(
                "Wall-OSS-0.5 batch produced neither action nor text supervision. "
                "Check the selected recipe and target annotations."
            )
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("Wall-OSS-0.5 produced a non-finite training loss.")
        metrics = {"loss": float(loss.detach())}
        if flow_loss is not None:
            metrics["flow_loss"] = float(flow_loss.detach())
        if text_loss is not None:
            metrics["text_loss"] = float(text_loss.detach())
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        del kwargs
        inputs = self._pretokenized_inputs(batch)
        output = self.model.generate_flow_action(
            action_horizon=self.config.chunk_size,
            action_dim=self.config.max_action_dim,
            num_inference_timesteps=self.config.num_inference_timesteps,
            **inputs,
        )
        return output["predict_action"]

    def supports_text_generation(self) -> bool:
        return True

    @torch.no_grad()
    def generate_text(self, batch: dict[str, Tensor]) -> str:
        """Decode one response from contract-rendered messages."""
        if not hasattr(self, "model"):
            raise RuntimeError("Wall-OSS-0.5 model is not loaded.")
        processor = self._get_processor()
        inputs = self._pretokenized_inputs(batch)
        output_ids = self.model.generate_text(
            **inputs,
            max_new_tokens=100,
            min_new_tokens=0,
            temperature=self.config.text_temperature,
            top_p=self.config.text_top_p,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        outputs = [
            text.strip()
            for text in processor.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        ]
        return require_single_text_output(outputs, policy_name="Wall-OSS-0.5")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        if not self._queues[ACTION]:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self._queues[ACTION].extend(chunk[:, : self.config.n_action_steps].transpose(0, 1))
        return self._queues[ACTION].popleft()
