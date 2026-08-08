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

import json
import math
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as functional
from huggingface_hub import save_torch_state_dict, snapshot_download
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLProcessor
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
        Qwen2_5_VLRotaryEmbedding,
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )

from .configuration_wall_oss_05 import WallOSS05Config

_ACTION_TOKEN = "<|action|>"  # nosec B105
_IMAGE_TOKEN = "<|image_pad|>"  # nosec B105
_PROPRIO_TOKEN = "<|propri|>"  # nosec B105
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
        self.mrope_section = list(rope["mrope_section"])

    def forward(
        self,
        hidden_states: Tensor,
        token_types: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
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
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        projection_dtype = self.qkv_proj_experts[0].weight.dtype
        query = query.to(projection_dtype)
        key = key.to(projection_dtype)
        value = value.to(projection_dtype)

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
        return output


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
    ) -> Tensor:
        normalized = self._normalize(hidden_states, token_types, self.input_layernorms)
        hidden_states = hidden_states + self.self_attn(
            normalized, token_types, attention_mask, position_embeddings
        )
        normalized = self._normalize(hidden_states, token_types, self.post_attention_layernorms)
        return hidden_states + self.moe(normalized, token_types)


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
    ) -> Tensor:
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, token_types, attention_mask, position_embeddings)
        output = torch.zeros_like(hidden_states)
        for expert_index, (norm, width) in enumerate(zip(self.norms, self.dim_inputs, strict=True)):
            mask = token_types == expert_index
            if mask.any():
                output[mask, :width] = norm(hidden_states[mask, :width])
        return output


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
        processor: Qwen2_5_VLProcessor,
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.visual = Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)
        for block in self.visual.blocks:
            block.mlp = WallOSS05VisionMLP(
                int(config.vision_config.hidden_size),
                int(config.vision_config.intermediate_size),
            )
        self.model = WallOSS05Decoder(config)
        self.action_preprocessor = WallOSS05ActionProcessor(config)
        self.action_token_id = processor.tokenizer.convert_tokens_to_ids(_ACTION_TOKEN)
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
            chunks = []
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
        return self.model(
            embeddings,
            moe_token_types.long(),
            self._causal_mask(attention_mask),
            position_ids,
        )

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
            eos_token_id = self.processor.tokenizer.eos_token_id
        pad_token_id = self.processor.tokenizer.pad_token_id
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
        noisy_actions = None
        active_action_chunk = None
        active_dof_mask = None
        active_sample_time = None
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
        if action_rows.any():
            assert active_action_chunk is not None
            assert active_dof_mask is not None
            assert active_sample_time is not None
            assert noisy_actions is not None
            action_hidden = hidden_states[action_token_mask, : self.config.action_hidden_size]
            velocity = self.action_preprocessor.action_proj_back(action_hidden.float())
            expanded_time = active_sample_time[:, None, None].expand_as(active_action_chunk).reshape(-1, 26)
            prediction = (1 - expanded_time) * velocity + noisy_actions.reshape(-1, 26)
            loss = (prediction - active_action_chunk.reshape(-1, 26).float()).square()
            loss = loss * active_dof_mask.reshape(-1, 26).float()
            if flow_loss_mask is not None:
                loss = loss * flow_loss_mask[action_rows].reshape(-1, 1).float()
            flow_loss = loss.mean()

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

        for index in range(num_inference_timesteps):
            timestep = times[index].expand(batch_size)
            action_embeddings = self.action_preprocessor.embed(timestep, noisy_actions, dof_mask)
            hidden_states = self._hidden_states(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                moe_token_types=moe_token_types,
                action_embeddings=action_embeddings,
            )
            action_hidden = hidden_states[input_ids == self.action_token_id, : self.config.action_hidden_size]
            velocity = self.action_preprocessor.action_proj_back(action_hidden.float())
            velocity = velocity.view(batch_size, action_horizon, action_dim)
            noisy_actions = noisy_actions + (times[index + 1] - times[index]) * velocity

        return {"predict_action": noisy_actions}


_CHECKPOINT_METADATA = (
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class WallOSS05Policy(PreTrainedPolicy):
    """Native LeRobot implementation of Wall-OSS-0.5."""

    config_class = WallOSS05Config
    name = "wall_oss_05"

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
        self._source_checkpoint: Path | None = None
        if load_model:
            self._load_native_model()
        self.reset()

    def _resolve_checkpoint(self) -> Path:
        candidate = Path(self.config.pretrained_name_or_path)
        if candidate.is_dir():
            return candidate.resolve()
        return Path(
            snapshot_download(
                repo_id=self.config.pretrained_name_or_path,
                revision=self.config.pretrained_revision,
            )
        )

    def _load_native_model(self) -> None:
        require_package("transformers", extra="wall_oss_05")

        source = self._resolve_checkpoint()
        processor = Qwen2_5_VLProcessor.from_pretrained(source, fix_mistral_regex=True)
        # These two control tokens were added dynamically by the training
        # loader and therefore are not present in the base Qwen tokenizer
        # files stored with the checkpoint.
        processor.tokenizer.add_tokens([_PROPRIO_TOKEN, _ACTION_TOKEN])
        processor.tokenizer.padding_side = "left"

        if self.config.vlm_config is None:
            checkpoint_config = WallOSS05Config.from_pretrained(source)
            self.config.vlm_config = checkpoint_config.vlm_config
        model_config = self.config.vlm_backbone_config
        model_config.vision_config._attn_implementation = "sdpa"

        state_dict = load_file(source / "model.safetensors", device="cpu")
        model_config.text_config.vocab_size = state_dict["model.embed_tokens.weight"].shape[0]
        model_config.text_config.pad_token_id = processor.tokenizer.pad_token_id

        model = WallOSS05Model(model_config, processor)
        for key in [key for key in state_dict if ".normalizer_" in key]:
            state_dict.pop(key)
        self._fuse_vision_mlp_weights(state_dict)
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.endswith("rotary_emb.inv_freq") and key != "lm_head.weight"
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Converted Wall-OSS-0.5 weights do not match the native model: "
                f"missing={missing[:10]}, unexpected={incompatible.unexpected_keys[:10]}"
            )
        model.tie_weights()

        self.processor = processor
        self.model = model.to(self.config.device)
        self._source_checkpoint = source

    @staticmethod
    def _fuse_vision_mlp_weights(state_dict: dict[str, Tensor]) -> None:
        gate_keys = [
            key for key in state_dict if key.startswith("visual.blocks.") and ".mlp.gate_proj." in key
        ]
        for gate_key in gate_keys:
            up_key = gate_key.replace("gate_proj", "up_proj")
            state_dict[gate_key.replace("gate_proj", "gate_up_proj")] = torch.cat(
                (state_dict.pop(gate_key), state_dict.pop(up_key)), dim=0
            )

    def _get_flow_prompt(self, instruction: str) -> tuple[str, str]:
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nObservation:"
        for camera_name in self.config.camera_key_mapping.values():
            label = _CAMERA_LABELS.get(camera_name, camera_name.replace("_", " "))
            prompt += f" {label}: <|vision_start|>{_IMAGE_TOKEN}<|vision_end|>"
        prompt += (
            f"\nInstruction: {instruction}\n"
            f"Predict the next action in robot action.\nProprioception: {_PROPRIO_TOKEN}\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        return prompt, _ACTION_TOKEN * self.config.chunk_size

    def _get_text_prompt(self, instruction: str, kind: str) -> str:
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nObservation:"
        for camera_name in self.config.camera_key_mapping.values():
            label = _CAMERA_LABELS.get(camera_name, camera_name.replace("_", " "))
            prompt += f" {label}: <|vision_start|>{_IMAGE_TOKEN}<|vision_end|>"
        if kind == "subtask":
            # The wording comes from the checkpoint's recipe (config.recipe),
            # token-exact with the trained subtask turn.
            prompt += f"\nInstruction: {self.build_prompt('subtask', task=instruction)}"
        else:
            prompt += f"\nQuestion: {instruction}\n"
        return prompt + "<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def _message_content(message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        elif content is None:
            text = ""
        else:
            text = str(content)

        tool_calls = message.get("tool_calls")
        if tool_calls:
            say_texts = []
            for call in tool_calls:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                if function.get("name") != "say":
                    continue
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (TypeError, ValueError):
                        arguments = {}
                if isinstance(arguments, dict) and arguments.get("text"):
                    say_texts.append(str(arguments["text"]))
            suffix = "".join(f"<say>{value}</say>" for value in say_texts)
            if suffix:
                text = f"{text}\n{suffix}" if text else suffix
        return text

    def _observation_prompt(self) -> str:
        prompt = "Observation:"
        for camera_name in self.config.camera_key_mapping.values():
            label = _CAMERA_LABELS.get(camera_name, camera_name.replace("_", " "))
            prompt += f" {label}: <|vision_start|>{_IMAGE_TOKEN}<|vision_end|>"
        return prompt

    def _format_recipe_prompt(
        self,
        messages: list[dict[str, Any]],
        message_streams: list[str | None],
        target_message_indices: list[int],
        task: str,
    ) -> tuple[str, str, bool]:
        if len(messages) != len(message_streams):
            raise ValueError("Recipe messages and message_streams must have equal length.")
        invalid_targets = [index for index in target_message_indices if not 0 <= index < len(messages)]
        if invalid_targets:
            raise ValueError(f"Recipe target indices are out of range: {invalid_targets}.")

        predict_actions = any(stream == "low_level" for stream in message_streams)
        if predict_actions and not target_message_indices:
            prefix, postfix = self._get_flow_prompt(task)
            return prefix, postfix, True

        target_indices = set(target_message_indices)
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        observation_injected = False
        for index, message in enumerate(messages):
            role = str(message.get("role", "user"))
            content = self._message_content(message)
            if _TEXT_TARGET_START in content or _TEXT_TARGET_END in content:
                raise ValueError("Recipe content contains a reserved Wall text-target marker.")
            if role == "user" and not observation_injected:
                content = f"{self._observation_prompt()}\n{content}"
                observation_injected = True

            prompt += f"<|im_start|>{role}\n"
            if index in target_indices:
                prompt += f"{_TEXT_TARGET_START}{content}<|im_end|>{_TEXT_TARGET_END}\n"
            else:
                prompt += f"{content}<|im_end|>\n"

        if not observation_injected:
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{self._observation_prompt()}<|im_end|>\n"
                + prompt.removeprefix("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
            )

        postfix = ""
        if predict_actions:
            prompt += (
                "<|im_start|>user\n"
                f"Instruction: {task}\n"
                f"Predict the next action in robot action.\nProprioception: {_PROPRIO_TOKEN}\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            postfix = _ACTION_TOKEN * self.config.chunk_size
        return prompt, postfix, predict_actions

    @staticmethod
    def _extract_text_target_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
        clean_parts: list[str] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        clean_length = 0
        while True:
            start = text.find(_TEXT_TARGET_START, cursor)
            if start < 0:
                suffix = text[cursor:]
                clean_parts.append(suffix)
                clean_length += len(suffix)
                break
            prefix = text[cursor:start]
            clean_parts.append(prefix)
            clean_length += len(prefix)
            payload_start = start + len(_TEXT_TARGET_START)
            end = text.find(_TEXT_TARGET_END, payload_start)
            if end < 0:
                raise ValueError("Wall text-target start marker has no matching end marker.")
            payload = text[payload_start:end]
            span_start = clean_length
            clean_parts.append(payload)
            clean_length += len(payload)
            spans.append((span_start, clean_length))
            cursor = end + len(_TEXT_TARGET_END)
        if _TEXT_TARGET_END in "".join(clean_parts):
            raise ValueError("Wall text-target end marker has no matching start marker.")
        return "".join(clean_parts), spans

    @staticmethod
    def _resize_image(image: np.ndarray, target: int = 448) -> Image.Image:
        pil_image = Image.fromarray(image)
        width, height = pil_image.size
        if width > height:
            resized = (target, int(target * height / width))
        else:
            resized = (int(target * width / height), target)
        return pil_image.resize(resized)

    @staticmethod
    def _expand_image_tokens(texts: list[str], image_grid_thw: Tensor, merge_size: int) -> list[str]:
        expanded = list(texts)
        image_index = 0
        for text_index, text in enumerate(expanded):
            while _IMAGE_TOKEN in text:
                token_count = int(image_grid_thw[image_index].prod() // merge_size**2)
                text = text.replace(_IMAGE_TOKEN, "<|wall_image_placeholder|>" * token_count, 1)
                image_index += 1
            expanded[text_index] = text.replace("<|wall_image_placeholder|>", _IMAGE_TOKEN)
        return expanded

    def _construct_model_input(
        self,
        observations: list[dict[str, Any]],
        prefix_list: list[str],
        postfix_list: list[str],
    ) -> dict[str, Any]:
        if self.processor is None or not hasattr(self, "model"):
            raise RuntimeError("Wall-OSS-0.5 model is not loaded.")

        device = next(self.model.parameters()).device
        state = torch.cat([torch.from_numpy(observation["proprioception"]) for observation in observations])
        state_mask = torch.cat(
            [torch.from_numpy(observation["agent_pos_mask"]) for observation in observations]
        ).bool()
        dof_mask = torch.cat(
            [torch.from_numpy(observation["dof_mask"]) for observation in observations]
        ).bool()
        images = []
        for observation in observations:
            for camera_name in self.config.camera_key_mapping.values():
                images.append(self._resize_image(observation[camera_name]))
        image_inputs = self.processor.image_processor(images=images, return_tensors="pt")
        prefix_list = self._expand_image_tokens(
            prefix_list,
            image_inputs["image_grid_thw"],
            int(self.processor.image_processor.merge_size),
        )

        bins = np.linspace(-1, 1, 513)[:-1]
        discrete_state = np.digitize(state.cpu().numpy(), bins=bins) - 1
        discrete_state = discrete_state[:, 0]
        active_state = state_mask[:, 0].cpu().numpy()
        for index in range(len(prefix_list)):
            state_text = " ".join(map(str, discrete_state[index, active_state[index]]))
            prefix_list[index] = prefix_list[index].replace(_PROPRIO_TOKEN, state_text)

        texts = [prefix + postfix for prefix, postfix in zip(prefix_list, postfix_list, strict=True)]
        target_spans: list[list[tuple[int, int]]] = []
        clean_texts = []
        for text in texts:
            clean_text, spans = self._extract_text_target_spans(text)
            clean_texts.append(clean_text)
            target_spans.append(spans)
        has_text_targets = any(target_spans)
        text_inputs = self.processor.tokenizer(
            clean_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_offsets_mapping=has_text_targets,
        )
        input_ids = text_inputs["input_ids"]
        model_inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": text_inputs["attention_mask"].to(device),
            "pixel_values": image_inputs["pixel_values"].to(device),
            "image_grid_thw": image_inputs["image_grid_thw"].to(device),
            "moe_token_types": (input_ids == self.model.action_token_id).to(device),
            "dof_mask": dof_mask.to(device),
        }
        if has_text_targets:
            offsets = text_inputs["offset_mapping"]
            labels = torch.full_like(input_ids, -100)
            for batch_index, spans in enumerate(target_spans):
                for span_start, span_end in spans:
                    token_overlap = (
                        (offsets[batch_index, :, 1] > span_start)
                        & (offsets[batch_index, :, 0] < span_end)
                        & text_inputs["attention_mask"][batch_index].bool()
                    )
                    labels[batch_index, token_overlap] = input_ids[batch_index, token_overlap]
            labels[labels == self.model.action_token_id] = -100
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            model_inputs["text_labels"] = labels.to(device)
        return model_inputs

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _tasks(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        task = batch.get("task")
        if isinstance(task, str):
            return [task] * batch_size
        if (
            isinstance(task, list)
            and len(task) == batch_size
            and all(isinstance(value, str) for value in task)
        ):
            return task
        raise ValueError(f"Expected exactly {batch_size} already-selected task strings.")

    @staticmethod
    def _image_to_numpy(image: Tensor) -> np.ndarray:
        if image.ndim == 4:
            image = image[-1]
        if image.ndim != 3:
            raise ValueError(f"Expected CHW image, got shape {tuple(image.shape)}.")
        image = image.detach().cpu()
        if image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        if image.dtype.is_floating_point:
            image = image.clamp(0, 1).mul(255).round()
        return image.to(torch.uint8).numpy()

    def _build_observations(
        self, batch: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], Tensor, Tensor, list[str]]:
        if self.processor is None or not hasattr(self, "model"):
            raise RuntimeError("Wall-OSS-0.5 model is not loaded.")

        state = batch[OBS_STATE]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2 or state.shape[-1] != self.config.max_state_dim:
            raise ValueError(f"Expected canonical state shaped (B,26), got {tuple(state.shape)}.")
        state_mask = torch.ones_like(state, dtype=torch.bool)
        batch_size = state.shape[0]
        tasks = self._tasks(batch, batch_size)

        action_mask = torch.ones(
            (batch_size, self.config.chunk_size, self.config.max_action_dim),
            device=state.device,
            dtype=torch.bool,
        )
        observations: list[dict[str, Any]] = []
        for row in range(batch_size):
            observation = {
                "proprioception": state[row : row + 1].unsqueeze(1).detach().cpu().numpy(),
                "agent_pos_mask": state_mask[row : row + 1].unsqueeze(1).detach().cpu().numpy(),
                "dof_mask": action_mask[row : row + 1].detach().cpu().numpy(),
            }
            for lerobot_key, author_slot in self.config.camera_key_mapping.items():
                observation[author_slot] = self._image_to_numpy(batch[lerobot_key][row])
            observations.append(observation)
        return observations, state_mask, action_mask, tasks

    def _build_model_inputs(self, batch: dict[str, Any]) -> tuple[dict[str, Any], Tensor, Tensor, list[str]]:
        observations, state_mask, action_mask, tasks = self._build_observations(batch)

        prefix_list, postfix_list = [], []
        for task in tasks:
            prefix, postfix = self._get_flow_prompt(task)
            prefix_list.append(prefix)
            postfix_list.append(postfix)
        inputs = self._construct_model_input(observations, prefix_list, postfix_list)
        return inputs, state_mask, action_mask, tasks

    @staticmethod
    def _batched_recipe_field(value: Any, batch_size: int, field_name: str) -> list[list[Any]]:
        if not isinstance(value, list):
            raise TypeError(f"{field_name} must be a list.")
        if len(value) == batch_size and all(isinstance(row, list) for row in value):
            return value
        if batch_size == 1:
            return [value]
        raise ValueError(f"Expected {field_name} for exactly {batch_size} samples.")

    def _build_recipe_model_inputs(
        self,
        batch: dict[str, Any],
    ) -> tuple[dict[str, Any], Tensor, Tensor, list[str], Tensor]:
        observations, state_mask, action_mask, tasks = self._build_observations(batch)
        batch_size = len(observations)
        messages_batch = self._batched_recipe_field(batch["messages"], batch_size, "messages")
        streams_batch = self._batched_recipe_field(
            batch.get("message_streams", []),
            batch_size,
            "message_streams",
        )
        targets_batch = self._batched_recipe_field(
            batch.get("target_message_indices", []),
            batch_size,
            "target_message_indices",
        )

        prefix_list = []
        postfix_list = []
        predict_actions = []
        for messages, streams, targets, task in zip(
            messages_batch,
            streams_batch,
            targets_batch,
            tasks,
            strict=True,
        ):
            prefix, postfix, predicts_action = self._format_recipe_prompt(
                messages,
                streams,
                targets,
                task,
            )
            prefix_list.append(prefix)
            postfix_list.append(postfix)
            predict_actions.append(predicts_action)
        inputs = self._construct_model_input(observations, prefix_list, postfix_list)
        return (
            inputs,
            state_mask,
            action_mask,
            tasks,
            torch.tensor(predict_actions, device=action_mask.device, dtype=torch.bool),
        )

    def _build_text_model_inputs(
        self,
        batch: dict[str, Any],
        *,
        kind: str,
        user_text: str | list[str] | None,
    ) -> dict[str, Any]:
        observations, _, _, tasks = self._build_observations(batch)
        if user_text is None:
            instructions = (
                ["Describe what you observe in the current scene."] * len(tasks)
                if kind == "caption"
                else tasks
            )
        elif isinstance(user_text, str):
            instructions = [user_text] * len(tasks)
        elif len(user_text) == len(tasks) and all(isinstance(value, str) for value in user_text):
            instructions = user_text
        else:
            raise ValueError(f"Expected exactly {len(tasks)} text prompts.")
        prefixes = [self._get_text_prompt(instruction, kind) for instruction in instructions]
        return self._construct_model_input(observations, prefixes, [""] * len(prefixes))

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

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if "messages" in batch:
            inputs, _, action_mask, _, predict_actions = self._build_recipe_model_inputs(batch)
        else:
            inputs, _, action_mask, _ = self._build_model_inputs(batch)
            predict_actions = torch.ones(
                action_mask.shape[0],
                device=action_mask.device,
                dtype=torch.bool,
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
        inputs, _, _, _ = self._build_model_inputs(batch)
        output = self.model.generate_flow_action(
            action_horizon=self.config.chunk_size,
            action_dim=self.config.max_action_dim,
            num_inference_timesteps=self.config.num_inference_timesteps,
            **inputs,
        )
        return output["predict_action"]

    @torch.no_grad()
    def generate_text(
        self,
        batch: dict[str, Any],
        *,
        kind: str = "subtask",
        user_text: str | None = None,
    ) -> str:
        """Single-sample text generation, as the interactive language runtime calls it."""
        outputs = self.generate_texts(
            batch,
            kind=kind,
            user_text=user_text,
            temperature=self.config.text_temperature,
            top_p=self.config.text_top_p,
        )
        if len(outputs) != 1:
            raise ValueError(f"The interactive runtime expected one Wall text output, got {len(outputs)}.")
        return outputs[0]

    def generate_texts(
        self,
        batch: dict[str, Any],
        *,
        kind: str = "vqa",
        user_text: str | list[str] | None = None,
        max_new_tokens: int = 100,
        min_new_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[str]:
        """Generate Wall text for VQA, grounding, or language subtasks over a batch."""
        if self.processor is None or not hasattr(self, "model"):
            raise RuntimeError("Wall-OSS-0.5 model is not loaded.")
        if kind not in {"vqa", "caption", "grounding", "text", "subtask"}:
            raise ValueError("kind must be one of: 'vqa', 'caption', 'grounding', 'text', 'subtask'.")
        inputs = self._build_text_model_inputs(batch, kind=kind, user_text=user_text)
        output_ids = self.model.generate_text(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        return [
            text.strip()
            for text in self.processor.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        ]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        if not self._queues[ACTION]:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self._queues[ACTION].extend(chunk[:, : self.config.n_action_steps].transpose(0, 1))
        return self._queues[ACTION].popleft()

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        if self._source_checkpoint is None or not hasattr(self, "model"):
            raise RuntimeError("Load the model before saving a Wall-OSS-0.5 checkpoint.")
        save_directory.mkdir(parents=True, exist_ok=True)

        if state_dict is None:
            model_state = dict(self.model.state_dict())
        else:
            model_state = {
                key.removeprefix("model."): value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }

        # The public checkpoint stores the tied vocabulary matrix only under
        # ``model.embed_tokens.weight``.
        if "model.embed_tokens.weight" in model_state and "lm_head.weight" in model_state:
            model_state.pop("lm_head.weight")
        total_bytes = sum(value.numel() * value.element_size() for value in model_state.values())
        save_torch_state_dict(
            model_state,
            str(save_directory),
            max_shard_size=max(total_bytes, 1),
        )

        for name in _CHECKPOINT_METADATA:
            shutil.copy2(self._source_checkpoint / name, save_directory / name)

        saved_source = self.config.pretrained_name_or_path
        self.config.pretrained_name_or_path = "."
        try:
            self.config._save_pretrained(save_directory)
        finally:
            self.config.pretrained_name_or_path = saved_source

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs: Any,
    ) -> WallOSS05Policy:
        dataset_stats = kwargs.pop("dataset_stats", None)
        dataset_meta = kwargs.pop("dataset_meta", None)
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        if not isinstance(config, WallOSS05Config):
            raise TypeError("Converted checkpoint config is not type='wall_oss_05'.")
        config.pretrained_name_or_path = str(pretrained_name_or_path)
        policy = cls(config, dataset_stats=dataset_stats, dataset_meta=dataset_meta)
        policy.eval()
        return policy
