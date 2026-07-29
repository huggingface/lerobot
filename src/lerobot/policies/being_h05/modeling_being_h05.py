#!/usr/bin/env python

# Copyright 2026 BeingBeyond Ltd. and/or its affiliates.
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

"""Native Being-H0.5 policy and vision-language-action model."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_being_h05 import BeingH05Config

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3MLP,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
        apply_rotary_pos_emb,
    )
else:
    # Keep the policy package importable without its optional Transformers extra.
    Qwen3Attention = nn.Module
    Qwen3MLP = None
    Qwen3RMSNorm = None
    Qwen3RotaryEmbedding = None
    apply_rotary_pos_emb = None

_compiled_flex_attention = torch.compile(flex_attention)


class BeingH05PackedMoTAttention(Qwen3Attention):
    """Packed Qwen3 attention with a separate action-expert projection stream."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        expert_config = config.expert_config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.q_norm_mot_gen = Qwen3RMSNorm(self.head_dim, eps=expert_config.rms_norm_eps)
        self.k_norm_mot_gen = Qwen3RMSNorm(self.head_dim, eps=expert_config.rms_norm_eps)
        self.q_proj_mot_gen = nn.Linear(
            expert_config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj_mot_gen = nn.Linear(
            expert_config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj_mot_gen = nn.Linear(
            expert_config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj_mot_gen = nn.Linear(
            self.num_heads * self.head_dim,
            expert_config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        packed_sequence_und: Tensor,
        packed_sequence_gen: Tensor,
        sample_lens: Sequence[int],
        attention_mask: BlockMask,
        packed_position_embeddings: tuple[Tensor, Tensor],
        packed_und_token_indexes: Tensor,
        packed_gen_token_indexes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        total_length = packed_sequence_und.shape[0] + packed_sequence_gen.shape[0]
        dtype, device = packed_sequence_und.dtype, packed_sequence_und.device

        queries = torch.zeros((total_length, self.num_heads * self.head_dim), dtype=dtype, device=device)
        keys = torch.zeros(
            (total_length, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device
        )
        values = torch.zeros_like(keys)

        queries[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        queries[packed_gen_token_indexes] = self.q_proj_mot_gen(packed_sequence_gen)
        keys[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        keys[packed_gen_token_indexes] = self.k_proj_mot_gen(packed_sequence_gen)
        values[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        values[packed_gen_token_indexes] = self.v_proj_mot_gen(packed_sequence_gen)

        queries = queries.view(-1, self.num_heads, self.head_dim)
        keys = keys.view(-1, self.num_key_value_heads, self.head_dim)
        values = values.view(-1, self.num_key_value_heads, self.head_dim)

        normalized_queries = torch.zeros_like(queries)
        normalized_keys = torch.zeros_like(keys)
        normalized_queries[packed_und_token_indexes] = self.q_norm(queries[packed_und_token_indexes])
        normalized_queries[packed_gen_token_indexes] = self.q_norm_mot_gen(queries[packed_gen_token_indexes])
        normalized_keys[packed_und_token_indexes] = self.k_norm(keys[packed_und_token_indexes])
        normalized_keys[packed_gen_token_indexes] = self.k_norm_mot_gen(keys[packed_gen_token_indexes])

        cos, sin = packed_position_embeddings
        normalized_queries, normalized_keys = apply_rotary_pos_emb(
            normalized_queries, normalized_keys, cos, sin, unsqueeze_dim=1
        )

        padding = sum(sample_lens) - total_length
        normalized_queries = _pad_sequence(normalized_queries.permute(1, 0, 2), padding)
        normalized_keys = _pad_sequence(normalized_keys.permute(1, 0, 2), padding)
        values = _pad_sequence(values.permute(1, 0, 2), padding)
        attention_output = _compiled_flex_attention(
            normalized_queries.unsqueeze(0),
            normalized_keys.unsqueeze(0),
            values.unsqueeze(0),
            enable_gqa=True,
            block_mask=attention_mask,
        )
        attention_output = attention_output[0, :, :total_length].transpose(0, 1)
        attention_output = attention_output.reshape(total_length, self.num_heads * self.head_dim)

        return (
            self.o_proj(attention_output[packed_und_token_indexes]),
            self.o_proj_mot_gen(attention_output[packed_gen_token_indexes]),
        )


class BeingH05MoTDecoderLayer(nn.Module):
    """One Qwen3 understanding layer paired with one smaller action-expert layer."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        expert_config = config.expert_config
        self.self_attn = BeingH05PackedMoTAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.mlp_mot_gen = Qwen3MLP(expert_config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(expert_config.hidden_size, eps=expert_config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(
            expert_config.hidden_size, eps=expert_config.rms_norm_eps
        )

    def forward(
        self,
        packed_sequence_und: Tensor,
        packed_sequence_gen: Tensor,
        sample_lens: Sequence[int],
        attention_mask: BlockMask,
        packed_position_embeddings: tuple[Tensor, Tensor],
        packed_und_token_indexes: Tensor,
        packed_gen_token_indexes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual_und, residual_gen = packed_sequence_und, packed_sequence_gen
        hidden_und, hidden_gen = self.self_attn(
            self.input_layernorm(packed_sequence_und),
            self.input_layernorm_mot_gen(packed_sequence_gen),
            sample_lens,
            attention_mask,
            packed_position_embeddings,
            packed_und_token_indexes,
            packed_gen_token_indexes,
        )
        packed_sequence_und = residual_und + hidden_und
        packed_sequence_gen = residual_gen + hidden_gen
        packed_sequence_und = packed_sequence_und + self.mlp(
            self.post_attention_layernorm(packed_sequence_und)
        )
        packed_sequence_gen = packed_sequence_gen + self.mlp_mot_gen(
            self.post_attention_layernorm_mot_gen(packed_sequence_gen)
        )
        return packed_sequence_und, packed_sequence_gen


class BeingH05Qwen3MoTModel(nn.Module):
    """Checkpoint-compatible packed Qwen3-MoT decoder."""

    def __init__(self, config):
        super().__init__()
        expert_config = config.expert_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            BeingH05MoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = Qwen3RMSNorm(expert_config.hidden_size, eps=expert_config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    def forward(
        self,
        packed_sequence_und: Tensor,
        packed_sequence_gen: Tensor,
        sample_lens: Sequence[int],
        attention_mask: BlockMask,
        packed_position_ids: Tensor,
        packed_und_token_indexes: Tensor,
        packed_gen_token_indexes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cos, sin = self.rotary_emb(packed_sequence_und, packed_position_ids.unsqueeze(0))
        position_embeddings = (cos.squeeze(0), sin.squeeze(0))
        for layer in self.layers:
            packed_sequence_und, packed_sequence_gen = layer(
                packed_sequence_und,
                packed_sequence_gen,
                sample_lens,
                attention_mask,
                position_embeddings,
                packed_und_token_indexes,
                packed_gen_token_indexes,
            )
        return self.norm(packed_sequence_und), self.norm_mot_gen(packed_sequence_gen)


class BeingH05Qwen3ForCausalLM(nn.Module):
    """Minimal CausalLM owner retaining the released checkpoint's tensor names."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BeingH05Qwen3MoTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, **kwargs) -> tuple[Tensor, Tensor]:
        return self.model(**kwargs)


def _pad_sequence(sequence: Tensor, padding: int) -> Tensor:
    if padding == 0:
        return sequence
    return torch.cat([sequence, sequence.new_zeros(sequence.shape[0], padding, sequence.shape[2])], dim=1)


SYSTEM_MESSAGE = (
    "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, "
    "是一个有用无害的人工智能助手。"
)


class InternVLConnector(nn.Module):
    def __init__(self, llm_hidden_size: int, vit_hidden_size: int, downsample_ratio: float):
        super().__init__()
        expanded_size = vit_hidden_size * int(1 / downsample_ratio) ** 2
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(expanded_size),
            nn.Linear(expanded_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp1(features)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.layer2(F.relu(self.layer1(features)))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        frequencies = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(frequencies), torch.cos(frequencies)], dim=-1)


class ActionEncoder(nn.Module):
    """Embed noisy action tokens together with their flow timesteps."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(action_dim, hidden_size)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: Tensor, timesteps: Tensor) -> Tensor:
        if actions.ndim == 3:
            batch_size, chunk_size = actions.shape[:2]
            if timesteps.shape != (batch_size,):
                raise ValueError(
                    f"Expected one timestep per sample, got actions={actions.shape}, "
                    f"timesteps={timesteps.shape}."
                )
            timesteps = timesteps[:, None].expand(-1, chunk_size)
        elif actions.ndim == 2:
            if timesteps.shape != (actions.shape[0],):
                raise ValueError(
                    f"Expected one timestep per packed action, got actions={actions.shape}, "
                    f"timesteps={timesteps.shape}."
                )
        else:
            raise ValueError(f"Actions must be 2D or 3D, got {actions.ndim}D.")

        action_embedding = self.W1(actions)
        time_embedding = self.pos_encoding(timesteps).to(action_embedding.dtype)
        hidden = self.W2(torch.cat([action_embedding, time_embedding], dim=-1))
        return self.W3(hidden * torch.sigmoid(hidden))


class SlicedWassersteinDistance(nn.Module):
    def __init__(self, num_projections: int):
        super().__init__()
        self.num_projections = num_projections

    def forward(self, first: Tensor, second: Tensor, seed: int | None = None) -> Tensor:
        feature_dim = first.shape[-1]
        generator = torch.Generator(device=first.device).manual_seed(seed) if seed is not None else None
        transport_cost = 0.0
        for _ in range(self.num_projections):
            direction = torch.randn(
                feature_dim,
                device=first.device,
                dtype=first.dtype,
                generator=generator,
            )
            direction = direction / torch.norm(direction)
            first_sorted = torch.sort(torch.matmul(first, direction).reshape(-1))[0]
            second_sorted = torch.sort(torch.matmul(second, direction).reshape(-1))[0]
            min_length = min(first_sorted.shape[0], second_sorted.shape[0])
            transport_cost += torch.mean((first_sorted[:min_length] - second_sorted[:min_length]) ** 2)
        return transport_cost / self.num_projections


class MPGEnhancement(nn.Module):
    """Manifold-preserving gating used by the RoboCasa checkpoint."""

    def __init__(
        self,
        obs_feature_dim: int,
        action_feature_dim: int,
        embedding_dim: int,
        num_projections: int,
        lambda_strength: float,
        use_stop_gradient: bool,
        gate_temperature: float,
    ):
        super().__init__()
        self.lambda_strength = lambda_strength
        self.use_stop_gradient = use_stop_gradient
        self.gate_temperature = gate_temperature
        self.obs_proj = nn.Linear(obs_feature_dim, embedding_dim)
        self.action_proj = nn.Linear(action_feature_dim, embedding_dim)
        self.obs_layer_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.action_layer_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.enhancement_proj = nn.Linear(embedding_dim, obs_feature_dim)
        self.sliced_wasserstein = SlicedWassersteinDistance(num_projections)

    def forward(
        self,
        obs_features: Tensor,
        action_features: Tensor,
        return_gate: bool = False,
        return_metrics: bool = False,
    ):
        if self.lambda_strength == 0:
            zero = torch.tensor(0.0, device=obs_features.device)
            if return_metrics:
                return obs_features, torch.tensor(1.0, device=obs_features.device), zero
            if return_gate:
                return obs_features, torch.tensor(1.0, device=obs_features.device), None
            return obs_features

        observation_embedding = self.obs_proj(obs_features)
        action_embedding = self.action_proj(action_features)
        if observation_embedding.shape[0] != action_embedding.shape[0]:
            action_embedding = action_embedding.mean(dim=0, keepdim=True)
        normalized_observation = self.obs_layer_norm(observation_embedding)
        normalized_action = self.action_layer_norm(action_embedding)
        aligned_action = normalized_action.mean(dim=1, keepdim=True).expand(
            -1, normalized_observation.shape[1], -1
        )
        transport_cost = self.sliced_wasserstein(normalized_observation, aligned_action)
        gate = torch.exp(-transport_cost / self.gate_temperature)
        gated_observation = observation_embedding * (gate.detach() if self.use_stop_gradient else gate)
        enhanced = obs_features + self.lambda_strength * self.enhancement_proj(gated_observation)
        if return_metrics:
            return enhanced, gate, transport_cost
        if return_gate:
            return enhanced, gate, None
        return enhanced


class BeingH05Model(nn.Module):
    """Native owner of the released Being-H0.5 checkpoint architecture."""

    def __init__(
        self,
        language_model: BeingH05Qwen3ForCausalLM,
        vit_model: nn.Module,
        connector: InternVLConnector,
        author_config: dict[str, Any],
        num_inference_steps: int,
    ):
        super().__init__()
        llm_config = language_model.config
        expert_config = llm_config.expert_config
        self.vit_model = vit_model
        self.language_model = language_model
        self.connector = connector
        self.hidden_size = llm_config.hidden_size
        self.action_hidden_size = expert_config.hidden_size
        self.action_chunk_length = author_config["action_chunk_length"]
        self.unified_state_dim = 200
        self.unified_action_dim = 200
        self.select_layer = author_config.get("select_layer", -1)
        self.downsample_ratio = author_config.get("downsample_ratio", 0.5)
        self.system_message = author_config.get("system_message") or SYSTEM_MESSAGE
        self.num_timestep_buckets = author_config.get("num_timestep_buckets", 1000)
        self.noise_s = author_config.get("noise_s", 0.999)
        self.num_inference_timesteps = num_inference_steps
        self.mpg_refinement_iters = author_config.get("mpg_refinement_iters", 1)
        self.noise_beta_alpha = author_config.get("noise_beta_alpha", 1.5)
        self.noise_beta_beta = author_config.get("noise_beta_beta", 1.0)

        self.proprio_encoder_robot = SimpleMLP(
            self.unified_state_dim, self.action_hidden_size, self.action_hidden_size
        )
        self.action_encoder = ActionEncoder(self.unified_action_dim, self.action_hidden_size)
        self.action_decoder = SimpleMLP(
            self.action_hidden_size, self.action_hidden_size, self.unified_action_dim
        )

        self.use_mpg = author_config.get("use_mpg", False)
        if self.use_mpg:
            self.action_to_vlm_proj = nn.Linear(self.action_hidden_size, self.hidden_size)
            self.vlm_to_action_proj = nn.Linear(self.hidden_size, self.action_hidden_size)
            self.mpg = MPGEnhancement(
                obs_feature_dim=self.hidden_size,
                action_feature_dim=self.action_hidden_size,
                embedding_dim=self.hidden_size,
                num_projections=author_config.get("mpg_num_projections", 32),
                lambda_strength=author_config.get("mpg_lambda", 0.0),
                use_stop_gradient=author_config.get("mpg_use_stop_gradient", True),
                gate_temperature=author_config.get("mpg_gate_temperature", 2.0),
            )
        else:
            self.action_to_vlm_proj = None
            self.vlm_to_action_proj = None
            self.mpg = None

        self._init_action_weights()

    def _init_action_weights(self) -> None:
        module_names = (
            "action_decoder",
            "proprio_encoder_robot",
            "action_encoder",
            "action_to_vlm_proj",
            "vlm_to_action_proj",
        )
        for name, parameter in self.named_parameters():
            if not any(module_name in name for module_name in module_names):
                continue
            if name.endswith("weight"):
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.normal_(parameter, mean=1.0, std=0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(parameter)

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        distribution = torch.distributions.Beta(self.noise_beta_alpha, self.noise_beta_beta)
        sample = distribution.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: Tensor,
        packed_text_indexes: Tensor,
        sample_lens: Sequence[int],
        packed_position_ids: Tensor,
        split_lens: Sequence[int],
        attn_modes: Sequence[str],
        packed_vit_tokens: Tensor,
        packed_vit_token_indexes: Tensor,
        padded_action: Tensor,
        padded_action_mask: Tensor,
        packed_action_indexes: Tensor,
        padded_state: Tensor,
        packed_state_indexes: Tensor,
        **kwargs,
    ) -> dict[str, Tensor]:
        del kwargs
        device = packed_text_ids.device
        text_embedding = self.language_model.get_input_embeddings()(packed_text_ids)
        packed_sequence = text_embedding.new_zeros((sequence_length, self.hidden_size))
        packed_sequence_gen = text_embedding.new_zeros((sequence_length, self.action_hidden_size))
        packed_sequence[packed_text_indexes] = text_embedding
        packed_sequence[packed_vit_token_indexes] = self.extract_feature(packed_vit_tokens).reshape(
            -1, self.hidden_size
        )

        batch_size = padded_state.shape[0]
        packed_sequence_gen[packed_state_indexes] = self.proprio_encoder_robot(padded_state)
        action_target = padded_action
        noise = torch.randn(action_target.shape, device=device, dtype=action_target.dtype)
        timestep = self.sample_time(batch_size, device, action_target.dtype)
        target_3d = action_target.view(batch_size, self.action_chunk_length, -1)
        noise_3d = noise.view(batch_size, self.action_chunk_length, -1)
        noisy_actions = (1 - timestep[:, None, None]) * noise_3d + timestep[:, None, None] * target_3d
        velocity_target = action_target - noise
        timestep_bucket = (timestep * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_actions, timestep_bucket).reshape(
            batch_size * self.action_chunk_length, -1
        )
        packed_sequence_gen[packed_action_indexes] = action_features
        if self.use_mpg and self.mpg is not None and self.mpg.lambda_strength > 0:
            clean_timestep = torch.zeros(batch_size, dtype=torch.long, device=device)
            clean_action_embedding = self.action_encoder(target_3d, clean_timestep)
            state_features = packed_sequence[packed_state_indexes]
            projected_action = self.action_to_vlm_proj(packed_sequence_gen[packed_action_indexes])
            suffix = torch.cat([state_features, projected_action], dim=0).unsqueeze(0)
            enhanced_suffix, gate, transport_cost = self.mpg(
                suffix,
                clean_action_embedding,
                return_gate=True,
                return_metrics=True,
            )
            enhanced_suffix = enhanced_suffix.squeeze(0)
            state_count = len(packed_state_indexes)
            packed_sequence[packed_state_indexes] = enhanced_suffix[:state_count].to(packed_sequence.dtype)
            packed_sequence_gen[packed_action_indexes] = self.vlm_to_action_proj(
                enhanced_suffix[state_count:]
            ).to(packed_sequence_gen.dtype)
            self.last_mpg_gate = gate.detach()
            self.last_mpg_transport_cost = transport_cost.detach()

        packed_und_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes])
        packed_gen_indexes = torch.cat([packed_state_indexes, packed_action_indexes])
        attention_mask = _create_attention_mask(
            sample_lens,
            sample_lens,
            attn_modes,
            self.language_model.config.num_attention_heads,
            device,
        )
        hidden_und, hidden_gen = self.language_model(
            packed_sequence_und=packed_sequence[packed_und_indexes],
            packed_sequence_gen=packed_sequence_gen[packed_gen_indexes],
            sample_lens=split_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            packed_und_token_indexes=packed_und_indexes,
            packed_gen_token_indexes=packed_gen_indexes,
        )
        del hidden_und
        action_hidden = hidden_gen[len(packed_state_indexes) :]
        predicted_velocity = self.action_decoder(
            action_hidden.reshape(batch_size, self.action_chunk_length, -1)
        )
        masked_loss = (
            F.mse_loss(
                predicted_velocity.reshape(batch_size * self.action_chunk_length, -1),
                velocity_target,
                reduction="none",
            )
            * padded_action_mask.float()
        )
        action_loss = masked_loss.sum() / (padded_action_mask.sum() + 1e-8)
        return {"action_loss": action_loss, "und_loss": torch.tensor(0.0, device=device)}

    @torch.no_grad()
    def get_action(
        self,
        sequence_length: int,
        packed_text_ids: Tensor,
        packed_text_indexes: Tensor,
        sample_lens: Sequence[int],
        packed_position_ids: Tensor,
        split_lens: Sequence[int],
        attn_modes: Sequence[str],
        packed_vit_tokens: Tensor,
        packed_vit_token_indexes: Tensor,
        packed_action_indexes: Tensor,
        padded_state: Tensor,
        packed_state_indexes: Tensor,
        **kwargs,
    ) -> dict[str, Tensor]:
        del split_lens, kwargs
        self.eval()
        device = packed_text_ids.device
        text_embedding = self.language_model.get_input_embeddings()(packed_text_ids)
        packed_sequence = text_embedding.new_zeros((sequence_length, self.hidden_size))
        packed_sequence_gen = text_embedding.new_zeros((sequence_length, self.action_hidden_size))
        packed_sequence[packed_text_indexes] = text_embedding
        packed_sequence[packed_vit_token_indexes] = self.extract_feature(packed_vit_tokens).reshape(
            -1, self.hidden_size
        )
        packed_sequence_gen[packed_state_indexes] = self.proprio_encoder_robot(
            padded_state.to(text_embedding.dtype)
        )

        sample_lens = list(sample_lens)
        attention_mask = _create_attention_mask(
            sample_lens,
            sample_lens,
            attn_modes,
            self.language_model.config.num_attention_heads,
            device,
        )
        packed_und_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes])
        packed_gen_indexes = torch.cat([packed_state_indexes, packed_action_indexes])
        packed_sequence_und = packed_sequence[packed_und_indexes]
        base_sequence_gen = packed_sequence_gen.clone()

        batch_size = 1
        action_shape = (batch_size, self.action_chunk_length, self.unified_action_dim)
        # Retain the author solver's initial draw before its baseline iteration.
        actions = torch.randn(action_shape, device=device, dtype=text_embedding.dtype)
        dt = 1.0 / self.num_inference_timesteps
        use_mpg_refinement = (
            self.use_mpg
            and self.mpg is not None
            and self.mpg.lambda_strength > 0
            and self.mpg_refinement_iters > 0
        )
        total_iterations = 1 + (self.mpg_refinement_iters if use_mpg_refinement else 0)
        predicted_action_embedding = None
        for iteration in range(total_iterations):
            actions = torch.randn(action_shape, device=device, dtype=text_embedding.dtype)
            for step in range(self.num_inference_timesteps):
                timestep = step / float(self.num_inference_timesteps)
                timestep_bucket = int(timestep * self.num_timestep_buckets)
                timesteps = torch.full((batch_size,), timestep_bucket, device=device)
                action_features = self.action_encoder(actions, timesteps).reshape(
                    batch_size * self.action_chunk_length, -1
                )
                current_sequence = packed_sequence.clone()
                current_sequence_gen = base_sequence_gen.clone()
                current_sequence_gen[packed_action_indexes] = action_features
                if iteration > 0 and predicted_action_embedding is not None:
                    state_features = current_sequence[packed_state_indexes]
                    projected_action = self.action_to_vlm_proj(current_sequence_gen[packed_action_indexes])
                    suffix = torch.cat([state_features, projected_action], dim=0).unsqueeze(0)
                    enhanced_suffix = self.mpg(suffix, predicted_action_embedding).squeeze(0)
                    state_count = len(packed_state_indexes)
                    current_sequence[packed_state_indexes] = enhanced_suffix[:state_count].to(
                        current_sequence.dtype
                    )
                    current_sequence_gen[packed_action_indexes] = self.vlm_to_action_proj(
                        enhanced_suffix[state_count:]
                    ).to(current_sequence_gen.dtype)
                _, hidden_gen = self.language_model(
                    packed_sequence_und=packed_sequence_und,
                    packed_sequence_gen=current_sequence_gen[packed_gen_indexes],
                    sample_lens=sample_lens,
                    attention_mask=attention_mask,
                    packed_position_ids=packed_position_ids,
                    packed_und_token_indexes=packed_und_indexes,
                    packed_gen_token_indexes=packed_gen_indexes,
                )
                action_hidden = hidden_gen[len(packed_state_indexes) :]
                velocity = self.action_decoder(
                    action_hidden.reshape(batch_size, self.action_chunk_length, -1)
                )
                actions = actions + dt * velocity
            if use_mpg_refinement and iteration < total_iterations - 1:
                clean_timestep = torch.zeros(batch_size, dtype=torch.long, device=device)
                predicted_action_embedding = self.action_encoder(actions, clean_timestep)
        return {"action_pred": actions.reshape(batch_size * self.action_chunk_length, -1)}

    def extract_feature(self, pixel_values: Tensor) -> Tensor:
        vision_output = self.vit_model(
            pixel_values=pixel_values,
            output_hidden_states=self.select_layer != -1,
            return_dict=True,
        )
        if self.select_layer == -1:
            vision_features = vision_output.last_hidden_state
        else:
            vision_features = vision_output.hidden_states[self.select_layer]
        vision_features = vision_features[:, 1:]
        height = width = int(vision_features.shape[1] ** 0.5)
        vision_features = vision_features.reshape(
            vision_features.shape[0], height, width, vision_features.shape[-1]
        )
        vision_features = self.pixel_shuffle(vision_features, self.downsample_ratio)
        vision_features = vision_features.reshape(vision_features.shape[0], -1, vision_features.shape[-1])
        return self.connector(vision_features)

    @staticmethod
    def pixel_shuffle(features: Tensor, scale_factor: float = 0.5) -> Tensor:
        batch_size, width, height, channels = features.shape
        features = features.view(batch_size, width, int(height * scale_factor), int(channels / scale_factor))
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.view(
            batch_size,
            int(height * scale_factor),
            int(width * scale_factor),
            int(channels / (scale_factor * scale_factor)),
        )
        return features.permute(0, 2, 1, 3).contiguous()


def _create_attention_mask(
    document_lens: Sequence[int],
    split_lens: Sequence[int],
    attn_modes: Sequence[str],
    num_heads: int,
    device: torch.device,
):
    split_ids = []
    for index, (length, mode) in enumerate(zip(split_lens, attn_modes, strict=False)):
        split_ids.extend([index if mode == "full" else -1] * length)
    split_id = torch.tensor(split_ids, dtype=torch.float32, device=device)
    document_id = torch.cat(
        [torch.full((length,), index, dtype=torch.float32) for index, length in enumerate(document_lens)]
    ).to(device)

    def causal_mask(batch, head, query_index, key_index):
        return query_index >= key_index

    def full_mask(batch, head, query_index, key_index):
        return (split_id[query_index] == split_id[key_index]) & (split_id[query_index] >= 0)

    def sample_mask(batch, head, query_index, key_index):
        return document_id[query_index] == document_id[key_index]

    sparse_mask = and_masks(or_masks(causal_mask, full_mask), sample_mask)
    sequence_length = sum(document_lens)
    return create_block_mask(
        sparse_mask,
        B=1,
        H=num_heads,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        BLOCK_SIZE=128,
        _compile=True,
    )


class BeingH05Policy(PreTrainedPolicy):
    """Native LeRobot implementation of the released Being-H0.5 architecture."""

    config_class = BeingH05Config
    name = "being_h05"

    def __init__(
        self,
        config: BeingH05Config,
        tokenizer_path: str | None = None,
        tokenizer_load_revision: str | None = None,
        **kwargs: Any,
    ):
        require_package("transformers", extra="being_h05")
        from transformers import AutoTokenizer, Qwen3Config
        from transformers.models.internvl.configuration_internvl import InternVLVisionConfig
        from transformers.models.internvl.modeling_internvl import InternVLVisionModel

        super().__init__(config)
        config.validate_features()
        self.config = config
        if not config.author_config:
            raise ValueError(
                "author_config is required; load a published LeRobot checkpoint or provide "
                "the released Being-H0.5 config payload."
            )
        author_config = config.author_config
        unsupported = {
            "use_expert": author_config.get("use_expert") is not True,
            "use_flow_matching": author_config.get("use_flow_matching") is not True,
            "Qwen3MoTDecoderLayer": "Qwen3MoTDecoderLayer"
            not in author_config["llm_config"].get("layer_module", ""),
            "training_time_rtc=False": bool(author_config.get("use_training_time_rtc", False)),
        }
        unsupported = [name for name, is_unsupported in unsupported.items() if is_unsupported]
        if unsupported:
            raise ValueError(
                "This native implementation supports the released Being-H0.5 checkpoint path; "
                f"unsupported settings: {', '.join(unsupported)}."
            )
        llm_dict = config.author_config["llm_config"]
        llm_config = Qwen3Config.from_dict(llm_dict)
        if llm_dict.get("expert_config"):
            llm_config.expert_config = Qwen3Config.from_dict(llm_dict["expert_config"])
        llm_config.qk_norm = llm_dict.get("qk_norm", True)
        llm_config.use_mot = llm_dict.get("use_mot", True)
        vit_dict = dict(config.author_config["vit_config"])
        vit_config = InternVLVisionConfig.from_dict(vit_dict)
        vit_config.attention_bias, vit_config.use_qk_norm = (
            vit_dict["qkv_bias"],
            vit_dict["qk_normalization"],
        )
        vit_config.hidden_dropout_prob = vit_config.projection_dropout = vit_dict["dropout"]
        vit_config.layer_scale_init_value = vit_dict["initializer_factor"]
        vit_config._attn_implementation = "eager"
        language_model = BeingH05Qwen3ForCausalLM(llm_config)
        vit_model = InternVLVisionModel(vit_config)
        connector = InternVLConnector(
            llm_hidden_size=llm_config.hidden_size,
            vit_hidden_size=vit_config.hidden_size,
            downsample_ratio=author_config["downsample_ratio"],
        )
        self.model = BeingH05Model(
            language_model,
            vit_model,
            connector,
            author_config,
            num_inference_steps=config.num_inference_steps,
        )
        patch_size = vit_config.patch_size[0]
        self.model.num_image_token = int(
            (author_config["force_image_size"] // patch_size) ** 2 * author_config["downsample_ratio"] ** 2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or config.tokenizer_name,
            revision=tokenizer_load_revision if tokenizer_path else config.tokenizer_revision,
            use_fast=False,
        )
        special = self.tokenizer.convert_tokens_to_ids(
            ["<|im_start|>", "<|im_end|>", "<img>", "</img>", "<|state_start|>", "<|state_end|>"]
        )
        self._bos, self._eos, self._image_start, self._image_end, self._state_start, self._state_end = special
        newline = self.tokenizer.encode("\n")
        if len(newline) != 1:
            raise ValueError("Being-H0.5 checkpoint tokenizer must encode newline as one token.")
        self._newline = newline[0]
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, revision=None, **kwargs):
        kwargs.setdefault("tokenizer_path", str(pretrained_name_or_path))
        kwargs.setdefault("tokenizer_load_revision", revision)
        return super().from_pretrained(pretrained_name_or_path, revision=revision, **kwargs)

    def _save_pretrained(self, save_directory, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        super()._save_pretrained(save_directory, state_dict=state_dict)
        self.tokenizer.save_pretrained(save_directory)

    def reset(self) -> None:
        self._action_queue: deque[torch.Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    def _model_kwargs(self, batch: dict[str, Any]) -> dict[str, Any]:
        kwargs = batch.get("being_h05.model_inputs")
        return kwargs if kwargs is not None else self._pack_model_inputs(batch, training=ACTION in batch)

    def _pack_model_inputs(self, batch: dict[str, Any], training: bool) -> dict[str, Any]:
        states = batch["being_h05.state"]
        pixels = batch["being_h05.pixel_values"]
        prompts = batch["being_h05_prompt"]
        device = states.device
        bsz, views = pixels.shape[:2]
        image_valid = batch.get(
            "being_h05.image_valid",
            torch.ones((bsz, views), dtype=torch.bool, device=device),
        )
        text_ids: list[int] = []
        text_indexes: list[int] = []
        vision_indexes: list[int] = []
        state_indexes: list[int] = []
        action_indexes: list[int] = []
        position_ids: list[int] = []
        sample_lens: list[int] = []
        split_lens: list[int] = []
        attn_modes: list[str] = []
        packed_images: list[torch.Tensor] = []
        cursor = 0
        system_ids = self.tokenizer.encode(f"system\n{self.model.system_message}")
        user_ids = self.tokenizer.encode("user\n")
        assistant_ids = self.tokenizer.encode("assistant\n")
        for sample in range(bsz):
            sample_images = pixels[sample, image_valid[sample]]
            if sample_images.shape[0] == 0:
                raise ValueError("Being-H0.5 requires at least one present camera per sample.")
            packed_images.extend(sample_images.unbind(0))
            num_image_tokens = self.model.num_image_token * sample_images.shape[0]
            sample_start = cursor
            rope = 0
            block = [self._bos, *system_ids, self._eos, self._newline]
            text_ids.extend(block)
            text_indexes.extend(range(cursor, cursor + len(block)))
            position_ids.extend(range(rope, rope + len(block)))
            cursor += len(block)
            rope += len(block)
            split_lens.append(len(block))
            attn_modes.append("causal")

            block_start = cursor
            block = [self._bos, *user_ids, self._image_start]
            text_ids.extend(block)
            text_indexes.extend(range(cursor, cursor + len(block)))
            cursor += len(block)
            vision_indexes.extend(range(cursor, cursor + num_image_tokens))
            cursor += num_image_tokens
            text_ids.extend([self._image_end, self._state_start])
            text_indexes.extend([cursor, cursor + 1])
            cursor += 2
            state_indexes.append(cursor)
            cursor += 1
            instruction = self.tokenizer.encode(prompts[sample])
            tail = [self._state_end, *instruction, self._eos, self._newline]
            text_ids.extend(tail)
            text_indexes.extend(range(cursor, cursor + len(tail)))
            cursor += len(tail)
            content_len = cursor - block_start
            position_ids.extend(range(rope, rope + content_len))
            rope += content_len
            split_lens.append(content_len)
            attn_modes.append("causal")

            block_start = cursor
            block = [self._bos, *assistant_ids]
            text_ids.extend(block)
            text_indexes.extend(range(cursor, cursor + len(block)))
            cursor += len(block)
            action_indexes.extend(range(cursor, cursor + self.config.chunk_size))
            cursor += self.config.chunk_size
            text_ids.append(self._eos)
            text_indexes.append(cursor)
            cursor += 1
            action_len = cursor - block_start
            position_ids.extend(range(rope, rope + action_len))
            split_lens.append(action_len)
            attn_modes.append("causal")
            sample_lens.append(cursor - sample_start)

        padding = (-cursor) % 128
        if padding:
            sample_lens.append(padding)
            split_lens.append(padding)
            attn_modes.append("causal")
        result = {
            "sequence_length": cursor,
            "packed_text_ids": torch.tensor(text_ids, dtype=torch.long, device=device),
            "packed_text_indexes": torch.tensor(text_indexes, dtype=torch.long, device=device),
            "sample_lens": sample_lens,
            "packed_position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
            "split_lens": split_lens,
            "attn_modes": attn_modes,
            "packed_vit_tokens": torch.stack(packed_images).to(device),
            "packed_vit_token_indexes": torch.tensor(vision_indexes, dtype=torch.long, device=device),
            "packed_action_indexes": torch.tensor(action_indexes, dtype=torch.long, device=device),
            "padded_state": states,
            "packed_state_indexes": torch.tensor(state_indexes, dtype=torch.long, device=device),
            "embodiment_ids": torch.full((bsz,), self.config.embodiment_id, dtype=torch.long, device=device),
        }
        if training:
            actions = batch[ACTION]
            result["padded_action"] = actions.reshape(-1, actions.shape[-1])
            valid = batch.get("being_h05.action_valid", torch.ones_like(actions, dtype=torch.bool))
            result["padded_action_mask"] = valid.reshape(-1, valid.shape[-1])
        return result

    def forward(self, batch: dict[str, Any], reduction: str = "mean"):
        output = self.model(**self._model_kwargs(batch))
        if isinstance(output, dict):
            loss = output.get("loss")
            if loss is None:
                loss = output["action_loss"] + output["und_loss"]
        else:
            loss = output.loss
        if reduction == "none" and loss.ndim == 0:
            loss = loss.unsqueeze(0)
        return loss, {"loss": float(loss.detach().mean())}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        if batch["being_h05.state"].shape[0] == 1:
            output = self.model.get_action(**self._model_kwargs(batch), **kwargs)
            chunks = output["action_pred"].reshape(1, self.config.chunk_size, self.config.unified_action_dim)
        else:
            # The released solver operates on one packed sample. Preserve its numerics by
            # evaluating batch elements independently.
            chunk_list = []
            for index in range(batch["being_h05.state"].shape[0]):
                single = {
                    key: (value[index : index + 1] if isinstance(value, torch.Tensor) else [value[index]])
                    for key, value in batch.items()
                }
                output = self.model.get_action(**self._model_kwargs(single), **kwargs)
                chunk_list.append(
                    output["action_pred"].reshape(1, self.config.chunk_size, self.config.unified_action_dim)
                )
            chunks = torch.cat(chunk_list)
        return chunks

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
