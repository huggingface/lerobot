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

"""Native Being-H0.5 vision-language-action model.

Adapted from BeingBeyond's Apache-2.0 Being-H0.5 implementation. Only the
released Qwen3-MoT, flow-matching, and zero-strength MPG checkpoint path is
implemented here; standard vision and Qwen3 primitives are provided by
Transformers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.attention.flex_attention import and_masks, create_block_mask, or_masks

from .modeling_mot import BeingH05Qwen3ForCausalLM

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
