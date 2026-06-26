#!/usr/bin/env python

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

from __future__ import annotations

import contextlib
import logging
import math
from collections import deque
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy
from .configuration_eo1 import EO1Config

if TYPE_CHECKING or _transformers_available:
    from transformers.activations import ACT2FN
    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers.utils import torch_compilable_check
else:
    ACT2FN = None
    Qwen2_5_VLForConditionalGeneration = None
    torch_compilable_check = None

logger = logging.getLogger(__name__)


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


class EO1Policy(PreTrainedPolicy):
    """EO1 policy wrapper for LeRobot robot-only training/evaluation."""

    config_class = EO1Config
    name = "eo1"

    def __init__(self, config: EO1Config, **kwargs):
        require_package("transformers", extra="eo1")
        super().__init__(config)
        config.validate_features()
        self.config = config

        if config.pretrained_path is None:
            # Initialize from pretrained VLM
            vlm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.vlm_base,
                dtype=config.dtype,
                attn_implementation=config.attn_implementation,
            )
        else:
            vlm_backbone = Qwen2_5_VLForConditionalGeneration._from_config(
                config.vlm_backbone_config,
                dtype=config.vlm_backbone_config.dtype if config.dtype == "auto" else config.dtype,
            )

        self.model = EO1VisionFlowMatchingModel(config, vlm_backbone)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    @staticmethod
    def _get_model_inputs(batch: dict[str, Tensor], excluded_keys: set[str]) -> dict[str, Tensor]:
        return {key: value for key, value in batch.items() if key not in excluded_keys}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        state = self.prepare_state(batch[OBS_STATE])
        actions = self.prepare_action(batch[ACTION])
        model_inputs = self._get_model_inputs(batch, {OBS_STATE, ACTION})
        loss = self.model(states=state, action=actions, **model_inputs)

        loss_dict = {"loss": loss.item()}
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        states = self.prepare_state(batch[OBS_STATE])
        model_inputs = self._get_model_inputs(batch, {OBS_STATE})
        actions = self.model.sample_actions(states=states, **model_inputs).to(torch.float32)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    def prepare_state(self, state: Tensor) -> Tensor:
        return pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, action: Tensor) -> Tensor:
        return pad_vector(action, self.config.max_action_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def get_optim_params(self) -> dict:
        return self.parameters()


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


class EO1VisionActionProjector(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        activation_layer: str = "linear",
        bias: bool = True,
        device: Any = None,
        dtype: torch.dtype = torch.float32,
    ):
        layers = []
        in_dim = in_channels
        hidden_channels = [in_dim] * (num_layers - 1) + [out_channels]
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias, dtype=dtype, device=device))
            layers.append(ACT2FN[activation_layer])
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias, dtype=dtype, device=device))
        super().__init__(*layers)

    @property
    def dtype(self):
        return self[0].weight.dtype


class EO1VisionFlowMatchingModel(nn.Module):
    def __init__(
        self,
        config: EO1Config,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration | None = None,
    ):
        require_package("transformers", extra="eo1")
        super().__init__()

        self.config = config
        # Preserve the backbone dtype selected at construction time so Qwen's fp32 rotary buffers stay intact.
        self.vlm_backbone = vlm_backbone
        self.hidden_size = self.vlm_backbone.config.text_config.hidden_size
        max_state_dim = config.max_state_dim
        max_action_dim = config.max_action_dim
        self.state_proj = nn.Linear(max_state_dim, self.hidden_size, dtype=torch.float32)
        self.action_in_proj = nn.Linear(max_action_dim, self.hidden_size, dtype=torch.float32)
        self.action_out_proj = EO1VisionActionProjector(
            self.hidden_size,
            max_action_dim,
            config.num_action_layers,
            config.action_act,
            dtype=torch.float32,
        )
        self.action_time_mlp_in = nn.Linear(self.hidden_size * 2, self.hidden_size, dtype=torch.float32)
        self.action_time_mlp_out = nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.float32)
        self.gradient_checkpointing_enabled = False

    def get_input_embeddings(self):
        return self.vlm_backbone.get_input_embeddings()

    def flow_head_autocast_context(self):
        if self.config.force_fp32_autocast:
            return torch.autocast(
                device_type=self.state_proj.weight.device.type,
                enabled=False,
            )
        return contextlib.nullcontext()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the Qwen2.5-VL backbone."""
        self.gradient_checkpointing_enabled = True
        self.vlm_backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Enabled gradient checkpointing for EO1VisionFlowMatchingModel")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the Qwen2.5-VL backbone."""
        self.gradient_checkpointing_enabled = False
        self.vlm_backbone.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing for EO1VisionFlowMatchingModel")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Apply manual gradient checkpointing to EO1 flow-head computations when training."""
        if self.gradient_checkpointing_enabled and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor | None,
        state_features: torch.FloatTensor | None = None,
        action_features: torch.FloatTensor | None = None,
        *,
        state_token_id: int,
        action_token_id: int,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        """Return EO1 state/action placeholder masks, following Qwen's multimodal mask style."""
        if input_ids is None:
            special_state_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(state_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_state_mask = special_state_mask.all(-1)
            special_action_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(action_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_action_mask = special_action_mask.all(-1)
        else:
            special_state_mask = input_ids == state_token_id
            special_action_mask = input_ids == action_token_id

        n_state_tokens = special_state_mask.sum()
        special_state_mask = (
            special_state_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        if state_features is not None:
            torch_compilable_check(
                inputs_embeds[special_state_mask].numel() == state_features.numel(),
                f"State features and state tokens do not match, tokens: {n_state_tokens}, features: {state_features.shape[0]}",
            )

        n_action_tokens = special_action_mask.sum()
        special_action_mask = (
            special_action_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        if action_features is not None:
            torch_compilable_check(
                inputs_embeds[special_action_mask].numel() == action_features.numel(),
                f"Action features and action tokens do not match, tokens: {n_action_tokens}, features: {action_features.shape[0]}",
            )

        return special_state_mask, special_action_mask

    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        states: torch.Tensor,
        *,
        state_token_id: int,
        action_token_id: int,
    ) -> torch.FloatTensor:
        """Embed the EO1 prefix tokens before native Qwen injects multimodal features."""

        # Get the input embeddings for the input IDs
        def input_embed_func(input_ids: torch.LongTensor) -> torch.FloatTensor:
            return self.get_input_embeddings()(input_ids)

        inputs_embeds = self._apply_checkpoint(input_embed_func, input_ids)

        # Project the states to the hidden size
        def state_proj_func(states: torch.Tensor) -> torch.FloatTensor:
            with self.flow_head_autocast_context():
                states = states.to(dtype=self.state_proj.weight.dtype)
                return self.state_proj(states)

        state_embs = self._apply_checkpoint(state_proj_func, states)
        state_mask, _ = self.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            state_features=state_embs,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )
        state_embs = state_embs.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(state_mask, state_embs)
        return inputs_embeds

    def embed_suffix(
        self,
        timestep: torch.Tensor,
        noisy_actions: torch.Tensor,
    ) -> torch.FloatTensor:
        """Embed the suffix"""

        def action_proj_func(noisy_actions: torch.Tensor) -> torch.FloatTensor:
            with self.flow_head_autocast_context():
                noisy_actions = noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
                return self.action_in_proj(noisy_actions)

        action_embs = self._apply_checkpoint(action_proj_func, noisy_actions)
        time_embs = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=action_embs.device,
        )
        time_embs = time_embs.to(dtype=action_embs.dtype)
        time_embs = time_embs[:, None, :].expand_as(action_embs)
        action_time_embs = torch.cat([action_embs, time_embs], dim=2)

        def mlp_func(action_time_embs: torch.Tensor) -> torch.FloatTensor:
            with self.flow_head_autocast_context():
                action_time_embs = action_time_embs.to(dtype=self.action_time_mlp_in.weight.dtype)
                action_time_embs = self.action_time_mlp_in(action_time_embs)
                action_time_embs = F.silu(action_time_embs)
                return self.action_time_mlp_out(action_time_embs)

        action_time_embs = self._apply_checkpoint(mlp_func, action_time_embs)
        return action_time_embs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        states: torch.FloatTensor | None = None,
        action: torch.FloatTensor | None = None,
        action_is_pad: torch.BoolTensor | None = None,
        *,
        state_token_id: int,
        action_token_id: int,
        **kwargs,
    ) -> Tensor:
        """Run the EO1 training forward pass and compute the flow-matching loss."""

        # 1. Build the EO1 prefix with state placeholders resolved.
        inputs_embeds = self.embed_prefix(
            input_ids,
            states=states,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )

        # 2. Sample the diffusion target and replace the action placeholders.
        time = self.sample_time(action.shape[0], inputs_embeds.device)
        noise = self.sample_noise(action.shape, inputs_embeds.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * action
        u_t = noise - action
        action_time_embs = self.embed_suffix(time, x_t)
        _, action_mask = self.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            action_features=action_time_embs,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )
        action_time_embs = action_time_embs.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(action_mask, action_time_embs)

        # 3. Optionally drop padded action tokens from backbone attention.
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        if not self.config.supervise_padding_actions:
            action_is_pad = action_is_pad.to(device=inputs_embeds.device, dtype=torch.bool)
            action_token_mask = action_mask[..., 0]
            action_padding_mask = torch.zeros_like(action_token_mask)
            action_padding_mask = action_padding_mask.masked_scatter(
                action_token_mask,
                action_is_pad.reshape(-1),
            )
            attention_mask = attention_mask.masked_fill(action_padding_mask, 0)

        # 4. Run the Qwen backbone on the fused EO1 sequence.
        def vlm_forward_func(
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None,
            inputs_embeds: torch.FloatTensor,
            pixel_values: torch.Tensor | None,
            image_grid_thw: torch.LongTensor | None,
            mm_token_type_ids: torch.IntTensor | None,
        ) -> torch.FloatTensor:
            outputs = self.vlm_backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            return outputs.last_hidden_state

        hidden_states = self._apply_checkpoint(
            vlm_forward_func,
            input_ids,
            attention_mask,
            inputs_embeds,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
        )
        action_hidden_states = hidden_states[action_mask[..., 0]]

        # 5. Project the action-token hidden states back to the flow target space.
        def action_out_proj_func(action_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
            with self.flow_head_autocast_context():
                action_hidden_states = action_hidden_states.to(dtype=self.action_out_proj.dtype)
                return self.action_out_proj(action_hidden_states)

        v_t = self._apply_checkpoint(action_out_proj_func, action_hidden_states)
        v_t = v_t.reshape(u_t.shape).to(dtype=u_t.dtype)
        losses = F.mse_loss(u_t, v_t, reduction="none")

        # 6. Apply the configured supervision mask and reduce the loss.
        if not self.config.supervise_padding_action_dims:
            original_action_dim = self.config.output_features[ACTION].shape[0]
            losses = losses[..., :original_action_dim]

        if not self.config.supervise_padding_actions:
            losses = losses[~action_is_pad]

        return losses.mean()

    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        states: torch.Tensor | None = None,
        *,
        state_token_id: int,
        action_token_id: int,
        **kwargs,
    ) -> Tensor:
        """Sample actions from the model."""
        if states is None:
            raise ValueError("states are required for EO1 action sampling.")
        if mm_token_type_ids is None:
            raise ValueError("mm_token_type_ids are required for EO1 action sampling.")

        # 1. Resolve the left-padded rollout prompt and locate the action span.
        chunk_size = self.config.chunk_size

        inputs_embeds = self.embed_prefix(
            input_ids,
            states=states,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        ).clone()
        _, action_placeholder_mask = self.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )
        action_mask = action_placeholder_mask[..., 0]
        token_counts = action_mask.sum(dim=1)
        if not torch.all(token_counts == chunk_size):
            raise ValueError(
                f"Each sample must contain exactly {chunk_size} action tokens, got {token_counts.tolist()}."
            )
        if action_mask.ne(action_mask[:1]).any():
            raise ValueError(
                "Batch inference expects all samples to share the same action token mask after left padding."
            )
        act_start = int(action_mask[0].to(torch.int64).argmax().item())
        act_end = act_start + self.config.chunk_size
        if not torch.all(action_mask[:, act_start:act_end]):
            raise ValueError("Action tokens must form a contiguous chunk of length chunk_size.")
        act_slice = slice(act_start, act_end)

        # 2. Encode the fixed prefix once and cache its KV state.
        batch_size = input_ids.shape[0]
        device = inputs_embeds.device
        attention_mask = attention_mask.to(device)
        mm_token_type_ids = mm_token_type_ids.to(device)
        position_ids, _ = self.vlm_backbone.model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
        position_ids = position_ids.to(device)

        outputs = self.vlm_backbone.model(
            input_ids=input_ids[:, :act_start],
            attention_mask=attention_mask[:, :act_start],
            position_ids=position_ids[..., :act_start],
            inputs_embeds=inputs_embeds[:, :act_start],
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids[:, :act_start],
            use_cache=True,
            return_dict=True,
        )

        x_t = self.sample_noise(
            (batch_size, chunk_size, self.config.max_action_dim),
            device,
        ).to(dtype=self.action_in_proj.weight.dtype)
        dt = -1.0 / self.config.num_denoise_steps
        past_key_values = outputs.past_key_values

        # 3. Denoise only the action chunk while keeping the prefix cache invariant.
        for step in range(self.config.num_denoise_steps):
            time = torch.full(
                (batch_size,),
                1.0 + step * dt,
                device=device,
                dtype=torch.float32,
            )
            action_time_embs = self.embed_suffix(time, x_t)
            inputs_embeds[:, act_slice] = action_time_embs.to(inputs_embeds.dtype)

            # Keep the prefix KV cache invariant across denoising steps.
            past_key_values.crop(act_start)
            outputs = self.vlm_backbone.model(
                attention_mask=attention_mask[:, :act_end],
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[:, act_slice],
                position_ids=position_ids[..., act_slice],
                use_cache=True,
                return_dict=True,
            )
            with self.flow_head_autocast_context():
                hidden_states = outputs.last_hidden_state[:, :chunk_size]
                hidden_states = hidden_states.to(dtype=self.action_out_proj.dtype)
                v_t = self.action_out_proj(hidden_states)

            x_t += dt * v_t.reshape(x_t.shape)

        return x_t
