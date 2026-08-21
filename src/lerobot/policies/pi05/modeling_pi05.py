#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import builtins
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from ..pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..common.flow_matching import (
    euler_integrate,
    sample_noise,
    sample_time_beta,
    staircase_substep,
    staircase_time,
)
from ..common.vla_utils import (
    clone_past_key_values,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    prepare_attention_masks_4d,
    resize_with_pad_torch,
)
from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config
from .memory import encode_video_with_mem, sample_observation_history


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def _prepare_trained_rtc_prefix(
    x_t: Tensor,
    prev_chunk_left_over: Tensor | None,
    inference_delay: int,
    training_max_delay: int,
) -> tuple[Tensor | None, Tensor | None]:
    """Pad and validate a hard prefix for training-time RTC inference."""
    if prev_chunk_left_over is None or inference_delay <= 0:
        return None, None
    if training_max_delay <= 0:
        raise ValueError(
            "RTC mode='trained' requires a checkpoint trained with policy.rtc_training_max_delay > 0."
        )
    if inference_delay > training_max_delay:
        raise ValueError(
            f"Measured RTC inference delay ({inference_delay}) exceeds the checkpoint's "
            f"rtc_training_max_delay ({training_max_delay})."
        )
    if inference_delay >= x_t.shape[1]:
        raise ValueError(
            f"RTC inference delay ({inference_delay}) must be smaller than chunk_size ({x_t.shape[1]})."
        )

    previous = prev_chunk_left_over.to(device=x_t.device, dtype=x_t.dtype)
    if not torch.isfinite(previous).all():
        raise ValueError("RTC prefix contains NaN or Inf values.")
    if previous.ndim == 2:
        previous = previous.unsqueeze(0)
    if previous.ndim != 3:
        raise ValueError(f"Expected RTC prefix shape (B, T, A), got {tuple(previous.shape)}")
    if previous.shape[0] == 1 and x_t.shape[0] > 1:
        previous = previous.expand(x_t.shape[0], -1, -1)
    if previous.shape[0] != x_t.shape[0]:
        raise ValueError(
            f"RTC prefix batch size ({previous.shape[0]}) does not match policy batch ({x_t.shape[0]})."
        )
    if previous.shape[1] < inference_delay:
        raise ValueError(f"RTC prefix has {previous.shape[1]} steps, but inference_delay={inference_delay}.")
    if previous.shape[2] > x_t.shape[2]:
        raise ValueError(
            f"RTC prefix action dimension ({previous.shape[2]}) exceeds model dimension ({x_t.shape[2]})."
        )

    padded_prefix = torch.zeros_like(x_t)
    padded_prefix[:, :inference_delay, : previous.shape[2]] = previous[:, :inference_delay]
    prefix_mask = torch.arange(x_t.shape[1], device=x_t.device) < inference_delay
    prefix_mask = prefix_mask[None, :, None].expand(x_t.shape[0], -1, x_t.shape[2])
    return padded_prefix, prefix_mask


def _sample_training_rtc_prefix_mask(
    batch_size: int,
    action_horizon: int,
    max_delay: int,
    device: torch.device,
) -> Tensor | None:
    """Sample a clean action-prefix length independently for each training example."""
    if max_delay <= 0:
        return None
    delays = torch.randint(0, max_delay + 1, (batch_size,), device=device)
    positions = torch.arange(action_horizon, device=device)
    return positions.unsqueeze(0) < delays.unsqueeze(1)


@dataclass
class PiR2SlowChannel:
    """Cached vision-language prefix, valid across many action-expert calls.

    Holding this instead of re-running the backbone every call is what takes the VLM off the
    control-rate critical path (arXiv 2607.26055, Sec. 3.2).
    """

    prefix_pad_masks: Tensor
    past_key_values: object
    # ``time.perf_counter()`` at capture, so the engine can tell the expert how stale this is.
    captured_at: float | None = None


def _build_staircase_schedule(
    batch_size: int,
    action_horizon: int,
    max_delay: int,
    shared_time: Tensor,
    *,
    time_jitter: float = 0.0,
    warmup_prob: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Sample the piR2 latency-adaptive staircase (arXiv 2607.26055, Eq. 3).

    The paper writes the schedule with tau=1 clean and tau=0 noise, in three regions: a clean
    front of ``delay`` in-flight actions, a linear ramp across the interior, and a pure-noise
    tail of ``delay`` freshly appended slots. Under this file's opposite convention (t=0 clean,
    t=1 noise) all three collapse into one clamped ramp, since the clamp reproduces the flat
    front and tail exactly.

    Returns ``(prefix_mask, position_time)``: the front positions to clamp and drop from the
    loss, and the per-position flow timestep.
    """
    device = shared_time.device
    delays = torch.randint(0, max_delay + 1, (batch_size, 1), device=device)
    positions = torch.arange(action_horizon, device=device).unsqueeze(0)
    # Ramp slope 1 / (H - 2d); the widest delay is validated to leave a non-empty interior.
    interior = (action_horizon - 2 * delays).clamp(min=1)
    position_time = ((positions - delays) / interior).clamp(0.0, 1.0).to(shared_time.dtype)

    if time_jitter > 0.0:
        # Paper Alg. 1 line 11 jitters every position, including the front, and then overwrites
        # the front with ground-truth actions without restoring its timestep.
        jitter = torch.empty_like(position_time).uniform_(-time_jitter, time_jitter)
        position_time = (position_time + jitter).clamp(0.0, 1.0)

    prefix_mask = positions < delays

    if warmup_prob > 0.0:
        # Warm-up branch: a standard shared-timestep flow batch with no clamped front, so the
        # same weights can still denoise a chunk from pure noise to initialize the buffer.
        # The paper draws this once per batch; drawing per example gives the same expectation
        # with lower variance.
        is_warmup = torch.rand((batch_size, 1), device=device) < warmup_prob
        position_time = torch.where(is_warmup, shared_time[:, None].expand_as(position_time), position_time)
        prefix_mask = prefix_mask & ~is_warmup

    return prefix_mask, position_time


def _build_flow_matching_inputs(
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    prefix_mask: Tensor | None,
    position_time: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Keep the sampled RTC prefix clean while noising the remaining action chunk."""
    if position_time is not None:
        model_time = position_time
        expanded_time = position_time.unsqueeze(-1)
    elif prefix_mask is None:
        model_time = time
        expanded_time = time[:, None, None]
    else:
        model_time = time[:, None].expand_as(prefix_mask)
        model_time = torch.where(prefix_mask, torch.zeros_like(model_time), model_time)
        expanded_time = model_time.unsqueeze(-1)
    x_t = expanded_time * noise + (1 - expanded_time) * actions
    if position_time is not None and prefix_mask is not None:
        # A jittered front carries a timestep slightly off zero, so the clean values have to be
        # written in explicitly rather than falling out of the interpolation.
        x_t = torch.where(prefix_mask.unsqueeze(-1), actions, x_t)
    return x_t, model_time


def _reduce_training_rtc_loss(
    losses: Tensor,
    prefix_mask: Tensor | None,
    reduction: str,
) -> Tensor:
    """Average flow loss over predicted postfix actions, excluding the clean RTC prefix."""
    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unsupported loss reduction: {reduction!r}")
    if prefix_mask is None:
        return losses.mean() if reduction == "mean" else losses.mean(dim=(1, 2))

    postfix_mask = (~prefix_mask).unsqueeze(-1).expand_as(losses)
    if reduction == "none":
        numerator = (losses * postfix_mask).sum(dim=(1, 2))
        denominator = postfix_mask.sum(dim=(1, 2))
        return numerator / denominator.clamp(min=1)
    return (losses * postfix_mask).sum() / postfix_mask.sum().clamp(min=1)


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(inputs_embeds, attention_mask, position_ids, adarms_cond, layers, rotary_emb):
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    paligemma_layer = layers[0]
    scaling = paligemma_layer.self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(
        self,
        image: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        temporal_attention_every: int = 4,
    ):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        if image.ndim == 5:
            if frame_mask is None:
                frame_mask = torch.ones(image.shape[:2], dtype=torch.bool, device=image.device)
            vision_transformer = self.paligemma.model.vision_tower.vision_model
            features = encode_video_with_mem(
                vision_transformer,
                image,
                frame_mask,
                temporal_attention_every=temporal_attention_every,
            )
            features = self.paligemma.model.multi_modal_projector(features)
        else:
            image_outputs = self.paligemma.model.get_image_features(image)
            features = image_outputs.pooler_output
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            paligemma_layers = self.paligemma.model.language_model.layers
            gemma_expert_layers = self.gemma_expert.model.layers
            rotary_emb = self.paligemma.model.language_model.rotary_emb

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layers in zip(paligemma_layers, gemma_expert_layers, strict=True):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )

            # final norm
            final_norms = (
                self.paligemma.model.language_model.norm,
                self.gemma_expert.model.norm,
            )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(final_norms[i], hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.proprio_history_proj = (
            nn.Linear(config.max_state_dim, paligemma_config.width)
            if config.use_proprioceptive_memory
            else None
        )

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def sample_noise(self, shape, device):
        return sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return sample_time_beta(
            bsize,
            device,
            alpha=self.config.time_sampling_beta_alpha,
            beta=self.config.time_sampling_beta_beta,
            scale=self.config.time_sampling_scale,
            offset=self.config.time_sampling_offset,
        )

    def embed_prefix(
        self, images, img_masks, tokens, masks, states=None, state_masks=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images, optional MEM state history, and language tokens."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img, img_mask):
                if img.ndim == 5:
                    return self.paligemma_with_expert.embed_image(
                        img,
                        frame_mask=img_mask,
                        temporal_attention_every=self.config.memory_temporal_attention_every,
                    )
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img, img_mask)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            current_img_mask = img_mask[:, -1] if img_mask.ndim == 2 else img_mask
            pad_masks.append(current_img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        proprio_history_proj = getattr(self, "proprio_history_proj", None)
        if proprio_history_proj is not None:
            if states is None or state_masks is None:
                raise ValueError("proprioceptive memory requires states and state_masks")
            state_embs = self._apply_checkpoint(proprio_history_proj, states)
            embs.append(state_embs)
            pad_masks.append(state_masks)
            att_masks += [0] * state_embs.shape[1]

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        adarms_cond = time_emb

        bsize, action_time_dim = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))
        att_masks = torch.tensor(att_masks, dtype=action_emb.dtype, device=action_emb.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return action_emb, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        prefix_mask: Tensor | None = None,
        states=None,
        state_masks=None,
        position_time: Tensor | None = None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        x_t, model_time = _build_flow_matching_inputs(actions, noise, time, prefix_mask, position_time)
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, model_time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        states=None,
        state_masks=None,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_pad_masks, past_key_values = self.prefill_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )

        rtc_mode = "guided"
        trained_prefix = trained_prefix_mask = None
        if self._rtc_enabled():
            rtc_mode = self.rtc_processor.rtc_config.mode
            if rtc_mode == "trained":
                training_max_delay = int(getattr(self.config, "rtc_training_max_delay", 0))
                if training_max_delay <= 0:
                    raise ValueError(
                        "RTC mode='trained' requires a checkpoint trained with "
                        "policy.rtc_training_max_delay > 0."
                    )
                trained_prefix, trained_prefix_mask = _prepare_trained_rtc_prefix(
                    noise,
                    kwargs.get("prev_chunk_left_over"),
                    int(kwargs.get("inference_delay") or 0),
                    training_max_delay,
                )

        return euler_integrate(
            self._prefix_denoise_fn(prefix_pad_masks, past_key_values),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled() and rtc_mode == "guided",
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
            hard_prefix=trained_prefix,
            hard_prefix_mask=trained_prefix_mask,
        )

    def prefill_prefix(self, images, img_masks, tokens, masks, states=None, state_masks=None):
        """Run the vision-language prefix once and return its attention mask and KV cache.

        This is piR2's slow channel (arXiv 2607.26055): the cache is valid for as many action
        expert calls as you care to make against it, so a background thread can refresh it on
        its own cadence while the expert keeps denoising against the last one.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_pad_masks, past_key_values

    def _prefix_denoise_fn(self, prefix_pad_masks, past_key_values):
        return lambda x_t, timestep: self.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )

    def warm_start_staircase_buffer(
        self,
        prefix_pad_masks,
        past_key_values,
        delay,
        *,
        noise=None,
        num_steps=None,
    ) -> Tensor:
        """Denoise a full chunk from pure noise, then re-noise it onto the staircase.

        Episode start has no in-flight actions to condition on, so piR2 falls back to a standard
        full denoise and then puts the result back at the per-position noise levels the steady
        state expects. The 20% shared-timestep branch during training is what keeps the weights
        able to do this first pass.
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        if noise is None:
            shape = (prefix_pad_masks.shape[0], self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(shape, prefix_pad_masks.device)

        clean = euler_integrate(self._prefix_denoise_fn(prefix_pad_masks, past_key_values), noise, num_steps)

        time = staircase_time(delay, clean.shape[1], device=clean.device, dtype=clean.dtype)
        fresh = self.sample_noise(clean.shape, clean.device).to(dtype=clean.dtype)
        return time[None, :, None] * fresh + (1 - time[None, :, None]) * clean

    def staircase_denoise_step(self, prefix_pad_masks, past_key_values, x_t, delay, *, noise=None):
        """One piR2 call against a cached prefix: emit ``delay`` actions and slide the buffer."""
        return staircase_substep(
            self._prefix_denoise_fn(prefix_pad_masks, past_key_values),
            x_t,
            delay,
            noise=noise,
        )

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def supports_rtc(self) -> bool:
        return True

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            remapped_state_dict = model._prepare_pretrained_state_dict(remapped_state_dict)

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _prepare_pretrained_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        # MEM's continuous proprioceptive projection is new relative to
        # lerobot/pi05_base. Preserve its fresh initialization on first load,
        # while loading learned values from subsequent MEM checkpoints.
        if getattr(self.config, "use_proprioceptive_memory", False):
            current = self.state_dict()
            for key in (
                "model.proprio_history_proj.weight",
                "model.proprio_history_proj.bias",
            ):
                state_dict.setdefault(key, current[key])
        return state_dict

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state at the shared boundary of a batched rollout.

        ``lerobot-eval`` calls this before every rollout and before resetting the
        vector environment. MEM inference queues therefore assume all batch rows
        share episode boundaries; independently autoresetting rows is unsupported.
        """
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.use_visual_memory or self.config.use_proprioceptive_memory:
            # The sampled ages are relative to the current step, so every step shifts
            # all of them: a dense ring buffer spanning the whole horizon is the
            # smallest structure that keeps inference aligned with the training
            # `delta_indices`. Storing only `memory_frames` observations would pin the
            # history to an absolute grid and drift up to `memory_stride` out of phase.
            history_length = (self.config.memory_frames - 1) * self.config.memory_stride + 1
            memory_keys = []
            if self.config.use_visual_memory:
                memory_keys.extend(self.config.image_features)
            if self.config.use_proprioceptive_memory:
                memory_keys.append(OBS_STATE)
            self._memory_queues = {key: deque(maxlen=history_length) for key in memory_keys}
            self._memory_steps_seen = 0
            self._memory_batch_size = None

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(
                self.config.rtc_config,
                trained_mode_supported=int(getattr(self.config, "rtc_training_max_delay", 0)) > 0,
            )

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # Handle [B,C,H,W], [B,H,W,C], and their [B,T,...] memory variants.
            is_video = img.ndim == 5
            channel_dim = 2 if is_video else 1
            is_channels_first = img.shape[channel_dim] == 3

            if is_channels_first:
                img = img.permute(0, 1, 3, 4, 2) if is_video else img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            spatial_shape = img.shape[2:4] if is_video else img.shape[1:3]
            if spatial_shape != self.config.image_resolution:
                if is_video:
                    batch_size, num_frames = img.shape[:2]
                    img = resize_with_pad_torch(img.flatten(0, 1), *self.config.image_resolution).unflatten(
                        0, (batch_size, num_frames)
                    )
                else:
                    img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 1, 4, 2, 3) if is_video else img.permute(0, 3, 1, 2)

            images.append(img)
            bsize = img.shape[0]
            pad_key = f"{key}_is_pad"
            if is_video and pad_key in batch:
                mask = ~batch[pad_key].bool()
            else:
                mask_shape = img.shape[:2] if is_video else (bsize,)
                mask = torch.ones(mask_shape, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def _prepare_memory_states(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        if not self.config.use_proprioceptive_memory:
            return None, None
        states = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        if states.ndim == 2:
            states = states[:, None]
        pad_key = f"{OBS_STATE}_is_pad"
        state_masks = (
            ~batch[pad_key].bool()
            if pad_key in batch
            else torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
        )
        return states, state_masks

    def _stack_inference_memory(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Record one homogeneous-batch observation and attach MEM histories."""
        if not (self.config.use_visual_memory or self.config.use_proprioceptive_memory):
            return batch
        result = dict(batch)
        for key, queue in self._memory_queues.items():
            if key not in batch:
                continue
            value = batch[key]
            batch_size = value.shape[0]
            if self._memory_batch_size is None:
                self._memory_batch_size = batch_size
            elif batch_size != self._memory_batch_size:
                raise ValueError(
                    "MEM inference batch size changed without policy.reset(); "
                    f"expected {self._memory_batch_size}, got {batch_size}"
                )
            if not queue:
                # Queue entries are snapshots that are never mutated in place, so the
                # pre-episode fill can share one clone instead of `maxlen` copies.
                queue.extend([value.clone()] * queue.maxlen)
            else:
                queue.append(value.clone())
            result[key], result[f"{key}_is_pad"] = sample_observation_history(
                list(queue),
                num_frames=self.config.memory_frames,
                stride=self.config.memory_stride,
                steps_seen=self._memory_steps_seen + 1,
            )
        self._memory_steps_seen += 1
        return result

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        if self.config.use_visual_memory or self.config.use_proprioceptive_memory:
            batch = self._stack_inference_memory(batch)

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Direct chunk callers provide single observations. ``select_action``
        # already supplies a temporal batch, so avoid recording it twice.
        has_temporal_input = any(
            key in batch and batch[key].ndim == 5 for key in self.config.image_features
        ) or (OBS_STATE in batch and batch[OBS_STATE].ndim == 3)
        if (
            self.config.use_visual_memory or self.config.use_proprioceptive_memory
        ) and not has_temporal_input:
            batch = self._stack_inference_memory(batch)

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        states, state_masks = self._prepare_memory_states(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(
            images, img_masks, tokens, masks, states=states, state_masks=state_masks, **kwargs
        )

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def encode_slow_channel(self, batch: dict[str, Tensor]) -> PiR2SlowChannel:
        """Run the vision-language prefix and return a cache the action expert can reuse.

        In piR2 this is the only work that scales with backbone size, and it is meant to run on
        a background thread at whatever rate it can manage while the expert keeps stepping.
        """
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        states, state_masks = self._prepare_memory_states(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        prefix_pad_masks, past_key_values = self.model.prefill_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        return PiR2SlowChannel(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            captured_at=time.perf_counter(),
        )

    @torch.no_grad()
    def warm_start_realtime_buffer(self, slow: PiR2SlowChannel, delay: int) -> Tensor:
        """Build the initial action buffer at episode start, when nothing is in flight yet."""
        self.eval()
        return self.model.warm_start_staircase_buffer(slow.prefix_pad_masks, slow.past_key_values, delay)

    @torch.no_grad()
    def realtime_substep(self, slow: PiR2SlowChannel, buffer: Tensor, delay: int) -> tuple[Tensor, Tensor]:
        """Advance the buffer by one denoising step, returning ``delay`` finished actions.

        ``slow`` may have been encoded several control steps ago; the clamped clean front of
        ``buffer`` is what tells the expert where the robot currently is.
        """
        self.eval()
        emitted, next_buffer = self.model.staircase_denoise_step(
            slow.prefix_pad_masks,
            slow.past_key_values,
            buffer,
            delay,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return emitted[:, :, :original_action_dim], next_buffer

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        states, state_masks = self._prepare_memory_states(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        position_time = None
        if self.config.rtc_training_schedule == "staircase" and self.config.rtc_training_max_delay > 0:
            prefix_mask, position_time = _build_staircase_schedule(
                actions.shape[0],
                actions.shape[1],
                self.config.rtc_training_max_delay,
                time,
                time_jitter=self.config.staircase_time_jitter,
                warmup_prob=self.config.staircase_warmup_prob,
            )
        else:
            prefix_mask = _sample_training_rtc_prefix_mask(
                actions.shape[0],
                actions.shape[1],
                self.config.rtc_training_max_delay,
                actions.device,
            )

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            prefix_mask=prefix_mask,
            states=states,
            state_masks=state_masks,
            position_time=position_time,
        )

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        if prefix_mask is None:
            loss_per_dim = losses.mean(dim=(0, 1))
        else:
            postfix_mask = (~prefix_mask).unsqueeze(-1).expand_as(losses)
            loss_per_dim = (losses * postfix_mask).sum(dim=(0, 1)) / postfix_mask.sum(dim=(0, 1)).clamp(min=1)
        loss_dict = {"loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist()}

        if reduction == "none":
            per_sample_loss = _reduce_training_rtc_loss(losses, prefix_mask, reduction="none")
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict

        loss = _reduce_training_rtc_loss(losses, prefix_mask, reduction="mean")
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        # MEM's proprioceptive projection does not exist in `lerobot/pi05_base`, so a
        # LoRA adapter cannot start from pretrained weights for it. Train and save it
        # in full, otherwise it stays frozen at its random init and is absent from
        # adapter checkpoints.
        modules_to_save = ["model.proprio_history_proj"] if self.config.use_proprioceptive_memory else []
        return {
            "target_modules": target_modules,
            "modules_to_save": modules_to_save,
        }
