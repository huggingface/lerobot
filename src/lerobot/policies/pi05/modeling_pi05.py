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
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_HIGH_LEVEL_TASK_TOKENS,
    OBS_LANGUAGE_HIGH_LEVEL_TASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ONLY_TOKENS,
    OBS_LANGUAGE_SUBTASK_ONLY_ATTENTION_MASK,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


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
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
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
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
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
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

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
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

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
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            # prefix_output to be used for the language head
            # shape: [batch_size, seq_len, hidden_size] with hidden_size = 2048
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
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
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

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # # FAST action token embedding and prediction head
        # self.fast_action_embedding = nn.Embedding(config.fast_vocab_size, paligemma_config.width)
        # self.fast_action_lm_head = nn.Linear(paligemma_config.width, config.fast_vocab_size)

        # # Apply dtype conversion to FAST layers to match model precision
        # if config.dtype == "bfloat16":
        #     self.fast_action_embedding = self.fast_action_embedding.to(dtype=torch.bfloat16)
        #     self.fast_action_lm_head = self.fast_action_lm_head.to(dtype=torch.bfloat16)
        # elif config.dtype == "float32":
        #     self.fast_action_embedding = self.fast_action_embedding.to(dtype=torch.float32)
        #     self.fast_action_lm_head = self.fast_action_lm_head.to(dtype=torch.float32)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
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

    def _prepare_attention_masks_4d(self, att_2d_masks, dtype=None):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result

    def _create_custom_attention_mask(self, att_mask_segments, pad_masks, bsize):
        """Create custom 2D attention mask for the new attention pattern.
        
        Attention rules:
        - Images + Language: bidirectional among themselves, don't attend to subtask or FAST
        - Subtask: attend to images + language, causal among themselves, don't attend to FAST
        - FAST: attend to images + language + subtask, causal among themselves
        
        Args:
            att_mask_segments: List of (type, length) tuples
            pad_masks: Padding masks [B, total_seq_len]
            bsize: Batch size
        
        Returns:
            att_2d_masks: 2D attention mask [B, total_seq_len, total_seq_len]
        """
        total_len = sum(length for _, length in att_mask_segments)
        device = pad_masks.device
        
        # Initialize attention mask as False (cannot attend)
        att_2d_masks = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)
        
        # Track positions for each segment
        positions = []
        current_pos = 0
        for seg_type, seg_len in att_mask_segments:
            positions.append((seg_type, current_pos, current_pos + seg_len))
            current_pos += seg_len
        
        # Apply attention rules
        for i, (query_type, query_start, query_end) in enumerate(positions):
            for j, (key_type, key_start, key_end) in enumerate(positions):
                # Images and Language can attend to each other bidirectionally
                if query_type in ['image', 'language'] and key_type in ['image', 'language']:
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = True
                
                # Subtask tokens attend to images + language
                elif query_type == 'subtask' and key_type in ['image', 'language']:
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = True
                
                # Subtask tokens attend causally to themselves
                elif query_type == 'subtask' and key_type == 'subtask':
                    # Create causal mask for subtask tokens
                    subtask_len = query_end - query_start
                    causal_mask = torch.tril(torch.ones(subtask_len, subtask_len, dtype=torch.bool, device=device))
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = causal_mask[None, :, :]
                
                # FAST tokens attend to images + language + subtask
                elif query_type == 'fast' and key_type in ['image', 'language', 'subtask']:
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = True
                
                # FAST tokens attend causally to themselves
                elif query_type == 'fast' and key_type == 'fast':
                    fast_len = query_end - query_start
                    causal_mask = torch.tril(torch.ones(fast_len, fast_len, dtype=torch.bool, device=device))
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = causal_mask[None, :, :]
        
        # Apply padding masks
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        att_2d_masks = att_2d_masks & pad_2d_masks
        
        return att_2d_masks

    def visualize_attention_mask(
        self,
        att_mask_segments,
        att_2d_masks,
        save_path,
        batch_idx=0,
        dpi=150,
        max_display_tokens=None
    ):
        """Visualize the attention mask with labeled segments.
        
        Args:
            att_mask_segments: List of (type, length) tuples defining the segments
            att_2d_masks: 2D attention mask tensor [B, total_seq_len, total_seq_len]
            save_path: Path where to save the visualization image
            batch_idx: Which batch item to visualize (default: 0)
            dpi: DPI for the saved image (default: 150)
            max_display_tokens: Maximum number of tokens to display (for very long sequences)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            logging.warning("matplotlib not available, skipping attention mask visualization")
            return
        
        # Extract the mask for the specified batch
        mask = att_2d_masks[batch_idx].cpu().float().numpy()
        
        # If sequence is too long, downsample for visualization
        if max_display_tokens is not None and mask.shape[0] > max_display_tokens:
            # Simple downsampling by taking every Nth token
            step = mask.shape[0] // max_display_tokens
            mask = mask[::step, ::step]
            # Adjust segments accordingly
            att_mask_segments = [(seg_type, max(1, seg_len // step)) for seg_type, seg_len in att_mask_segments]
        
        # Calculate positions for each segment
        positions = []
        current_pos = 0
        for seg_type, seg_len in att_mask_segments:
            positions.append((seg_type, current_pos, current_pos + seg_len))
            current_pos += seg_len
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create custom colormap: white for False (no attention), blue for True (attention)
        colors = ['white', '#2E86AB']
        n_bins = 2
        cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
        
        # Display the mask
        im = ax.imshow(mask, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Enabled', rotation=270, labelpad=20)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['No', 'Yes'])
        
        # Define colors for each segment type
        segment_colors = {
            'image': '#A23B72',
            'language': '#F18F01',
            'subtask': '#C73E1D',
            'fast': '#6A994E'
        }
        
        # Draw segment boundaries and labels
        for seg_type, start, end in positions:
            color = segment_colors.get(seg_type, '#666666')
            
            # Draw vertical lines for columns (keys)
            ax.axvline(x=start - 0.5, color=color, linewidth=2, alpha=0.7)
            ax.axvline(x=end - 0.5, color=color, linewidth=2, alpha=0.7)
            
            # Draw horizontal lines for rows (queries)
            ax.axhline(y=start - 0.5, color=color, linewidth=2, alpha=0.7)
            ax.axhline(y=end - 0.5, color=color, linewidth=2, alpha=0.7)
            
            # Add labels at the top
            mid_pos = (start + end) / 2
            ax.text(mid_pos, -mask.shape[0] * 0.02, f"{seg_type.upper()}\n({end - start})",
                   ha='center', va='top', fontsize=10, fontweight='bold', color=color)
            
            # Add labels on the left
            ax.text(-mask.shape[1] * 0.02, mid_pos, f"{seg_type.upper()}\n({end - start})",
                   ha='right', va='center', fontsize=10, fontweight='bold', color=color, rotation=0)
        
        # Set axis labels
        ax.set_xlabel('Key Position (tokens being attended to)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Position (tokens attending)', fontsize=12, fontweight='bold')
        ax.set_title('Attention Mask Pattern\n(White = No Attention, Blue = Attention Allowed)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Create legend for segment types
        legend_patches = []
        attention_rules = {
            'image': 'Bidirectional with lang',
            'language': 'Bidirectional with images',
            'subtask': 'Attends to img+lang, causal self',
            'fast': 'Attends to all, causal self'
        }
        for seg_type, color in segment_colors.items():
            if any(seg[0] == seg_type for seg in att_mask_segments):
                rule = attention_rules.get(seg_type, '')
                legend_patches.append(mpatches.Patch(color=color, label=f'{seg_type.upper()}: {rule}'))
        
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1.0),
                 framealpha=0.9, fontsize=9)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Ensure the directory exists
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Attention mask visualization saved to: {save_path}")

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)
    
    def embed_prefix(
        self, 
        images, 
        img_masks, 
        tokens, 
        subtask_tokens, 
        masks, 
        subtask_masks, 
        fast_action_tokens=None, 
        fast_action_masks=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """Embed images with SigLIP, tokens, and optionally subtask tokens with embedding layer.
        
        Args:
            images: List of image tensors
            img_masks: List of image masks
            tokens: Language instruction tokens
            subtask_tokens: Subtask tokens to predict (can be None for inference)
            masks: Attention masks for tokens
            fast_action_tokens: FAST action tokens for auxiliary prediction (can be None) - discrete token IDs
            fast_action_masks: Padding masks for FAST action tokens (can be None)
            
        Returns:
            embs: Concatenated embeddings [images, tokens, (subtask_tokens if provided), (fast_action_tokens if provided)]
            pad_masks: Padding masks
            att_masks: Custom 2D attention mask implementing the required pattern
            total_T_images: Total number of image tokens
            num_subtask_embs: Number of subtask token embeddings
            num_fast_embs: Number of FAST action token embeddings
        """
        embs = []
        pad_masks = []
        att_mask_segments = []  # Store info about each segment for custom mask creation
        total_T_images = 0
        num_subtask_embs = 0
        num_fast_embs = 0
        
        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_mask_segments.append(('image', num_img_embs))
            total_T_images += num_img_embs
            
        # Process language instruction tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_mask_segments.append(('language', num_lang_embs))

        # Process subtask tokens if provided (these are predicted, so use causal masking)
        if subtask_tokens is not None:
            def subtask_embed_func(subtask_tokens):
                subtask_emb = self.paligemma_with_expert.embed_language_tokens(subtask_tokens)
                subtask_emb_dim = subtask_emb.shape[-1]
                return subtask_emb * math.sqrt(subtask_emb_dim)

            subtask_emb = self._apply_checkpoint(subtask_embed_func, subtask_tokens)
            embs.append(subtask_emb)
            
            # Create subtask pad masks (non-zero tokens are valid)
            pad_masks.append(subtask_masks)

            num_subtask_embs = subtask_emb.shape[1]
            att_mask_segments.append(('subtask', num_subtask_embs))
        # Process FAST action tokens if provided (these are discrete token IDs)
        if fast_action_tokens is not None:
            def fast_action_embed_func(fast_action_tokens):
                fast_emb = self.fast_action_embedding(fast_action_tokens)
                fast_emb_dim = fast_emb.shape[-1]
                return fast_emb * math.sqrt(fast_emb_dim)
            
            fast_action_emb = self._apply_checkpoint(fast_action_embed_func, fast_action_tokens)
            embs.append(fast_action_emb)
            
            # Use provided mask or create default (all valid)
            if fast_action_masks is not None:
                fast_pad_mask = fast_action_masks
            else:
                bsize = fast_action_tokens.shape[0]
                num_fast_embs = fast_action_tokens.shape[1]
                fast_pad_mask = torch.ones(bsize, num_fast_embs, dtype=torch.bool, device=fast_action_tokens.device)
            
            num_fast_embs = fast_action_tokens.shape[1]
            pad_masks.append(fast_pad_mask)
            att_mask_segments.append(('fast', num_fast_embs))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        
        # Create custom 2D attention mask
        # Attention rules:
        # - Images + Language: bidirectional among themselves, don't attend to subtask or FAST
        # - Subtask: attend to images + language, causal among themselves, don't attend to FAST
        # - FAST: attend to images + language + subtask, causal among themselves
        att_masks = self._create_custom_attention_mask(att_mask_segments, pad_masks, bsize)

        # # Optionally visualize the attention mask
        # self.visualize_attention_mask(
        #     att_mask_segments=att_mask_segments,
        #     att_2d_masks=att_masks,
        #     save_path="/admin/home/jade_choghari/lerobot/src/lerobot/policies/pi05/attention_mask_visualization.png",
        #     batch_idx=0,
        #     max_display_tokens=512  # Limit display for very long sequences
        # )

        return embs, pad_masks, att_masks, total_T_images, num_subtask_embs, num_fast_embs

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
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
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    #  loss_dict = self.model.forward(images, img_masks, high_level_task, tokens, masks, subtask_tokens, subtask_masks, actions, fast_action_tokens, fast_action_masks)
    def forward(self, images, img_masks, high_level_task, high_level_task_masks, subtask_tokens, subtask_masks, actions, fast_action_tokens=None, fast_action_masks=None, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss.
        
        Args:
            images: List of image tensors
            img_masks: List of image masks
            high_level_task: Instruction tokens WITHOUT subtask (e.g., "High level task: X; State: Y; Subtask:")
            high_level_task_masks: Attention masks for high_level_task
            subtask_tokens: Subtask tokens to predict (e.g., tokens for "pick up the cup")
            subtask_masks: Attention masks for subtask_tokens
            actions: Ground truth actions [B, chunk_size, action_dim]
            fast_action_tokens: Discrete action token IDs [B, max_action_tokens]
            fast_action_masks: Padding masks for fast action tokens [B, max_action_tokens]
            noise: Optional noise for flow matching
            time: Optional time for flow matching
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Initialize FAST loss to 0 (will be computed only if FAST tokens are provided)
        fast_loss = torch.tensor(0.0, device=actions.device, dtype=actions.dtype)

        # ========== PASS 1: Prefix with FAST tokens for subtask + FAST prediction ==========
        # Only run this pass if FAST action tokens are provided
        if fast_action_tokens is not None and fast_action_masks is not None:
            # Embed prefix (images + high_level_task + subtask_tokens + FAST tokens)
            # FAST tokens are provided as discrete token IDs
            prefix_with_fast_embs, prefix_with_fast_pad_masks, prefix_with_fast_att_masks, total_T_images, num_subtask_embs, num_fast_embs = self.embed_prefix(
                images, img_masks, high_level_task, subtask_tokens, high_level_task_masks, subtask_masks, 
                fast_action_tokens=fast_action_tokens, fast_action_masks=fast_action_masks
            )

            # Convert embeddings to bfloat16 if needed for the model
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_with_fast_embs = prefix_with_fast_embs.to(dtype=torch.bfloat16)

            # Prepare attention masks for prefix pass with FAST tokens
            position_ids_prefix_with_fast = torch.cumsum(prefix_with_fast_pad_masks, dim=1) - 1
            att_2d_prefix_with_fast_4d = self._prepare_attention_masks_4d(prefix_with_fast_att_masks, dtype=prefix_with_fast_embs.dtype)

            # Forward pass through paligemma for subtask + FAST prediction
            (prefix_with_fast_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_prefix_with_fast_4d,
                position_ids=position_ids_prefix_with_fast,
                past_key_values=None,
                inputs_embeds=[prefix_with_fast_embs, None],  # SUFFIX = None
                use_cache=False,
                adarms_cond=[None, None],
            )

            # LM HEAD → SUBTASK LOGITS
            lm_head = self.paligemma_with_expert.paligemma.lm_head
            logits = lm_head(prefix_with_fast_out)  # (B, T_prefix_with_fast, vocab)

            # Extract logits for subtask token prediction
            T_high_level_task = high_level_task.size(1)
            T_subtask = subtask_tokens.size(1)
            start_index = total_T_images + T_high_level_task
            end_index = start_index + T_subtask
            logits_subtask = logits[:, start_index-1:end_index-1, :]  # (B, T_subtask, vocab)

            targets = subtask_tokens  # (B, T_subtask)
            # Compute cross-entropy loss for subtask
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits_flat = logits_subtask.reshape(-1, logits_subtask.size(-1))
            targets_flat = targets.reshape(-1)
            loss_per_token = loss_fct(logits_flat, targets_flat)
            loss_per_token = loss_per_token.reshape(targets.shape)
            masked_loss = loss_per_token * subtask_masks.float()
            subtask_loss = masked_loss.sum() / subtask_masks.sum().clamp(min=1)

            # Extract outputs for FAST action token prediction and compute auxiliary loss
            # FAST outputs start after subtask tokens
            # Similar to subtask, we use autoregressive prediction where position i predicts token i+1
            fast_start_index = end_index
            fast_end_index = fast_start_index + num_fast_embs
            
            # Get logits for FAST action tokens using the FAST LM head
            fast_logits = self.fast_action_lm_head(prefix_with_fast_out)  # (B, T_prefix_with_fast, fast_vocab_size)
            
            # Extract logits for FAST token prediction (autoregressive: position i predicts token i+1)
            # - Position (fast_start_index-1) predicts fast_action_tokens[0]
            # - Position (fast_start_index) predicts fast_action_tokens[1], etc.
            fast_logits_for_pred = fast_logits[:, fast_start_index-1:fast_end_index-1, :]  # (B, max_action_tokens, fast_vocab_size)
            
            # Compute cross-entropy loss for FAST action tokens
            fast_targets = fast_action_tokens  # (B, max_action_tokens)
            loss_fct_fast = torch.nn.CrossEntropyLoss(reduction='none')
            fast_logits_flat = fast_logits_for_pred.reshape(-1, fast_logits_for_pred.size(-1))  # (B*max_action_tokens, fast_vocab_size)
            fast_targets_flat = fast_targets.reshape(-1)  # (B*max_action_tokens)
            
            fast_loss_per_token = loss_fct_fast(fast_logits_flat, fast_targets_flat)  # (B*max_action_tokens)
            fast_loss_per_token = fast_loss_per_token.reshape(fast_targets.shape)  # (B, max_action_tokens)
            
            # Apply mask and compute mean loss over valid tokens
            masked_fast_loss = fast_loss_per_token * fast_action_masks.float()
            fast_loss = masked_fast_loss.sum() / fast_action_masks.sum().clamp(min=1)
        else:
            # If no FAST tokens provided, compute subtask loss without FAST tokens
            # This is the fallback for backward compatibility
            prefix_embs_for_subtask, prefix_pad_masks_for_subtask, prefix_att_masks_for_subtask, total_T_images, _, _ = self.embed_prefix(
                images, img_masks, high_level_task, subtask_tokens, high_level_task_masks, subtask_masks,
                fast_action_tokens=None, fast_action_masks=None
            )
            
            # Convert embeddings to bfloat16 if needed for the model
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs_for_subtask = prefix_embs_for_subtask.to(dtype=torch.bfloat16)
            
            position_ids_prefix = torch.cumsum(prefix_pad_masks_for_subtask, dim=1) - 1
            att_2d_prefix_4d = self._prepare_attention_masks_4d(prefix_att_masks_for_subtask, dtype=prefix_embs_for_subtask.dtype)
            
            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_prefix_4d,
                position_ids=position_ids_prefix,
                past_key_values=None,
                inputs_embeds=[prefix_embs_for_subtask, None],
                use_cache=False,
                adarms_cond=[None, None],
            )
            
            lm_head = self.paligemma_with_expert.paligemma.lm_head
            logits = lm_head(prefix_out)
            
            T_high_level_task = high_level_task.size(1)
            T_subtask = subtask_tokens.size(1)
            start_index = total_T_images + T_high_level_task
            end_index = start_index + T_subtask
            logits_subtask = logits[:, start_index-1:end_index-1, :]
            
            targets = subtask_tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits_flat = logits_subtask.reshape(-1, logits_subtask.size(-1))
            targets_flat = targets.reshape(-1)
            loss_per_token = loss_fct(logits_flat, targets_flat)
            loss_per_token = loss_per_token.reshape(targets.shape)
            masked_loss = loss_per_token * subtask_masks.float()
            subtask_loss = masked_loss.sum() / subtask_masks.sum().clamp(min=1)

        # ========== PASS 2: Full forward WITHOUT FAST tokens for flow matching ==========
        # Embed prefix WITHOUT FAST tokens (images + high_level_task + subtask_tokens)
        prefix_embs_no_fast, prefix_pad_masks_no_fast, prefix_att_masks_no_fast, _, _, _ = self.embed_prefix(
            images, img_masks, high_level_task, subtask_tokens, high_level_task_masks, subtask_masks, 
            fast_action_tokens=None, fast_action_masks=None
        )
        
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        # Convert embeddings to bfloat16 if needed for the model
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs_no_fast = prefix_embs_no_fast.to(dtype=torch.bfloat16)

        # For the flow matching pass, we need custom attention where:
        # - prefix follows the custom pattern (images+lang bidirectional, subtask causal, no cross-attention)
        # - suffix attends to all prefix + causal to itself
        # We'll construct this by extending prefix_att_masks_no_fast to include suffix
        
        # prefix_att_masks_no_fast is already a 2D boolean mask [B, prefix_len, prefix_len]
        # We need to extend it to [B, prefix_len + suffix_len, prefix_len + suffix_len]
        
        bsize = prefix_pad_masks_no_fast.shape[0]
        prefix_len = prefix_pad_masks_no_fast.shape[1]
        suffix_len = suffix_pad_masks.shape[1]
        total_len = prefix_len + suffix_len
        device = prefix_pad_masks_no_fast.device
        
        # Create full attention mask
        full_att_2d_masks = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)
        
        # Copy prefix attention pattern
        full_att_2d_masks[:, :prefix_len, :prefix_len] = prefix_att_masks_no_fast
        
        # Suffix attends to all prefix
        full_att_2d_masks[:, prefix_len:, :prefix_len] = True
        
        # Suffix has causal attention among itself
        suffix_causal_mask = torch.tril(torch.ones(suffix_len, suffix_len, dtype=torch.bool, device=device))
        full_att_2d_masks[:, prefix_len:, prefix_len:] = suffix_causal_mask[None, :, :]
        
        # Apply padding masks
        pad_masks = torch.cat([prefix_pad_masks_no_fast, suffix_pad_masks], dim=1)
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        full_att_2d_masks = full_att_2d_masks & pad_2d_masks
        
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks, dtype=prefix_embs_no_fast.dtype)

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
            forward_func, prefix_embs_no_fast, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        fm_loss = F.mse_loss(u_t, v_t, reduction="none")

        return {
            "flow_loss": fm_loss,
            "subtask_loss": subtask_loss,
            "fast_loss": fast_loss,
            "loss": fm_loss.mean() + 0.1 * subtask_loss + 0.05 * fast_loss, # ref: b1k winner
        }

    def embed_prefix_fast(
        self,
        images,
        img_masks,
        tokens,
        masks,
        fast_action_tokens=None,
        fast_action_masks=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Embed images, language tokens, and FAST action tokens for FAST-only mode.
        
        This is a simplified version of embed_prefix without subtask tokens.
        Attention pattern:
        - Images + Language: bidirectional among themselves
        - FAST: attend to images + language, causal among themselves
        
        Args:
            images: List of image tensors
            img_masks: List of image masks
            tokens: Language instruction tokens
            masks: Attention masks for tokens
            fast_action_tokens: FAST action tokens (discrete token IDs)
            fast_action_masks: Padding masks for FAST action tokens
            
        Returns:
            embs: Concatenated embeddings [images, tokens, fast_action_tokens]
            pad_masks: Padding masks
            att_masks: 2D attention mask
            total_T_images: Total number of image tokens
            num_fast_embs: Number of FAST action token embeddings
        """
        embs = []
        pad_masks = []
        att_mask_segments = []
        total_T_images = 0
        num_fast_embs = 0
        
        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_mask_segments.append(('image', num_img_embs))
            total_T_images += num_img_embs
            
        # Process language instruction tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_mask_segments.append(('language', num_lang_embs))

        # Process FAST action tokens (discrete token IDs)
        if fast_action_tokens is not None:
            def fast_action_embed_func(fast_action_tokens):
                fast_emb = self.paligemma_with_expert.embed_language_tokens(fast_action_tokens)
                fast_emb_dim = fast_emb.shape[-1]
                return fast_emb * math.sqrt(fast_emb_dim)
            
            fast_action_emb = self._apply_checkpoint(fast_action_embed_func, fast_action_tokens)
            embs.append(fast_action_emb)
            
            if fast_action_masks is not None:
                fast_pad_mask = fast_action_masks
            else:
                bsize = fast_action_tokens.shape[0]
                num_fast_embs = fast_action_tokens.shape[1]
                fast_pad_mask = torch.ones(bsize, num_fast_embs, dtype=torch.bool, device=fast_action_tokens.device)
            
            num_fast_embs = fast_action_tokens.shape[1]
            pad_masks.append(fast_pad_mask)
            att_mask_segments.append(('fast', num_fast_embs))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        
        # Create custom 2D attention mask for FAST-only mode:
        # - Images + Language: bidirectional among themselves
        # - FAST: attend to images + language, causal among themselves
        att_masks = self._create_custom_attention_mask_fast(att_mask_segments, pad_masks, bsize)

        return embs, pad_masks, att_masks, total_T_images, num_fast_embs

    def _create_custom_attention_mask_fast(self, att_mask_segments, pad_masks, bsize):
        """Create custom 2D attention mask for FAST-only mode.
        
        Attention rules:
        - Images + Language: bidirectional among themselves
        - FAST: attend to images + language, causal among themselves
        """
        total_len = sum(length for _, length in att_mask_segments)
        device = pad_masks.device
        
        att_2d_masks = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)
        
        positions = []
        current_pos = 0
        for seg_type, seg_len in att_mask_segments:
            positions.append((seg_type, current_pos, current_pos + seg_len))
            current_pos += seg_len
        
        for i, (query_type, query_start, query_end) in enumerate(positions):
            for j, (key_type, key_start, key_end) in enumerate(positions):
                # Images and Language can attend to each other bidirectionally
                if query_type in ['image', 'language'] and key_type in ['image', 'language']:
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = True
                
                # FAST tokens attend to images + language
                elif query_type == 'fast' and key_type in ['image', 'language']:
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = True
                
                # FAST tokens attend causally to themselves
                elif query_type == 'fast' and key_type == 'fast':
                    fast_len = query_end - query_start
                    causal_mask = torch.tril(torch.ones(fast_len, fast_len, dtype=torch.bool, device=device))
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = causal_mask[None, :, :]
        
        # Apply padding masks
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        att_2d_masks = att_2d_masks & pad_2d_masks
        
        return att_2d_masks

    def forward_fast_only(
        self,
        images,
        img_masks,
        tokens,
        masks,
        fast_action_tokens,
        fast_action_masks,
    ) -> dict:
        """Forward pass for FAST-only mode (no flow matching, no subtask).
        
        This implements the Pi0FAST training objective: predict next action token
        using cross-entropy loss.
        
        Args:
            images: List of image tensors
            img_masks: List of image masks
            tokens: Language instruction tokens
            masks: Attention masks for tokens
            fast_action_tokens: Discrete action token IDs [B, max_action_tokens]
            fast_action_masks: Padding masks for fast action tokens [B, max_action_tokens]
            
        Returns:
            Dictionary with 'fast_loss' and 'loss' keys
        """
        if fast_action_tokens is None or fast_action_masks is None:
            raise ValueError("fast_action_tokens and fast_action_masks are required for FAST-only mode")

        # Embed prefix with FAST tokens
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_T_images, num_fast_embs = self.embed_prefix_fast(
            images, img_masks, tokens, masks,
            fast_action_tokens=fast_action_tokens,
            fast_action_masks=fast_action_masks
        )

        # Convert embeddings to bfloat16 if needed
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # For next-token prediction, we input tokens [0:T-1] to predict tokens [1:T]
        # So we remove the last token from input
        input_embs = prefix_embs[:, :-1]
        input_pad_masks = prefix_pad_masks[:, :-1]
        input_att_masks = prefix_att_masks[:, :-1, :-1]

        position_ids = torch.cumsum(input_pad_masks, dim=1) - 1
        att_2d_4d = self._prepare_attention_masks_4d(input_att_masks, dtype=input_embs.dtype)

        # Forward pass through paligemma (language model only, no action expert)
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[input_embs, None],  # No suffix/action expert
            use_cache=False,
            adarms_cond=[None, None],
        )

        # Get logits for FAST action tokens using the FAST LM head
        # We only compute logits for the positions that predict FAST tokens
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        # The FAST tokens start at position (total_T_images + num_lang_tokens)
        # For next-token prediction:
        # - Position (fast_start - 1) in input predicts fast_action_tokens[0]
        # - Position (fast_start) in input predicts fast_action_tokens[1], etc.
        T_lang = masks.shape[1]
        fast_start = total_T_images + T_lang
        
        # Extract logits for FAST token prediction
        # Input positions [fast_start-1 : fast_start-1+num_fast_embs] predict FAST tokens
        fast_hidden = prefix_out[:, fast_start-1:fast_start-1+num_fast_embs, :]  # (B, num_fast_embs, hidden_dim)
        fast_logits_for_pred = lm_head(fast_hidden)  # (B, num_fast_embs, gemma_vocab_size)
        
        # Targets are the FAST action tokens
        fast_targets = fast_action_tokens  # (B, num_fast_embs)

        # Compute cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        fast_logits_flat = fast_logits_for_pred.reshape(-1, fast_logits_for_pred.size(-1))
        fast_targets_flat = fast_targets.reshape(-1)
        
        fast_loss_per_token = loss_fct(fast_logits_flat, fast_targets_flat)
        fast_loss_per_token = fast_loss_per_token.reshape(fast_targets.shape)
        
        # Apply mask and compute mean loss
        masked_fast_loss = fast_loss_per_token * fast_action_masks.float()
        fast_loss = masked_fast_loss.sum() / fast_action_masks.sum().clamp(min=1)

        # breakpoint()
        # from transformers import AutoTokenizer, AutoProcessor
        # _paligemma_tokenizer = AutoTokenizer.from_pretrained(
        #     "google/paligemma-3b-pt-224", 
        #     trust_remote_code=True, 
        #     add_eos_token=True, 
        #     add_bos_token=False
        # )
        # # 257152
        # # # Decode predicted output tokens
        # # # fast_logits_for_pred.argmax(dim=-1)
        # def _paligemma_tokens_to_act_tokens(tokens: torch.Tensor) -> torch.Tensor:
        #     """
        #     Converts PaliGemma tokens back to action tokens (inverse of _act_tokens_to_paligemma_tokens).
        #     """
        #     return _paligemma_tokenizer.vocab_size - 1 - 128 - tokens
        # # # target = _paligemma_tokens_to_act_tokens(fast_targets)
        # decoded_tokens = _paligemma_tokenizer.batch_decode(fast_targets, skip_special_tokens=False)
        # decoded_tokens = [
        #     _paligemma_tokenizer.convert_ids_to_tokens(seq.tolist())
        #     for seq in fast_logits_for_pred.argmax(dim=-1)
        # ]
        # cleaned_tokens = []
        # for token_seq in decoded_tokens:
        #     if "|" in token_seq:
        #         token_seq = token_seq[:token_seq.index("|")]
        #     cleaned_tokens.append(token_seq)
        # raw_action_tokens = [
        #     torch.tensor(
        #         _paligemma_tokenizer.convert_tokens_to_ids(token_seq),
        #         dtype=torch.long,
        #         device=fast_targets.device,
        #     )
        #     for token_seq in cleaned_tokens
        # ]
        
        # action_tokens = [
        #     _paligemma_tokens_to_act_tokens(raw_action_token) 
        #     for raw_action_token in raw_action_tokens
        # ]
        # breakpoint()
        # # Clean the decoded tokens by removing "Action:" prefix and extracting the relevant part
        # cleaned_tokens = [
        #     tokens_sequence.strip().split("|")[0].strip()
        #     for tokens_sequence in decoded_tokens
        # ]
        
        # # Re-encode the cleaned text to get raw action tokens
        # raw_action_tokens = [
        #     _paligemma_tokenizer.encode(sample_tokens, return_tensors="pt", padding=False).squeeze(0)
        #     for sample_tokens in cleaned_tokens
        # ]
        # # Convert PaliGemma tokens back to action tokens
        # action_tokens = [
        #     _paligemma_tokens_to_act_tokens(raw_action_token) 
        #     for raw_action_token in raw_action_tokens
        # ]
        # # # Decode each sample's tokens to continuous actions
        # action_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
        # # breakpoint()
        # decoded_actions = action_tokenizer.decode(
        #     action_tokens, 
        #     time_horizon=self.config.chunk_size, 
        #     action_dim=6
        # )
        # breakpoint()
        # def decode_actions_with_fast(
        #     token_ids: list[int], 
        #     time_horizon: int, 
        #     action_dim: int, 
        #     relaxed_decoding: bool = False
        # ) -> list:
        #     """
        #     Decodes action token IDs back to continuous action values using the FAST tokenizer.

        #     Args:
        #         token_ids: List of token IDs to decode.
        #         time_horizon: The number of timesteps for actions.
        #         action_dim: The dimensionality of each action.
        #         relaxed_decoding: Whether to use relaxed decoding (allows partial sequences).

        #     Returns:
        #         A list representing the decoded actions.
        #     """
        #     # Use the action tokenizer's decode method
        #     # The FAST tokenizer should have a decode method that converts tokens back to actions
        #     try:
        #         decoded_actions = action_tokenizer.decode(
        #             token_ids, 
        #             time_horizon=time_horizon, 
        #             action_dim=action_dim
        #         )
        #         return decoded_actions
        #     except Exception as e:
        #         if relaxed_decoding:
        #             # If relaxed decoding is enabled, try to decode as much as possible
        #             import logging
        #             logging.warning(f"Relaxed decoding: {e}. Returning partial decode.")
        #             try:
        #                 # Try to decode with whatever tokens we have
        #                 partial_decoded = action_tokenizer.decode(
        #                     token_ids[:len(token_ids)], 
        #                     time_horizon=time_horizon, 
        #                     action_dim=action_dim
        #                 )
        #                 return partial_decoded
        #             except:
        #                 # Return zeros if decoding completely fails
        #                 return [[0.0] * action_dim for _ in range(time_horizon)]
        #         else:
        #             raise e
        
        # valid = fast_logits_for_pred.argmax(dim=-1) <= (self._paligemma_tokenizer.vocab_size - 1 - 128)
        # fast_region = fast_logits_for_pred.argmax(dim=-1).masked_fill(~valid, 0)
        # fast_tokens = _paligemma_tokens_to_act_tokens(fast_region)
        # actions = decode_actions_with_fast(fast_tokens.tolist(), time_horizon=self.config.chunk_size, action_dim=7, relaxed_decoding=True)[0]
        # breakpoint()
        # decoded_actions = [
        #     torch.tensor(
        #         decode_actions_with_fast(
        #             tok[0].tolist(),
        #             time_horizon=self.config.chunk_size,
        #             action_dim=7,
        #             relaxed_decoding=True,
        #         ),
        #         device=tokens.device,
        #     ).squeeze(0)
        #     for tok in action_tokens
        # ]
        # breakpoint()
        # # Stack into a batch
        # result = torch.stack(decoded_actions, dim=0)
        # breakpoint()
        return {
            "fast_loss": fast_loss,
            "loss": fast_loss,
        }
    
    @torch.no_grad()
    def sample_actions_fast(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps=None,
        temperature=0.0,
    ) -> torch.Tensor:
        """
        Inefficient but safe autoregressive decoding for FAST tokens.
        Matches the pattern of _generate_subtask_tokens.
        """
        if max_decoding_steps is None:
            max_decoding_steps = self.config.max_action_tokens

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        # 1. Initial Embedding (Matches Training Prefix)
        # prefix_embs will include [Images, Language Prompt]
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_T_images, _ = self.embed_prefix_fast(
            images, img_masks, tokens, masks,
            fast_action_tokens=None,
            fast_action_masks=None
        )

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        generated_action_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)

        # 2. Decoding Loop (Re-computes full sequence every step)
        for t in range(max_decoding_steps):
            # Always re-calculate position IDs from the current pad mask (matches training)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            att_4d = self._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

            # Full forward pass (No KV Cache)
            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )

            # Predict next token from the very last sequence position
            last_logits = lm_head(prefix_out[:, -1:, :]) # (B, 1, vocab_size)

            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            generated_action_tokens[:, t] = next_token.squeeze(-1)

            # 3. Update Sequence for next iteration (unless it's the last step)
            if t < max_decoding_steps - 1:
                # Embed the newly generated token
                next_token_emb = self.paligemma_with_expert.embed_language_tokens(next_token)
                next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
                if prefix_embs.dtype == torch.bfloat16:
                    next_token_emb = next_token_emb.to(dtype=torch.bfloat16)

                # Append to embeddings
                prefix_embs = torch.cat([prefix_embs, next_token_emb], dim=1)

                # Update padding mask (New token is always valid/1)
                prefix_pad_masks = torch.cat([
                    prefix_pad_masks,
                    torch.ones((bsize, 1), dtype=torch.bool, device=device)
                ], dim=1)

                # Update 2D attention mask: Grow the matrix
                old_len = prefix_att_masks.shape[1]
                new_len = old_len + 1
                new_att_masks = torch.zeros((bsize, new_len, new_len), dtype=torch.bool, device=device)
                new_att_masks[:, :old_len, :old_len] = prefix_att_masks
                # New token attends to all non-padding tokens in the updated sequence
                new_att_masks[:, -1, :] = prefix_pad_masks 
                prefix_att_masks = new_att_masks
        return generated_action_tokens

    @torch.no_grad()
    def sample_actions_fast_kv_cache(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps=None,
        temperature=0.0,
    ) -> torch.Tensor:
        """
        Efficient autoregressive decoding for FAST tokens using KV-caching.
        Only computes the prefix once, then incrementally generates tokens.
        """
        if max_decoding_steps is None:
            max_decoding_steps = self.config.max_action_tokens

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        # 1. Initial Embedding (Matches Training Prefix)
        # prefix_embs will include [Images, Language Prompt]
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_T_images, _ = self.embed_prefix_fast(
            images, img_masks, tokens, masks,
            fast_action_tokens=None,
            fast_action_masks=None
        )

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        generated_action_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)

        # 2. Initial forward pass to populate KV cache
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

        # First forward pass with full prefix (caching enabled)
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        # Predict first token from the last sequence position
        last_logits = lm_head(prefix_out[:, -1:, :])  # (B, 1, vocab_size)

        if temperature > 0:
            probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

        generated_action_tokens[:, 0] = next_token.squeeze(-1)

        # Track current sequence length for position IDs and maintain the padding mask
        current_seq_len = prefix_embs.shape[1]
        # Keep track of valid positions: prefix_pad_masks tells us which positions are valid
        current_pad_mask = prefix_pad_masks.clone()  # (B, seq_len)

        # 3. Incremental Decoding Loop (using KV cache)
        for t in range(1, max_decoding_steps):
            # Embed the newly generated token
            next_token_emb = self.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            if prefix_embs.dtype == torch.bfloat16:
                next_token_emb = next_token_emb.to(dtype=torch.bfloat16)

            # Update padding mask: new generated token is always valid
            current_pad_mask = torch.cat([
                current_pad_mask,
                torch.ones((bsize, 1), dtype=torch.bool, device=device)
            ], dim=1)  # (B, seq_len+1)

            # Position ID for the new token (continues from where we left off)
            new_position_id = torch.full((bsize, 1), current_seq_len, dtype=torch.long, device=device)

            # For KV-cache: attention mask for the new token should only attend to valid positions
            # Shape: (B, 1, past_len+1) where the new token attends to valid prefix + all generated tokens
            new_att_mask_2d = current_pad_mask.unsqueeze(1)  # (B, 1, seq_len+1)
            att_4d_incremental = self._prepare_attention_masks_4d(new_att_mask_2d, dtype=next_token_emb.dtype)

            # Forward pass with only the new token embedding (reusing cached KVs)
            (new_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=att_4d_incremental,
                position_ids=new_position_id,
                past_key_values=past_key_values,
                inputs_embeds=[next_token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )

            # Predict next token
            last_logits = lm_head(new_out[:, -1:, :])  # (B, 1, vocab_size)

            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            generated_action_tokens[:, t] = next_token.squeeze(-1)

            # Update sequence length
            current_seq_len += 1

        return generated_action_tokens
    
    
    
    @torch.no_grad()
    def _generate_subtask_tokens(
        self, images, img_masks, tokens, masks, tokenizer, max_length, device
    ):
        bsize = tokens.shape[0]
        lm_head = self.paligemma_with_expert.paligemma.lm_head
        
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_T_images, _, _ = self.embed_prefix(
            images, img_masks, tokens, subtask_tokens=None, masks=masks, subtask_masks=None, 
            fast_action_tokens=None, fast_action_masks=None
        )

        generated_tokens = torch.zeros((bsize, max_length), dtype=torch.long, device=device)
        
        # tracking mask: False = still generating, True = finished
        finished = torch.zeros(bsize, dtype=torch.bool, device=device)
        
        for t in range(max_length):
            position_ids_prefix = torch.cumsum(prefix_pad_masks, dim=1) - 1
            att_2d_prefix_4d = self._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_prefix_4d, 
                position_ids=position_ids_prefix,
                inputs_embeds=[prefix_embs, None],
                # ...
            )
            
            logits = lm_head(prefix_out)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1) # (B,)
            
            # 1. if a row was already finished, force the next token to be PAD (0)
            next_token = torch.where(finished, torch.tensor(0, device=device), next_token)
            
            # 2. store the token
            generated_tokens[:, t] = next_token
            
            # 3. update the finished mask
            if tokenizer.eos_token_id is not None:
                finished |= (next_token == tokenizer.eos_token_id)
            
            # 4. break only if everyone is finished
            if finished.all():
                break
                
            next_token_unsqueezed = next_token.unsqueeze(1)
            
            def next_token_embed_func(next_token_unsqueezed):
                next_emb = self.paligemma_with_expert.embed_language_tokens(next_token_unsqueezed)
                return next_emb * math.sqrt(next_emb.shape[-1])
            
            next_emb = self._apply_checkpoint(next_token_embed_func, next_token_unsqueezed)
            
            # update embeddings
            prefix_embs = torch.cat([prefix_embs, next_emb], dim=1)
            
            # update padding masks
            prefix_pad_masks = torch.cat([
                prefix_pad_masks,
                torch.ones((bsize, 1), dtype=torch.bool, device=device)
            ], dim=1)
            
            # update attention masks
            old_seq_len = prefix_att_masks.shape[1]
            new_seq_len = old_seq_len + 1
            new_att_masks = torch.zeros((bsize, new_seq_len, new_seq_len), dtype=torch.bool, device=device)
            new_att_masks[:, :old_seq_len, :old_seq_len] = prefix_att_masks
            new_att_masks[:, -1, :] = prefix_pad_masks
            prefix_att_masks = new_att_masks

        return generated_tokens

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        tokenizer=None,
        max_subtask_tokens=50,
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

        # Generate subtask tokens autoregressively during inference
        generated_subtask_tokens = None
        if tokenizer is not None:
            generated_subtask_tokens = self._generate_subtask_tokens(
                images, img_masks, tokens, masks, tokenizer, max_subtask_tokens, device
            )
            
            # Decode and print the generated subtask tokens
            for i in range(bsize):
                # Remove padding tokens (0) and special tokens
                valid_tokens = generated_subtask_tokens[i][generated_subtask_tokens[i] != 0]
                decoded_text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                print(f"[Inference] Generated subtask {i}: {decoded_text}")

        # Create mask for generated tokens (all valid)
        subtask_masks = torch.ones_like(generated_subtask_tokens, dtype=torch.bool)

        # During inference, we don't have subtask_tokens yet, so pass None
        # Also no FAST tokens during inference
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, _ = self.embed_prefix(
            images, img_masks, tokens, subtask_tokens=generated_subtask_tokens, masks=masks, subtask_masks=subtask_masks, 
            fast_action_tokens=None, fast_action_masks=None
        )
        
        # Convert embeddings to bfloat16 if needed for the model
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        # prefix_att_masks is already a 2D attention mask from embed_prefix
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Define a closure function to properly capture expanded_time
            # This avoids the lambda expression (E731) and loop variable binding (B023) issues
            def denoise_step_partial_call(input_x_t, current_timestep=expanded_time):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            # Euler step
            x_t += dt * v_t

            # Record x_t and v_t after Euler step
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

            time += dt

        return x_t

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

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks, dtype=suffix_embs.dtype)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

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

    def __init__(
        self,
        config: PI05Config,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
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
        
        # Load tokenizer for subtask decoding
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        except Exception as e:
            logging.warning(f"Could not load tokenizer for subtask decoding: {e}")
            self.tokenizer = None

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

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
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

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)
            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

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
            print(f"Warning: Could not remap state dict keys: {e}")

        return model

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

            if key == "model.paligemma_with_expert.paligemma.lm_head.weight":
                fixed_state_dict["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

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

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

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

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        
        # FAST-only mode: use autoregressive decoding
        if self.config.fast_only:
            tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
            masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
            
            # Get optional parameters
            temperature = kwargs.get("temperature", 0.0)
            max_decoding_steps = 256
            
            # Sample action tokens autoregressively
            action_tokens = self.model.sample_actions_fast_kv_cache(
                images, img_masks, tokens, masks,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
            )
            # Return the action tokens - these need to be decoded by the FAST tokenizer
            # The caller is responsible for decoding tokens to continuous actions
            return action_tokens

        # Full mode: use flow matching with optional subtask generation
        # Use high_level_task tokens (WITHOUT subtask) for inference - we'll generate the subtask
        high_level_task = batch[f"{OBS_LANGUAGE_HIGH_LEVEL_TASK_TOKENS}"]
        high_level_task_masks = batch[f"{OBS_LANGUAGE_HIGH_LEVEL_TASK_ATTENTION_MASK}"]
        
        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(
            images, img_masks, high_level_task, high_level_task_masks, 
            tokenizer=self.tokenizer,
            **kwargs
        )

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        
        # Get FAST action tokens from batch
        fast_action_tokens = batch.get("action.tokens", None)  # (B, max_action_tokens)
        fast_action_masks = batch.get("action.token_mask", None)  # (B, max_action_tokens)
        
        # FAST-only mode: only use discrete action token prediction
        if self.config.fast_only:
            # Use full language tokens (no separation into high_level_task and subtask)
            tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
            masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
            
            if fast_action_tokens is None or fast_action_masks is None:
                raise ValueError("FAST-only mode requires action.tokens and action.token_mask in the batch")
            
            loss_dict = self.model.forward_fast_only(
                images, img_masks, tokens, masks,
                fast_action_tokens=fast_action_tokens,
                fast_action_masks=fast_action_masks
            )
            
            loss = loss_dict["loss"]
            detailed_loss_dict = {
                "loss": loss.item(),
                "fast_loss": loss_dict["fast_loss"].item(),
            }
            return loss, detailed_loss_dict

        # Full mode: flow matching + subtask + FAST
        high_level_task = batch[f"{OBS_LANGUAGE_HIGH_LEVEL_TASK_TOKENS}"]
        high_level_task_masks = batch[f"{OBS_LANGUAGE_HIGH_LEVEL_TASK_ATTENTION_MASK}"]
        subtask_tokens, subtask_masks = batch[f"{OBS_LANGUAGE_SUBTASK_ONLY_TOKENS}"], batch[f"{OBS_LANGUAGE_SUBTASK_ONLY_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        
        # Compute loss (no separate state needed for PI05)
        # high_level_task = instruction tokens WITHOUT subtask (e.g., "High level task: X; State: Y; Subtask:")
        # subtask_tokens = subtask tokens to predict (e.g., "pick up the cup")
        # fast_action_tokens = discrete action token IDs to predict
        loss_dict = self.model.forward(
            images, img_masks, high_level_task, high_level_task_masks, subtask_tokens, subtask_masks, actions,
            fast_action_tokens=fast_action_tokens, fast_action_masks=fast_action_masks
        )

        # Extract the total loss
        loss = loss_dict["loss"]
        
        # Prepare detailed loss dictionary for logging
        detailed_loss_dict = {
            "loss": loss.item(),
            "flow_loss": loss_dict["flow_loss"].mean().item(),
            "subtask_loss": loss_dict["subtask_loss"].item(),
            "fast_loss": loss_dict["fast_loss"].item(),
        }

        return loss, detailed_loss_dict
