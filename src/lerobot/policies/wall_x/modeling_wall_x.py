#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
Wall-X: Cross-embodiment robotic control using Qwen2.5-VL with flow matching.

[Paper](https://github.com/x2-robot/wall-x)

Install wall-x extra dependencies:
```bash
pip install -e ".[wall_x]"
```

Example of finetuning a wall-x model:
```bash
lerobot-train \
--policy.type=wall_x \
--dataset.repo_id=your/dataset \
--batch_size=32 \
--steps=100000
```
"""

import glob
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch import Tensor
from torch.distributions import Beta
from torch.nn import CrossEntropyLoss
from torchdiffeq import odeint
from transformers import AutoConfig, AutoProcessor
from transformers.activations import ACT2FN
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_torchdynamo_compiling, logging
from transformers import AutoProcessor, BatchFeature
from qwen_vl_utils.vision_process import smart_resize

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from lerobot.policies.wall_x.utils import *
from lerobot.policies.wall_x.constant import *
from lerobot.policies.wall_x.qwen_model.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2RMSNorm,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
)

from lerobot.policies.wall_x.qwen_model.qwen2_5_vl_moe import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLDecoderLayer_with_MoE,
    Qwen2_5_VLACausalLMOutputWithPast,
)

# Add wall-x repo to path if available
WALL_X_PATH = Path("/x2robot_v2/vincent/workspace/lerobot_opensource/wall-x")
if WALL_X_PATH.exists():
    sys.path.insert(0, str(WALL_X_PATH))


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionHead(nn.Module):
    """
    Action prediction head with flow matching.

    Implements Beta-distributed noise scheduling and temporal embeddings
    for action sequence prediction.
    """

    def __init__(self, config: WallXConfig):
        super().__init__()

        self.config = config
        self.action_dim = sum(config.dof_config.values())
        self.propri_dim = sum(config.agent_pos_config.values())
        self.hidden_size = config.hidden_size

        # Beta distribution for noise scheduling
        noise_config = config.noise_scheduler
        self.beta_alpha = noise_config.get("beta_alpha", 1.5)
        self.beta_beta = noise_config.get("beta_beta", 1.0)
        self.s = noise_config.get("s", 0.999)

        # Sinusoidal timestep embedding
        self.time_embed = SinusoidalPosEmb(config.hidden_size)

        # Action embedding network
        # *2 for action + DOF mask concatenation
        self.w1 = nn.Linear(self.action_dim * 2, self.hidden_size, bias=False)
        self.w2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)  # *2 for action + time
        self.w3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        # Project back to action space
        self.action_proj_back = nn.Linear(self.hidden_size, self.action_dim, bias=False)

        # Proprioception projection
        self.propri_proj = nn.Linear(self.propri_dim * 2, self.hidden_size, bias=False)

    def sample_time(self, batch_size, device, dtype):
        """Sample timesteps using Beta distribution."""
        beta_dist = Beta(
            torch.tensor(self.beta_alpha, dtype=dtype, device=device),
            torch.tensor(self.beta_beta, dtype=dtype, device=device)
        )
        sample = beta_dist.sample([batch_size])
        time = (1 - sample) * self.s
        return time

    def forward(self, action_chunk, dof_mask=None):
        """
        Process action sequences with noise injection for training.

        Args:
            action_chunk: Action sequences [batch, seq_len, action_dim]
            dof_mask: DOF mask [batch, seq_len, action_dim]

        Returns:
            tuple: (action_embeddings, flow_target)
        """
        batch_size = action_chunk.shape[0]
        device = action_chunk.device
        dtype = action_chunk.dtype

        # Add noise using flow matching
        noise = torch.randn_like(action_chunk)
        time = self.sample_time(batch_size, device, dtype)
        t = time.unsqueeze(-1).unsqueeze(-1)

        # Linear interpolation
        noisy_action = (1 - t) * noise + t * action_chunk
        flow = action_chunk - noise

        # Generate time embeddings
        time_embed = self.time_embed(time)

        # Project noisy actions
        if dof_mask is not None:
            noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)

        noisy_action = noisy_action.to(dtype=self.w1.weight.dtype)
        action_embed = self.w1(noisy_action)

        # Combine with time embeddings
        time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
        time_embed = time_embed.to(dtype=self.w2.weight.dtype)

        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = self.w2(concat_embed)
        embed = self.w3(self.act_fn(concat_embed))

        return embed, flow

    def step(self, timestep, noisy_action, dof_mask=None):
        """Single denoising step for inference."""
        if dof_mask is not None:
            noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)

        time_embed = self.time_embed(timestep)
        action_embed = self.w1(noisy_action)

        time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
        time_embed = time_embed.to(device=noisy_action.device, dtype=noisy_action.dtype)

        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = self.w2(concat_embed)
        embed = self.w3(self.act_fn(concat_embed))

        return embed

    def flow_loss(self, action_hidden_states, flow, dof_mask=None):
        """Compute flow matching loss."""
        action_pred = self.action_proj_back(action_hidden_states)
        loss = F.mse_loss(action_pred, flow, reduction="none")

        if dof_mask is not None:
            dof_mask = dof_mask.reshape(-1, dof_mask.shape[-1])
            loss = loss * dof_mask

        return loss

    def proprioception_proj(self, proprioception, dof_mask=None, use_history=False):
        """Project proprioceptive data to hidden space."""
        # Ensure proper device and dtype alignment
        proprioception = proprioception.to(device=self.propri_proj.weight.device).to(
            dtype=self.propri_proj.weight.dtype
        )

        if dof_mask is not None:
            # Concatenate proprioception with DOF mask
            # TODO: Use variable-based dimension checking for better flexibility
            if use_history:
                proprioception = torch.cat([proprioception, dof_mask], dim=-1)
            else:
                proprioception = torch.cat([proprioception, dof_mask], dim=-1)

        proprioception = proprioception.to(device=self.propri_proj.weight.device).to(
            dtype=self.propri_proj.weight.dtype
        )
        return self.propri_proj(proprioception)


class Qwen2_5_VLMoEModel(Qwen2_5_VLPreTrainedModel):
    """Qwen2.5-VL model with Mixture of Experts (MoE) architecture.

    This model extends the base Qwen2.5-VL model by incorporating MoE layers
    for improved scalability and specialization across different token types.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_experts: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """Load a pretrained model with optional MoE configuration.

        Args:
            pretrained_model_name_or_path: Path or name of the pretrained model
            num_experts: Number of experts for MoE layers (if not in config)
            *args: Additional arguments passed to parent class
            **kwargs: Additional keyword arguments passed to parent class

        Returns:
            Initialized model instance with MoE configuration
        """
        config = kwargs.get("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Override number of experts if specified
        if num_experts is not None:
            config.num_experts = num_experts

        kwargs["config"] = config
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def __init__(self, config: Qwen2_5_VLConfig):
        """Initialize the Qwen2.5-VL MoE model.

        Args:
            config: Model configuration containing architecture parameters
        """
        super().__init__(config)

        # Basic model parameters
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Model components
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        # Decoder layers with MoE support
        self.layers = nn.ModuleList(
            [
                Qwen2_5_VLDecoderLayer_with_MoE(config, layer_idx, config.num_experts)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Model configuration
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embedding layer.

        Returns:
            The token embedding layer
        """
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the input embedding layer.

        Args:
            value: New embedding layer to use
        """
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        moe_token_types: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Set default output options
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Validate inputs
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if moe_token_types is None:
            raise ValueError("moe_token_types must be provided for MoE routing")

        # Handle gradient checkpointing compatibility
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Initialize cache if needed
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Set up cache position
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Set up position IDs (hardcoded 3 dimensions for temporal, height, width)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Create causal attention mask
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
            moe_token_types,
        )

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Initialize output collections
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # Process through decoder layers
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing during training
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    moe_token_types,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                # Regular forward pass
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    token_types=moe_token_types,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            # Update cache if using it
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            # Collect attention weights if requested
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Apply final layer normalization
        hidden_states = self.norm(hidden_states)

        # Add final hidden states if collecting all states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # Return outputs in requested format
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        moe_token_types: Optional[torch.LongTensor] = None,
    ):
        """Update causal attention mask with support for bidirectional attention for specific token types.

        This method creates and modifies attention masks to support different attention patterns:
        - Standard causal (unidirectional) attention for most tokens
        - Bidirectional attention for specific token types (e.g., MoE routing tokens)

        Args:
            attention_mask: Input attention mask to avoid attending to padding tokens
            input_tensor: Input embeddings tensor for shape and device information
            cache_position: Position indices for caching mechanisms
            past_key_values: Cached key-value pairs from previous forward passes
            output_attentions: Whether attention weights will be returned
            moe_token_types: Optional tensor indicating token types for MoE routing
                            (type 1 tokens will use bidirectional attention)

        Returns:
            Updated causal attention mask, or None if using Flash Attention 2
        """
        # Flash Attention 2 handles masking internally
        if self.config._attn_implementation == "flash_attention_2":
            return None

        # Calculate sequence lengths for cache management
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # For SDPA (Scaled Dot Product Attention), use `is_causal` argument when possible
        # instead of explicit attention mask to enable Flash Attention 2 dispatch
        # Note: This optimization is not compatible with static cache
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            # Check if we can ignore the causal mask and rely on SDPA's internal handling
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        # Extract tensor properties for mask creation
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        # Determine target length based on cache type
        if using_sliding_window_cache or using_static_cache:
            # Use maximum cache shape for sliding window or static caches
            target_length = past_key_values.get_max_cache_shape()
        else:
            # For dynamic cache or no cache, calculate based on attention mask or sequence length
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # Generate 4D causal attention mask from 2D input mask if provided
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        # Modify mask to support bidirectional attention for specific token types
        if moe_token_types is not None:
            # Identify positions of type 1 tokens (MoE routing tokens)
            type1_tokens = (
                (moe_token_types == 1).unsqueeze(1).unsqueeze(2)
            )  # Shape: [B, 1, 1, S]

            # Create bidirectional attention region for type 1 tokens
            # This allows type 1 tokens to attend to each other bidirectionally
            type1_mask = torch.zeros_like(causal_mask)  # Shape: [B, num_heads, S, S]
            type1_region = type1_tokens & type1_tokens.transpose(
                -1, -2
            )  # Shape: [B, 1, S, S]
            type1_mask = type1_mask.masked_fill(type1_region, 1.0).to(torch.bool)

            # Apply bidirectional attention: zero out causal constraints in type 1 regions
            causal_mask = torch.where(
                type1_mask,  # Where type 1 tokens interact with each other
                torch.zeros_like(
                    causal_mask
                ),  # Remove causal masking (allow bidirectional)
                causal_mask,  # Keep original causal masking for other regions
            )

        # Handle special case for SDPA with CUDA/XPU devices
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Ensure attention to all tokens in fully masked rows for memory-efficient attention
            # This is required for F.scaled_dot_product_attention's memory-efficient path
            # when using left padding. See: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5_VLConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2_5_VLConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            diagonal_attend_mask = torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if (
                    not isinstance(past_key_values, SlidingWindowCache)
                    or sequence_length > target_length
                ):
                    sliding_attend_mask = torch.arange(
                        target_length, device=device
                    ) <= (cache_position.reshape(-1, 1) - config.sliding_window)
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask


class Qwen2_5_VLMoEForAction(Qwen2_5_VLForConditionalGeneration):
    """
    Qwen2.5 Vision-Language Mixture of Experts model for action processing.

    This model extends the base Qwen2.5 VL model with action token processing capabilities
    and optional LoRA fine-tuning support.
    """

    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer_with_MoE", "Qwen2_5_VLVisionBlock"]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        train_config,
        config_path=None,
        processor_path=None,
        action_tokenizer_path=None,
        **kwargs,
    ):
        """
        Load model from pretrained model path.

        Args:
            pretrained_model_path (str): Model directory path containing model.safetensors file
            config_path (str, optional): Configuration file path, if None will look for qwen25_config.json in pretrained_model_path
            processor_path (str, optional): Processor path, if None will load from default config
            action_tokenizer_path (str, optional): Action tokenizer path, if None will load from default config
            **kwargs: Additional arguments

        Returns:
            Qwen2_5_VLMoEForAction: Loaded model instance
        """
        # Load model components from pretrained path
        config_path = os.path.join(pretrained_model_path, "config.json")
        config = cls.config_class.from_pretrained(config_path)
        processor = AutoProcessor.from_pretrained(pretrained_model_path, use_fast=True)
        if action_tokenizer_path is not None:
            processor.action_processor = AutoProcessor.from_pretrained(
                action_tokenizer_path, trust_remote_code=True
            )

        # Initialize model with configuration and processor
        model = cls(config, processor=processor, **kwargs)

        # Resize token embeddings to match processor tokenizer vocabulary size
        model.resize_token_embeddings(len(processor.tokenizer))

        # Load model state dict from safetensors file
        safetensor_files = glob.glob(
            os.path.join(pretrained_model_path, "*.safetensors")
        )
        state_dict = {}
        for file in safetensor_files:
            sd = load_file(file, device="cpu")
            # filter normalizer statistic params
            del_keys = []
            for key in sd.keys():
                if "action_preprocessor.normalizer" in key:
                    print(f"filter load model weight {key}")
                    del_keys.append(key)
            for key in del_keys:
                del sd[key]
            state_dict.update(sd)

        model.load_state_dict(state_dict, strict=False)

        return model

    def __init__(
        self,
        config,
        use_fast_tokenizer=False,
        processor=None,
        action_tokenizer=None,
        action_mapper=None,
        flow_loss_weight=1.0,
    ):
        """
        Initialize the Qwen2.5 VLMoE model for action processing.

        Args:
            config: Model configuration
            use_fast_tokenizer (bool): Whether to use fast tokenizer
            processor: Text and image processor
            action_tokenizer: Action-specific tokenizer
            action_mapper: Action mapping utility
            flow_loss_weight (float): Weight for flow loss computation
        """
        super().__init__(config)

        # Initialize vision transformer and language model components
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            config.vision_config
        )
        self.model = Qwen2_5_VLMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize loss function without reduction for channel-wise loss computation
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.flow_loss_weight = flow_loss_weight
        self.use_fast_tokenizer = use_fast_tokenizer
        self.processor = processor

        # Define action token IDs
        self.define_action_token_id()

        # Cache for rope deltas
        self.rope_deltas = None

        # Initialize action preprocessor
        self.action_preprocessor = ActionHead(config)

        # Apply LoRA if specified in configuration
        if hasattr(config, "use_lora") and config.use_lora:
            self.add_lora(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
            )

        # Initialize weights and apply final processing
        self.post_init()

    def to_bfloat16_for_selected_params(self):
        self.to(dtype=torch.bfloat16)

        params_to_keep_float32 = []

        for name, param in self.named_parameters():
            if "input_layernorm" in name or "post_attention_layernorm" in name or "model.norm" in name:
                params_to_keep_float32.append(name)
            if "action_preprocessor" in name:
                params_to_keep_float32.append(name)
        
        for name, param in self.named_parameters():
            if name in params_to_keep_float32:
                param.data = param.data.to(torch.float32)

    def define_action_token_id(self):
        """
        Define action token IDs based on tokenizer configuration.

        Creates mappings for fast action tokens, proprioception tokens, and general action tokens.
        """
        # Create list of fast action token IDs
        fast_action_token_list = []
        if self.use_fast_tokenizer:
            for i in range(
                self.processor.tokenizer.init_kwargs["action_token_vocab_size"]
            ):
                action_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                    f"<|action_token_{i}|>"
                )
                fast_action_token_list.append(action_token_id)

        # Get special action token IDs
        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        propri_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|propri|>")

        # Store action token ID mappings
        self.action_token_id_set = {
            "fast_action_token_list": fast_action_token_list,
            "propri_token_id": propri_token_id,
            "action_token_id": action_token_id,
        }

    def add_lora(
        self, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1
    ):
        """
        Add LoRA (Low-Rank Adaptation) adapters to the model.

        Args:
            r (int): Rank of adaptation
            lora_alpha (int): LoRA scaling parameter
            target_modules (list): List of module names to apply LoRA to
            lora_dropout (float): Dropout probability for LoRA layers
        """
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

        # Print information about trainable parameters
        self.model.print_trainable_parameters()

    def get_input_embeddings(self):
        """Get input embeddings layer."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Set input embeddings layer."""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Get output embeddings layer."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings layer."""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """Set the decoder model."""
        self.model = decoder

    def get_decoder(self):
        """Get the decoder model."""
        return self.model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate 3D RoPE (Rotary Position Embedding) indices for vision and text tokens.

        This method computes position embeddings that account for the temporal, height, and width
        dimensions of vision tokens (images/videos) while maintaining standard 1D position embeddings
        for text tokens.

        For vision tokens, 3D position embeddings are calculated based on:
        - Temporal dimension: Time patches in videos
        - Height dimension: Vertical patches in images/video frames
        - Width dimension: Horizontal patches in images/video frames

        For text tokens, standard 1D position embeddings are used, continuing from the maximum
        vision position ID plus 1.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs of shape (batch_size, sequence_length)
            image_grid_thw (torch.LongTensor, optional): Image grid dimensions (num_images, 3) for [temporal, height, width]
            video_grid_thw (torch.LongTensor, optional): Video grid dimensions (num_videos, 3) for [temporal, height, width]
            second_per_grid_ts (torch.Tensor, optional): Time interval per temporal grid (num_videos,)
            attention_mask (torch.Tensor, optional): Attention mask (batch_size, sequence_length)

        Returns:
            tuple:
                - position_ids (torch.LongTensor): 3D position IDs of shape (3, batch_size, sequence_length)
                - mrope_position_deltas (torch.Tensor): Position deltas for mRoPE of shape (batch_size, 1)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)

            # Initialize 3D position IDs tensor
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)

            # Process each sequence in the batch
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0

                # Find vision tokens and count images/videos
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                # Process each vision token (image or video)
                for _ in range(image_nums + video_nums):
                    # Find next image or video token
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1

                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1

                    # Determine if processing image or video token
                    if ed_image < ed_video:
                        # Process image token
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        # Process video token
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    # Calculate grid dimensions after spatial merging
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    # Add position IDs for text tokens before vision token
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    # Calculate 3D position embeddings for vision tokens
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    # Calculate temporal position IDs with time scaling
                    time_tensor = (
                        expanded_range
                        * second_per_grid_t
                        * self.config.vision_config.tokens_per_second
                    )
                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    # Calculate spatial position IDs
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )

                    # Add 3D position IDs for vision tokens
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # Add position IDs for remaining text tokens
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                # Concatenate all position IDs for this sequence
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )

            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            # Handle case without vision tokens - use standard 1D position embeddings
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def train_step_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        moe_token_types: Optional[
            torch.LongTensor
        ] = None,  # MoE token type assignments
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        action_chunk: Optional[torch.FloatTensor] = None,  # Action trajectory chunks
        proprioception: Optional[
            torch.FloatTensor
        ] = None,  # Joint position/orientation data
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        dataset_names: Optional[str] = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLACausalLMOutputWithPast]:
        """
        Forward pass for training with multi-modal inputs including vision, text, and action data.

        This method handles the complete forward pass during training, processing various input modalities
        including images, videos, text, proprioceptive data, and action sequences. It computes losses
        for both language modeling and action prediction using flow matching.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask for input tokens
            position_ids (torch.LongTensor, optional): Position IDs for tokens
            past_key_values (List[torch.FloatTensor], optional): Cached key-value pairs for generation
            inputs_embeds (torch.FloatTensor, optional): Pre-computed input embeddings
            moe_token_types (torch.LongTensor, optional): Token type assignments for MoE routing
            labels (torch.LongTensor, optional): Target labels for loss computation
            use_cache (bool, optional): Whether to use key-value caching
            output_attentions (bool, optional): Whether to return attention weights
            output_hidden_states (bool, optional): Whether to return hidden states
            return_dict (bool, optional): Whether to return structured output
            pixel_values (torch.Tensor, optional): Image pixel values
            pixel_values_videos (torch.FloatTensor, optional): Video pixel values
            image_grid_thw (torch.LongTensor, optional): Image grid dimensions (temporal, height, width)
            video_grid_thw (torch.LongTensor, optional): Video grid dimensions (temporal, height, width)
            action_chunk (torch.FloatTensor, optional): Action trajectory data chunks
            proprioception (torch.FloatTensor, optional): Proprioceptive sensor data (joint positions, etc.)
            rope_deltas (torch.LongTensor, optional): RoPE position deltas
            cache_position (torch.LongTensor, optional): Cache position indices
            second_per_grid_ts (torch.Tensor, optional): Time interval per temporal grid
            dataset_names (str, optional): Names of datasets in the current batch
            dof_mask (torch.FloatTensor, optional): Degrees of freedom mask for action tokens
            agent_pos_mask (torch.FloatTensor, optional): Agent position mask for proprioceptive data
            **kwargs: Additional keyword arguments

        Returns:
            Union[Tuple, Qwen2_5_VLACausalLMOutputWithPast]: Model outputs including losses, logits,
                and auxiliary information, or tuple if return_dict=False
        """
        batch_size, seq_length = input_ids.shape

        # Set output configuration from model config if not specified
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Calculate RoPE position IDs if not provided
        # Note: Cannot calculate rope deltas with 4D attention mask. TODO: Fix this limitation
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # Calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # Use previously calculated rope deltas to get correct position IDs
            else:
                delta = (
                    (cache_position[0] + self.rope_deltas).to(self.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=self.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Process input embeddings with multi-modal data
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            # Process image embeddings
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Process video embeddings
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]

                # Validate video token and feature count match
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Process proprioceptive data (joint positions, orientations, etc.)
            if proprioception is not None:
                proprioception = proprioception.to(inputs_embeds.device).to(
                    inputs_embeds.dtype
                )
                agent_pos_mask = agent_pos_mask.to(inputs_embeds.device).to(
                    inputs_embeds.dtype
                )
                proprioception = self.action_preprocessor.proprioception_proj(
                    proprioception,
                    dataset_names,
                    agent_pos_mask,
                    use_history=proprioception.shape[1] > 1,
                )
                mask = input_ids == self.action_token_id_set["propri_token_id"]
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                proprioception_mask = mask_expanded.to(inputs_embeds.device)

                proprioception = proprioception.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    proprioception_mask, proprioception
                )
            elif self.training:
                # Dummy forward pass to ensure gradient registration in DDP
                # This handles cases where one process has proprioception data while another doesn't
                # Without this, DDP would hang waiting for a gradient that will never be computed
                dummy_input = torch.randn(
                    2,
                    self.action_preprocessor.propri_dim * 2,
                    device=inputs_embeds.device,
                )
                dummy_forward = self.action_preprocessor.proprioception_proj(
                    dummy_input
                )
                dummy_loss = sum(p.sum() for p in dummy_forward)
                inputs_embeds = inputs_embeds + 0 * dummy_loss

            # Process action chunk data
            if action_chunk is not None:
                action_chunk = action_chunk.to(inputs_embeds.device).to(
                    inputs_embeds.dtype
                )
                dof_mask = dof_mask.to(inputs_embeds.device).to(inputs_embeds.dtype)
                noisy_action_emb, flow = self.action_preprocessor(
                    action_chunk, dataset_names, dof_mask
                )
                mask = input_ids == self.action_token_id_set["action_token_id"]
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                action_mask = mask_expanded.to(inputs_embeds.device)

                noisy_action_emb = noisy_action_emb.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    action_mask, noisy_action_emb
                )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # Forward pass through the main model
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            moe_token_types=moe_token_types,  # Pass token types for MoE routing
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden_states)

        # Initialize loss computation variables
        loss = None
        cross_entropy_loss, flow_loss = None, None
        channel_loss_dict = None
        channel_loss_count_dict = None

        # Compute losses if labels are provided
        if labels is not None:
            loss = 0
            action_accuracy = 0
            unique_datasets_name = list(set(dataset_names))

            # Initialize per-dataset loss tracking dictionaries
            channel_loss_dict = {
                dataset_name: torch.tensor(0.0, device=logits.device)
                for dataset_name in ACTION_DATASET_NAMES + MULTIMODAL_DATASET_NAMES
            }
            channel_loss_count_dict = {
                dataset_name: torch.tensor(0, device=logits.device)
                for dataset_name in ACTION_DATASET_NAMES + MULTIMODAL_DATASET_NAMES
            }

            # Compute standard cross-entropy loss for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism by moving labels to correct device
            shift_labels = shift_labels.to(shift_logits.device)
            non_ignored_mask = shift_labels != -100
            _cross_entropy_loss = self.loss_fct(shift_logits, shift_labels)
            cross_entropy_loss = (
                _cross_entropy_loss[non_ignored_mask].mean()
                if non_ignored_mask.any()
                else torch.tensor(0.0, device=shift_logits.device)
            )

            # Compute per-dataset channel losses
            _cross_entropy_loss = _cross_entropy_loss.view(batch_size, seq_length - 1)
            non_ignored_mask = non_ignored_mask.view(batch_size, seq_length - 1)
            for dataset_name_i in unique_datasets_name:
                dataset_mask = torch.tensor(
                    [name == dataset_name_i for name in dataset_names],
                    device=logits.device,
                )
                combined_mask = dataset_mask.unsqueeze(1) & non_ignored_mask
                channel_loss_dict[dataset_name_i] = (
                    _cross_entropy_loss[combined_mask].sum()
                    if combined_mask.any()
                    else torch.tensor(0.0, device=shift_logits.device)
                )
                channel_loss_count_dict[dataset_name_i] += combined_mask.sum()

            # Add cross-entropy loss to total loss if valid
            if not torch.isnan(cross_entropy_loss):
                loss += cross_entropy_loss
            else:
                with torch.no_grad():
                    cross_entropy_loss.detach()

            # Compute action token prediction accuracy
            shift_logits = logits[..., :-1, :].contiguous()
            action_preds = shift_logits.argmax(dim=-1)
            shift_labels = labels[..., 1:].contiguous()
            if self.use_fast_tokenizer:
                action_mask = (
                    shift_labels > self.action_token_id_set["fast_action_token_list"][0]
                )
                correct_preds = (action_preds == shift_labels) & action_mask
                action_accuracy = (
                    correct_preds.sum().float() / action_mask.sum().float()
                )
                channel_loss_dict["action_accuracy"] = action_accuracy

        if action_chunk is not None:
            action_mask = input_ids == self.action_token_id_set["action_token_id"]
            if action_mask.any():
                action_hidden_states = hidden_states[action_mask].to(torch.float32)
                flow = flow.reshape(-1, flow.shape[-1])
                _flow_loss = self.action_preprocessor.flow_loss(
                    action_hidden_states, flow, dof_mask
                )
                if isinstance(_flow_loss, torch.Tensor):
                    flow_loss = _flow_loss.mean()
                if loss is not None:
                    loss += self.flow_loss_weight * flow_loss
                else:
                    loss = self.flow_loss_weight * flow_loss
                _flow_loss = _flow_loss.view(
                    dof_mask.shape[0], dof_mask.shape[1], dof_mask.shape[2]
                )

        # Return outputs based on return_dict setting
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLACausalLMOutputWithPast(
            loss=loss,
            cross_entropy_loss=(
                cross_entropy_loss.clone() if cross_entropy_loss is not None else None
            ),
            flow_loss=flow_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            channel_loss_dict=channel_loss_dict,
            channel_loss_count_dict=channel_loss_count_dict,
        )

    def predict_action(self, predict_mode: str, **kwargs):
        """
        Predict actions using specified prediction mode.

        Args:
            predict_mode (str): Prediction mode, either "fast" or "diffusion"
            **kwargs: Additional arguments passed to the predict method

        Returns:
            tuple: (predicted_action, ground_truth_action) where ground_truth_action may be None
        """
        assert predict_mode in ["fast", "diffusion"]

        output = self.predict(predict_mode=predict_mode, **kwargs)

        return output["predict_action"], output.get("gt_action", None)

    @torch.no_grad()
    def predict(
        self,
        predict_mode: str,
        pred_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        moe_token_types: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        action_chunk: Optional[torch.FloatTensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        num_inference_timesteps: Optional[int] = 10,
        dataset_names: Optional[str] = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        re_generate: bool = False,
        **kwargs,
    ):
        """
        Multi-modal prediction method supporting text generation, fast action prediction, and diffusion-based action prediction.

        This method handles three prediction modes:
        1. "text": Pure text generation using autoregressive decoding
        2. "fast": Fast action prediction using discrete action tokens
        3. "diffusion": Continuous action prediction using diffusion/flow matching

        Args:
            predict_mode (str): Prediction mode ("text", "fast", or "diffusion")
            pred_horizon (int, optional): Prediction horizon for action sequences
            action_dim (int, optional): Dimensionality of action space
            input_ids (torch.LongTensor, optional): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask for input tokens
            position_ids (torch.LongTensor, optional): Position IDs for tokens
            past_key_values (List[torch.FloatTensor], optional): Cached key-value pairs
            inputs_embeds (torch.FloatTensor, optional): Pre-computed input embeddings
            moe_token_types (torch.LongTensor, optional): Token type assignments for MoE routing
            labels (torch.LongTensor, optional): Target labels for evaluation
            use_cache (bool, optional): Whether to use key-value caching
            output_attentions (bool, optional): Whether to return attention weights
            output_hidden_states (bool, optional): Whether to return hidden states
            return_dict (bool, optional): Whether to return structured output
            pixel_values (torch.Tensor, optional): Image pixel values
            pixel_values_videos (torch.FloatTensor, optional): Video pixel values
            image_grid_thw (torch.LongTensor, optional): Image grid dimensions
            video_grid_thw (torch.LongTensor, optional): Video grid dimensions
            action_chunk (torch.FloatTensor, optional): Ground truth action sequences
            proprioception (torch.FloatTensor, optional): Proprioceptive sensor data
            rope_deltas (torch.LongTensor, optional): RoPE position deltas
            cache_position (torch.LongTensor, optional): Cache position indices
            second_per_grid_ts (torch.Tensor, optional): Time interval per temporal grid
            num_inference_timesteps (int, optional): Number of diffusion inference steps
            dataset_names (str, optional): Dataset names for normalization
            dof_mask (torch.FloatTensor, optional): Degrees of freedom mask
            agent_pos_mask (torch.FloatTensor, optional): Agent position mask
            re_generate (bool, optional): Whether to use sampling for regeneration
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing prediction results with keys like:
                - 'predict_action': Predicted action sequences
                - 'gt_action': Ground truth actions (if available)
                - 'input_text': Input text (for text/fast modes)
                - 'predict_output_text': Generated text (for text/fast modes)
                - 'gt_output_text': Ground truth text (for text/fast modes)
        """
        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )

        # Text and fast modes require batch size 1 for autoregressive generation
        if predict_mode in ["text", "fast"]:
            assert (
                batch_size == 1
            ), "predict only support batch size 1 for ar generation"

        # Set output configuration from model config if not specified
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Process input embeddings with multi-modal data
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            # Process image embeddings
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]

                # Validate image token and feature count match
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Process video embeddings
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]

                # Validate video token and feature count match
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Process proprioceptive data
            if proprioception is not None:
                proprioception = proprioception.to(inputs_embeds.device).to(
                    inputs_embeds.dtype
                )
                agent_pos_mask = agent_pos_mask.to(inputs_embeds.device).to(
                    inputs_embeds.dtype
                )
                proprio_embed = self.action_preprocessor.proprioception_proj(
                    proprioception,
                    dataset_names,
                    agent_pos_mask,
                    use_history=proprioception.shape[1] > 1,
                )
                proprioception_mask = (
                    input_ids == self.action_token_id_set["propri_token_id"]
                )
                proprio_embed = proprio_embed.to(torch.bfloat16)
                inputs_embeds[proprioception_mask] = proprio_embed.reshape(
                    -1, inputs_embeds.shape[-1]
                )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # Calculate RoPE position IDs if not provided
        # Note: Cannot calculate rope deltas with 4D attention mask. TODO: Fix this limitation
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # Calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # Use previously calculated rope deltas to get correct position IDs
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Prepare action chunk data if provided
        if action_chunk is not None:
            action_chunk = action_chunk.to(inputs_embeds.device).to(torch.float32)

        output = {}

        # Split input sequence for text and fast modes (not needed for diffusion)
        if predict_mode == "text" or predict_mode == "fast":
            # Look for generation prompt tokens: <|im_start|>assistant
            generation_prompt_ids = torch.tensor(
                [151644, 77091], device=input_ids.device, dtype=input_ids.dtype
            )
            matches = (input_ids[0, :-1] == generation_prompt_ids[0]) & (
                input_ids[0, 1:] == generation_prompt_ids[1]
            )

            if matches.any():
                split_pos = torch.nonzero(matches, as_tuple=True)[0][0].item()
                # Extract ground truth output tokens (including newline)
                gt_output_ids = input_ids[:, split_pos + 3 :]
                # Remove output part from input, keeping prompt
                input_ids = input_ids[:, : split_pos + 3]
                inputs_embeds = inputs_embeds[:, : split_pos + 3, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, : split_pos + 3]
                if labels is not None:
                    labels = labels[:, split_pos + 3 :]
            else:
                raise Warning(
                    "input_ids does not contain the generation prompt tokens <|im_start|>assistant"
                )

            # Decode input text for output
            input_text = self.processor.batch_decode(
                input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
            output["input_text"] = input_text

        # Handle text and fast prediction modes using autoregressive generation
        if predict_mode == "text" or predict_mode == "fast":
            # Initialize MoE token types for generation
            moe_token_types = torch.zeros_like(input_ids)
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "moe_token_types": moe_token_types,
                "image_grid_thw": image_grid_thw,
                "dof_mask": dof_mask,
                "agent_pos_mask": agent_pos_mask,
                "proprioception": proprioception,
                "dataset_names": dataset_names,
            }

            # Generate output tokens
            predict_output_ids = self.generate(
                **batch,
                max_new_tokens=100,
                eos_token_id=[self.processor.tokenizer.eos_token_id],
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                temperature=(
                    1.0 if not re_generate else 0.7
                ),  # Higher temperature for regeneration
                do_sample=(
                    False if not re_generate else True
                ),  # Enable sampling for regeneration
            )

            # Decode generated and ground truth text
            gt_output_text = self.processor.batch_decode(
                gt_output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            predict_output_text = self.processor.batch_decode(
                predict_output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            output["gt_output_text"] = gt_output_text
            output["predict_output_text"] = predict_output_text

        # Convert tokens to actions for fast prediction mode
        if predict_mode == "fast":
            action_id = []
            # Extract action tokens from generated sequence
            for token_id_i in predict_output_ids[0]:
                if (
                    token_id_i.item()
                    >= self.processor.tokenizer.init_kwargs["action_token_start_index"]
                ):
                    action_id.append(
                        token_id_i.item()
                        - self.processor.tokenizer.init_kwargs[
                            "action_token_start_index"
                        ]
                    )

            predict_action = self.processor.action_processor.decode(
                [action_id], time_horizon=pred_horizon, action_dim=action_dim
            )
            # Handle action decoding errors
            if np.sum(predict_action) == 0:
                print("Error in decoding action, predict_action is None")
                output["predict_action"] = None
            else:
                # Convert discrete tokens to continuous actions
                predict_action = torch.tensor(predict_action, device=self.device)
                dof_mask = dof_mask.to(self.device).to(pixel_values.dtype)
                # removed unnormalization step for now
                predict_action = (
                    predict_action[:, :, dof_mask[0, 0, :].bool()]
                )
                output["predict_action"] = predict_action

            # Process ground truth actions if available
            if action_chunk is not None:
                # Apply DOF mask to get ground truth actions
                # removed unnormalization step for now
                action_chunk = action_chunk[:, :, dof_mask[0, 0, :].bool()]
                output["gt_action"] = action_chunk
            else:
                output["gt_action"] = None

        # Handle diffusion-based action prediction
        if predict_mode == "diffusion":
            # Initialize with random noise
            noisy_action = torch.randn(
                size=(batch_size, pred_horizon, action_dim),
                dtype=torch.float32,
                device=inputs_embeds.device,
            )
            dof_mask = dof_mask.to(inputs_embeds.device).to(torch.float32)

            def step(timestep, noisy_action):
                """
                Single denoising step for diffusion process.

                Args:
                    timestep: Current diffusion timestep
                    noisy_action: Current noisy action estimate

                Returns:
                    torch.Tensor: Predicted clean action
                """
                action_mask = input_ids == self.action_token_id_set["action_token_id"]
                assert action_mask.any(), "No action token found in input_ids"

                # Prepare timestep for batch processing
                timestep = timestep.unsqueeze(0).repeat(noisy_action.shape[0])
                action_embed = self.action_preprocessor.step(
                    timestep=timestep, noisy_action=noisy_action, dof_mask=dof_mask
                )
                action_embed = action_embed.reshape(-1, inputs_embeds.shape[-1])
                
                # Ensure action_embed has the correct dtype and device before assignment
                action_embed = action_embed.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

                # Create temporary copy of embeddings (clone preserves dtype)
                temp_inputs_embeds = inputs_embeds.clone()
                temp_inputs_embeds[action_mask] = action_embed

                # Forward pass through transformer
                transformer_outputs = self.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=temp_inputs_embeds,
                    moe_token_types=moe_token_types,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                # Extract action predictions from hidden states
                hidden_states = transformer_outputs.last_hidden_state
                action_mask = input_ids == self.action_token_id_set["action_token_id"]
                action_hidden_states = hidden_states[action_mask].to(torch.float32)
                pred = self.action_preprocessor.action_proj_back(action_hidden_states)
                return pred.reshape(batch_size, pred_horizon, action_dim)

            # Perform ODE integration for diffusion sampling
            times = torch.linspace(
                0,
                1,
                num_inference_timesteps+1,
                device=inputs_embeds.device,
                dtype=torch.float32,
            )
            action_trajectory = odeint(step, noisy_action, times, method="euler")

            # Extract final predicted action
            # Removed unnormalization step for now
            predict_action = action_trajectory[-1]
            output["predict_action"] = predict_action

            # Process ground truth actions if available
            # removed unnormalization step for now
            if action_chunk is not None:
                output["gt_action"] = action_chunk[:, :, dof_mask[0, 0, :].bool()]

        return output

    def forward(
        self, mode: Optional[str] = None, predict_mode: Optional[str] = "text", **kwargs
    ):
        """
        Main forward pass dispatcher for different execution modes.

        This method routes execution to appropriate forward functions based on the specified mode:
        - No mode (None): Training step with gradient disabled
        - 'predict': Prediction/inference mode
        - 'train': Training mode with gradients enabled
        - 'validate': Validation mode with gradients disabled

        Args:
            mode (str, optional): Execution mode. If None, defaults to training step without gradients
            predict_mode (str, optional): Prediction mode for 'predict' mode ("text", "fast", or "diffusion")
            **kwargs: Additional arguments passed to the selected forward function

        Returns:
            Model outputs appropriate for the selected mode

        Todo:
            - Add support for distinguishing multi-modal data types in prediction mode
        """
        if not mode:
            with torch.no_grad():
                return self.train_step_forward(**kwargs)
        elif mode == "predict":
            return self.predict(predict_mode=predict_mode, **kwargs)
        elif mode == "train":
            return self.train_step_forward(use_cache=False, **kwargs)
        elif mode == "validate":
            with torch.no_grad():
                return self.train_step_forward(use_cache=False, **kwargs)
        else:
            raise NotImplementedError("invalid key")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        moe_token_types=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        dataset_names=None,
        proprioception=None,
        dof_mask=None,
        agent_pos_mask=None,
        **kwargs,
    ):
        """
        Prepare inputs for autoregressive generation with multi-modal support.

        This method handles input preparation for generation, including proper slicing of inputs
        based on cache position, MoE token type management, and multi-modal data handling.
        Vision inputs are selectively forwarded only when needed during generation.

        Args:
            input_ids: Input token IDs
            past_key_values: Cached key-value pairs from previous generation steps
            attention_mask: Attention mask for input tokens
            inputs_embeds: Pre-computed input embeddings
            moe_token_types: Token type assignments for MoE routing
            cache_position: Current cache position for generation
            position_ids: Position IDs for tokens
            use_cache: Whether to use key-value caching
            pixel_values: Image pixel values
            pixel_values_videos: Video pixel values
            image_grid_thw: Image grid dimensions
            video_grid_thw: Video grid dimensions
            second_per_grid_ts: Time interval per temporal grid
            dataset_names: Dataset names for processing
            proprioception: Proprioceptive sensor data
            dof_mask: Degrees of freedom mask
            agent_pos_mask: Agent position mask
            **kwargs: Additional arguments

        Returns:
            dict: Prepared model inputs for generation step

        Todo:
            - Test this function thoroughly with various input configurations

        Note:
            This is an overridden method that handles specific cases for multi-modal generation:
            - Slices input_ids through cache_position to keep only unprocessed tokens
            - Handles special cases for input_embeds, generation methods, and GPU synchronization
            - Manages vision inputs to avoid unnecessary forward passes
        """
        # Initialize MoE token types if not provided
        if moe_token_types is None:
            moe_token_types = torch.zeros_like(
                input_ids
            )  # FIXME: Handle case when input_embeds is used instead
        else:
            # Ensure moe_token_types length matches input_ids
            if moe_token_types.shape[1] < input_ids.shape[1]:
                # Calculate required padding length
                pad_length = input_ids.shape[1] - moe_token_types.shape[1]
                # Create padding tensor with default token type (0)
                pad_tensor = torch.zeros(
                    (moe_token_types.shape[0], pad_length),
                    dtype=moe_token_types.dtype,
                    device=moe_token_types.device,
                )
                # Concatenate padding to existing moe_token_types
                moe_token_types = torch.cat([moe_token_types, pad_tensor], dim=1)

        # Handle input slicing based on cache state and special cases
        if past_key_values is not None:
            if (
                inputs_embeds is not None and input_ids.shape[1] == 0
            ):  # Exception 4: input_embeds case
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
                moe_token_types = moe_token_types[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or (  # Exception 1: input_embeds provided
                is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1]
            ):  # Exception 3: GPU sync edge case
                input_ids = input_ids[:, -cache_position.shape[0] :]
                moe_token_types = moe_token_types[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (Exception 2 is no-op)
                cache_pos = cache_position.clone()
                input_ids = input_ids[:, cache_pos]
                moe_token_types = moe_token_types[:, cache_pos]

        # Skip vision inputs for continuation steps (not initial generation)
        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # Determine whether to use inputs_embeds or input_ids for this generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        # Prepare 4D causal attention mask for static cache
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = (
                self.model._prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.lm_head.weight.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
            )

        # Assemble all model inputs for generation
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "moe_token_types": moe_token_types,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
                "proprioception": proprioception,
                "dataset_names": dataset_names,
                "dof_mask": dof_mask,
                "agent_pos_mask": agent_pos_mask,
            }
        )
        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate tensor separation lengths.

        These parameters are computed directly from input_ids rather than being passed through
        the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (torch.LongTensor): Input token IDs of shape (batch_size, sequence_length)

        Returns:
            tuple:
                - image_nums (torch.LongTensor): Number of images per sample
                - video_nums (torch.LongTensor): Number of videos per sample
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        # Find vision start tokens and their following tokens
        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id

        # Count images and videos following vision start tokens
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """
        Expand inputs for generation with support for multi-modal tensors.

        This is an overridden method that supports expanding tensors without a standard batch
        size dimension, specifically for vision-related tensors:
        - pixel_values.shape[0] = sum(sequence_lengths for all image samples)
        - image_grid_thw.shape[0] = sum(num_images for all samples)
        - Similar patterns for video tensors

        Args:
            expand_size (int): Factor by which to expand inputs (for beam search, etc.)
            is_encoder_decoder (bool): Whether using encoder-decoder architecture
            input_ids (torch.LongTensor, optional): Input token IDs
            **model_kwargs: Additional model arguments to expand

        Returns:
            tuple: (expanded_input_ids, expanded_model_kwargs)
        """
        if expand_size == 1:
            return input_ids, model_kwargs

        # Define keys for vision-related tensors that need special handling
        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            """Expand vision-related tensors based on image/video counts per sample."""
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                """Split tensor by lengths and repeat each sample."""
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat(
                    [sample.repeat(*repeat_args) for sample in samples], dim=0
                )
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # Split images into samples and compute sequence lengths
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # Expand based on number of images per sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    # Split videos into samples and compute sequence lengths
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    # Expand based on number of videos per sample
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    # Handle list-type temporal grid data
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(
                        tensor, lengths=lengths, repeat_times=expand_size
                    )
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            """Expand standard tensors using repeat_interleave."""
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(
                        expand_size, dim=0
                    )
            return dict_to_expand

        # Expand visual inputs only if input_ids is available for counting images/videos
        # If input_ids is unavailable, visual inputs won't be used, so no expansion needed
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        # Expand input_ids using standard repeat_interleave
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        # Expand all other model arguments
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        # Handle encoder-decoder specific expansion
        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(
                model_kwargs["encoder_outputs"]
            )

        return input_ids, model_kwargs


class WallXPolicy(PreTrainedPolicy):
    """
    Wall-X policy for cross-embodiment robotic control.

    Integrates Qwen2.5-VL vision-language model with action prediction
    using flow matching for continuous action spaces.
    """

    config_class = WallXConfig
    name = "wall_x"

    def __init__(self, config: WallXConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize VLM wrapper
        self.model = Qwen2_5_VLMoEForAction(config)

        self.reset()

    def reset(self):
        """Reset action queue."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self):
        """Get parameters for optimization."""
        params = []

        if self.model.visual.available:
            # Add VLM parameters
            if not self.config.train_expert_only:
                params.extend(self.vlm.model.parameters())

        # Always add action head parameters
        if self.config.train_action_head:
            params.extend(self.action_head.parameters())

        return params

    def preprocess_inputs(
        self,
        batch: List[Dict[str, Any]],
        config: Dict[str, Any],
        dataload_config: Dict[str, Any],
        norm_stats: Dict[str, Any],
        lerobot_config: Dict[str, Any],
        processor: Any,
        action_tokenizer: Optional[Any] = None,
        camera_keys: Optional[List[str]] = None,
    ) -> BatchFeature:
        """
        Convert a batch of LeRobot dataset items to Wall-X model input format.
        
        This is the batch version of convert_lerobot_to_wallx_format, processing multiple 
        samples together for efficient batched inference or training.
        
        Args:
            batch: List of items from LeRobot dataset
            config: Model configuration dict
            dataload_config: Data loading configuration
            norm_stats: Normalization statistics
            lerobot_config: LeRobot config containing 'repo_id'
            processor: Hugging Face processor for tokenization
            action_tokenizer: Optional action tokenizer
            camera_keys: List of camera keys in the dataset
        
        Returns:
            BatchFeature containing batched model inputs
        
        Example:
            >>> batch = [dataset[i] for i in range(8)]
            >>> result = preprocess_inputs(
            ...     batch=batch,
            ...     config=config,
            ...     dataload_config=dataload_config,
            ...     norm_stats=norm_stats,
            ...     lerobot_config={'repo_id': 'lerobot/aloha_mobile_cabinet'},
            ...     processor=processor,
            ... )
        """
        repo_id = lerobot_config["repo_id"]
        use_fast_tokenizer = config.get("use_fast_tokenizer", False)
        
        # Get key mappings
        cam_key_mapping = KEY_MAPPINGS[repo_id]["camera"]
        state_key = KEY_MAPPINGS[repo_id]["state"]
        action_key = KEY_MAPPINGS[repo_id]["action"]
        
        # Build data config
        data_config = X2RDataProcessingConfig().update(
            train_test_split=dataload_config.get("train_test_split", 0.95),
            split_seed=dataload_config.get("split_seed", 42),
            predict_action_keys=dataload_config.get("predict_action_keys", []),
            obs_action_keys=dataload_config.get("obs_action_keys", []),
            resolution=dataload_config.get("resolution", None),
            priority_order=dataload_config.get("priority_order", None),
        )
        
        if camera_keys is None:
            camera_keys = list(cam_key_mapping.keys())
        
        # ==================== PROCESS ALL SAMPLES ====================
        all_image_inputs = []
        all_texts = []
        all_agent_pos = []
        all_actions = []
        all_frame_indices = []
        
        for data in batch:
            # Vision preprocessing per sample
            processed_frames = []
            orig_height, orig_width = None, None
            resized_height, resized_width = None, None
            
            for key in camera_keys:
                current_obs = data[key].clone()
                if current_obs.dim() == 3:
                    current_obs = current_obs.permute(1, 2, 0)
                
                img_pil = Image.fromarray((current_obs * 255).to(torch.uint8).cpu().numpy())
                orig_width, orig_height = img_pil.size
                
                cam_name = cam_key_mapping.get(key, key)
                target_size = data_config.resolution.get(cam_name, -1)
                if target_size != -1:
                    if orig_width > orig_height:
                        new_width = target_size
                        new_height = int(target_size * orig_height / orig_width)
                    else:
                        new_height = target_size
                        new_width = int(target_size * orig_width / orig_height)
                    img_pil = img_pil.resize((new_width, new_height))
                
                current_width, current_height = img_pil.size
                resized_height, resized_width = smart_resize(
                    current_height,
                    current_width,
                    factor=data_config.image_factor,
                    min_pixels=data_config.min_pixels,
                    max_pixels=data_config.max_pixels,
                )
                resized_img = img_pil.resize((resized_width, resized_height))
                processed_frames.append(resized_img)
            
            all_image_inputs.append(processed_frames)
            
            # Text preprocessing
            frame_index = data["frame_index"]
            instruction_info = {"instruction": data["task"]}
            
            complete_text, _ = get_wallx_normal_text(
                instruction_info,
                dataload_config.get("action_horizon", 33) - 1,
                frame_index,
                data_config.priority_order,
                cam_key_mapping,
                generate_subtask_ratio=data_config.generate_subtask_ratio,
            )
            
            text = process_grounding_points(
                complete_text, orig_height, orig_width, resized_height, resized_width,
                data_config.model_type
            )
            all_texts.append(text)
            
            # Collect raw values
            all_agent_pos.append(data[state_key])
            all_actions.append(data[action_key])
            all_frame_indices.append(frame_index)
        
        # ==================== BATCH NORMALIZATION ====================
        action_min_stat = norm_stats["action"].min
        action_delta = norm_stats["action"].delta
        state_min_stat = norm_stats["state"].min
        state_delta = norm_stats["state"].delta
        
        def normalize(x, min_stat, delta):
            delta = torch.where(delta == 0, torch.ones_like(delta), delta)
            x = (x - min_stat) / delta
            x = x * 2 - 1
            return torch.clamp(x, -1, 1)
        
        # Stack and normalize agent_pos
        agent_pos = torch.stack(all_agent_pos)
        if agent_pos.dim() == 2:
            agent_pos = agent_pos.unsqueeze(1)
        agent_pos_mask = (~torch.isnan(agent_pos)).float()
        agent_pos = agent_pos.nan_to_num(nan=0.0)
        agent_pos = normalize(agent_pos, state_min_stat, state_delta)
        
        if agent_pos.shape[-1] != 20:
            pad_size = 20 - agent_pos.shape[-1]
            agent_pos = torch.cat([
                agent_pos,
                torch.zeros(agent_pos.shape[0], agent_pos.shape[1], pad_size)
            ], dim=-1)
            agent_pos_mask = torch.cat([
                agent_pos_mask,
                torch.zeros(agent_pos_mask.shape[0], agent_pos_mask.shape[1], pad_size)
            ], dim=-1)
        
        # Stack and normalize actions
        action = torch.stack(all_actions)
        if action.dim() == 2:
            action = action.unsqueeze(1)
        dof_mask = (~torch.isnan(action)).float()
        action = action.nan_to_num(nan=0.0)
        action = normalize(action, action_min_stat, action_delta)
        
        if action.shape[-1] != 20:
            pad_size = 20 - action.shape[-1]
            action = torch.cat([
                action,
                torch.zeros(action.shape[0], action.shape[1], pad_size)
            ], dim=-1)
            dof_mask = torch.cat([
                dof_mask,
                torch.zeros(dof_mask.shape[0], dof_mask.shape[1], pad_size)
            ], dim=-1)
        
        # ==================== ACTION TOKEN REPLACEMENT ====================
        all_texts = replace_action_token(
            all_texts,
            action,
            action_tokenizer if use_fast_tokenizer else None,
            [repo_id] * len(batch),
            dof_mask,
        )
        
        # ==================== TOKENIZATION ====================
        inputs = preprocesser_call(
            processor=processor,
            text=all_texts,
            images=all_image_inputs,
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=dataload_config.get("max_length", 768),
        )
        
        # ==================== ADDITIONAL INPUTS ====================
        action_token_id = processor.tokenizer.convert_tokens_to_ids("<|action|>")
        moe_token_types = inputs.input_ids == action_token_id
        
        inputs["proprioception"] = agent_pos
        inputs["agent_pos_mask"] = agent_pos_mask
        inputs["action_chunk"] = action
        inputs["dof_mask"] = dof_mask
        inputs["moe_token_types"] = moe_token_types
        inputs["dataset_names"] = [repo_id] * len(batch)
        inputs["frame_index"] = torch.stack([
            torch.tensor(fi) if not isinstance(fi, torch.Tensor) else fi 
            for fi in all_frame_indices
        ])
        
        return inputs

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Training forward pass.

        Args:
            batch: Dictionary containing observations and actions

        Returns:
            tuple: (loss, loss_dict)
        """
        # Prepare inputs
        images = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        batch_size = actions.shape[0]
        device = actions.device
        dtype = actions.dtype

        # Create DOF mask
        dof_mask = self._create_dof_mask(batch_size, device, dtype)

        # Process actions through action head (adds noise, gets embeddings)
        action_embeds, flow_target = self.action_head(actions, dof_mask)

        # For now, use simplified loss computation
        # In full implementation, would pass through VLM transformer
        loss_dict = {}

        # Compute flow matching loss
        # Note: In full wall-x, action_embeds would go through VLM transformer first
        flow_loss = self.action_head.flow_loss(action_embeds, flow_target, dof_mask)
        loss = flow_loss.mean()

        loss_dict["loss"] = loss.item()
        loss_dict["flow_loss"] = loss.item()

        return loss, loss_dict

    def _sample_actions_flow(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Sample actions using flow matching / diffusion.

        Args:
            batch: Dictionary containing observations

        Returns:
            Predicted actions [batch, chunk_size, action_dim]
        """
        batch_size = 1  # Typically inference is single sample
        device = self.config.device
        dtype = torch.float32

        # Initialize with noise
        noisy_action = torch.randn(
            batch_size,
            self.config.chunk_size,
            sum(self.config.dof_config.values()),
            device=device,
            dtype=dtype
        )

        # Create DOF mask
        dof_mask = self._create_dof_mask(batch_size, device, dtype)

        # ODE integration for denoising
        num_steps = self.config.num_inference_timesteps
        dt = 1.0 / num_steps

        for step_idx in range(num_steps + 1):
            t = torch.tensor(step_idx * dt, device=device, dtype=dtype)
            timestep = t.unsqueeze(0).repeat(batch_size)

            # Single denoising step
            action_embeds = self.action_head.step(timestep, noisy_action, dof_mask)

            # Predict flow (in full implementation, would go through VLM)
            flow_pred = self.action_head.action_proj_back(action_embeds)

            # Euler integration step
            noisy_action = noisy_action + dt * flow_pred

        return noisy_action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk for evaluation."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self.config.prediction_mode == "flow":
            actions = self._sample_actions_flow(batch)
        else:
            raise NotImplementedError(f"Prediction mode {self.config.prediction_mode} not implemented")

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action for environment execution."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Use action queue
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[:self.config.n_action_steps])

        return self._queues[ACTION].popleft()
