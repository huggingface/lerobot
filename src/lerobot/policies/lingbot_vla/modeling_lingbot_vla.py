#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from collections import deque
from collections.abc import Callable
from functools import partial
from typing import TypedDict

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_lingbot_vla import LingbotVLAConfig
from .flex_attention import flex_attention_forward
from .qwen_model.qwenvl_in_vla import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
)
from .utils import (
    apply_rope,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    our_eager_attention_forward,
    sample_beta,
)


# `LossKwargs` was removed from `transformers.utils` in transformers>=5.0; it is only
# used here as a TypedDict base for loss-related forward kwargs, so define a local shim.
class LossKwargs(TypedDict, total=False):
    num_items_in_batch: int | None


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-qwen2/Qwen2-2-7b-hf"
_CONFIG_FOR_DOC = "Qwen2Config"


# Depth-alignment heads are only needed for the LingBot-VLA-*-Depth checkpoints. They are
# imported lazily (see FlowMatching.init_depth_heads) so the no-depth model has no extra deps.


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class FixQwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        FixQwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.norm_qkv:
            self.q_layernorm = Qwen2RMSNorm(self.self_attn.head_dim, eps=config.rms_norm_eps)
            self.k_layernorm = Qwen2RMSNorm(self.self_attn.head_dim, eps=config.rms_norm_eps)

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: torch.Tensor | None = None,
        start: int | None = 0,
        end: int | None = 0,
        compute_kqv: bool = False,
        norm_qkv: bool = False,
        output_atten: bool = False,
        ada_cond: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        if compute_kqv:
            if ada_cond is not None:
                hidden_states = self.input_layernorm(hidden_states, ada_cond)
                gate = None
            else:
                hidden_states = self.input_layernorm(hidden_states)
                gate = None
            hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

            query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
            if norm_qkv:
                query_state = self.q_layernorm(query_state)
                key_state = self.k_layernorm(key_state)

            return query_state, key_state, value_state, gate

        elif output_atten:
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
            out_emb = self.self_attn.o_proj(att_output[:, start:end])

            # first residual
            if gate is not None:
                out_emb = out_emb * gate + hidden_states
            else:
                out_emb += hidden_states
            after_first_residual = out_emb.clone()
            if ada_cond is not None:
                out_emb = self.post_attention_layernorm(out_emb, ada_cond)
                after_gate = None
            else:
                out_emb = self.post_attention_layernorm(out_emb)
                after_gate = None
            out_emb = self.mlp(out_emb)

            # second residual
            if after_gate is not None:
                out_emb = out_emb * after_gate + after_first_residual
            else:
                out_emb += after_first_residual

            return out_emb

        else:
            raise ValueError(
                f"Invaild Operation compute_kqv={compute_kqv} and output_atten={output_atten} with Qwen2DecoderLayer in LingBot-VLA"
            )


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # transformers>=5.0 dropped the "default" key from ROPE_INIT_FUNCTIONS.
        # In the dual-stream VLA path rope is applied externally via utils.apply_rope,
        # so this embedding only needs to instantiate; fall back to the standard
        # (non-scaled) inv_freq computation for "default"/unknown rope types.
        if self.rope_type in ROPE_INIT_FUNCTIONS:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        else:
            base = getattr(config, "rope_theta", None)
            if base is None and config.rope_scaling is not None:
                base = config.rope_scaling.get("rope_theta", 10000.0)
            base = base if base is not None else 10000.0
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device).float() / head_dim)
            )
            self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config, eval=False):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = FixQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        if eval:
            self._init_weights = lambda module: None
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
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

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

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
        config: Qwen2Config,
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
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
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
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    # transformers>=5.0 expects a dict mapping {output: input} for tied weights.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, eval, **kwargs):
        super().__init__(config)
        self.model = Qwen2Model(config, eval)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QwenvlWithExpertConfig(PretrainedConfig):
    model_type = "QwenvlWithExpertModel"
    sub_configs = {"qwenvl_config": AutoConfig, "qwen_expert_config": AutoConfig}

    def __init__(
        self,
        qwenvl_config: dict | None = None,
        qwen_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        vocab_size: int = 257152,
        use_lm_head: bool = False,
        attention_implementation: str = "eager",
        tokenizer_path: str | None = None,
        use_cache: bool = False,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        if qwenvl_config is None:
            self.qwenvl_config = CONFIG_MAPPING["qwen2_5_vl"](
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151645,
                vision_start_token_id=151652,
                vision_end_token_id=151653,
                vision_token_id=151654,
                image_token_id=151655,
                video_token_id=151656,
                hidden_act="silu",
                hidden_size=2048,
                initializer_range=0.02,
                intermediate_size=11008,
                max_position_embeddings=128000,
                max_window_layers=70,
                model_type="qwen2_5_vl",
                num_attention_heads=16,
                num_hidden_layers=36,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=32768,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                transformers_version="4.41.2",
                use_cache=use_cache,
                use_sliding_window=False,
                vision_config={
                    "depth": 32,
                    "hidden_act": "silu",
                    "hidden_size": 1280,
                    "intermediate_size": 3420,
                    "num_heads": 16,
                    "in_chans": 3,
                    "out_hidden_size": 2048,
                    "patch_size": 14,
                    "spatial_merge_size": 2,
                    "spatial_patch_size": 14,
                    "window_size": 112,
                    "fullatt_block_indexes": [7, 15, 23, 31],
                    "tokens_per_second": 2,
                    "temporal_patch_size": 2,
                },
                rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
                vocab_size=151936,
            )
        elif isinstance(self.qwenvl_config, dict):
            if "model_type" not in qwen_expert_config:
                qwenvl_config["model_type"] = "qwen2_5_vl"

            cfg_cls = CONFIG_MAPPING[qwenvl_config["model_type"]]
            self.qwenvl_config = cfg_cls(**qwenvl_config)

        if qwen_expert_config is None:
            self.qwen_expert_config = CONFIG_MAPPING["qwen2"](
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151645,
                hidden_act="silu",
                hidden_size=768,
                head_dim=128,
                initializer_range=0.02,
                intermediate_size=2752,
                max_position_embeddings=32768,
                max_window_layers=21,
                model_type="qwen2",
                num_attention_heads=16,
                num_hidden_layers=36,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=32768,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                transformers_version="4.43.1",
                use_cache=use_cache,
                use_sliding_window=False,
                vocab_size=151936,
            )
        elif isinstance(self.qwen_expert_config, dict):
            if "model_type" not in qwen_expert_config:
                qwen_expert_config["model_type"] = "qwen2"

            cfg_cls = CONFIG_MAPPING[qwenvl_config["model_type"]]
            self.qwen_expert_config = cfg_cls(**qwen_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )


class AdaRMSNorm(nn.Module):
    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        """
        AdaRMSNorm: RMSNorm + FiLM
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

        # DiT style init: gamma.weight=0, gamma.bias=1; beta.weight=0, beta.bias=0
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, 0.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight * hidden_states
        gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(cond).unsqueeze(1)  # [B, 1, H]
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype)


# HACK: show directly use this norm during initialization
def replace_lnorm_with_adanorm(
    module, hidden_size, cond_dim, split_gate_liner, no_split_gate_liner, final_norm_adanorm
):
    for name, child in module.named_children():
        if isinstance(child, Qwen2RMSNorm):
            if "q_layernorm" not in name and "k_layernorm" not in name:
                setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
        else:
            replace_lnorm_with_adanorm(
                child, hidden_size, cond_dim, split_gate_liner, no_split_gate_liner, final_norm_adanorm
            )


class QwenvlWithExpertModel(PreTrainedModel):
    config_class = QwenvlWithExpertConfig

    def __init__(self, config: QwenvlWithExpertConfig, eval=False):
        super().__init__(config=config)
        self.config = config
        vlm_config = AutoConfig.from_pretrained(self.config.tokenizer_path)
        # transformers>=5.0 nests the text-model fields (hidden_size, vocab_size,
        # pad_token_id, rope_scaling, ...) under `text_config`. The vendored Qwen2.5-VL
        # reads them flat, so promote any missing field onto the top-level config. The
        # vision tower keeps reading from `vlm_config.vision_config` (left untouched).
        text_config = getattr(vlm_config, "text_config", None)
        if text_config is not None:
            for k, v in vars(text_config).items():
                if k.startswith("_"):
                    continue
                if getattr(vlm_config, k, None) is None:
                    setattr(vlm_config, k, v)
        if getattr(vlm_config, "pad_token_id", None) is None:
            vlm_config.pad_token_id = None
        vlm_config.vision_config.initializer_range = 0.02
        vlm_config.norm_qkv = self.config.norm_qkv
        if (
            self.config.vocab_size != 0
            and self.config.vocab_size != 257152
            and vlm_config.vocab_size != self.config.vocab_size
        ):
            vlm_config.vocab_size = self.config.vocab_size

        # Map the dual-stream attention setting to a HF-recognised attn implementation.
        # Runtime attention is dispatched via get_attention_interface(); this only sets the
        # sub-models' internal default. flash_attention_2 requires the flash-attn package.
        _hf_attn_impl = "flash_attention_2" if self.config.attention_implementation == "fa2" else "eager"
        vlm_config._attn_implementation = _hf_attn_impl
        self.qwenvl = Qwen2_5_VLForConditionalGeneration._from_config(vlm_config)
        if self.config.use_lm_head:
            self.qwenvl.tie_weights()
        self.config.qwen_expert_config.norm_qkv = self.config.norm_qkv
        self.config.qwen_expert_config._attn_implementation = _hf_attn_impl
        self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config, eval=eval)

        self.rotary_pos_emb = None
        self.window_index = None
        self.cu_window_seqlens = None
        self.cu_seqlens = None

        if getattr(self.config, "adanorm_time", False):
            replace_lnorm_with_adanorm(
                self.qwen_expert,
                self.config.qwen_expert_config.hidden_size,
                self.config.qwen_expert_config.hidden_size,
                config.split_gate_liner,
                config.no_split_gate_liner,
                config.final_norm_adanorm,
            )
        # Remove unused embed_tokens
        del self.qwen_expert.model.embed_tokens
        self.attention_interface = self.get_attention_interface()

        # self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    def set_requires_grad(self):
        """sets the requires_grad attribute of the model parameters based on the configuration.
        If `freeze_vision_encoder` is True, the vision tower parameters are frozen.
        If `train_expert_only` is True, the entire Qwenvl model is frozen.
        """
        if self.config.freeze_vision_encoder:
            self.qwenvl.visual.eval()
            for params in self.qwenvl.visual.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.qwenvl.eval()
            for params in self.qwenvl.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.qwenvl.visual.eval()
        if self.config.train_expert_only:
            self.qwenvl.eval()

    def to_bfloat16_like_physical_intelligence(self):
        """casts the model to bfloat16.

        Modules not casted to bfloat16:
        - qwenvl.model.embed_tokens.weight
        - qwenvl.model.norm.weight
        - qwen_expert.model.norm.weight
        - qwen_expert.lm_head.weight
        """
        self.qwenvl = self.qwenvl.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "qwenvl.model.layers",
            "qwen_expert.model.layers",
            "visual",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        if (
            self.rotary_pos_emb is None
            or self.window_index is None
            or self.cu_window_seqlens is None
            or self.cu_seqlens is None
        ):
            (self.rotary_pos_emb, self.window_index, self.cu_window_seqlens, self.cu_seqlens) = (
                self.qwenvl.visual.preprcess_grid_thw(grid_thw=image_grid_thw)
            )
        image_embeds = self.qwenvl.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            rotary_pos_emb=self.rotary_pos_emb,
            window_index=self.window_index,
            cu_window_seqlens=self.cu_window_seqlens,
            cu_seqlens=self.cu_seqlens,
        )
        split_sizes = (image_grid_thw.prod(-1) // self.qwenvl.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        image_embeds = torch.stack(image_embeds, dim=0)
        return image_embeds

    def embed_image(self, image: torch.Tensor, patch_size=14, temporal_patch_size=2):
        h = w = int(image.shape[1] ** 0.5)
        image_grid_thw = torch.tensor([[1, h, w]] * image.shape[0], device=image.device)
        image_embeds = self.get_image_features(image, image_grid_thw=image_grid_thw)
        return image_embeds
        # return torch.randn(72, 64, 2048).to(device=image.device, dtype=torch.bfloat16)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.model.embed_tokens(tokens)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}

            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat(
                    [past_key_values[layer_idx]["value_states"], value_states],
                    dim=1,
                )
        return key_states, value_states, past_key_values

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        vlm_position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        ada_cond: list[torch.FloatTensor] = None,
        norm_qkv: bool = False,
    ):
        """
        Args:
            attention_mask (Optional[torch.Tensor], optional):
                Attention mask with shape (b, seq_len, seq_len). Defaults to None.
            position_ids (Optional[torch.LongTensor], optional):
                Position indices for applying RoPE. Defaults to None.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]], optional):
                Optional kv cache. Defaults to None.
            inputs_embeds (List[torch.FloatTensor], optional):
                Input embeddings. Defaults to None.
            use_cache (Optional[bool], optional):
                Whether to use kv cache. Defaults to None.
            fill_kv_cache (Optional[bool], optional):
                Whether to return kv tensors in this forward pass as cache. Defaults to None.

        Returns:
            outputs_embeds (torch.Tensor): Output embeddings.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]]):
                Optional kv cache.
        """
        models = [self.qwenvl.model, self.qwen_expert.model]

        # RMSNorm
        num_layers = self.qwenvl.config.num_hidden_layers  # 36
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            gates = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                if i == 1:  # For action expert
                    query_state, key_state, value_state, gate = models[i].layers[layer_idx](
                        hidden_states, compute_kqv=True, ada_cond=ada_cond, norm_qkv=norm_qkv
                    )
                else:  # For VLM
                    query_state, key_state, value_state = models[i].layers[layer_idx](
                        hidden_states, compute_kqv=True, norm_qkv=norm_qkv
                    )
                    gate = None

                if query_state.dtype != torch.float32:
                    query_state, key_state, value_state = (
                        query_state.to(torch.float32),
                        key_state.to(torch.float32),
                        value_state.to(torch.float32),
                    )
                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)
                gates.append(gate)

            # B,L,H,D with L sequence length (img, lang, state, action), H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            key_states, value_states, past_key_values = self.handle_kv_cache(
                key_states,
                value_states,
                layer_idx,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )

            att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is not None:
                    end = start + hidden_states.shape[1]
                    if i == 1:
                        out_emb = models[i].layers[layer_idx](
                            hidden_states,
                            att_output,
                            start,
                            end,
                            output_atten=True,
                            ada_cond=ada_cond,
                            gate=(gates[0] if len(gates) == 1 else gates[i]),
                        )
                    else:
                        out_emb = models[i].layers[layer_idx](
                            hidden_states, att_output, start, end, output_atten=True
                        )
                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                if self.config.final_norm_adanorm:
                    if i == 1:
                        out_emb, _ = models[i].norm(hidden_states, ada_cond)
                    else:
                        out_emb = models[i].norm(hidden_states)
                else:
                    out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            raise NotImplementedError("FA2 is not implemented (yet)")
        elif self.config.attention_implementation == "flex":
            attention_interface = flex_attention_forward
        elif self.config.attention_implementation == "eager":
            attention_interface = our_eager_attention_forward
        elif self.config.attention_implementation == "xformer":
            # attention_interface = xformer_attention_forward
            raise NotImplementedError("Xformer attention is not implemented (yet)")
        else:
            raise ValueError(
                f"Invalid attention implementation: {self.config.attention_implementation}. "
                "Expected one of ['fa2', 'flex', 'eager', 'xformer']."
            )
        return attention_interface


class LingbotVLAPolicy(PreTrainedPolicy):
    """LingBot-VLA policy for cross-embodiment robotic control.

    Couples a Qwen2.5-VL vision-language backbone with a narrow action expert
    (pi0-style dual-stream) and predicts action chunks via flow matching.

    The model expects already model-ready tensors in the batch (produced by the
    lingbot_vla processor / feature transform):
        - ``images``: (B, num_views, num_patches, patch_dim) patchified pixels
        - ``img_masks``: (B, num_views) per-view validity mask
        - ``lang_tokens`` / ``lang_masks``: (B, L) tokenized instruction + mask
        - ``observation.state``: (B, max_state_dim) padded state
        - ``action``: (B, chunk_size, max_action_dim) padded action (training only)
    """

    config_class = LingbotVLAConfig
    name = "lingbot_vla"
    _no_split_modules = ["Qwen2DecoderLayer", "FixQwen2RMSNorm"]

    def __init__(self, config: LingbotVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.language_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.model = FlowMatching(config, eval=False)

        if not getattr(self.config, "use_lm_head", False):
            del self.model.qwenvl_with_expert.qwenvl.lm_head
        del self.model.qwenvl_with_expert.qwen_expert.lm_head

        self.reset()
        torch.set_float32_matmul_precision("high")

    def reset(self):
        """Reset the rolling action queue used by select_action."""
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _extract_model_inputs(self, batch: dict[str, Tensor]):
        dtype = next(self.parameters()).dtype
        images = batch["images"].to(dtype=dtype)
        img_masks = batch["img_masks"]
        lang_tokens = batch["lang_tokens"]
        lang_masks = batch["lang_masks"]
        state = batch[OBS_STATE].to(dtype=dtype)
        state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))
        return images, img_masks, lang_tokens, lang_masks, state

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward pass returning the flow-matching loss in lerobot convention."""
        images, img_masks, lang_tokens, lang_masks, state = self._extract_model_inputs(batch)
        actions = batch[ACTION].to(dtype=state.dtype)
        action_dim = actions.shape[-1]
        actions = F.pad(actions, (0, self.config.max_action_dim - action_dim))

        losses, loss_depth, _ = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            label=None,
            noise=batch.get("noise"),
            time=batch.get("time"),
            vlm_causal=False,
            loss_type=self.config.loss_type,
            depth_targets=batch.get("depth_targets"),
            norm_qkv=False,
        )

        losses = losses[:, :, :action_dim]
        loss_vla = losses.mean()

        loss_dict: dict[str, Tensor] = {"l2_loss": loss_vla.item()}
        total_loss = loss_vla
        if not isinstance(loss_depth, int):
            loss_dict["depth_loss"] = loss_depth.item()
            total_loss = total_loss + loss_depth
        loss_dict["loss"] = total_loss.item()

        return total_loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Run flow-matching denoising and return an action chunk (B, chunk, action_dim)."""
        self.eval()
        images, img_masks, lang_tokens, lang_masks, state = self._extract_model_inputs(batch)

        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            num_steps=self.config.num_steps,
        )
        action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action for environment execution, buffering chunks in a queue."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()


class FlowMatching(nn.Module):
    def __init__(self, config, eval):
        super().__init__()
        self.config = config
        # qwenvl with action expert
        qwenvl_with_export_config = QwenvlWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            vocab_size=getattr(self.config, "vocab_size", 0),
            use_lm_head=getattr(self.config, "use_lm_head", False),
            attention_implementation=self.config.attention_implementation,
            tokenizer_path=self.config.tokenizer_path,
            use_cache=getattr(self.config, "use_cache", True),
        )
        qwenvl_with_export_config.adanorm_time = getattr(config, "adanorm_time", False)
        qwenvl_with_export_config.split_gate_liner = getattr(config, "split_gate_liner", False)
        qwenvl_with_export_config.no_split_gate_liner = getattr(config, "nosplit_gate_liner", False)
        qwenvl_with_export_config.separate_time_proj = getattr(config, "separate_time_proj", False)
        qwenvl_with_export_config.final_norm_adanorm = getattr(config, "final_norm_adanorm", False)
        qwenvl_with_export_config.norm_qkv = getattr(config, "norm_qkv", False)
        self.qwenvl_with_expert = QwenvlWithExpertModel(qwenvl_with_export_config, eval)
        self.config.proj_width = qwenvl_with_export_config.qwen_expert_config.hidden_size
        self.config.initializer_range = getattr(
            qwenvl_with_export_config.qwen_expert_config, "initializer_range", None
        )
        # projection layers
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        if getattr(config, "separate_time_proj", False):
            self.time_mlp_in = nn.Linear(self.config.proj_width, self.config.proj_width)
            self.time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        else:
            self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
            self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        self.config.align_params = getattr(self.config, "align_params", {})
        if self.config.align_params != {}:
            self.steps = 0
            self.use_depth_align = True
            self.init_depth_heads(self.config.align_params)
        else:
            self.use_depth_align = False

        self.set_requires_grad()

    def init_depth_heads(self, config):
        # Depth-alignment heads live in the LingBot source tree; import lazily so the
        # no-depth checkpoint (align_params={}) carries no extra dependency.
        from lingbotvla.models.vla.vision_models.align_heads.depth_head import (
            TaskTokenDepthHead,
        )

        self.llm_image_token_size = config["llm"]["image_token_size"]
        self.llm_image_input_size = config["llm"]["image_input_size"]
        self.depth_token_size = config["depth"]["token_size"]
        self.depth_input_size = config["depth"]["input_size"]
        self.align_type = config.get("mode", None)
        self.model_type = config["depth"]["model_type"]

        if self.align_type == "direct":
            self.depth_align_head = nn.Sequential(
                nn.Linear(config["llm"]["dim_out"], config["depth"]["dim_out"] * 2),
                nn.GELU(),
                nn.Linear(config["depth"]["dim_out"] * 2, config["depth"]["dim_out"]),
            )
            for p in self.depth_align_head.parameters():
                p.requires_grad = True
        elif self.align_type == "query":
            self.num_task_tokens = config["num_task_tokens"]
            assert config["depth"]["num_backbone_tokens"] % self.num_task_tokens == 0
            self.depth_align_embs = nn.Parameter(
                torch.randn(config["depth"]["num_backbone_tokens"], config["llm"]["dim_out"])
            ).to(dtype=torch.bfloat16)
            self.depth_align_embs.requires_grad_ = True

            self.depth_align_head = TaskTokenDepthHead(
                config["depth"], llm_hidden_size=config["llm"]["dim_out"]
            ).to(dtype=torch.bfloat16)

            for p in self.depth_align_head.parameters():
                p.requires_grad = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, vlm_causal, label=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize = images.shape[0]
        device = images.device

        # embed image
        if images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        elif images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 3:  # For inference bs=1
            bsize = 1

        img_emb = self.qwenvl_with_expert.embed_image(images)
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b (n l) d", b=bsize)  # bsize = 24
        num_img_embs = img_emb.shape[1]
        if img_masks.ndim == 1:  # For inference bs=1
            img_masks = img_masks.unsqueeze(0)
        if self.use_depth_align and self.align_type == "query":
            align_masks = einops.repeat(img_masks, "b n -> b (n l)", l=self.num_task_tokens)
        img_masks = einops.repeat(img_masks, "b n -> b (n l)", l=num_patch)

        # embed language
        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens)
        num_lang_embs = lang_emb.shape[1]

        if self.use_depth_align and self.align_type == "query":

            def _get_align_tokens(tokens):
                tk_weights = tokens.view(
                    self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1]
                )
                tk_weights = tk_weights.mean(dim=1)
                return tk_weights

            align_embs = (
                _get_align_tokens(self.depth_align_embs)
                .repeat(img_emb.size(0), 1, 1)
                .to(img_emb.device, img_emb.dtype)
            )
            # align_masks = einops.rearrange(img_masks, "b (n l) -> b n l", n=3)
            # align_masks = align_masks[:, :, 0]
            # align_masks = einops.repeat(align_masks, "b n -> b (n l)", l=self.num_task_tokens)
            embs = torch.cat([img_emb, align_embs, align_embs, align_embs, lang_emb], dim=1)
            pad_masks = torch.cat([img_masks, align_masks, lang_masks], dim=1)
        else:
            # assemble embeddings
            embs = torch.cat([img_emb, lang_emb], dim=1)
            pad_masks = torch.cat([img_masks, lang_masks], dim=1)

        # (see `make_att_2d_masks` to understand why zeros means bidirection)
        if not vlm_causal:
            if self.use_depth_align and self.align_type == "query":
                att_masks = torch.zeros(
                    (img_emb.size(0), num_img_embs + 3 * self.num_task_tokens + num_lang_embs),
                    device=device,
                    dtype=torch.bool,
                )
            else:
                att_masks = torch.zeros(
                    (img_emb.size(0), num_img_embs + num_lang_embs), device=device, dtype=torch.bool
                )
        else:
            if self.use_depth_align and self.align_type == "query":
                att_masks = torch.ones(
                    (img_emb.size(0), num_img_embs + 3 * self.num_task_tokens + num_lang_embs),
                    device=device,
                    dtype=torch.bool,
                )
            else:
                att_masks = torch.ones(
                    (img_emb.size(0), num_img_embs + num_lang_embs), device=device, dtype=torch.bool
                )
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        bsize = state.shape[0]  # state_bs = img_bs
        device = state.device
        dtype = state.dtype
        # embed state
        state_emb = self.state_proj(state)  # torch.Size([state_bs, 1024])

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(  # 1, 1024
            timestep,  # torch.Size([1]))
            self.config.proj_width,  # 1024
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb_ori = time_emb

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)  # torch.Size([1, state_bs*50, 1024])
        if getattr(self.config, "separate_time_proj", False):
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb_ori = F.silu(self.time_mlp_out(time_emb))  # [1, 1024]
            action_time_emb = action_emb
        else:
            time_emb = einops.repeat(
                time_emb, "b d -> b n d", n=action_emb.shape[1]
            )  # [1, 1024] -> [1, state_bs*50, 1024]
            action_time_emb = torch.cat([action_emb, time_emb], dim=-1)  # [1, state_bs*50, 2048]

            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)  # swish == silu
            action_time_emb = self.action_time_mlp_out(action_time_emb)  # [1, state_bs*50, 1024]
        action_time_dim = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks = torch.ones((bsize, action_time_dim + 1), device=device, dtype=torch.bool)

        # Set attention masks for suffix tokens so that prefix tokens cannot attend to suffix tokens.
        # And state token cannot attend action tokens.
        # Action tokens use a bidirectional attention.
        att_masks = torch.zeros((bsize, action_time_dim + 1), device=device, dtype=torch.bool)
        att_masks[:, :2] = True

        return time_emb_ori, embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        label=None,
        noise=None,
        time=None,
        vlm_causal=False,
        loss_type="fm",
        depth_targets=None,
        norm_qkv=False,
    ) -> Tensor:
        dtype = state.dtype
        device = state.device
        if noise is None:
            noise = torch.randn(actions.shape, device=device, dtype=dtype)

        if time is None:
            time = self.sample_time(actions.size(0), device).to(dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal, label
        )  # 1,bs_img*(768+48),2048  1,bs_img*(768+48)  1,bs_img*(768+48)
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time
        )  # [1, state_bs*(50+1), 1024], [1, state_bs*(50+1)], [1, state_bs*(50+1)]   state_bs=bs_img

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # pad_masks = pad_masks.reshape(state.size(0), -1)
        # att_masks = att_masks.reshape(state.size(0), -1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        if self.use_depth_align and self.align_type == "query":
            att_2d_masks = self.make_att_2d_masks_with_query(
                att_2d_masks, prefix_pad_masks.shape[-1], img_masks
            )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # prefix_embs = prefix_embs.reshape(state.size(0), -1, prefix_embs.size(-1))
        # suffix_embs = suffix_embs.reshape(state.size(0), -1, suffix_embs.size(-1))
        (outputs_embeds, suffix_out), _ = self.qwenvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            vlm_position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
            norm_qkv=norm_qkv,
        )
        if self.config.align_params != {}:
            loss_depth, depth_preds = self.depth_emb_forward(outputs_embeds, depth_targets, img_masks)
            loss_depth = loss_depth * self.config.align_params["depth_loss_weight"]
            self.steps += 1
        else:
            loss_depth = 0
            depth_preds = None
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if suffix_out.dtype != self.action_out_proj.weight.dtype:
            suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)
        # u_t = u_t.reshape(images.size(0), -1, u_t.size(-1))
        if loss_type == "fm":
            losses = F.mse_loss(u_t, v_t, reduction="none")
        elif loss_type == "L1_fm":
            losses = F.l1_loss(u_t, v_t, reduction="none")

        return losses, loss_depth, depth_preds

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, vlm_causal=False, noise=None, num_steps=None
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal
        )
        prefix_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )  # bs, prefix_len, prefix_len
        if self.use_depth_align and self.align_type == "query":
            prefix_att_2d_masks = self.make_att_2d_masks_with_query(
                prefix_att_2d_masks, prefix_pad_masks.shape[-1], img_masks
            )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        num_steps = num_steps if num_steps is not None else self.config.num_steps
        dt = torch.tensor(-1.0 / num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        count = 0
        while time >= -dt / 2:
            count += 1
            expanded_time = time.expand(bsize)

            v_t = self.predict_velocity(state, prefix_pad_masks, past_key_values, x_t, expanded_time)

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """predict velocity at time t using the suffix model."""
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
        )  # bs, suffix_len, prefix_len+suffix_len

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def make_att_2d_masks_with_query(self, att_2d_masks, prefix_len, img_masks):
        if img_masks.ndim == 1:
            img_masks = img_masks.unsqueeze(0)

        num_image_tokens = self.llm_image_token_size * self.llm_image_token_size

        att_2d_masks[
            :,
            num_image_tokens * 0 : num_image_tokens * 3,
            num_image_tokens * 3 + self.num_task_tokens * 0 : num_image_tokens * 3 + self.num_task_tokens * 3,
        ] = False
        att_2d_masks[
            :,
            num_image_tokens * 3 + self.num_task_tokens * 3 : prefix_len,
            num_image_tokens * 3 + self.num_task_tokens * 0 : num_image_tokens * 3 + self.num_task_tokens * 3,
        ] = False

        att_2d_masks[
            :,
            num_image_tokens * 3 + self.num_task_tokens * 0 : num_image_tokens * 3 + self.num_task_tokens * 3,
            :,
        ] = False

        att_2d_masks[
            img_masks[:, 0],
            num_image_tokens * 3 + self.num_task_tokens * 0 : num_image_tokens * 3 + self.num_task_tokens * 1,
            num_image_tokens * 0 : num_image_tokens * 1,
        ] = True
        att_2d_masks[
            img_masks[:, 1],
            num_image_tokens * 3 + self.num_task_tokens * 1 : num_image_tokens * 3 + self.num_task_tokens * 2,
            num_image_tokens * 1 : num_image_tokens * 2,
        ] = True
        att_2d_masks[
            img_masks[:, 2],
            num_image_tokens * 3 + self.num_task_tokens * 2 : num_image_tokens * 3 + self.num_task_tokens * 3,
            num_image_tokens * 2 : num_image_tokens * 3,
        ] = True

        att_2d_masks[
            img_masks[:, 0],
            num_image_tokens * 3 + self.num_task_tokens * 0 : num_image_tokens * 3 + self.num_task_tokens * 1,
            num_image_tokens * 3 + self.num_task_tokens * 0 : num_image_tokens * 3 + self.num_task_tokens * 1,
        ] = True
        att_2d_masks[
            img_masks[:, 1],
            num_image_tokens * 3 + self.num_task_tokens * 1 : num_image_tokens * 3 + self.num_task_tokens * 2,
            num_image_tokens * 3 + self.num_task_tokens * 1 : num_image_tokens * 3 + self.num_task_tokens * 2,
        ] = True
        att_2d_masks[
            img_masks[:, 2],
            num_image_tokens * 3 + self.num_task_tokens * 2 : num_image_tokens * 3 + self.num_task_tokens * 3,
            num_image_tokens * 3 + self.num_task_tokens * 2 : num_image_tokens * 3 + self.num_task_tokens * 3,
        ] = True

        return att_2d_masks

    def depth_emb_forward(self, hidden_states, depth_targets=None, img_masks=None):
        chunk_size = self.llm_image_token_size * self.llm_image_token_size
        img_masks = einops.rearrange(img_masks, "b n -> (b n)")
        if self.align_type == "direct":
            image_embs = hidden_states[:, chunk_size * 0 : chunk_size * 3, :]
            image_embs = einops.rearrange(image_embs, "b (n l) c -> (b n) l c", n=3)

            depth_preds = self.depth_align_head(image_embs).contiguous().float()
        elif self.align_type == "query":
            align_embs = hidden_states[:, chunk_size * 3 : chunk_size * 3 + self.num_task_tokens * 3, :]
            align_embs = einops.rearrange(align_embs, "b (n l) c -> (b n) l c", n=3)

            image_embs = hidden_states[:, chunk_size * 0 : chunk_size * 3, :]
            image_embs = einops.rearrange(image_embs, "b (n l) c -> (b n) l c", n=3)

            align_embs = torch.cat([image_embs, align_embs], dim=1)
            depth_preds = self.depth_align_embs.repeat(align_embs.shape[0], 1, 1).to(
                dtype=align_embs.dtype, device=align_embs.device
            )

            depth_preds = self.depth_align_head(align_embs, depth_preds).contiguous().float()

        loss = self._emb_loss(depth_preds[img_masks], depth_targets[img_masks])

        return loss, depth_preds

    def _emb_loss(self, emb_preds, emb_targets):
        if self.align_type == "direct":
            S, L, D = emb_targets.shape
            emb_preds = emb_preds.contiguous().view(
                S * self.llm_image_token_size * self.llm_image_token_size, emb_preds.shape[-1]
            )

            emb_targets = emb_targets.to(emb_preds.dtype).to(emb_preds.device)
            emb_targets = (
                emb_targets.view(S, self.depth_token_size, self.depth_token_size, D)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            emb_targets = F.adaptive_avg_pool2d(
                emb_targets, (self.llm_image_token_size, self.llm_image_token_size)
            ).view(S, D, self.llm_image_token_size, self.llm_image_token_size)
            emb_targets = (
                emb_targets.view(S, D, self.llm_image_token_size * self.llm_image_token_size)
                .permute(0, 2, 1)
                .contiguous()
                .view(S * self.llm_image_token_size * self.llm_image_token_size, D)
            )

            l1_loss = F.l1_loss(emb_preds.float(), emb_targets.float().detach(), reduction="none")

            emb_preds_norm = F.normalize(emb_preds.float(), p=2, dim=-1, eps=1e-6)
            emb_targets_norm = F.normalize(emb_targets.float(), p=2, dim=-1, eps=1e-6)
            emb_preds_matrix = torch.matmul(emb_preds_norm, emb_preds_norm.transpose(0, 1))
            emb_targets_matrix = torch.matmul(emb_targets_norm, emb_targets_norm.transpose(0, 1))

            sim_loss = F.l1_loss(
                emb_preds_matrix.float(), emb_targets_matrix.float().detach(), reduction="none"
            )

            emb_loss = sim_loss.mean() + l1_loss.mean()
        elif self.align_type == "query":
            l1_loss = F.smooth_l1_loss(emb_preds.float(), emb_targets.float().detach(), reduction="none")
            emb_loss = l1_loss.mean()
            if self.model_type == "MoRGBD":
                emb_loss = emb_loss
        return emb_loss


__all__ = [
    "LingbotVLAPolicy",
    "FlowMatching",
    "QwenvlWithExpertModel",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2ForCausalLM",
    "Qwen2_5_VLPreTrainedModel",
]
