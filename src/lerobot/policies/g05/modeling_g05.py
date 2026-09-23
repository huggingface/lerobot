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

"""Native G0.5 policy, model, and ActionCodec implementation."""

from __future__ import annotations

import itertools
import json
import math
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
    Qwen3_5VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import OptimizerParams
from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, MESSAGES_RENDERED, OBS_STATE
from lerobot.utils.device_utils import resolve_safetensors_device

from .action_codec_g05 import G05NativeActionCodec
from .configuration_g05 import (
    G05_POLICY_PARTS,
    G05Config,
    make_g05_cot_prompt_template,
    make_g05_prompt_template,
)
from .tokenizer_g05 import (
    G05_INPUT_IDS,
    G05_LABELS,
    G05_RUNTIME_PREDICT_COT,
    G05_SPLIT_INDEX,
    G05_TOKEN_TYPES,
    IGNORE_INDEX,
    G05SequenceBatch,
    G05Tokenizer,
    G05TokenType,
)


class G05GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Qwen3.5 linear attention with the numerical contract used by G0.5."""

    def forward(
        self,
        hidden_states: Tensor,
        cache_params: DynamicCache | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Run the gated delta-net mixer over the sequence."""
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None]

        batch_size, sequence_length, _ = hidden_states.shape
        use_cached_state = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_cached_state:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        gate = self.in_proj_z(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_v_heads,
            self.head_v_dim,
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = self.in_proj_a(hidden_states)

        if use_cached_state:
            if sequence_length == 1:
                mixed_qkv = self.causal_conv1d_update(
                    mixed_qkv,
                    conv_state,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.activation,
                )
            else:
                conv_input = torch.cat((conv_state[..., 1:], mixed_qkv), dim=-1)
                mixed_qkv = functional.silu(
                    functional.conv1d(
                        conv_input,
                        self.conv1d.weight,
                        self.conv1d.bias,
                        groups=self.conv1d.groups,
                    )
                )
                cache_params.layers[self.layer_idx].conv_states.copy_(
                    conv_input[..., -self.conv_kernel_size :]
                )
        else:
            if cache_params is not None:
                conv_state = functional.pad(
                    mixed_qkv,
                    (self.conv_kernel_size - mixed_qkv.shape[-1], 0),
                )
                cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = functional.silu(self.conv1d(mixed_qkv)[:, :, :sequence_length])

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2),
            (self.key_dim, self.key_dim, self.value_dim),
            dim=-1,
        )
        query = query.reshape(batch_size, sequence_length, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, sequence_length, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, sequence_length, self.num_v_heads, self.head_v_dim)

        with torch.autocast(hidden_states.device.type, enabled=False):
            decay = -self.A_log.float().exp() * functional.softplus(decay.float() + self.dt_bias.float())
        head_repeats = self.num_v_heads // self.num_k_heads
        if head_repeats > 1:
            query = query.repeat_interleave(head_repeats, dim=2)
            key = key.repeat_interleave(head_repeats, dim=2)

        if use_cached_state and sequence_length == 1:
            attended, recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            initial_state = recurrent_state.clone() if use_cached_state else None
            attended, recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                chunk_size=32,
                initial_state=initial_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(recurrent_state, self.layer_idx)

        attended = attended.reshape(-1, self.head_v_dim)
        gate = gate.reshape(-1, self.head_v_dim)
        with torch.autocast(hidden_states.device.type, enabled=False):
            attended = self.norm(attended.float(), gate.float())
        attended = attended.reshape(batch_size, sequence_length, self.value_dim)
        return self.out_proj(attended)


@dataclass
class G05TextGeneration:
    """Generated tokens plus the state needed to continue into ActionCodec decoding."""

    token_ids: Tensor
    stop_tokens: Tensor
    history: Tensor | None
    history_mask: Tensor | None


def _autoregressive_ce_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    ce_weight: float,
    z_loss_scale: float,
) -> Tensor:
    """Apply G0.5's checkpoint-configured autoregressive objective."""

    if ce_weight < 0:
        raise ValueError("G0.5 ar.ce_weight must be non-negative.")
    if z_loss_scale < 0:
        raise ValueError("G0.5 ar.ce_z_loss_scale must be non-negative.")
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "G0.5 autoregressive CE expects logits [N,V] and labels [N], "
            f"got {tuple(logits.shape)} and {tuple(labels.shape)}."
        )
    if logits.shape[0] == 0 or ce_weight == 0:
        return logits.sum() * 0

    token_loss = functional.cross_entropy(logits, labels, reduction="none")
    if z_loss_scale:
        log_z = torch.logsumexp(logits.float(), dim=-1)
        token_loss = token_loss.float() + z_loss_scale * log_z.square()
    return token_loss.mean() * ce_weight


def _qwen_text_config(values: Mapping[str, Any], *, vocab_size: int | None = None):
    """Translate the serialized G0.5 Qwen config into a Transformers config."""

    return Qwen3_5TextConfig(
        vocab_size=int(vocab_size if vocab_size is not None else values.get("vocab_size", 1)),
        hidden_size=int(values["hidden_size"]),
        intermediate_size=int(values["intermediate_size"]),
        num_hidden_layers=int(values["num_hidden_layers"]),
        num_attention_heads=int(values["num_attention_heads"]),
        num_key_value_heads=int(values["num_key_value_heads"]),
        head_dim=int(values["head_dim"]),
        rms_norm_eps=float(values["rms_norm_eps"]),
        max_position_embeddings=int(values["max_position_embeddings"]),
        attention_bias=bool(values.get("attention_bias", False)),
        hidden_act=str(values.get("hidden_act", "silu")),
        rope_parameters=dict(values["rope_parameters"]),
        linear_conv_kernel_dim=int(values.get("linear_conv_kernel_dim", 4)),
        linear_key_head_dim=int(values.get("linear_key_head_dim", 128)),
        linear_value_head_dim=int(values.get("linear_value_head_dim", 128)),
        linear_num_key_heads=int(values.get("linear_num_key_heads", 16)),
        linear_num_value_heads=int(values.get("linear_num_value_heads", 16)),
        layer_types=list(values["layer_types"]),
        pad_token_id=values.get("pad_token_id"),
        tie_word_embeddings=True,
    )


def _qwen_vision_config(values: Mapping[str, Any]):
    """Translate the serialized G0.5 vision config into Transformers."""

    config = Qwen3_5VisionConfig(
        depth=int(values["depth"]),
        hidden_size=int(values["hidden_size"]),
        num_heads=int(values["num_heads"]),
        patch_size=int(values["patch_size"]),
        temporal_patch_size=int(values["temporal_patch_size"]),
        spatial_merge_size=int(values["spatial_merge_size"]),
        in_channels=int(values.get("in_channels", 3)),
        intermediate_size=int(values["intermediate_size"]),
        out_hidden_size=int(values["out_hidden_size"]),
        num_position_embeddings=int(values["num_position_embeddings"]),
        hidden_act=str(values.get("hidden_act", "gelu_pytorch_tanh")),
    )
    config.temporal_freq = int(values.get("temporal_freq", 0))
    config.spacetime_mode = str(values.get("spacetime_mode", "factorized"))
    config.token_drop_layer = values.get("token_drop_layer")
    config.temporal_pe_pretrain_frames = values.get("temporal_pe_pretrain_frames")
    config.batch_all_cameras = bool(values.get("batch_all_cameras", False))
    return config


class G05ProprioEmbedder(nn.Module):
    """Project the padded G0.5 proprioception vector into the VLM hidden size."""

    def __init__(self, proprio_dim: int, hidden_size: int) -> None:
        """Build the proprioception MLP."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, proprio: Tensor) -> Tensor:
        """Embed proprioception in float32."""
        with torch.autocast(proprio.device.type, enabled=False):
            return self.mlp(proprio.float())


class G05QwenTextModel(nn.Module):
    """Qwen3.5 text stack with G0.5's checkpoint-compatible module names."""

    def __init__(self, values: Mapping[str, Any], *, vocab_size: int) -> None:
        """Build the Qwen backbone and its projection."""
        super().__init__()

        self.config = _qwen_text_config(values, vocab_size=vocab_size)
        self.input_proj = nn.Embedding(vocab_size, self.config.hidden_size, self.config.pad_token_id)
        layers = []
        for layer_idx in range(self.config.num_hidden_layers):
            layer = Qwen3_5DecoderLayer(self.config, layer_idx)
            if self.config.layer_types[layer_idx] == "linear_attention":
                layer.linear_attn = G05GatedDeltaNet(self.config, layer_idx)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.norm = Qwen3_5RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.config)

    def embed(self, input_ids: Tensor) -> Tensor:
        """Project input ids to hidden states."""
        return self.input_proj(input_ids)

    def logits(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states back to vocabulary logits."""
        return torch.nn.functional.linear(hidden_states, self.input_proj.weight)

    def forward(
        self,
        inputs_embeds: Tensor,
        *,
        full_attention_mask: Tensor,
        linear_attention_mask: Tensor,
        position_ids: Tensor,
        cache=None,
    ) -> tuple[Tensor, Any]:
        """Run the backbone, creating a cache when none is given."""
        if cache is None:
            cache = DynamicCache(config=self.config)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer_index, layer in enumerate(self.layers):
            attention_mask = (
                linear_attention_mask
                if self.config.layer_types[layer_index] == "linear_attention"
                else full_attention_mask
            )
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids[0],
                past_key_values=cache,
                use_cache=True,
            )
        return self.norm(hidden_states), cache


class G05ActionDecoderLayer(nn.Module):
    """Qwen3.5 decoder layer with G0.5 adaptive RMSNorm conditioning."""

    def __init__(self, config, layer_idx: int) -> None:
        """Build the time-conditioned decoder layer."""
        super().__init__()

        if config.layer_types[layer_idx] != "full_attention":
            raise ValueError("The released G0.5 action expert requires full-attention layers.")
        self.layer_idx = layer_idx
        self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = PiGemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cond_dim=config.hidden_size,
        )
        self.post_attention_layernorm = PiGemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cond_dim=config.hidden_size,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_ids: Tensor,
        cache,
        time_cond: Tensor,
    ) -> Tensor:
        """Apply time-conditioned attention and feed-forward."""
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond=time_cond)
        key_length = cache.layers[self.layer_idx].get_seq_length() + hidden_states.shape[1]
        layer_attention_mask = (
            attention_mask[..., -key_length:] if attention_mask.shape[-1] != key_length else attention_mask
        )
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=layer_attention_mask,
            position_ids=position_ids[0],
            past_key_values=cache,
            position_embeddings=position_embeddings,
            use_cache=True,
        )
        hidden_states = residual + hidden_states if gate is None else residual + hidden_states * gate

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=time_cond)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states if gate is None else residual + hidden_states * gate


class G05ActionExpert(nn.Module):
    """Continuous G0.5 action expert with checkpoint-compatible parameter names."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        """Build the action expert layers and projections."""
        super().__init__()

        self.config = _qwen_text_config(values)
        input_dim = int(values["input_dim"])
        output_dim = int(values["output_dim"])
        hidden_size = int(values["hidden_size"])
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.layers = nn.ModuleList(
            [
                G05ActionDecoderLayer(self.config, layer_idx)
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = PiGemmaRMSNorm(
            hidden_size,
            eps=self.config.rms_norm_eps,
            cond_dim=hidden_size,
        )
        self.output_proj = nn.Linear(hidden_size, output_dim)
        self.time_mlp_in = nn.Linear(hidden_size, hidden_size)
        self.time_mlp_out = nn.Linear(hidden_size, hidden_size)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.config)

    def embed(self, actions: Tensor) -> Tensor:
        """Project actions to hidden states."""
        return self.input_proj(actions)

    def encode_time(self, timesteps: Tensor) -> Tensor:
        """Embed the flow timesteps."""
        half = self.config.hidden_size // 2
        fraction = torch.linspace(0.0, 1.0, half, device=timesteps.device, dtype=torch.float32)
        periods = 4e-3 * (4.0 / 4e-3) ** fraction
        phase = timesteps.float().unsqueeze(-1) * (2 * math.pi / periods)
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        with torch.autocast(timesteps.device.type, enabled=False):
            return torch.nn.functional.silu(
                self.time_mlp_out(torch.nn.functional.silu(self.time_mlp_in(embedding)))
            )

    def forward(
        self,
        inputs_embeds: Tensor,
        *,
        attention_mask: Tensor,
        position_ids: Tensor,
        cache,
        time_cond: Tensor,
    ) -> Tensor:
        """Run the action expert layers."""
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache=cache,
                time_cond=time_cond,
            )
        hidden_states, _ = self.norm(hidden_states, cond=time_cond)
        return hidden_states

    def decode(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states back to actions in float32."""
        with torch.autocast(hidden_states.device.type, enabled=False):
            return self.output_proj(hidden_states.float())


class G05NativeModel(nn.Module):
    """Weight-owning native G0.5 model assembled from serialized checkpoint config."""

    def __init__(self, model_config: Mapping[str, Any], *, vocab_size: int) -> None:
        """Build the vision-language model, action expert and embedders."""
        super().__init__()

        self.vision_tower = Qwen3_5VisionModel(_qwen_vision_config(model_config["vision"]))
        self.vlm = G05QwenTextModel(model_config["vlm"], vocab_size=vocab_size)
        self.action_expert = G05ActionExpert(model_config["action_expert"])
        self.proprio_embedder = G05ProprioEmbedder(
            int(model_config["proprio_dim"]),
            int(model_config["vlm"]["hidden_size"]),
        )


def _temporal_embedding(timesteps: Tensor, dimension: int) -> Tensor:
    """Sinusoidal embedding for a temporal index."""
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    phase = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.stack((phase.sin(), phase.cos() - 1), dim=-1).reshape(len(timesteps), dimension)


class G05NativeBackend(nn.Module):
    """Native LeRobot backend for G0.5.

    Inference and training routing are added around this checkpoint-compatible
    model core; no OpenGalaxea Python package is imported.
    """

    def __init__(
        self,
        model_config: Mapping[str, Any],
        *,
        vocab_size: int,
        processor_path: str | Path,
        action_tokenizer_path: str | Path,
    ) -> None:
        """Store the model configuration and build the native model."""
        super().__init__()
        self.model_config = dict(model_config)
        self.action_tokenizer_path = Path(action_tokenizer_path)
        self.model = G05NativeModel(self.model_config, vocab_size=vocab_size)
        attention_implementation = str(self.model_config.get("attn_implementation", "eager"))
        self.model.vlm.config._attn_implementation = attention_implementation
        self.model.action_expert.config._attn_implementation = attention_implementation
        self.model.vision_tower.config._attn_implementation = attention_implementation
        self.processor = G05Tokenizer(processor_path, self.model_config)
        if len(self.processor) != vocab_size:
            raise ValueError(
                f"G0.5 tokenizer has {len(self.processor)} rows, but model expects {vocab_size}."
            )
        self.action_tokenizer = None
        action_config = self.model_config.get("AT_CONFIG")
        if (
            isinstance(action_config, Mapping)
            and self.action_tokenizer_path.is_file()
            and not next(self.model.parameters()).is_meta
        ):
            self.action_tokenizer = G05NativeActionCodec.load(
                action_config,
                action_token_begin=self.processor.action_token_begin,
                ckpt_path=self.action_tokenizer_path,
            )
        self._last_vision_grids: list[tuple[int, int, int]] = []

    def materialize_runtime_buffers(self, device: torch.device | str) -> None:
        """Rebuild non-persistent Transformers buffers after meta construction."""

        for module in self.modules():
            if isinstance(module, Qwen3_5TextRotaryEmbedding) and module.inv_freq.is_meta:
                # The Transformers initializer creates its arange before moving it
                # to ``device``. Override the outer meta-device construction context
                # so that intermediate is materialized on the final device as well.
                with torch.device(device):
                    inverse_frequency, attention_scaling = module.compute_default_rope_parameters(
                        module.config,
                        device,
                    )
                module.inv_freq = inverse_frequency
                module.original_inv_freq = inverse_frequency.clone()
                module.attention_scaling = attention_scaling
            elif isinstance(module, Qwen3_5VisionRotaryEmbedding) and module.inv_freq.is_meta:
                frequency = torch.arange(0, module.dim, 2, dtype=torch.float32, device=device)
                module.inv_freq = 1.0 / (module.theta ** (frequency / module.dim))

        remaining = [name for name, buffer in self.named_buffers() if buffer.is_meta]
        if remaining:
            raise RuntimeError(f"G0.5 meta loading left runtime buffers unmaterialized: {remaining}")

    def apply_fp32_params(self) -> None:
        """Restore the FP32 islands used by the released mixed-precision runtime."""

        patterns = (
            "vision_tower.patch_embed",
            "vision_tower.pos_embed",
            "vision_tower.merger",
            "norm1",
            "norm2",
            "input_layernorm",
            "post_attention_layernorm",
            "q_norm",
            "k_norm",
            "linear_attn.norm",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "vlm.norm",
            "action_expert.norm",
            "action_expert.input_proj",
            "action_expert.output_proj",
            "action_expert.time_mlp",
            "proprio_embedder",
        )
        for name, parameter in self.named_parameters():
            if any(pattern in name for pattern in patterns):
                parameter.data = parameter.data.float()

    @staticmethod
    def _should_apply_weight_decay(
        owner_module: nn.Module | None,
        leaf_name: str,
        parameter: nn.Parameter,
    ) -> bool:
        """Whether a parameter takes weight decay."""
        return leaf_name != "bias" and parameter.ndim > 1 and not isinstance(owner_module, nn.Embedding)

    def get_optim_param_groups(
        self,
        *,
        lr: float,
        weight_decay: float,
        apply_decay_on_norm_and_bias: bool = False,
        backbone_lr_multiplier: float = 1.0,
        vision_lr_multiplier: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Build the released six native backbone/action/vision parameter groups."""

        action_parameters = {id(parameter) for parameter in self.model.action_expert.parameters()}
        vision_parameters = {id(parameter) for parameter in self.model.vision_tower.parameters()}
        modules = dict(self.model.named_modules())
        grouped: dict[str, list[nn.Parameter]] = {
            "backbone_decay": [],
            "action_decay": [],
            "vision_decay": [],
            "backbone_no_decay": [],
            "action_no_decay": [],
            "vision_no_decay": [],
        }
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            owner_name, _, leaf_name = name.rpartition(".")
            decay = apply_decay_on_norm_and_bias or self._should_apply_weight_decay(
                modules.get(owner_name),
                leaf_name,
                parameter,
            )
            if id(parameter) in action_parameters:
                family = "action"
            elif id(parameter) in vision_parameters:
                family = "vision"
            else:
                family = "backbone"
            grouped[f"{family}_{'decay' if decay else 'no_decay'}"].append(parameter)

        learning_rates = {
            "backbone": lr * backbone_lr_multiplier,
            "action": lr,
            "vision": lr * backbone_lr_multiplier * vision_lr_multiplier,
        }
        parameter_groups = [
            {
                "params": grouped[name],
                "lr": learning_rates[name.split("_", 1)[0]],
                "weight_decay": weight_decay if not name.endswith("_no_decay") else 0.0,
                "name": name,
            }
            for name in (
                "backbone_decay",
                "action_decay",
                "vision_decay",
                "backbone_no_decay",
                "action_no_decay",
                "vision_no_decay",
            )
        ]
        expected = sum(parameter.requires_grad for parameter in self.model.parameters())
        actual = sum(len(group["params"]) for group in parameter_groups)
        if actual != expected:
            raise RuntimeError(
                f"G0.5 optimizer grouping lost parameters: grouped {actual}, expected {expected}."
            )
        return parameter_groups

    @classmethod
    def from_config(cls, model_config: Mapping[str, Any], checkpoint_dir: str | Path) -> G05NativeBackend:
        """Build the backend from the checkpoint's model configuration."""
        checkpoint_dir = Path(checkpoint_dir)
        processor_path = checkpoint_dir / "hf_processor"
        tokenizer_config = processor_path / "tokenizer_config.json"
        if not tokenizer_config.is_file():
            raise FileNotFoundError(f"G0.5 tokenizer config not found: {tokenizer_config}")

        tokenizer_metadata = json.loads(tokenizer_config.read_text())
        added = tokenizer_metadata.get("added_tokens_decoder") or {}
        base_vocab_size = max((int(token_id) for token_id in added), default=-1) + 1
        at_config = model_config["AT_CONFIG"]
        codebook_size = int(at_config["model_arch"]["codebook_size"])
        parts = at_config["parts_meta"]
        rule_patterns = tuple(at_config.get("rule_based_key_patterns") or ())
        rule_parts = [name for name in parts if any(pattern in name for pattern in rule_patterns)]
        neural_parts = [name for name in parts if name not in rule_parts]
        residuals = int(at_config["model_arch"]["n_codebooks"])
        marker_count = len(neural_parts) * residuals + len(rule_parts)
        # Action-code tokens, group markers, <EOV>, and the MLP <state> token.
        vocab_size = base_vocab_size + codebook_size + marker_count + 2
        return cls(
            model_config,
            vocab_size=vocab_size,
            processor_path=processor_path,
            action_tokenizer_path=checkpoint_dir / "action_tokenizer.safetensors",
        )

    @staticmethod
    def _patchify(images: Tensor, patch_size: int, temporal_patch_size: int, merge_size: int) -> Tensor:
        """Split images into vision patches."""
        batch_frames, channels, height, width = images.shape
        grid_h, grid_w = height // patch_size, width // patch_size
        temporal = images.unsqueeze(2).expand(-1, -1, temporal_patch_size, -1, -1)
        return (
            temporal.reshape(
                batch_frames,
                temporal_patch_size,
                channels,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshape(batch_frames * grid_h * grid_w, -1)
        )

    def _vision_temporal_block(
        self,
        block,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor],
        batch_size: int,
        num_frames: int,
        patches_per_frame: int,
        temporal_pe: Tensor,
        temporal_mask: Tensor,
    ) -> Tensor:
        """Run one vision block with attention across frames."""
        total, hidden_size = hidden_states.shape
        num_heads = block.attn.num_heads
        head_dim = block.attn.head_dim
        residual = hidden_states
        conditioned = (
            hidden_states.view(batch_size, num_frames, patches_per_frame, hidden_size)
            + temporal_pe[None, :, None, :]
        ).reshape(total, hidden_size)
        normed = block.norm1(conditioned)
        query, key, value = (
            block.attn.qkv(normed).reshape(total, 3, num_heads, head_dim).permute(1, 0, 2, 3).unbind(0)
        )

        def temporal_view(tensor: Tensor) -> Tensor:
            """Reshape a tensor to attend across frames."""
            return (
                tensor.view(batch_size, num_frames, patches_per_frame, num_heads, head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(batch_size * patches_per_frame, num_heads, num_frames, head_dim)
            )

        query_t, key_t, value_t = (temporal_view(tensor) for tensor in (query, key, value))
        weights = torch.matmul(query_t, key_t.transpose(-2, -1)) * block.attn.scaling
        weights = functional.softmax(weights + temporal_mask[None, None], dim=-1, dtype=torch.float32).to(
            query_t.dtype
        )
        mixed_value = torch.matmul(weights, value_t)
        mixed_value = (
            mixed_value.view(batch_size, patches_per_frame, num_heads, num_frames, head_dim)
            .permute(0, 3, 1, 2, 4)
            .reshape(total, num_heads, head_dim)
        )

        cosine, sine = position_embeddings
        query, key = apply_rotary_pos_emb_vision(query, key, cosine, sine)
        spatial_outputs = []
        for start in range(0, total, patches_per_frame):
            stop = start + patches_per_frame
            spatial_outputs.append(
                functional.scaled_dot_product_attention(
                    query[start:stop].transpose(0, 1).unsqueeze(0),
                    key[start:stop].transpose(0, 1).unsqueeze(0),
                    mixed_value[start:stop].transpose(0, 1).unsqueeze(0),
                    scale=block.attn.scaling,
                )
            )
        spatial = torch.cat(spatial_outputs, dim=2).squeeze(0).transpose(0, 1).reshape(total, hidden_size)
        hidden_states = residual + block.attn.proj(spatial)
        return hidden_states + block.mlp(block.norm2(hidden_states))

    def _encode_camera_transformers(self, frames: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        """Encode a single-frame camera through Transformers' native Qwen3.5 vision forward."""

        tower = self.model.vision_tower
        batch_size, num_frames, _, height, width = frames.shape
        if num_frames != 1:
            raise ValueError("the native Transformers vision path requires one frame per camera")
        patch_size = int(tower.config.patch_size)
        merge_size = int(tower.config.spatial_merge_size)
        temporal_patch_size = int(tower.config.temporal_patch_size)
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = self._patchify(
            frames.reshape(batch_size, *frames.shape[2:]),
            patch_size,
            temporal_patch_size,
            merge_size,
        )
        grid = torch.tensor((1, grid_h, grid_w), dtype=torch.long, device=frames.device).expand(
            batch_size, -1
        )
        encoded = tower(patches, grid).pooler_output
        tokens_per_frame = (grid_h // merge_size) * (grid_w // merge_size)
        return encoded.reshape(batch_size, tokens_per_frame, -1), (1, grid_h, grid_w)

    def _encode_camera_temporal(self, frames: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        """Encode one camera with G0.5's causal temporal-memory extension."""

        tower = self.model.vision_tower
        batch_size, num_frames, _, height, width = frames.shape
        patch_size = int(tower.config.patch_size)
        merge_size = int(tower.config.spatial_merge_size)
        temporal_patch_size = int(tower.config.temporal_patch_size)
        grid_h, grid_w = height // patch_size, width // patch_size
        patches_per_frame = grid_h * grid_w
        flattened = frames.reshape(batch_size * num_frames, *frames.shape[2:])
        patches = self._patchify(flattened, patch_size, temporal_patch_size, merge_size)
        grid = torch.tensor(
            [[1, grid_h, grid_w]] * (batch_size * num_frames),
            dtype=torch.long,
            device=frames.device,
        )

        with torch.autocast(frames.device.type, enabled=False):
            hidden_states = tower.patch_embed(patches)
            hidden_states = hidden_states + tower.fast_pos_embed_interpolate(grid)
        rotary = tower.rot_pos_emb(grid).reshape(hidden_states.shape[0], -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        cu_seqlens = torch.arange(
            0,
            (batch_size * num_frames + 1) * patches_per_frame,
            patches_per_frame,
            dtype=torch.int32,
            device=frames.device,
        )

        temporal_frequency = int(getattr(tower.config, "temporal_freq", 0))
        if num_frames > 1 and temporal_frequency > 0:
            timesteps = torch.arange(-(num_frames - 1), 1, device=frames.device)
            temporal_pe = _temporal_embedding(timesteps, hidden_states.shape[-1]).to(hidden_states.dtype)
            temporal_mask = torch.triu(
                torch.full(
                    (num_frames, num_frames),
                    float("-inf"),
                    device=frames.device,
                    dtype=hidden_states.dtype,
                ),
                diagonal=1,
            )
            drop_layer = int(getattr(tower.config, "token_drop_layer", None) or len(tower.blocks)) - 1
        else:
            temporal_pe = temporal_mask = None
            drop_layer = -1

        for layer_index, block in enumerate(tower.blocks):
            use_temporal = (
                temporal_pe is not None
                and layer_index <= drop_layer
                and (drop_layer - layer_index) % temporal_frequency == 0
            )
            if use_temporal:
                hidden_states = self._vision_temporal_block(
                    block,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    patches_per_frame=patches_per_frame,
                    temporal_pe=temporal_pe,
                    temporal_mask=temporal_mask,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )
            if layer_index == drop_layer and num_frames > 1:
                hidden_states = hidden_states.view(batch_size, num_frames, patches_per_frame, -1)[
                    :, -1
                ].reshape(batch_size * patches_per_frame, -1)
                num_frames = 1
                cu_seqlens = torch.arange(
                    0,
                    (batch_size + 1) * patches_per_frame,
                    patches_per_frame,
                    dtype=torch.int32,
                    device=frames.device,
                )
                cosine, sine = position_embeddings
                position_embeddings = (
                    cosine[: batch_size * patches_per_frame],
                    sine[: batch_size * patches_per_frame],
                )

        with torch.autocast(frames.device.type, enabled=False):
            merged = tower.merger(hidden_states)
        tokens_per_frame = (grid_h // merge_size) * (grid_w // merge_size)
        return merged.reshape(batch_size, tokens_per_frame, -1), (1, grid_h, grid_w)

    def _encode_camera(self, frames: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        """Use upstream Qwen vision for one frame and G0.5 temporal vision for history."""

        if frames.shape[1] == 1:
            return self._encode_camera_transformers(frames)
        temporal_frequency = int(getattr(self.model.vision_tower.config, "temporal_freq", 0))
        if temporal_frequency <= 0:
            raise ValueError(
                "multi-frame G0.5 vision requires a checkpoint with temporal_freq > 0; "
                "single-frame checkpoints use Transformers' native Qwen3.5 vision path"
            )
        return self._encode_camera_temporal(frames)

    def _encode_vision(self, pixel_values: Mapping[str, Tensor]) -> Tensor:
        """Encode the camera images to vision features."""
        features = []
        grids = []
        for frames in pixel_values.values():
            feature, grid = self._encode_camera(frames)
            features.append(feature)
            grids.append(grid)
        self._last_vision_grids = grids
        return torch.cat(features, dim=1)

    def _embed(
        self,
        sequence: G05SequenceBatch,
        pixel_values: Mapping[str, Tensor],
        proprio: Tensor,
    ) -> Tensor:
        """Combine the vision, text and proprioception embeddings."""
        image_features = self._encode_vision(pixel_values)
        text_features = self.model.vlm.embed(sequence.input_ids).to(image_features.dtype)
        embeddings = text_features.clone()

        image_mask = sequence.token_types == G05TokenType.IMAGE
        image_indices = (image_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        if image_mask.any() and int(image_indices[image_mask].max()) >= image_features.shape[1]:
            raise ValueError("G0.5 prompt image-token count does not match the native vision encoder output.")
        gathered_images = torch.gather(
            image_features,
            1,
            image_indices.unsqueeze(-1).expand(-1, -1, image_features.shape[-1]),
        )
        embeddings[image_mask] = gathered_images[image_mask]

        state_mask = sequence.token_types == G05TokenType.PROPRIO
        state_features = self.model.proprio_embedder(proprio).to(embeddings.dtype)
        state_indices = (state_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        gathered_state = torch.gather(
            state_features,
            1,
            state_indices.unsqueeze(-1).expand(-1, -1, state_features.shape[-1]),
        )
        embeddings[state_mask] = gathered_state[state_mask]
        return embeddings

    def _mrope_positions(self, token_types: Tensor) -> Tensor:
        """Build multimodal rotary position ids from the token types."""
        batch_size, sequence_length = token_types.shape
        positions = torch.zeros(
            3,
            batch_size,
            sequence_length,
            dtype=torch.long,
            device=token_types.device,
        )
        position_mode = str(self.model_config.get("position_ids_type", "pi0fast"))
        for batch_index in range(batch_size):
            cursor = 0
            grid_index = 0
            values = token_types[batch_index].detach().cpu().tolist()
            for token_type, entries in itertools.groupby(enumerate(values), key=lambda item: item[1]):
                entries = list(entries)
                start, stop = entries[0][0], entries[-1][0] + 1
                if int(token_type) == G05TokenType.PADDING:
                    continue
                length = stop - start
                if int(token_type) == G05TokenType.IMAGE:
                    if grid_index >= len(self._last_vision_grids):
                        raise ValueError("G0.5 MRoPE received more image segments than vision grids.")
                    _, raw_h, raw_w = self._last_vision_grids[grid_index]
                    grid_index += 1
                    merge = int(self.model_config["vision"]["spatial_merge_size"])
                    grid_h, grid_w = raw_h // merge, raw_w // merge
                    height = (
                        torch.arange(grid_h, device=token_types.device).repeat_interleave(grid_w)[:length]
                        + cursor
                    )
                    width = torch.arange(grid_w, device=token_types.device).repeat(grid_h)[:length] + cursor
                    positions[0, batch_index, start:stop] = cursor
                    positions[1, batch_index, start:stop] = height
                    positions[2, batch_index, start:stop] = width
                    cursor += max(grid_h, grid_w)
                    continue

                if position_mode == "gaussian":
                    if self.training:
                        steps = (
                            torch.normal(
                                mean=2.0,
                                std=0.5,
                                size=(length,),
                                device=token_types.device,
                            )
                            .round()
                            .clamp(1, 3)
                            .long()
                        )
                    else:
                        steps = torch.full((length,), 2, dtype=torch.long, device=token_types.device)
                else:
                    steps = torch.ones(length, dtype=torch.long, device=token_types.device)
                text_positions = cursor + steps.cumsum(0) - steps[0]
                positions[:, batch_index, start:stop] = text_positions
                cursor = int(text_positions[-1]) + int(steps[-1])
        return positions

    @staticmethod
    def _causal_mask(token_types: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Build the causal attention mask from the token types."""
        valid = token_types != G05TokenType.PADDING
        sequence_length = token_types.shape[1]
        causal = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=token_types.device,
        ).tril()
        allowed = causal[None] & valid[:, None, :] & valid[:, :, None]
        full = torch.zeros(
            token_types.shape[0],
            1,
            sequence_length,
            sequence_length,
            dtype=dtype,
            device=token_types.device,
        )
        full.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
        return full, valid.to(dtype)

    @staticmethod
    def _proprio(samples: list[dict[str, Any]], device: torch.device) -> Tensor:
        """Stack the samples' proprioception rows."""
        rows = []
        for sample in samples:
            value = sample["proprio"]
            value = value["value"] if isinstance(value, Mapping) else value
            value = torch.as_tensor(value, dtype=torch.float32, device=device)
            rows.append(value.unsqueeze(0) if value.ndim == 1 else value)
        return torch.stack(rows)

    def _prefill(
        self,
        sequence: G05SequenceBatch,
        pixel_values: Mapping[str, Tensor],
        proprio: Tensor,
    ) -> tuple[Tensor, Any, Tensor]:
        """Run the prefix through the backbone and fill the cache."""
        embeddings = self._embed(sequence, pixel_values, proprio)
        positions = self._mrope_positions(sequence.token_types)
        full_mask, linear_mask = self._causal_mask(sequence.token_types, embeddings.dtype)
        hidden_states, cache = self.model.vlm(
            embeddings,
            full_attention_mask=full_mask,
            linear_attention_mask=linear_mask,
            position_ids=positions,
        )
        return hidden_states, cache, positions

    def _decode_token(
        self,
        token_ids: Tensor,
        *,
        token_types: Tensor,
        positions: Tensor,
        cache,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode one generated token."""
        embeddings = self.model.vlm.embed(token_ids[:, None])
        batch_size = token_ids.shape[0]
        if active_mask is None:
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=token_ids.device)
        elif active_mask.shape != (batch_size,):
            raise ValueError("active_mask must have shape [batch]")
        next_positions = positions.amax(dim=-1, keepdim=True) + 1
        prefix_length = token_types.shape[1]
        prefix_mask = (token_types == G05TokenType.PADDING).to(embeddings.dtype)
        full_mask = torch.zeros(
            batch_size,
            1,
            1,
            prefix_length + 1,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        full_mask[..., :prefix_length].masked_fill_(
            prefix_mask[:, None, None].bool(), torch.finfo(embeddings.dtype).min
        )
        frozen_linear_states = []
        inactive = ~active_mask
        if inactive.any():
            for layer in cache.layers:
                if hasattr(layer, "keys") or not hasattr(layer, "conv_states"):
                    continue
                frozen_linear_states.append(
                    (
                        layer,
                        layer.conv_states[inactive].clone(),
                        layer.recurrent_states[inactive].clone(),
                    )
                )
        hidden_states, cache = self.model.vlm(
            embeddings,
            full_attention_mask=full_mask,
            linear_attention_mask=active_mask[:, None].to(embeddings.dtype),
            position_ids=next_positions,
            cache=cache,
        )
        for layer, conv_states, recurrent_states in frozen_linear_states:
            layer.conv_states[inactive] = conv_states
            layer.recurrent_states[inactive] = recurrent_states
        next_types = torch.full(
            (batch_size, 1),
            float(G05TokenType.PRED_TEXT),
            dtype=token_types.dtype,
            device=token_types.device,
        )
        next_types.masked_fill_(inactive[:, None], float(G05TokenType.PADDING))
        return (
            hidden_states[:, -1],
            torch.cat((token_types, next_types), dim=1),
            torch.cat((positions, next_positions), dim=-1),
        )

    def _sample_next_token(
        self,
        logits: Tensor,
        history: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply the checkpoint's autoregressive sampling contract."""

        ar_config = self.model_config.get("ar") or {}
        do_sample = bool(ar_config.get("do_sample", False))
        temperature = float(ar_config.get("temperature", 0.7))
        if not do_sample or temperature == 0:
            return logits.float().argmax(dim=-1)

        scores = logits.float().clone()
        if history is not None and history.numel():
            if history_mask is not None and history_mask.shape != history.shape:
                raise ValueError("history_mask must have the same shape as history")
            repetition_penalty = float(ar_config.get("repetition_penalty", 1.0))
            if repetition_penalty != 1:
                if history_mask is None:
                    previous_scores = scores.gather(1, history)
                    previous_scores = torch.where(
                        previous_scores < 0,
                        previous_scores * repetition_penalty,
                        previous_scores / repetition_penalty,
                    )
                    scores.scatter_(1, history, previous_scores)
                else:
                    for batch_index in range(history.shape[0]):
                        row = history[batch_index, history_mask[batch_index]]
                        if not row.numel():
                            continue
                        previous_scores = scores[batch_index, row]
                        previous_scores = torch.where(
                            previous_scores < 0,
                            previous_scores * repetition_penalty,
                            previous_scores / repetition_penalty,
                        )
                        scores[batch_index].scatter_(0, row, previous_scores)

            ngram_size = int(ar_config.get("no_repeat_ngram_size", 0))
            if ngram_size > 0:
                histories = (
                    history.tolist()
                    if history_mask is None
                    else [history[index, history_mask[index]].tolist() for index in range(history.shape[0])]
                )
                for batch_index, row in enumerate(histories):
                    if len(row) < ngram_size - 1:
                        continue
                    prefix = tuple(row[-(ngram_size - 1) :]) if ngram_size > 1 else ()
                    banned = {
                        row[index + ngram_size - 1]
                        for index in range(len(row) - ngram_size + 1)
                        if tuple(row[index : index + ngram_size - 1]) == prefix
                    }
                    if banned:
                        scores[batch_index, list(banned)] = -torch.inf

        scores /= temperature
        top_k = min(int(ar_config.get("top_k", 128)), scores.shape[-1])
        if top_k > 0:
            threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
            scores.masked_fill_(scores < threshold, -torch.inf)
        top_p = float(ar_config.get("top_p", 0.95))
        if top_p < 1:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            scores.scatter_(1, sorted_indices, sorted_scores.masked_fill(remove, -torch.inf))
        return torch.multinomial(scores.softmax(dim=-1), num_samples=1).squeeze(1)

    def _generate_text(
        self,
        last_hidden: Tensor,
        *,
        token_types: Tensor,
        positions: Tensor,
        cache,
        max_new_tokens: int,
        stop_token_ids: int | tuple[int, ...],
        initial_history: Tensor | None = None,
        initial_history_mask: Tensor | None = None,
        forced_first_tokens: Tensor | None = None,
    ) -> tuple[G05TextGeneration, Any, Tensor, Tensor, Tensor]:
        """Generate the chain-of-thought text tokens."""
        generated = []
        batch_size = last_hidden.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=last_hidden.device)
        stop_tokens = torch.full((batch_size,), -1, dtype=torch.long, device=last_hidden.device)
        stop_ids = (stop_token_ids,) if isinstance(stop_token_ids, int) else stop_token_ids
        history = initial_history
        history_mask = initial_history_mask
        track_history_mask = batch_size > 1 and self.model_config.get("embodiment") not in {
            "so100",
            "so101",
        }
        if track_history_mask and history is not None and history_mask is None:
            history_mask = torch.ones_like(history, dtype=torch.bool)

        for step in range(max_new_tokens):
            logits = self.model.vlm.logits(last_hidden)
            next_token = self._sample_next_token(logits, history, history_mask)
            if step == 0 and forced_first_tokens is not None:
                next_token = torch.where(forced_first_tokens.ge(0), forced_first_tokens, next_token)
            next_token = next_token.masked_fill(finished, self.processor.pad_token_id)
            generated.append(next_token)

            is_stop = torch.zeros_like(finished)
            for token_id in stop_ids:
                is_stop |= next_token.eq(token_id)
            stop_tokens = torch.where(is_stop & (~finished), next_token, stop_tokens)
            finished |= is_stop
            if bool(finished.all()):
                break

            active = ~finished
            history_token = next_token
            if track_history_mask:
                history_token = history_token.masked_fill(~active, self.processor.pad_token_id)
                history_mask = (
                    active.unsqueeze(1)
                    if history_mask is None
                    else torch.cat((history_mask, active.unsqueeze(1)), dim=1)
                )
            history = (
                history_token.unsqueeze(1)
                if history is None
                else torch.cat((history, history_token.unsqueeze(1)), dim=1)
            )

            decoded_hidden, token_types, positions = self._decode_token(
                next_token.masked_fill(~active, self.processor.pad_token_id),
                token_types=token_types,
                positions=positions,
                cache=cache,
                active_mask=active,
            )
            last_hidden = torch.where(active[:, None], decoded_hidden, last_hidden)
        generated_ids = (
            torch.stack(generated, dim=1)
            if generated
            else torch.empty(last_hidden.shape[0], 0, dtype=torch.long, device=last_hidden.device)
        )
        return (
            G05TextGeneration(generated_ids, stop_tokens, history, history_mask),
            cache,
            last_hidden,
            token_types,
            positions,
        )

    def _action_cache(self, vlm_cache, prefix_length: int, *, repeats: int = 1):
        """Build the action expert's attention cache."""
        cache = DynamicCache(config=self.model.action_expert.config)
        layer_types = self.model.vlm.config.layer_types
        for layer_index, layer_type in enumerate(layer_types):
            if layer_type != "full_attention":
                continue
            source = vlm_cache.layers[layer_index]
            if not source.is_initialized:
                continue
            key = source.keys[..., :prefix_length, :].detach()
            value = source.values[..., :prefix_length, :].detach()
            if repeats > 1:
                key = key.repeat_interleave(repeats, dim=0)
                value = value.repeat_interleave(repeats, dim=0)
            cache.layers[layer_index].update(key, value)
        return cache

    def _action_mask_and_positions(
        self,
        token_types: Tensor,
        positions: Tensor,
        horizon: int,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Build the action expert's mask and position ids."""
        batch_size, prefix_length = token_types.shape
        prefix_mask = (token_types == G05TokenType.PADDING).to(dtype) * torch.finfo(dtype).min
        action_mask = torch.zeros(
            batch_size,
            horizon,
            horizon,
            dtype=dtype,
            device=token_types.device,
        )
        if bool(self.model_config["fm"].get("action_causal", False)):
            action_mask = torch.triu(torch.full_like(action_mask, torch.finfo(dtype).min), diagonal=1)
        mask = torch.cat((prefix_mask[:, None].expand(-1, horizon, -1), action_mask), dim=-1).unsqueeze(1)
        offset = positions.amax(dim=-1, keepdim=True)
        action_positions = torch.arange(1, horizon + 1, device=token_types.device)[None, None] + offset
        return mask, action_positions

    def _velocity(
        self,
        actions: Tensor,
        timesteps: Tensor,
        *,
        vlm_cache,
        token_types: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Evaluate the flow velocity field at one timestep."""
        action_embeddings = self.model.action_expert.embed(actions)
        time_cond = self.model.action_expert.encode_time(timesteps)
        mask, action_positions = self._action_mask_and_positions(
            token_types, positions, actions.shape[1], action_embeddings.dtype
        )
        cache = self._action_cache(vlm_cache, token_types.shape[1])
        hidden_states = self.model.action_expert(
            action_embeddings,
            attention_mask=mask,
            position_ids=action_positions,
            cache=cache,
            time_cond=time_cond,
        )
        return self.model.action_expert.decode(hidden_states)

    def _infer_flow(
        self,
        *,
        vlm_cache,
        token_types: Tensor,
        positions: Tensor,
        action_dim_is_pad: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        """Integrate the flow to sample an action chunk."""
        fm = self.model_config["fm"]
        batch_size = token_types.shape[0]
        horizon = int(fm["horizon_steps"])
        action_dim = int(fm["action_dim"])
        action = torch.randn(
            batch_size,
            horizon,
            action_dim,
            device=token_types.device,
            dtype=dtype,
        )
        dim_mask = (
            action_dim_is_pad.bool().unsqueeze(1)
            if action_dim_is_pad is not None and not bool(fm["zero_pad_action_target"])
            else None
        )
        if dim_mask is not None:
            action.masked_fill_(dim_mask, 0)
        steps = int(fm["num_inference_steps"])
        delta = 1.0 / steps
        pi_convention = fm["time_convention"] == "pi_convention"
        time_value = 1.0 if pi_convention else 0.0
        timesteps = torch.full((batch_size,), time_value, dtype=dtype, device=token_types.device)
        for _ in range(steps):
            velocity = self._velocity(
                action,
                timesteps,
                vlm_cache=vlm_cache,
                token_types=token_types,
                positions=positions,
            )
            action = action - delta * velocity if pi_convention else action + delta * velocity
            timesteps = timesteps - delta if pi_convention else timesteps + delta
            if dim_mask is not None:
                action.masked_fill_(dim_mask, 0)
        clip = fm.get("final_action_clip_value")
        return action.clamp(-float(clip), float(clip)) if clip is not None else action

    def _flow_loss(
        self,
        actions: Tensor,
        *,
        action_is_pad: Tensor,
        action_dim_is_pad: Tensor | None,
        vlm_cache,
        token_types: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Flow-matching training loss."""
        fm = self.model_config["fm"]
        samples = int(fm.get("num_flow_samples", 1))
        batch_size = actions.shape[0]
        beta = torch.distributions.Beta(1.5, 1.0)
        z = beta.sample((samples, batch_size)).to(actions.device, actions.dtype)
        if fm["time_convention"] == "pi_convention":
            timesteps = 1 - (1 - float(fm["flow_sig_min"])) * (1 - z)
        else:
            timesteps = (1 - float(fm["flow_sig_min"])) * (1 - z)
        timesteps = timesteps.reshape(-1)
        noise = torch.randn(
            samples,
            *actions.shape,
            device=actions.device,
            dtype=actions.dtype,
        ).flatten(0, 1)
        target_actions = actions.repeat(samples, 1, 1)
        t = timesteps[:, None, None]
        if fm["time_convention"] == "pi_convention":
            interpolated = (1 - t) * target_actions + t * noise
            target_velocity = noise - target_actions
        else:
            interpolated = t * target_actions + (1 - t) * noise
            target_velocity = target_actions - noise

        repeated_dim_mask = None
        if action_dim_is_pad is not None:
            repeated_dim_mask = action_dim_is_pad.repeat(samples, 1)
            if not bool(fm["zero_pad_action_target"]):
                interpolated = interpolated.masked_fill(repeated_dim_mask[:, None], 0)

        repeated_types = token_types.repeat(samples, 1)
        repeated_positions = positions.repeat(1, samples, 1)
        action_embeddings = self.model.action_expert.embed(interpolated)
        time_cond = self.model.action_expert.encode_time(timesteps)
        mask, action_positions = self._action_mask_and_positions(
            repeated_types, repeated_positions, actions.shape[1], action_embeddings.dtype
        )
        cache = self._action_cache(vlm_cache, token_types.shape[1], repeats=samples)
        predicted = self.model.action_expert.decode(
            self.model.action_expert(
                action_embeddings,
                attention_mask=mask,
                position_ids=action_positions,
                cache=cache,
                time_cond=time_cond,
            )
        )
        weights = torch.ones_like(predicted)
        weights[action_is_pad.repeat(samples, 1)] = float(fm["padding_action_weight"])
        if repeated_dim_mask is not None and not bool(fm["zero_pad_action_target"]):
            weights.masked_fill_(
                repeated_dim_mask[:, None],
                float(fm["padding_action_weight"]),
            )
        loss = (weights * (predicted - target_velocity).square()).sum() / weights.sum().clamp_min(1)
        return loss * float(fm["fm_weight"])

    def predict_action(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Predict one action chunk, optionally with chain of thought."""
        start = time.monotonic()
        samples = list(batch["samples"])
        pixel_values = batch["pixel_values"]
        first_image = next(iter(pixel_values.values()))
        sequence = G05SequenceBatch(
            input_ids=batch[G05_INPUT_IDS],
            labels=batch[G05_LABELS],
            token_types=batch[G05_TOKEN_TYPES],
            split_index=batch.get(G05_SPLIT_INDEX),
        )
        proprio = self._proprio(samples, first_image.device)
        hidden_states, cache, positions = self._prefill(sequence, pixel_values, proprio)
        result: dict[str, Any] = {}
        last_hidden = hidden_states[:, -1]
        token_types = sequence.token_types
        cot_generation: G05TextGeneration | None = None
        predict_cot = bool(batch.get(G05_RUNTIME_PREDICT_COT, self.model_config.get("predict_cot", False)))
        if predict_cot:
            cot_generation, cache, last_hidden, token_types, positions = self._generate_text(
                last_hidden,
                token_types=sequence.token_types,
                positions=positions,
                cache=cache,
                max_new_tokens=int(self.model_config["ar"].get("max_new_tokens", 300)),
                stop_token_ids=(self.processor.eov_token_id, self.processor.eos_token_id),
            )
            sequence.token_types = token_types
            result["generated_ids"] = cot_generation.token_ids
            result["cot_text"] = [
                self.processor.decode(
                    ids[
                        : next(
                            (
                                index
                                for index, token_id in enumerate(ids.tolist())
                                if token_id in {self.processor.eov_token_id, self.processor.eos_token_id}
                            ),
                            len(ids),
                        )
                    ]
                )
                for ids in cot_generation.token_ids
            ]
        sequence.token_types = token_types
        if bool(self.model_config.get("continuous_action", False)):
            dim_mask = batch.get("action_dim_is_pad")
            if not isinstance(dim_mask, Tensor):
                dim_mask = None
            result[ACTION] = self._infer_flow(
                vlm_cache=cache,
                token_types=sequence.token_types,
                positions=positions,
                action_dim_is_pad=dim_mask,
                dtype=first_image.dtype,
            )
        if bool(self.model_config.get("discrete_action", False)):
            if self.action_tokenizer is None:
                if ACTION not in result:
                    raise RuntimeError(
                        "The native G0.5 ActionCodec checkpoint has not been loaded; "
                        "select the continuous flow head for this checkpoint."
                    )
            else:
                action_generation, _, _, _, _ = self._generate_text(
                    last_hidden,
                    token_types=sequence.token_types,
                    positions=positions,
                    cache=cache,
                    max_new_tokens=self.action_tokenizer.action_token_length + 32,
                    stop_token_ids=self.processor.eos_token_id,
                    initial_history=cot_generation.history if cot_generation is not None else None,
                    initial_history_mask=(
                        cot_generation.history_mask if cot_generation is not None else None
                    ),
                    forced_first_tokens=(cot_generation.stop_tokens if cot_generation is not None else None),
                )
                decoded_actions = []
                decoded_tokens = []
                absent_keys = []
                for token_row in action_generation.token_ids:
                    is_action = (token_row >= self.processor.action_token_begin) & (
                        token_row < self.processor.action_token_end_with_markers
                    )
                    action_tokens = token_row[is_action]
                    decoded, absent = self.action_tokenizer.decode_language_tokens(
                        action_tokens,
                        horizon=int(self.model_config["fm"]["horizon_steps"]),
                        action_dim=int(self.model_config["fm"]["action_dim"]),
                    )
                    decoded_actions.append(decoded)
                    decoded_tokens.append(action_tokens)
                    absent_keys.append(absent)
                result["ar_action"] = torch.stack(decoded_actions)
                result["decoded_action_tokens"] = decoded_tokens
                result["ar_absent_keys"] = absent_keys
                if ACTION not in result:
                    result[ACTION] = result["ar_action"]
        result["_timing"] = {"forward_inference_total_ms": (time.monotonic() - start) * 1000}
        return result

    def forward(self, batch: Mapping[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the training forward pass."""
        samples = list(batch["samples"])
        pixel_values = batch["pixel_values"]
        first_image = next(iter(pixel_values.values()))
        sequence = G05SequenceBatch(
            input_ids=batch[G05_INPUT_IDS],
            labels=batch[G05_LABELS],
            token_types=batch[G05_TOKEN_TYPES],
            split_index=batch.get(G05_SPLIT_INDEX),
        )
        proprio = self._proprio(samples, first_image.device)
        hidden_states, cache, positions = self._prefill(sequence, pixel_values, proprio)
        loss_dict: dict[str, Tensor] = {}

        ar_config = self.model_config.get("ar") or {}
        ce_weight = float(ar_config.get("ce_weight", 1.0))
        z_loss_scale = float(ar_config.get("ce_z_loss_scale", 0.0))
        if ce_weight < 0:
            raise ValueError("G0.5 ar.ce_weight must be non-negative.")
        if z_loss_scale < 0:
            raise ValueError("G0.5 ar.ce_z_loss_scale must be non-negative.")
        skip_ce = (
            bool(self.model_config.get("continuous_action", False))
            and not bool(self.model_config.get("discrete_action", False))
            and not bool(self.model_config.get("predict_cot", False))
        )
        if not skip_ce:
            shift_labels = sequence.labels[:, 1:]
            valid = shift_labels != IGNORE_INDEX
            if valid.any() and ce_weight:
                shift_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
                valid_hidden = shift_hidden[valid.reshape(-1)]
                valid_labels = shift_labels.reshape(-1)[valid.reshape(-1)]
                loss_dict["ce_loss"] = _autoregressive_ce_loss(
                    self.model.vlm.logits(valid_hidden),
                    valid_labels,
                    ce_weight=ce_weight,
                    z_loss_scale=z_loss_scale,
                )
            else:
                loss_dict["ce_loss"] = hidden_states.sum() * 0

        if bool(self.model_config.get("continuous_action", False)):
            actions = batch.get(ACTION)
            if not isinstance(actions, Tensor):
                raise ValueError("G0.5 flow training requires an action tensor.")
            action_is_pad = batch.get("action_is_pad")
            if not isinstance(action_is_pad, Tensor):
                action_is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
            action_dim_is_pad = batch.get("action_dim_is_pad")
            if not isinstance(action_dim_is_pad, Tensor):
                action_dim_is_pad = None
            prefix = int(sequence.split_index)
            loss_dict["fm_loss"] = self._flow_loss(
                actions,
                action_is_pad=action_is_pad,
                action_dim_is_pad=action_dim_is_pad,
                vlm_cache=cache,
                token_types=sequence.token_types[:, :prefix],
                positions=positions[..., :prefix],
            )
        loss = sum(loss_dict.values())
        return loss, loss_dict


def _native_backend(config: G05Config, checkpoint_dir: str | Path | None) -> nn.Module:
    """Build the native backend for a policy config."""
    if not config.author_model_config:
        raise ValueError(
            "G0.5 author_model_config is empty. Load a packaged checkpoint, or "
            "inject a backend explicitly for testing."
        )
    if checkpoint_dir is None:
        raise ValueError(
            "G0.5 needs the checkpoint directory holding hf_processor/ and "
            "action_tokenizer.safetensors. Load a packaged checkpoint with "
            "from_pretrained(), or inject a backend explicitly for testing."
        )
    model_config = dict(config.author_model_config)
    model_config.update(
        {
            "embodiment": config.embodiment,
            "predict_cot": config.predict_cot,
            "discrete_action": config.discrete_action,
            "continuous_action": config.continuous_action,
            "return_continuous_action": config.return_continuous_action,
        }
    )
    return G05NativeBackend.from_config(model_config, checkpoint_dir)


def _first_cot_text(metadata: Mapping[str, Any]) -> str | None:
    """First non-empty chain-of-thought string in a batched inference result."""
    texts = metadata.get("cot_text")
    if isinstance(texts, str):
        return texts.strip() or None
    if isinstance(texts, list | tuple):
        for text in texts:
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


class G05Policy(PreTrainedPolicy):
    """LeRobot policy surface for G0.5's unified CoT and action stream."""

    config_class = G05Config
    name = "g05"

    def __init__(
        self,
        config: G05Config,
        backend: nn.Module | None = None,
        *,
        checkpoint_dir: str | Path | None = None,
        **kwargs,
    ):
        """Build the policy and its native backend."""
        super().__init__(config)
        config.validate_features()
        self.backend = backend if backend is not None else _native_backend(config, checkpoint_dir)
        if not isinstance(self.backend, nn.Module):
            raise TypeError(f"G0.5 backend must be an nn.Module, got {type(self.backend)}.")
        self._action_queue: deque[Tensor] = deque()

    def supports_text_generation(self) -> bool:
        """G0.5 can generate text."""
        return True

    @torch.no_grad()
    def generate_text(self, batch: dict[str, Tensor]) -> str:
        """Generate G0.5's native System-2 text from model-ready observations."""
        if not self.config.predict_cot:
            raise ValueError("G0.5 text generation requires a checkpoint with predict_cot=True.")
        _, metadata = self._run_inference(batch, system_mode="system2")
        text = _first_cot_text(metadata)
        if text is None:
            raise ValueError("G0.5 text generation returned no text.")
        return text

    @classmethod
    def _load_as_safetensor(
        cls,
        model: G05Policy,
        model_file: str,
        map_location: str,
        strict: bool,
    ) -> G05Policy:
        """Load the weights from a safetensors file."""
        device = resolve_safetensors_device(map_location)
        state_dict = load_file(model_file, device=device, backend="pread")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
        materialize = getattr(model.backend, "materialize_runtime_buffers", None)
        if callable(materialize):
            materialize(device)
        remaining_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        if remaining_meta:
            raise RuntimeError(f"G0.5 checkpoint did not materialize model parameters: {remaining_meta}")
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Error(s) loading G0.5 safetensors: missing={missing_keys}, unexpected={unexpected_keys}"
            )
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: G05Config | None = None,
        **kwargs,
    ) -> G05Policy:
        """Load a policy, downloading the checkpoint when needed."""
        resolved_path = Path(pretrained_name_or_path)
        if not resolved_path.is_dir():
            # The backend reads hf_processor/tokenizer_config.json to size the vocabulary
            # before any weight is loaded, so both sidecars must be on disk up front.
            resolved_path = Path(
                snapshot_download(
                    repo_id=str(pretrained_name_or_path),
                    token=kwargs.get("token"),
                    cache_dir=kwargs.get("cache_dir"),
                    local_files_only=kwargs.get("local_files_only", False),
                    revision=kwargs.get("revision"),
                )
            )
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                resolved_path,
                token=kwargs.get("token"),
                cache_dir=kwargs.get("cache_dir"),
                local_files_only=kwargs.get("local_files_only", False),
                revision=kwargs.get("revision"),
            )
        if not isinstance(config, G05Config):
            raise TypeError(f"Expected a G05Config, got {type(config).__name__}.")
        with torch.device("meta"):
            policy = super().from_pretrained(
                resolved_path,
                config=config,
                checkpoint_dir=resolved_path,
                **kwargs,
            )
        if isinstance(policy.backend, G05NativeBackend) and policy.backend.action_tokenizer is None:
            action_config = policy.backend.model_config.get("AT_CONFIG")
            if isinstance(action_config, Mapping) and policy.backend.action_tokenizer_path.is_file():
                # The backend is built on meta, where the codec cannot load, so bind it here.
                policy.backend.action_tokenizer = G05NativeActionCodec.load(
                    action_config,
                    action_token_begin=policy.backend.processor.action_token_begin,
                    ckpt_path=policy.backend.action_tokenizer_path,
                ).to(next(policy.backend.parameters()).device)
        return policy

    def reset(self) -> None:
        """Clear the queued actions and any backend state."""
        self._action_queue.clear()
        reset = getattr(self.backend, "reset", None)
        if callable(reset):
            reset()

    def _apply_author_inference_precision(self) -> None:
        """Match the released serving path's BF16 weights with declared FP32 islands."""

        self.backend.to(dtype=torch.bfloat16)
        apply_fp32_params = getattr(self.backend, "apply_fp32_params", None)
        if callable(apply_fp32_params):
            apply_fp32_params()
        if self.config.predict_cot:
            # The author Qwen3.5 final norm is an FP32 island and its fused CE
            # kernel disables autocast, so the tied output projection must be
            # FP32 as well. Otherwise CoT training reaches FLCE with FP32 hidden
            # states and a BF16 weight and fails before computing text loss.
            model = getattr(self.backend, "model", None)
            vlm = getattr(model, "vlm", None)
            output_proj = getattr(vlm, "output_proj", None)
            weight = getattr(output_proj, "weight", None)
            if isinstance(weight, nn.Parameter):
                weight.data = weight.data.float()

    def to(self, *args, **kwargs) -> G05Policy:
        """Apply the released inference precision and move the ActionCodec sidecar."""

        result = super().to(*args, **kwargs)
        explicit_dtype = "dtype" in kwargs or any(isinstance(arg, torch.dtype | Tensor) for arg in args)
        if (
            self.config.model_weights_to_bf16
            and not explicit_dtype
            and next(self.backend.parameters()).device.type == "cuda"
        ):
            self._apply_author_inference_precision()
        action_tokenizer = getattr(self.backend, "action_tokenizer", None)
        move_tokenizer = getattr(action_tokenizer, "to", None)
        if callable(move_tokenizer):
            device = next(self.backend.parameters()).device
            move_tokenizer(device)
        return result

    def get_optim_params(self) -> OptimizerParams:
        """Return the optimizer parameter groups."""
        get_param_groups = getattr(self.backend, "get_optim_param_groups", None)
        if callable(get_param_groups):
            return get_param_groups(
                lr=self.config.optimizer_lr,
                weight_decay=self.config.optimizer_weight_decay,
                apply_decay_on_norm_and_bias=self.config.optimizer_apply_decay_on_norm_and_bias,
                backbone_lr_multiplier=self.config.optimizer_backbone_lr_multiplier,
                vision_lr_multiplier=self.config.optimizer_vision_lr_multiplier,
            )
        get_params = getattr(self.backend, "get_optim_params", None)
        if callable(get_params):
            params = get_params()
            return [params] if isinstance(params, dict) and "params" in params else params
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @staticmethod
    def _task_values(batch: Mapping[str, Any], task: str | None, batch_size: int) -> list[str]:
        """Broadcast the task string across the batch."""
        if task is not None:
            return [task] * batch_size
        value = batch.get("task")
        if isinstance(value, str):
            return [value] * batch_size
        if isinstance(value, list | tuple) and len(value) == batch_size:
            return [str(item) for item in value]
        raise ValueError(
            "G0.5 requires the already-selected LeRobot task string; no task augmentation "
            "or model-local sampling is performed."
        )

    @staticmethod
    def _batch_item(value: Any, index: int, batch_size: int) -> Any:
        """Take one sample's value out of a batched field."""
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            return value[index]
        if isinstance(value, list | tuple) and len(value) == batch_size:
            return value[index]
        return value

    def _recipe_cot_targets(
        self,
        batch: Mapping[str, Any],
        index: int,
        batch_size: int,
    ) -> tuple[str | None, str | None]:
        """Read the selected recipe's supervised Subtask/BBox messages."""

        messages = batch.get(MESSAGES_RENDERED)
        target_indices = batch.get("target_message_indices")
        if messages is None or target_indices is None:
            return None, None

        sample_messages = messages
        if (
            isinstance(messages, list | tuple)
            and len(messages) == batch_size
            and (not messages or isinstance(messages[0], list | tuple))
        ):
            sample_messages = messages[index]
        sample_target_indices = target_indices
        has_batched_target_indices = (isinstance(target_indices, Tensor) and target_indices.ndim > 1) or (
            isinstance(target_indices, list | tuple)
            and len(target_indices) == batch_size
            and (not target_indices or isinstance(target_indices[0], list | tuple | Tensor))
        )
        if has_batched_target_indices:
            sample_target_indices = target_indices[index]
        if isinstance(sample_messages, Mapping):
            sample_messages = [sample_messages]
        if isinstance(sample_target_indices, Tensor):
            sample_target_indices = sample_target_indices.detach().cpu().tolist()
        if not isinstance(sample_messages, list | tuple) or not isinstance(
            sample_target_indices, list | tuple
        ):
            return None, None

        subtask: str | None = None
        bbox_json: str | None = None
        for target_index in sample_target_indices:
            message = sample_messages[int(target_index)]
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str):
                continue
            if content.startswith("Subtask:"):
                value = content.removeprefix("Subtask:").strip()
                if value:
                    subtask = value
            elif content.startswith("BBoxJSON:"):
                value = content.removeprefix("BBoxJSON:").strip()
                if value:
                    bbox_json = value
        return subtask, bbox_json

    @staticmethod
    def _format_bbox_target(bbox_json: str | None, image_size: tuple[int, int]) -> str | None:
        """Convert LeRobot grounded-VQA JSON into G0.5's location-token format."""

        if not bbox_json:
            return None
        try:
            payload = json.loads(bbox_json)
            if isinstance(payload, str):
                payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        if isinstance(payload.get("answer"), Mapping):
            payload = payload["answer"]

        height, width = image_size
        boxes: list[tuple[str, list[float]]] = []
        detections = payload.get("detections")
        if isinstance(detections, list):
            for detection in detections:
                if not isinstance(detection, Mapping) or detection.get("bbox_format", "xyxy") != "xyxy":
                    continue
                coords = detection.get("bbox")
                if not isinstance(coords, list | tuple) or len(coords) != 4:
                    continue
                label = str(detection.get("label") or "object")
                boxes.append((label, [float(value) for value in coords]))
        else:
            for label, coords in payload.items():
                if isinstance(coords, list | tuple) and len(coords) == 4:
                    boxes.append((str(label), [float(value) for value in coords]))
        if not boxes:
            return None

        def normalize(coords: list[float]) -> list[float]:
            """Scale box coordinates to the unit interval."""
            if max(abs(value) for value in coords) <= 1.0:
                return coords
            x1, y1, x2, y2 = coords
            return [x1 / width, y1 / height, x2 / width, y2 / height]

        def location_token(value: float) -> str:
            """Render a coordinate as a location token."""
            location = max(0, min(1023, round(value * 1024)))
            return f"<loc{location:04d}>"

        formatted = []
        for label, raw_coords in boxes:
            x1, y1, x2, y2 = normalize(raw_coords)
            locations = "".join(location_token(value) for value in (y1, x1, y2, x2))
            formatted.append(f"{label} {locations}")
        return "BBox: " + "; ".join(formatted)

    def _apply_recipe_cot(
        self,
        sample: dict[str, Any],
        batch: Mapping[str, Any],
        index: int,
        batch_size: int,
    ) -> bool:
        """Populate one author sample from recipe-rendered CoT targets."""

        subtask, bbox_json = self._recipe_cot_targets(batch, index, batch_size)
        image_size = batch.get("g05_bbox_image_size")
        if (
            isinstance(image_size, list | tuple)
            and len(image_size) == batch_size
            and image_size
            and isinstance(image_size[0], list | tuple | Tensor)
        ):
            image_size = image_size[index]
        if isinstance(image_size, Tensor):
            image_size = image_size.detach().cpu().tolist()
        if not isinstance(image_size, list | tuple) or len(image_size) != 2:
            camera = self.config.cot_bbox_camera or self.config.camera_order[0]
            image_size = self.config.camera_sizes[camera]
        bbox = self._format_bbox_target(bbox_json, (int(image_size[0]), int(image_size[1])))

        fields = tuple(field for field, value in (("bbox", bbox), ("subtask", subtask)) if value)
        if not fields:
            return False
        flow_only = "<action_action" not in self.config.prompt_template
        sample["template"] = make_g05_cot_prompt_template(
            self.config.num_prompt_images,
            fields=fields,
            flow_only=flow_only,
        )
        if bbox is not None:
            sample["bbox"] = bbox
        if subtask is not None:
            sample["atomic_task"] = f"Subtask: {subtask}"
        sample["prompt"] = {
            ("bbox",): "predict bbox",
            ("subtask",): "predict subtask",
            ("bbox", "subtask"): "predict bbox, subtask and action",
        }[fields]
        return True

    def _prepare_author_batch(
        self,
        batch: Mapping[str, Any],
        task: str | None = None,
        *,
        predict_cot: bool | None = None,
    ) -> dict[str, Any]:
        """Build the author-format sample batch."""
        run_predict_cot = self.config.predict_cot if predict_cot is None else predict_cot
        prepare = getattr(self.backend, "prepare_lerobot_batch", None)
        if callable(prepare):
            prepared = prepare(batch, task=task, config=self.config)
            prepared[G05_RUNTIME_PREDICT_COT] = run_predict_cot
            return prepared

        state = batch.get(OBS_STATE)
        if not isinstance(state, Tensor):
            raise ValueError(f"G0.5 requires tensor {OBS_STATE!r}.")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        tasks = self._task_values(batch, task, batch_size)
        state_mask = batch.get("proprio_dim_is_pad")
        if state_mask is None:
            state_mask = torch.zeros(
                batch_size, self.config.policy_state_dim, dtype=torch.bool, device=state.device
            )
        elif isinstance(state_mask, Tensor) and state_mask.ndim == 1:
            state_mask = state_mask.unsqueeze(0).expand(batch_size, -1)

        pixel_values: dict[str, Tensor] = {}
        for key in self.config.camera_order:
            image = batch.get(key)
            if not isinstance(image, Tensor):
                raise ValueError(f"G0.5 requires camera {key!r}; camera order is checkpoint state.")
            if image.ndim == 4:
                image = image.unsqueeze(1)
            pixel_values[key] = image
        image_count = sum(image.shape[1] for image in pixel_values.values())
        if image_count != self.config.num_input_images:
            raise ValueError(
                f"G0.5 received {image_count} camera/history frames, but the checkpoint "
                f"template requires {self.config.num_input_images}."
            )

        samples = []
        flow_only = "<action_action" not in self.config.prompt_template
        inference_template = (
            self.config.prompt_template
            if run_predict_cot
            else make_g05_prompt_template(
                self.config.num_prompt_images,
                predict_cot=False,
                flow_only=flow_only,
            )
        )
        for index, raw_task in enumerate(tasks):
            proprio = state[index]
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            sample = {
                "template": inference_template,
                # This is the author InputPreprocessor command slot. Keep it byte-for-byte
                # unchanged; checkpoint-specific chat formatting occurs downstream.
                "command": raw_task,
                "embodiment": self.config.embodiment,
                "proprio": {
                    "value": proprio,
                    "proprio_dim_is_pad": state_mask[index],
                },
            }
            frequency = self.config.processor_metadata.get("frequency")
            if frequency is not None:
                sample["frequency"] = frequency
            if run_predict_cot:
                rendered_recipe = MESSAGES_RENDERED in batch
                applied_recipe_cot = rendered_recipe and self._apply_recipe_cot(
                    sample, batch, index, batch_size
                )
                if not applied_recipe_cot:
                    # During mixed-recipe training an applicable no-CoT branch is a
                    # genuine target format. At inference, where actions are absent,
                    # retain the checkpoint's configured System 2 prompt.
                    if rendered_recipe and isinstance(batch.get(ACTION), Tensor):
                        sample["template"] = make_g05_prompt_template(
                            self.config.num_prompt_images,
                            predict_cot=False,
                            flow_only="<action_action" not in self.config.prompt_template,
                        )
                    else:
                        sample["prompt"] = "predict subtask"
                        atomic_task = batch.get("atomic_task")
                        if atomic_task is not None:
                            atomic_task = str(self._batch_item(atomic_task, index, batch_size))
                            sample["atomic_task"] = (
                                atomic_task
                                if atomic_task.startswith("Subtask:")
                                else f"Subtask: {atomic_task}"
                            )
            for image_index in range(self.config.num_prompt_images):
                camera = self.config.camera_order[image_index % len(self.config.camera_order)]
                sample[f"image{image_index}"] = self.config.camera_sizes[camera]
            action = batch.get(ACTION)
            if "<action_action" in self.config.prompt_template:
                if not isinstance(action, Tensor):
                    action = state.new_zeros(
                        batch_size, self.config.chunk_size, self.config.policy_action_dim
                    )
                action_dim_is_pad = batch.get("action_dim_is_pad")
                if action_dim_is_pad is None:
                    action_dim_is_pad = torch.zeros(
                        batch_size,
                        self.config.policy_action_dim,
                        dtype=torch.bool,
                        device=action.device,
                    )
                elif action_dim_is_pad.ndim == 1:
                    action_dim_is_pad = action_dim_is_pad.unsqueeze(0).expand(batch_size, -1)
                action_payload = {
                    "value": action[index],
                    "action_dim_is_pad": action_dim_is_pad[index],
                }
                action_op_mask = batch.get("action_op_mask")
                if isinstance(action_op_mask, Tensor):
                    action_payload["action_op_mask"] = (
                        action_op_mask[index] if action_op_mask.ndim > 1 else action_op_mask
                    )
                else:
                    action_payload["action_op_mask"] = ~action_dim_is_pad[index]
                action_payload["parts_meta"] = batch.get(
                    "action_parts_meta", G05_POLICY_PARTS[self.config.policy_action_dim]
                )
                sample["action"] = action_payload
            samples.append(sample)
        prepared = dict(batch)
        prepared["samples"] = samples
        prepared["pixel_values"] = pixel_values
        prepared[G05_RUNTIME_PREDICT_COT] = run_predict_cot
        return prepared

    def _run_inference(
        self,
        batch: Mapping[str, Any],
        *,
        task: str | None = None,
        system_mode: str | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Run inference in the requested system mode."""
        if system_mode is None:
            system_mode = self.config.runtime_system
        if system_mode not in {"system1", "system2"}:
            raise ValueError("G0.5 system_mode must be 'system1' or 'system2'.")
        if system_mode == "system2" and not self.config.predict_cot:
            raise ValueError("G0.5 System 2 requires predict_cot=True in the packaged checkpoint.")
        if task is not None:
            raise ValueError("G0.5 task overrides must run through the policy input processor.")
        prepared = dict(batch)
        preprocessed_predict_cot = bool(prepared.get(G05_RUNTIME_PREDICT_COT, False))
        if preprocessed_predict_cot != (system_mode == "system2"):
            raise ValueError(
                "G0.5 system mode does not match the token sequence emitted by its input processor."
            )
        predict = getattr(self.backend, "predict_action", None)
        device = next(self.backend.parameters()).device
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.model_weights_to_bf16 and device.type == "cuda",
        ):
            result = predict(prepared) if callable(predict) else self.backend(prepared)
        if isinstance(result, Tensor):
            result = {ACTION: result}
        if not isinstance(result, Mapping):
            raise TypeError("G0.5 backend inference must return a tensor or mapping.")

        if self.config.action_head == "actioncodec":
            action = result.get("ar_action", result.get(ACTION))
        else:
            action = result.get(ACTION)
        if not isinstance(action, Tensor):
            raise ValueError(f"G0.5 {self.config.action_head} output is missing its action tensor.")
        metadata_keys = ("decoded_action_tokens", "ar_absent_keys", "_timing")
        if system_mode == "system2":
            metadata_keys = ("cot_text", "generated_ids", *metadata_keys)
        metadata = {key: result[key] for key in metadata_keys if key in result}
        return action, metadata

    @torch.no_grad()
    def predict_action_chunk_with_runtime(
        self,
        batch: dict[str, Any],
        *,
        task: str,
        system_mode: str | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Run the selected system and return its action plus same-pass telemetry."""

        return self._run_inference(batch, task=task, system_mode=system_mode)

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Any], *, with_text: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, str | None]:
        """The action chunk, and with `with_text` the same-pass chain-of-thought.

        G0.5 emits reasoning and actions from one inference stream: `_generate_text`
        extends the prefill cache with the CoT tokens and the flow head then runs on
        that extended cache, so the action really is conditioned on this text rather
        than merely accompanied by it. This is why G0.5 honours `with_text` at all.
        System 1 generates no CoT and reports `None`.
        """
        action, metadata = self._run_inference(batch)
        return (action, _first_cot_text(metadata)) if with_text else action

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs) -> Tensor:
        """Return the next action, refilling the queue when it empties."""
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            if chunk.ndim != 3:
                raise ValueError(f"G0.5 action chunk must be [B,T,D], got {tuple(chunk.shape)}.")
            # LeRobot's synchronous select_action queue is intentionally batch-size one.
            if chunk.shape[0] != 1:
                raise ValueError(
                    "G0.5 select_action requires batch size 1; use predict_action_chunk for B>1."
                )
            self._action_queue.extend(chunk[0, : self.config.n_action_steps])
        return self._action_queue.popleft().unsqueeze(0)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any] | None]:
        """Run the training forward pass."""
        prepared = batch
        device = next(self.backend.parameters()).device
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.model_weights_to_bf16 and device.type == "cuda",
        ):
            result = self.backend(prepared)
        if isinstance(result, tuple) and len(result) == 2:
            loss, loss_dict = result
        elif isinstance(result, Mapping) and "loss" in result:
            loss = result["loss"]
            loss_dict = {key: value for key, value in result.items() if key != "loss"}
        else:
            raise TypeError("G0.5 training backend must return (loss, loss_dict) or {'loss': ...}.")
        if not isinstance(loss, Tensor):
            raise TypeError("G0.5 training loss must be a torch.Tensor.")
        logging_values = {
            key: value.detach().item() if isinstance(value, Tensor) and value.numel() == 1 else value
            for key, value in (loss_dict or {}).items()
        }
        return loss, logging_values


def prepare_g05_policy_batch(
    config: Any,
    batch: Mapping[str, Any],
    *,
    task: str | None = None,
    predict_cot: bool | None = None,
) -> dict[str, Any]:
    """Run G0.5's deterministic sample builder without constructing policy weights."""

    proxy = object.__new__(G05Policy)
    nn.Module.__init__(proxy)
    proxy.config = config
    proxy.backend = None
    return G05Policy._prepare_author_batch(proxy, batch, task=task, predict_cot=predict_cot)
