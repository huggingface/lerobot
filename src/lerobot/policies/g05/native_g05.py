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

"""Native G0.5 model components built on Transformers' Qwen3.5 implementation."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from lerobot.utils.constants import ACTION

from .action_codec_g05 import G05NativeActionCodec
from .processing_g05 import IGNORE_INDEX, G05SequenceBatch, G05Tokenizer, G05TokenType

G05_RUNTIME_PREDICT_COT = "g05_runtime_predict_cot"


def _qwen_text_config(values: Mapping[str, Any], *, vocab_size: int | None = None):
    """Translate the serialized G0.5 Qwen config into a Transformers config."""

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

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

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

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
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, proprio: Tensor) -> Tensor:
        with torch.autocast(proprio.device.type, enabled=False):
            return self.mlp(proprio.float())


class G05QwenTextModel(nn.Module):
    """Qwen3.5 text stack with G0.5's checkpoint-compatible module names."""

    def __init__(self, values: Mapping[str, Any], *, vocab_size: int) -> None:
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
            Qwen3_5TextRotaryEmbedding,
        )

        self.config = _qwen_text_config(values, vocab_size=vocab_size)
        self.input_proj = nn.Embedding(vocab_size, self.config.hidden_size, self.config.pad_token_id)
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(self.config, layer_idx)
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.config)

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.input_proj(input_ids)

    def logits(self, hidden_states: Tensor) -> Tensor:
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
        from transformers import DynamicCache

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
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5MLP

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
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

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
        return self.input_proj(actions)

    def encode_time(self, timesteps: Tensor) -> Tensor:
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
        with torch.autocast(hidden_states.device.type, enabled=False):
            return self.output_proj(hidden_states.float())


class G05NativeModel(nn.Module):
    """Weight-owning native G0.5 model assembled from serialized checkpoint config."""

    def __init__(self, model_config: Mapping[str, Any], *, vocab_size: int) -> None:
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        self.vision_tower = Qwen3_5VisionModel(_qwen_vision_config(model_config["vision"]))
        self.vlm = G05QwenTextModel(model_config["vlm"], vocab_size=vocab_size)
        self.action_expert = G05ActionExpert(model_config["action_expert"])
        self.proprio_embedder = G05ProprioEmbedder(
            int(model_config["proprio_dim"]),
            int(model_config["vlm"]["hidden_size"]),
        )


def _temporal_embedding(timesteps: Tensor, dimension: int) -> Tensor:
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
    ) -> None:
        super().__init__()
        self.model_config = dict(model_config)
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
        if isinstance(action_config, Mapping):
            checkpoint = Path(str(action_config.get("ckpt_dir", "")))
            if checkpoint.is_file():
                self.action_tokenizer = G05NativeActionCodec.load(
                    action_config,
                    action_token_begin=self.processor.action_token_begin,
                )
        self._last_vision_grids: list[tuple[int, int, int]] = []

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
    def from_config(cls, model_config: Mapping[str, Any]) -> G05NativeBackend:
        processor_path = Path(str(model_config["hf_processor_path"]))
        tokenizer_config = processor_path / "tokenizer_config.json"
        if not tokenizer_config.is_file():
            raise FileNotFoundError(f"G0.5 tokenizer config not found: {tokenizer_config}")
        import json

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
        return cls(model_config, vocab_size=vocab_size, processor_path=processor_path)

    @staticmethod
    def _patchify(images: Tensor, patch_size: int, temporal_patch_size: int, merge_size: int) -> Tensor:
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
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

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

    def _encode_camera(self, frames: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        """Encode one camera, including G0.5's causal temporal-memory mixing."""

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

    def _encode_vision(self, pixel_values: Mapping[str, Tensor]) -> Tensor:
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
        import itertools

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
    ) -> tuple[Tensor, Tensor, Tensor]:
        embeddings = self.model.vlm.embed(token_ids[:, None])
        batch_size = token_ids.shape[0]
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
        hidden_states, cache = self.model.vlm(
            embeddings,
            full_attention_mask=full_mask,
            linear_attention_mask=torch.ones(batch_size, 1, dtype=embeddings.dtype, device=embeddings.device),
            position_ids=next_positions,
            cache=cache,
        )
        next_types = torch.full(
            (batch_size, 1),
            float(G05TokenType.PRED_TEXT),
            dtype=token_types.dtype,
            device=token_types.device,
        )
        return (
            hidden_states[:, -1],
            torch.cat((token_types, next_types), dim=1),
            torch.cat((positions, next_positions), dim=-1),
        )

    def _generate_text(
        self,
        last_hidden: Tensor,
        *,
        token_types: Tensor,
        positions: Tensor,
        cache,
        max_new_tokens: int,
        stop_token_id: int,
    ) -> tuple[Tensor, Any, Tensor, Tensor, Tensor]:
        generated = []
        finished = torch.zeros(last_hidden.shape[0], dtype=torch.bool, device=last_hidden.device)
        for _ in range(max_new_tokens):
            logits = self.model.vlm.logits(last_hidden)
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, stop_token_id),
                next_token,
            )
            generated.append(next_token)
            last_hidden, token_types, positions = self._decode_token(
                next_token,
                token_types=token_types,
                positions=positions,
                cache=cache,
            )
            finished |= next_token == stop_token_id
            if bool(finished.all()):
                break
        generated_ids = (
            torch.stack(generated, dim=1)
            if generated
            else torch.empty(last_hidden.shape[0], 0, dtype=torch.long, device=last_hidden.device)
        )
        return generated_ids, cache, last_hidden, token_types, positions

    def _action_cache(self, vlm_cache, prefix_length: int, *, repeats: int = 1):
        from transformers import DynamicCache

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
        start = time.monotonic()
        samples = list(batch["samples"])
        pixel_values = batch["pixel_values"]
        first_image = next(iter(pixel_values.values()))
        sequence = self.processor.encode_inference(samples, device=first_image.device)
        proprio = self._proprio(samples, first_image.device)
        hidden_states, cache, positions = self._prefill(sequence, pixel_values, proprio)
        result: dict[str, Any] = {}
        last_hidden = hidden_states[:, -1]
        token_types = sequence.token_types
        predict_cot = bool(batch.get(G05_RUNTIME_PREDICT_COT, self.model_config.get("predict_cot", False)))
        if predict_cot:
            generated, cache, last_hidden, token_types, positions = self._generate_text(
                last_hidden,
                token_types=sequence.token_types,
                positions=positions,
                cache=cache,
                max_new_tokens=int(self.model_config["ar"].get("max_new_tokens", 300)),
                stop_token_id=self.processor.eov_token_id,
            )
            sequence.token_types = token_types
            result["generated_ids"] = generated
            result["cot_text"] = [
                self.processor.decode(
                    ids[
                        : next(
                            (
                                index
                                for index, token_id in enumerate(ids.tolist())
                                if token_id == self.processor.eov_token_id
                            ),
                            len(ids),
                        )
                    ]
                )
                for ids in generated
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
                generated_action, _, _, _, _ = self._generate_text(
                    last_hidden,
                    token_types=sequence.token_types,
                    positions=positions,
                    cache=cache,
                    max_new_tokens=self.action_tokenizer.action_token_length + 32,
                    stop_token_id=self.processor.eos_token_id,
                )
                decoded_actions = []
                decoded_tokens = []
                absent_keys = []
                for token_row in generated_action:
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
        samples = list(batch["samples"])
        pixel_values = batch["pixel_values"]
        first_image = next(iter(pixel_values.values()))
        sequence = self.processor.encode_train(
            samples,
            device=first_image.device,
            action_codec=self.action_tokenizer,
        )
        proprio = self._proprio(samples, first_image.device)
        hidden_states, cache, positions = self._prefill(sequence, pixel_values, proprio)
        loss_dict: dict[str, Tensor] = {}

        skip_ce = (
            bool(self.model_config.get("continuous_action", False))
            and not bool(self.model_config.get("discrete_action", False))
            and not bool(self.model_config.get("predict_cot", False))
        )
        if not skip_ce:
            shift_labels = sequence.labels[:, 1:]
            valid = shift_labels != IGNORE_INDEX
            if valid.any():
                logits = self.model.vlm.logits(hidden_states[:, :-1])
                loss_dict["ce_loss"] = functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
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
