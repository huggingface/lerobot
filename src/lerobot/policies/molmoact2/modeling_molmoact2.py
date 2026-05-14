from __future__ import annotations

import json
import os
import types
from collections import defaultdict, deque
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from huggingface_hub import snapshot_download
from torch import Tensor
from torch.distributions import Beta

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from ..rtc.modeling_rtc import RTCProcessor
from .configuration_molmoact2 import MolmoAct2Config

_MODEL_INPUT_KEYS = {
    "input_ids",
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
    "pixel_values_videos",
    "video_token_pooling",
    "video_grids",
    "attention_mask",
    "position_ids",
    "past_key_values",
    "token_type_ids",
    "inputs_embeds",
}


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")


def _resolve_checkpoint_location(
    checkpoint_path: str,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> str:
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        raise ValueError("MolmoAct2 policy requires `checkpoint_path`.")
    local_path = Path(checkpoint_path).expanduser()
    if local_path.exists():
        return str(local_path)
    return snapshot_download(
        repo_id=checkpoint_path,
        repo_type="model",
        revision=revision,
        force_download=force_download,
        token=_hf_token(),
    )


def _load_hf_norm_metadata_for_tag(
    checkpoint_path: str,
    *,
    revision: str | None,
    force_download: bool,
    norm_tag: str | None,
) -> dict[str, Any]:
    norm_tag = str(norm_tag or "").strip()
    if not norm_tag:
        return {}
    checkpoint_location = Path(
        _resolve_checkpoint_location(
            checkpoint_path,
            revision=revision,
            force_download=force_download,
        )
    )
    norm_stats_filename = "norm_stats.json"
    config_path = checkpoint_location / "config.json"
    if config_path.exists():
        with suppress(OSError, json.JSONDecodeError):
            norm_stats_filename = str(
                json.loads(config_path.read_text()).get("norm_stats_filename") or norm_stats_filename
            )
    stats_path = checkpoint_location / norm_stats_filename
    if not stats_path.exists():
        raise FileNotFoundError(
            f"MolmoAct2 HF checkpoint is missing {norm_stats_filename!r}; cannot resolve norm_tag={norm_tag!r}."
        )
    payload = json.loads(stats_path.read_text())
    metadata_by_tag = payload.get("metadata_by_tag")
    if not isinstance(metadata_by_tag, dict):
        raise ValueError(f"MolmoAct2 norm stats file {stats_path} has no metadata_by_tag mapping.")
    metadata = metadata_by_tag.get(norm_tag)
    if not isinstance(metadata, dict):
        available = sorted(str(tag) for tag in metadata_by_tag)
        raise ValueError(f"Unknown MolmoAct2 norm_tag={norm_tag!r}. Available tags: {available}.")
    return metadata


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _sample_beta_timesteps(
    *,
    batch_size: int,
    device: torch.device,
    cutoff: float,
    time_offset: float,
    time_scale: float,
    alpha: float,
    beta: float,
) -> Tensor:
    if cutoff < time_offset:
        raise ValueError(f"flow-matching cutoff must be >= time_offset, got {cutoff} < {time_offset}")
    if time_scale <= 0:
        raise ValueError(f"flow-matching time_scale must be > 0, got {time_scale}")
    upper = min(cutoff, time_offset + time_scale)
    dist = Beta(torch.tensor(alpha, device=device), torch.tensor(beta, device=device))
    samples = dist.sample((batch_size,))
    scale = upper - time_offset
    if scale == 0:
        return torch.full((batch_size,), time_offset, device=device, dtype=samples.dtype)
    return time_offset + scale * samples


def _patch_batched_image_attention_bias(backbone: Any) -> None:
    original = getattr(backbone, "_build_native_attention_bias", None)
    if original is None:
        return
    original_func = getattr(original, "__func__", original)
    original_globals = getattr(original_func, "__globals__", {})
    cache_seq_len = original_globals.get("_cache_seq_len_int")
    cache_max_len = original_globals.get("_cache_max_len_int")
    if cache_seq_len is None or cache_max_len is None:
        return

    def _build_native_attention_bias(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None,
        token_type_ids: Tensor | None,
        past_key_values: Any,
    ) -> Tensor:
        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask.to(device=inputs_embeds.device)
        batch_size, seq_len = inputs_embeds.shape[:2]
        past_length = int(cache_seq_len(past_key_values))
        current_length = past_length + int(seq_len)
        max_cache_len = int(cache_max_len(past_key_values))
        attention_mask_len = max_cache_len if max_cache_len > 0 else current_length
        device = inputs_embeds.device

        if attention_mask is None:
            positions = torch.arange(attention_mask_len, device=device)
            valid_mask = positions.unsqueeze(0) < current_length
            valid_mask = valid_mask.expand(batch_size, -1)
        elif attention_mask.ndim == 2:
            valid_mask = torch.zeros((batch_size, attention_mask_len), device=device, dtype=torch.bool)
            source_mask = attention_mask.to(device=device, dtype=torch.bool)
            copy_len = min(int(source_mask.shape[-1]), attention_mask_len)
            if copy_len > 0:
                valid_mask[:, :copy_len] = source_mask[:, :copy_len]
            if attention_mask_len > current_length:
                valid_mask[:, current_length:] = False
        else:
            raise ValueError(f"Unsupported attention_mask shape for MolmoAct2: {tuple(attention_mask.shape)}")

        valid_mask = valid_mask[:, None, None, :]
        causal_mask = torch.tril(
            torch.ones(attention_mask_len, attention_mask_len, device=device, dtype=torch.bool)
        )[None, None, past_length:current_length, :attention_mask_len]

        if token_type_ids is not None and past_length == 0:
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1).clone()
            image_mask = token_type_ids.to(device=device, dtype=torch.bool)
            can_attend_back = image_mask[:, :, None] & image_mask[:, None, :]
            image_len = min(int(token_type_ids.shape[1]), attention_mask_len)
            causal_mask[:, :, :, :image_len] = (
                causal_mask[:, :, :, :image_len] | can_attend_back[:, None, :, :image_len]
            )

        allowed = valid_mask & causal_mask
        return torch.where(
            allowed,
            torch.zeros((), device=device, dtype=inputs_embeds.dtype),
            torch.full((), torch.finfo(inputs_embeds.dtype).min, device=device, dtype=inputs_embeds.dtype),
        )

    backbone._build_native_attention_bias = types.MethodType(_build_native_attention_bias, backbone)


def _patch_leaf_safe_input_embedding_update(backbone: Any) -> None:
    if getattr(backbone, "_lerobot_leaf_safe_input_embedding_update_patched", False):
        return
    if not callable(getattr(backbone, "build_input_embeddings", None)):
        return

    def _build_input_embeddings(
        self,
        input_ids: Tensor,
        images: Tensor | None = None,
        token_pooling: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = self.transformer.wte(input_ids)

        image_features = None
        if images is not None:
            image_features = self.vision_backbone(images, token_pooling).to(x.device)
            is_image_patch = input_ids.reshape(-1) == self.config.image_patch_id
            if is_image_patch.sum() != len(image_features):
                raise RuntimeError(
                    f"Expected {int(is_image_patch.sum())} image patch embeddings, got {len(image_features)}."
                )
            flat_x = x.reshape(-1, x.shape[-1]).clone()
            flat_x[is_image_patch] = flat_x[is_image_patch] + image_features
            x = flat_x.reshape_as(x)

        x = self.transformer.emb_drop(x)
        return x, image_features

    backbone.build_input_embeddings = types.MethodType(_build_input_embeddings, backbone)
    backbone._lerobot_leaf_safe_input_embedding_update_patched = True


def _patch_memory_efficient_vision_backbone(backbone: Any, *, gradient_checkpointing: bool) -> None:
    vision_backbone = getattr(backbone, "vision_backbone", None)
    if vision_backbone is None or getattr(
        vision_backbone, "_lerobot_memory_efficient_vision_backbone_patched", False
    ):
        return

    image_vit = getattr(vision_backbone, "image_vit", None)
    transformer = getattr(image_vit, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None)
    if image_vit is None or transformer is None or resblocks is None:
        return
    if not hasattr(vision_backbone, "vit_layers"):
        return

    def _encode_image(self, images: Tensor) -> Tensor:
        batch_size, num_crops, num_patches, patch_dim = images.shape
        images = images.view(batch_size * num_crops, num_patches, patch_dim)

        x = self.image_vit.patch_embedding(images)
        x = self.image_vit.add_pos_emb(x, self.image_vit.config.image_num_patch)

        needed_layers = {int(layer) for layer in self.vit_layers}
        selected_features: dict[int, Tensor] = {}
        use_checkpoint = bool(
            self._lerobot_vision_gradient_checkpointing and self.training and torch.is_grad_enabled()
        )
        for layer_idx, block in enumerate(self.image_vit.transformer.resblocks):
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if layer_idx in needed_layers:
                selected_features[layer_idx] = x

        if len(selected_features) != len(needed_layers):
            missing = sorted(needed_layers - set(selected_features))
            raise RuntimeError(f"MolmoAct2 vision backbone did not produce requested layers: {missing}.")

        image_features = torch.cat([selected_features[int(layer)] for layer in self.vit_layers], dim=-1)
        if self.num_prefix_tokens > 0:
            image_features = image_features[:, 1:]
        image_features = image_features.view(batch_size, num_crops, num_patches, -1)
        return image_features

    vision_backbone.encode_image = types.MethodType(_encode_image, vision_backbone)
    vision_backbone._lerobot_vision_gradient_checkpointing = bool(gradient_checkpointing)
    vision_backbone._lerobot_memory_efficient_vision_backbone_patched = True


def _patch_training_kv_collection(backbone: Any) -> None:
    """Expose per-layer VLM KV tensors without enabling HF autoregressive cache."""
    if getattr(backbone, "_lerobot_training_kv_collection_patched", False):
        return

    transformer = getattr(backbone, "transformer", None)
    blocks = getattr(transformer, "blocks", None)
    if transformer is None or blocks is None:
        raise RuntimeError("MolmoAct2 checkpoint does not expose a patchable text transformer.")

    original_transformer_forward = transformer.forward
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def _patch_attention(attention: torch.nn.Module) -> None:
        if getattr(attention, "_lerobot_training_kv_collection_patched", False):
            return

        original_attention_forward = attention.forward
        original_attention_func = getattr(original_attention_forward, "__func__", original_attention_forward)
        attention_globals = getattr(original_attention_func, "__globals__", {})
        apply_rotary_pos_emb = attention_globals.get("apply_rotary_pos_emb")
        repeat_kv = attention_globals.get("repeat_kv")
        eager_attention_forward = attention_globals.get("eager_attention_forward")
        all_attention_functions = attention_globals.get("ALL_ATTENTION_FUNCTIONS")
        if (
            apply_rotary_pos_emb is None
            or repeat_kv is None
            or eager_attention_forward is None
            or all_attention_functions is None
        ):
            raise RuntimeError("MolmoAct2 attention internals changed; cannot patch KV collection.")

        def _attention_forward(
            self,
            hidden_states: Tensor,
            position_embeddings: tuple[Tensor, Tensor],
            attention_mask: Tensor | None,
            past_key_values: Any | None = None,
            cache_position: Tensor | None = None,
            **kwargs,
        ):
            collect_layer_kv_states = bool(kwargs.pop("collect_layer_kv_states", False))
            if not collect_layer_kv_states:
                return original_attention_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            qkv = self.att_proj(hidden_states)
            query_states, key_states, value_states = qkv.split(self.fused_dims, dim=-1)
            value_states = value_states.view(hidden_shape)

            if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type != "qwen3":
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)

            query_states = query_states.view(hidden_shape)
            key_states = key_states.view(hidden_shape)
            if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type == "qwen3":
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            collected_key_states = key_states
            collected_value_states = value_states
            dropout_p = 0.0 if not self.training else self.attention_dropout
            if self.config._attn_implementation == "sdpa" and (
                attention_mask is None or torch.is_tensor(attention_mask)
            ):
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    dropout_p=dropout_p,
                    is_causal=attention_mask is None,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_weights = None
            else:
                attention_interface = eager_attention_forward
                if self.config._attn_implementation != "eager":
                    attention_interface = all_attention_functions[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=dropout_p,
                    scaling=self.scaling,
                    **kwargs,
                )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.attn_out(attn_output)
            return attn_output, attn_weights, collected_key_states, collected_value_states

        attention.forward = types.MethodType(_attention_forward, attention)
        attention._lerobot_training_kv_collection_patched = True

    def _patch_decoder_layer(layer: torch.nn.Module) -> None:
        if getattr(layer, "_lerobot_training_kv_collection_patched", False):
            return

        _patch_attention(layer.self_attn)
        original_layer_forward = layer.forward
        is_post_norm = "PostNorm" in layer.__class__.__name__

        def _decoder_layer_forward(
            self,
            hidden_states: Tensor,
            position_embeddings: tuple[Tensor, Tensor],
            attention_mask: Tensor | None = None,
            position_ids: Tensor | None = None,
            past_key_values: Any | None = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Tensor | None = None,
            **kwargs,
        ):
            collect_layer_kv_states = bool(kwargs.pop("collect_layer_kv_states", False))
            if not collect_layer_kv_states:
                return original_layer_forward(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            residual = hidden_states
            attn_input = hidden_states if is_post_norm else self.attn_norm(hidden_states)
            attn_output, self_attn_weights, key_states, value_states = self.self_attn(
                hidden_states=attn_input,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                collect_layer_kv_states=True,
                **kwargs,
            )
            if is_post_norm:
                attn_output = self.attn_norm(attn_output)
            hidden_states = residual + self.dropout(attn_output)

            residual = hidden_states
            if is_post_norm:
                hidden_states = self.mlp(hidden_states)
                hidden_states = self.ff_norm(hidden_states)
            else:
                hidden_states = self.ff_norm(hidden_states)
                hidden_states = self.mlp(hidden_states)
            hidden_states = residual + self.dropout(hidden_states)

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            return outputs + (key_states, value_states)

        layer.forward = types.MethodType(_decoder_layer_forward, layer)
        layer._lerobot_training_kv_collection_patched = True

    for block in blocks:
        _patch_decoder_layer(block)

    def _transformer_forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: Tensor | None = None,
        **kwargs,
    ):
        collect_layer_kv_states = bool(kwargs.pop("collect_layer_kv_states", False))
        if not collect_layer_kv_states:
            return original_transformer_forward(
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
        if past_key_values is not None:
            raise ValueError("collect_layer_kv_states only supports full-sequence training forwards.")

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            cache_position = torch.arange(
                0,
                inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if torch.is_tensor(attention_mask) and attention_mask.ndim == 4:
            causal_mask_mapping = attention_mask
        elif not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            causal_mask_mapping = create_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        if self.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": self.rotary_embs["default"](hidden_states, position_ids),
                "scaling": self.rotary_embs["scaling"](hidden_states, position_ids),
            }
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        collected_kv_states = []

        for layer_idx, decoder_block in enumerate(self.blocks[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in self.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings

            layer_outputs = decoder_block(
                hidden_states,
                position_embeddings=position_embeddings_i,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                collect_layer_kv_states=True,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            output_idx = 1
            if output_attentions:
                all_self_attns += (layer_outputs[output_idx],)
                output_idx += 1
            collected_kv_states.append((layer_outputs[output_idx], layer_outputs[output_idx + 1]))

        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(collected_kv_states),
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    transformer.forward = types.MethodType(_transformer_forward, transformer)
    backbone._lerobot_training_kv_collection_patched = True


class MolmoAct2Policy(PreTrainedPolicy):
    config_class = MolmoAct2Config
    name = "molmoact2"

    def __init__(
        self,
        config: MolmoAct2Config,
        *inputs,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        dataset_meta: Any | None = None,
        **kwargs,
    ):
        super().__init__(config, *inputs, **kwargs)
        self._checkpoint_action_mode = self._load_saved_policy_action_mode()
        self._apply_norm_tag_metadata()
        self.config.validate_features()
        del inputs, kwargs, dataset_stats, dataset_meta
        self._action_queues: dict[int, deque[Tensor]] = defaultdict(deque)
        self._rollout_action_generator: torch.Generator | None = None
        self._rollout_task_key: tuple[Any, ...] | None = None
        self._rollout_index_for_task = -1
        self.rtc_processor: RTCProcessor | None = None
        self.action_tokenizer: Any | None = None
        self._load_hf_model()
        self._validate_inference_action_mode()
        if self.config.enable_lora_vlm:
            self._apply_lora_adapters()
        self.init_rtc_processor()

    def _load_saved_policy_action_mode(self) -> str | None:
        pretrained_path = getattr(self.config, "pretrained_path", None)
        if pretrained_path is None:
            return None
        config_path = Path(pretrained_path) / "config.json"
        if not config_path.exists():
            return None
        try:
            mode = json.loads(config_path.read_text()).get("action_mode")
        except (OSError, json.JSONDecodeError):
            return None
        if mode in {"continuous", "discrete", "both"}:
            return str(mode)
        return None

    def _training_action_mode(self) -> str:
        return getattr(self, "_checkpoint_action_mode", None) or self.config.action_mode

    def _validate_inference_action_mode(self) -> None:
        requested_mode = self.config.inference_action_mode
        if requested_mode is None:
            return
        training_mode = self._training_action_mode()
        if requested_mode == "continuous" and training_mode == "discrete":
            raise ValueError(
                "MolmoAct2 checkpoint was trained with action_mode='discrete' and cannot run "
                "continuous inference."
            )
        if requested_mode == "discrete" and training_mode == "continuous":
            raise ValueError(
                "MolmoAct2 checkpoint was trained with action_mode='continuous' and cannot run "
                "discrete inference. Train with action_mode='both' or action_mode='discrete' first."
            )

    def _apply_norm_tag_metadata(self) -> None:
        if not str(self.config.norm_tag or "").strip():
            return
        metadata = _load_hf_norm_metadata_for_tag(
            self.config.checkpoint_path,
            revision=self.config.checkpoint_revision,
            force_download=bool(self.config.checkpoint_force_download),
            norm_tag=self.config.norm_tag,
        )
        if metadata.get("action_horizon") is not None:
            self.config.chunk_size = int(metadata["action_horizon"])
        if metadata.get("n_action_steps") is not None:
            self.config.n_action_steps = int(metadata["n_action_steps"])
        if not self.config.setup_type and metadata.get("setup_type") is not None:
            self.config.setup_type = str(metadata["setup_type"])
        if not self.config.control_mode and metadata.get("control_mode") is not None:
            self.config.control_mode = str(metadata["control_mode"])
        if not self.config.image_keys and isinstance(metadata.get("camera_keys"), list):
            self.config.image_keys = [str(key) for key in metadata["camera_keys"]]

    def _load_hf_model(self) -> None:
        require_package("transformers", extra="molmoact2")
        from transformers import AutoModelForImageTextToText

        checkpoint_location = _resolve_checkpoint_location(
            self.config.checkpoint_path,
            revision=self.config.checkpoint_revision,
            force_download=bool(self.config.checkpoint_force_download),
        )
        model_dtype = _torch_dtype(self.config.model_dtype)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_location,
            trust_remote_code=self.config.trust_remote_code,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
            token=_hf_token(),
        )
        hf_max_action_dim = int(getattr(self.model.config, "max_action_dim", -1))
        if hf_max_action_dim != int(self.config.expected_max_action_dim):
            raise ValueError(
                "MolmoAct2 checkpoint max_action_dim mismatch: "
                f"checkpoint={hf_max_action_dim}, expected={self.config.expected_max_action_dim}."
            )
        if hf_max_action_dim != 32:
            raise ValueError(
                f"MolmoAct2 released checkpoints must have max_action_dim=32, got {hf_max_action_dim}."
            )

        if not hasattr(self.model.config, "max_action_horizon"):
            raise ValueError("MolmoAct2 HF checkpoints must define `max_action_horizon`.")
        self._override_loaded_max_action_horizon(int(self.config.chunk_size))

        if not hasattr(self.model.config, "action_mode"):
            raise ValueError(
                "MolmoAct2 HF checkpoints must define `action_mode`. If this is a released "
                "MolmoAct2 checkpoint, refresh the local Hub cache with "
                "`policy.checkpoint_force_download=true` after the updated files are pushed."
            )
        checkpoint_action_mode = str(self.model.config.action_mode)
        if self.config.action_mode == "both" and checkpoint_action_mode != "both":
            raise ValueError(
                f"action_mode='both' requires checkpoint action_mode='both', got {checkpoint_action_mode!r}."
            )
        if self.config.action_mode == "discrete" and checkpoint_action_mode not in {"discrete", "both"}:
            raise ValueError(
                f"action_mode='discrete' requires checkpoint action_mode in {{'discrete', 'both'}}, "
                f"got {checkpoint_action_mode!r}."
            )
        if self.config.action_mode in {"continuous", "both"} and not bool(
            getattr(self.model.config, "add_action_expert", False)
        ):
            raise ValueError("Continuous MolmoAct2 training requires an action expert checkpoint.")

        if self.config.freeze_embedding:
            self._freeze_input_embeddings()
        if self.config.train_action_expert_only:
            self._freeze_non_action_expert_parameters()
        _patch_batched_image_attention_bias(self._backbone())
        _patch_leaf_safe_input_embedding_update(self._backbone())
        _patch_memory_efficient_vision_backbone(
            self._backbone(),
            gradient_checkpointing=bool(self.config.gradient_checkpointing),
        )
        _patch_training_kv_collection(self._backbone())
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        self.train(self.training)

    def reset(self) -> None:
        self._action_queues = defaultdict(deque)
        self._rollout_action_generator = None

    def _set_inference_cuda_graph_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "model"):
            return
        hf_model = self._hf_model()
        enabled = bool(enabled and getattr(self.config, "enable_inference_cuda_graph", True))
        managers = [
            getattr(self._backbone(), "action_cuda_graph_manager", None),
            getattr(hf_model, "action_cuda_graph_manager", None),
            getattr(hf_model, "depth_decode_cuda_graph_manager", None),
        ]
        seen: set[int] = set()
        for manager in managers:
            if manager is None or id(manager) in seen:
                continue
            seen.add(id(manager))
            set_enabled = getattr(manager, "set_enabled", None)
            if callable(set_enabled):
                set_enabled(enabled)

    def init_rtc_processor(self) -> None:
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _action_expert(self) -> torch.nn.Module:
        return self._backbone()._require_action_expert()

    def _enable_gradient_checkpointing(self) -> None:
        enable_gradient_checkpointing = getattr(self._hf_model(), "gradient_checkpointing_enable", None)
        if callable(enable_gradient_checkpointing):
            try:
                enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                enable_gradient_checkpointing()
        else:
            transformer = getattr(self._backbone(), "transformer", None)
            if transformer is None:
                raise RuntimeError("gradient_checkpointing=true, but MolmoAct2 exposes no text transformer.")
            transformer.gradient_checkpointing = True

        transformer = getattr(self._backbone(), "transformer", None)
        if transformer is not None:
            transformer.gradient_checkpointing = True

    def _freeze_non_action_expert_parameters(self) -> None:
        trainable_params = 0
        for name, param in self.named_parameters():
            param.requires_grad = "action_expert" in name
            if param.requires_grad:
                trainable_params += param.numel()
        if trainable_params == 0:
            raise RuntimeError("train_action_expert_only=true, but no action_expert parameters were found.")

    def _unfreeze_action_expert_parameters(self) -> None:
        trainable_params = 0
        for name, param in self.named_parameters():
            if "action_expert" in name:
                param.requires_grad_(True)
                trainable_params += param.numel()
        if trainable_params == 0:
            raise RuntimeError("enable_lora_vlm=true, but no action_expert parameters were found.")

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.config, "train_action_expert_only", False) and hasattr(self, "model"):
            self._hf_model().eval()
            self._action_expert().train(mode)
        self._set_inference_cuda_graph_enabled(not mode)
        return self

    def _freeze_input_embeddings(self) -> None:
        embedding_modules: list[torch.nn.Module] = []
        seen_module_ids: set[int] = set()
        hf_model = self._hf_model()
        for module in (hf_model, self._backbone()):
            get_input_embeddings = getattr(module, "get_input_embeddings", None)
            if not callable(get_input_embeddings):
                continue
            embeddings = get_input_embeddings()
            if embeddings is None or id(embeddings) in seen_module_ids:
                continue
            embedding_modules.append(embeddings)
            seen_module_ids.add(id(embeddings))

        if not embedding_modules:
            raise RuntimeError("freeze_embedding=true, but MolmoAct2 checkpoint exposes no input embeddings.")

        lm_head = getattr(hf_model, "lm_head", None)
        lm_head_params = {id(param) for param in lm_head.parameters()} if lm_head is not None else set()
        embedding_params = [param for embeddings in embedding_modules for param in embeddings.parameters()]
        if any(id(param) in lm_head_params for param in embedding_params):
            raise RuntimeError(
                "freeze_embedding=true would also freeze lm_head because input embeddings and lm_head "
                "share parameters in this checkpoint."
            )
        for param in embedding_params:
            param.requires_grad = False

    def get_optim_params(self) -> list[dict[str, Any]]:
        vit_params: list[Tensor] = []
        connector_params: list[Tensor] = []
        action_expert_params: list[Tensor] = []
        vlm_params: list[Tensor] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "action_expert" in name:
                action_expert_params.append(param)
            elif any(part in name for part in ("image_pooling_2d", "image_projector")):
                connector_params.append(param)
            elif any(part in name for part in ("vision", "image_encoder", "vit")):
                vit_params.append(param)
            elif any(part in name for part in ("multi_modal_projector", "connector", "mm_projector")):
                connector_params.append(param)
            else:
                vlm_params.append(param)

        vlm_lr = 5e-5 if self.config.enable_lora_vlm else self.config.optimizer_lr
        vit_lr = 5e-5 if self.config.enable_lora_vlm else self.config.optimizer_vit_lr
        connector_lr = 5e-5 if self.config.enable_lora_vlm else self.config.optimizer_connector_lr

        groups: list[dict[str, Any]] = []
        if vlm_params:
            groups.append({"params": vlm_params, "lr": vlm_lr})
        if vit_params:
            groups.append({"params": vit_params, "lr": vit_lr})
        if connector_params:
            groups.append({"params": connector_params, "lr": connector_lr})
        if action_expert_params:
            groups.append({"params": action_expert_params, "lr": self.config.optimizer_action_expert_lr})
        return groups

    def _model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        compute_dtype = _torch_dtype(self.config.model_dtype)
        return {
            key: value.to(dtype=compute_dtype) if value.is_floating_point() else value
            for key, value in batch.items()
            if key in _MODEL_INPUT_KEYS and value is not None
        }

    def _output_action_dim(self, batch: dict[str, Tensor]) -> int:
        action_feature = self.config.output_features.get(ACTION)
        if action_feature is not None and action_feature.shape:
            action_dim = int(action_feature.shape[0])
            if action_dim > 0:
                return action_dim

        action_dim_is_pad = batch.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            valid_counts = (~action_dim_is_pad.to(dtype=torch.bool)).sum(dim=-1)
            if bool((valid_counts == valid_counts[0]).all()) and int(valid_counts[0]) > 0:
                return int(valid_counts[0])

        raise RuntimeError("MolmoAct2 inference requires a positive action dimension in output_features.")

    def _hf_model(self):
        base_model = getattr(self.model, "base_model", None)
        wrapped_model = getattr(base_model, "model", None) if base_model is not None else None
        return wrapped_model if wrapped_model is not None else self.model

    def _backbone(self):
        return self._hf_model().model

    def _override_loaded_max_action_horizon(self, action_horizon: int) -> None:
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}.")
        hf_model = self._hf_model()
        for cfg in (getattr(hf_model, "config", None), getattr(self._backbone(), "config", None)):
            if cfg is not None:
                cfg.max_action_horizon = int(action_horizon)

    def _generation_action_horizon(self) -> int:
        chunk_size = getattr(self.config, "chunk_size", None)
        if chunk_size is not None:
            return int(chunk_size)
        hf_model = self._hf_model()
        for cfg in (getattr(hf_model, "config", None), getattr(self._backbone(), "config", None)):
            if cfg is None:
                continue
            value = getattr(cfg, "max_action_horizon", None)
            if value is not None:
                return int(value)
        raise RuntimeError("MolmoAct2 could not resolve an action generation horizon.")

    @staticmethod
    def _mask_discrete_action_spans(
        *,
        input_ids: Tensor,
        mask: Tensor,
        start_token_id: int | None,
        end_token_id: int | None,
    ) -> Tensor:
        if start_token_id is None or end_token_id is None:
            return mask
        mask = mask.clone()
        for batch_idx in range(input_ids.shape[0]):
            row = input_ids[batch_idx]
            starts = (row == int(start_token_id)).nonzero(as_tuple=False).flatten().tolist()
            ends = (row == int(end_token_id)).nonzero(as_tuple=False).flatten().tolist()
            end_ptr = 0
            for start in starts:
                while end_ptr < len(ends) and ends[end_ptr] < start:
                    end_ptr += 1
                if end_ptr >= len(ends):
                    mask[batch_idx, start:] = False
                    break
                end = int(ends[end_ptr])
                mask[batch_idx, start : end + 1] = False
                end_ptr += 1
        return mask

    def _encoder_attention_mask_for_action_expert(
        self,
        *,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        backbone = self._backbone()
        get_encoder_attention_mask = getattr(backbone, "_get_encoder_attention_mask", None)
        if callable(get_encoder_attention_mask):
            mask = get_encoder_attention_mask(input_ids, attention_mask)
        elif attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool)
        elif input_ids is not None:
            mask = input_ids != -1
        else:
            return None

        if getattr(self.config, "action_mode", None) != "both" or input_ids is None or mask is None:
            return mask

        mask = mask.to(dtype=torch.bool).clone()
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if eos_token_id is not None:
            mask &= input_ids != int(eos_token_id)
        return self._mask_discrete_action_spans(
            input_ids=input_ids,
            mask=mask,
            start_token_id=getattr(self.model.config, "action_start_token_id", None),
            end_token_id=getattr(self.model.config, "action_end_token_id", None),
        )

    @staticmethod
    def _drop_trivial_attention_mask(model_inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        attention_mask = model_inputs.get("attention_mask")
        if torch.is_tensor(attention_mask) and bool(attention_mask.to(dtype=torch.bool).all().item()):
            model_inputs = dict(model_inputs)
            model_inputs.pop("attention_mask", None)
        return model_inputs

    def _load_discrete_action_tokenizer(self) -> Any:
        if self.action_tokenizer is None:
            require_package("transformers", extra="molmoact2")
            from transformers import AutoProcessor

            self.action_tokenizer = AutoProcessor.from_pretrained(
                self.config.discrete_action_tokenizer,
                trust_remote_code=self.config.trust_remote_code,
                token=_hf_token(),
            )
        return self.action_tokenizer

    def _resolve_inference_action_mode(self, requested_mode: str | None) -> str:
        training_mode = self._training_action_mode()
        if requested_mode is None:
            requested_mode = self.config.inference_action_mode
        if requested_mode is None:
            raise ValueError(
                "MolmoAct2 inference requires `inference_action_mode` to be set explicitly "
                "to either 'continuous' or 'discrete'."
            )
        if requested_mode not in {"continuous", "discrete"}:
            raise ValueError("MolmoAct2 inference_action_mode must be either 'continuous' or 'discrete'.")
        if requested_mode == "continuous" and training_mode == "discrete":
            raise ValueError("MolmoAct2 action_mode='discrete' checkpoint cannot run continuous inference.")
        if requested_mode == "discrete" and training_mode == "continuous":
            raise ValueError("MolmoAct2 action_mode='continuous' checkpoint cannot run discrete inference.")
        return requested_mode

    @staticmethod
    def _combine_rollout_seeds(first_seed: int, batch_size: int) -> int:
        seed = 0
        for idx in range(batch_size):
            seed = (seed + (idx + 1) * (first_seed + idx)) % (2**63 - 1)
        return seed

    @staticmethod
    def _rollout_task_signature(batch: dict[str, Any]) -> tuple[Any, ...] | None:
        task = batch.get("task")
        if task is None:
            task = batch.get("observation.language")
        if task is None:
            return None
        if isinstance(task, str):
            return (task,)
        if isinstance(task, (list, tuple)):
            return tuple(str(item) for item in task)
        return (str(task),)

    def _rollout_generator_for_inputs(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Generator | None:
        if not bool(getattr(self.config, "per_episode_seed", False)):
            return None
        if self._rollout_action_generator is not None:
            return self._rollout_action_generator

        task_signature = self._rollout_task_signature(batch)
        if task_signature != self._rollout_task_key:
            self._rollout_task_key = task_signature
            self._rollout_index_for_task = 0
        else:
            self._rollout_index_for_task += 1

        base_seed = int(getattr(self.config, "eval_seed", None) or 0)
        first_seed = base_seed + self._rollout_index_for_task * batch_size
        generator_device = (
            device if device.type == "cuda" and torch.cuda.is_available() else torch.device("cpu")
        )
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self._combine_rollout_seeds(first_seed, batch_size))
        self._rollout_action_generator = generator
        return generator

    @staticmethod
    def _expand_mask(mask: Tensor | None, num_flow_timesteps: int) -> Tensor | None:
        if mask is None:
            return None
        return (
            mask.unsqueeze(1)
            .expand(-1, num_flow_timesteps, *([-1] * (mask.ndim - 1)))
            .reshape(mask.shape[0] * num_flow_timesteps, *mask.shape[1:])
        )

    @staticmethod
    def _action_dim_valid_mask(target: Tensor, action_dim_is_pad: Tensor | None) -> Tensor | None:
        if action_dim_is_pad is None:
            return None
        mask = ~action_dim_is_pad.to(device=target.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[-1] != target.shape[-1]:
            raise ValueError(
                f"action_dim_is_pad width {mask.shape[-1]} does not match target width {target.shape[-1]}."
            )
        if mask.shape[0] == 1 and target.shape[0] != 1:
            mask = mask.expand(target.shape[0], -1)
        if mask.shape[0] != target.shape[0]:
            raise ValueError(
                f"action_dim_is_pad batch {mask.shape[0]} does not match target batch {target.shape[0]}."
            )
        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(1)
        return mask

    @classmethod
    def _mask_action_dim_tensor(cls, tensor: Tensor, action_dim_is_pad: Tensor | None) -> Tensor:
        if not cls._mask_enabled_static(action_dim_is_pad):
            return tensor
        valid_mask = cls._action_dim_valid_mask(tensor, action_dim_is_pad)
        if valid_mask is None:
            return tensor
        return tensor.masked_fill(~valid_mask, 0)

    @staticmethod
    def _mask_enabled_static(action_dim_is_pad: Tensor | None) -> bool:
        return action_dim_is_pad is not None

    @classmethod
    def _apply_action_dim_padding_mask(cls, loss: Tensor, action_dim_is_pad: Tensor | None) -> Tensor:
        valid_mask = cls._action_dim_valid_mask(loss, action_dim_is_pad)
        if valid_mask is None:
            return loss
        valid = valid_mask.to(dtype=loss.dtype)
        denom = valid.sum(dim=-1).clamp_min(1.0)
        return (loss * valid).sum(dim=-1) / denom

    @staticmethod
    def _apply_action_chunk_padding_mask(loss: Tensor, action_horizon_is_pad: Tensor | None) -> Tensor:
        if action_horizon_is_pad is None:
            return loss
        valid_action = (
            (~action_horizon_is_pad.to(device=loss.device, dtype=torch.bool)).unsqueeze(1).unsqueeze(-1)
        )
        return loss * valid_action

    def _prepare_flow_matching_tensors(
        self,
        *,
        actions: Tensor,
        action_dim_is_pad: Tensor | None,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_expert = self._backbone()._require_action_expert()
        action_dtype = next(action_expert.parameters()).dtype
        actions = actions.to(dtype=action_dtype)
        batch_size = int(actions.shape[0])
        device = actions.device
        num_flow_timesteps = max(1, int(self.config.num_flow_timesteps))

        if timesteps is None:
            timesteps = (
                _sample_beta_timesteps(
                    batch_size=batch_size * num_flow_timesteps,
                    device=device,
                    cutoff=self.config.flow_matching_cutoff,
                    time_offset=self.config.flow_matching_time_offset,
                    time_scale=self.config.flow_matching_time_scale,
                    alpha=self.config.flow_matching_beta_alpha,
                    beta=self.config.flow_matching_beta_beta,
                )
                .to(dtype=action_dtype)
                .view(batch_size, num_flow_timesteps)
            )
        else:
            expected_timesteps_shape = (batch_size, num_flow_timesteps)
            timesteps = timesteps.to(device=device, dtype=action_dtype)
            if tuple(timesteps.shape) != expected_timesteps_shape:
                raise ValueError(
                    f"flow timesteps must have shape {expected_timesteps_shape}, got {tuple(timesteps.shape)}."
                )

        if self.config.mask_action_dim_padding:
            actions = self._mask_action_dim_tensor(actions, action_dim_is_pad)

        expected_noise_shape = (batch_size, num_flow_timesteps, actions.shape[1], actions.shape[2])
        if noise is None:
            noise = torch.randn(*expected_noise_shape, device=device, dtype=actions.dtype)
        else:
            noise = noise.to(device=device, dtype=actions.dtype)
            if tuple(noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"flow noise must have shape {expected_noise_shape}, got {tuple(noise.shape)}."
                )
        if self.config.mask_action_dim_padding:
            noise = self._mask_action_dim_tensor(noise, action_dim_is_pad)

        t_broadcast = timesteps.view(batch_size, num_flow_timesteps, 1, 1)
        actions_expanded = actions.unsqueeze(1).expand(-1, num_flow_timesteps, -1, -1)
        xt = (1.0 - t_broadcast) * noise + t_broadcast * actions_expanded
        target_velocity = actions_expanded - noise
        return actions, timesteps, xt, target_velocity

    def _prepare_joint_training_backbone_inputs(
        self,
        model_inputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor | dict[str, Any], Tensor, Tensor]:
        backbone = self._backbone()
        input_ids = model_inputs.get("input_ids")
        inputs_embeds = model_inputs.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "MolmoAct2 joint flow training requires exactly one of input_ids or inputs_embeds."
            )

        images = None
        token_pooling = None
        merge_visual_inputs = getattr(backbone, "merge_visual_inputs", None)
        if callable(merge_visual_inputs):
            images, token_pooling = merge_visual_inputs(
                input_ids=input_ids,
                pixel_values=model_inputs.get("pixel_values"),
                image_token_pooling=model_inputs.get("image_token_pooling"),
                image_grids=model_inputs.get("image_grids"),
                image_num_crops=model_inputs.get("image_num_crops"),
                pixel_values_videos=model_inputs.get("pixel_values_videos"),
                video_token_pooling=model_inputs.get("video_token_pooling"),
                video_grids=model_inputs.get("video_grids"),
            )
        elif (
            model_inputs.get("pixel_values") is not None
            or model_inputs.get("pixel_values_videos") is not None
        ):
            raise RuntimeError("MolmoAct2 checkpoint does not expose merge_visual_inputs for joint training.")

        if images is not None and inputs_embeds is not None:
            raise ValueError("MolmoAct2 joint flow training cannot combine inputs_embeds with visual inputs.")
        if inputs_embeds is None:
            inputs_embeds, _image_features = backbone.build_input_embeddings(input_ids, images, token_pooling)

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = model_inputs.get("position_ids")
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        attention_mask = model_inputs.get("attention_mask")
        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            causal_mask_mapping = backbone._build_native_attention_bias(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=model_inputs.get("token_type_ids"),
                past_key_values=None,
            )
        return inputs_embeds, causal_mask_mapping, position_ids, cache_position

    @staticmethod
    def _decoder_layer_kv_outputs(
        layer_outputs: tuple[Any, ...], *, output_attentions: bool
    ) -> tuple[Tensor, Tensor]:
        output_idx = 2 if output_attentions else 1
        return layer_outputs[output_idx], layer_outputs[output_idx + 1]

    @staticmethod
    def _action_time_conditioning(action_expert: torch.nn.Module, timesteps: Tensor) -> Tensor:
        time_conditioning = getattr(action_expert, "_time_conditioning", None)
        if callable(time_conditioning):
            return time_conditioning(timesteps)
        return action_expert.time_embed(timesteps)

    def _compute_flow_matching_loss_joint_per_layer(
        self,
        *,
        batch: dict[str, Tensor],
        model_inputs: dict[str, Tensor],
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, Tensor]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}. Expected 'mean' or 'none'.")
        backbone = self._backbone()
        transformer = getattr(backbone, "transformer", None)
        action_expert = backbone._require_action_expert()
        if transformer is None:
            raise RuntimeError("MolmoAct2 joint flow training requires a patchable text transformer.")
        if len(action_expert.blocks) != int(transformer.config.num_hidden_layers):
            raise RuntimeError(
                "MolmoAct2 joint flow training requires one action expert block per text transformer layer."
            )

        actions, timesteps, xt, target_velocity = self._prepare_flow_matching_tensors(
            actions=batch[ACTION],
            action_dim_is_pad=batch.get("action_dim_is_pad"),
            timesteps=timesteps,
            noise=noise,
        )
        num_flow_timesteps = max(1, int(self.config.num_flow_timesteps))
        batch_size = int(actions.shape[0])
        device = actions.device
        xt_flat = xt.reshape(batch_size * num_flow_timesteps, actions.shape[1], actions.shape[2])
        timesteps_flat = timesteps.reshape(batch_size * num_flow_timesteps)

        hidden_states, causal_mask_mapping, position_ids, cache_position = (
            self._prepare_joint_training_backbone_inputs(model_inputs)
        )
        if hidden_states.shape[0] != batch_size:
            raise ValueError(
                f"Backbone batch size {hidden_states.shape[0]} does not match action batch size {batch_size}."
            )

        encoder_attention_mask = self._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        action_attention_mask = None
        if batch.get("action_horizon_is_pad") is not None:
            action_attention_mask = ~batch["action_horizon_is_pad"].to(device=device, dtype=torch.bool)

        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(device=device, dtype=actions.dtype).unsqueeze(-1)
            valid_action = self._expand_mask(valid_action, num_flow_timesteps)

        rope_cache = None
        if len(action_expert.blocks) > 0 and action_expert.blocks[0].self_attn.rope is not None:
            rope_cache = action_expert.blocks[0].self_attn.rope.build_cache(
                seq_len=actions.shape[1],
                device=device,
                dtype=actions.dtype,
            )

        cross_mask = action_expert._build_cross_attention_mask(
            encoder_attention_mask,
            batch_size,
            actions.dtype,
        )
        cross_mask = self._expand_mask(cross_mask, num_flow_timesteps)
        self_mask = action_expert._build_self_attention_mask(
            action_attention_mask,
            actions.shape[1],
            device,
            actions.dtype,
        )
        self_mask = self._expand_mask(self_mask, num_flow_timesteps)

        conditioning = self._action_time_conditioning(action_expert, timesteps_flat)
        action_hidden = action_expert.action_embed(xt_flat)
        if valid_action is not None:
            action_hidden = action_hidden * valid_action

        if transformer.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": transformer.rotary_embs["default"](hidden_states, position_ids),
                "scaling": transformer.rotary_embs["scaling"](hidden_states, position_ids),
            }
        else:
            position_embeddings = transformer.rotary_emb(hidden_states, position_ids)

        use_gradient_checkpointing = bool(
            getattr(self.config, "gradient_checkpointing", False)
            and self.training
            and torch.is_grad_enabled()
        )

        def run_layer(
            layer_idx: int, layer_hidden: Tensor, layer_action_hidden: Tensor
        ) -> tuple[Tensor, Tensor]:
            decoder_block = transformer.blocks[layer_idx]
            action_block = action_expert.blocks[layer_idx]
            if transformer.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in transformer.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings

            layer_outputs = decoder_block(
                layer_hidden,
                position_embeddings=position_embeddings_i,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                collect_layer_kv_states=True,
            )
            next_hidden = layer_outputs[0]
            key_states, value_states = self._decoder_layer_kv_outputs(layer_outputs, output_attentions=False)
            key_states = backbone._cache_to_sequence(key_states)
            value_states = backbone._cache_to_sequence(value_states)
            if self.config.enable_knowledge_insulation:
                key_states = key_states.detach()
                value_states = value_states.detach()

            k_ctx = action_expert._project_kv_tensor(key_states, action_expert.context_k_proj)
            v_ctx = action_expert._project_kv_tensor(value_states, action_expert.context_v_proj)
            k_norm = action_block.cross_attn.k_norm
            if k_norm is not None:
                k_ctx = k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)
            if num_flow_timesteps != 1:
                k_ctx = self._expand_mask(k_ctx, num_flow_timesteps)
                v_ctx = self._expand_mask(v_ctx, num_flow_timesteps)

            next_action_hidden = action_block(
                layer_action_hidden,
                conditioning,
                cross_kv=(k_ctx, v_ctx),
                self_attn_mask=self_mask,
                attn_mask=cross_mask,
                is_causal=action_expert.config.causal_attn,
                modulation=None,
                rope_cache=rope_cache,
            )
            if valid_action is not None:
                next_action_hidden = next_action_hidden * valid_action
            return next_hidden, next_action_hidden

        for layer_idx in range(int(transformer.config.num_hidden_layers)):
            if use_gradient_checkpointing:
                hidden_states, action_hidden = torch.utils.checkpoint.checkpoint(
                    lambda layer_hidden, layer_action_hidden, idx=layer_idx: run_layer(
                        idx,
                        layer_hidden,
                        layer_action_hidden,
                    ),
                    hidden_states,
                    action_hidden,
                    use_reentrant=False,
                )
            else:
                hidden_states, action_hidden = run_layer(layer_idx, hidden_states, action_hidden)

        hidden_states = transformer.ln_f(hidden_states)
        pred_velocity = action_expert.final_layer(action_hidden, conditioning)
        if valid_action is not None:
            pred_velocity = pred_velocity * valid_action
        pred_velocity = pred_velocity.reshape(
            batch_size, num_flow_timesteps, actions.shape[1], actions.shape[2]
        )

        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = self._apply_action_chunk_padding_mask(loss, batch.get("action_horizon_is_pad"))
        if self.config.mask_action_dim_padding:
            loss = self._apply_action_dim_padding_mask(loss, batch.get("action_dim_is_pad"))
        loss = loss.reshape(batch_size, -1).mean(dim=1)
        if reduction == "mean":
            loss = loss.mean()
        return loss, hidden_states

    def _discrete_token_weights(self, valid_positions: Tensor) -> Tensor | None:
        mode = self.config.discrete_loss_token_weighting
        if mode in {"none", "token", "root_subsegments"}:
            return None
        if mode != "root_subsegments_root_tokens" and mode != "root_tokens":
            raise ValueError(f"Unsupported discrete_loss_token_weighting={mode!r}.")

        token_counts = valid_positions.sum(dim=1).to(dtype=torch.float32)
        example_weights = torch.zeros_like(token_counts)
        nonempty = token_counts > 0
        example_weights[nonempty] = 2.0 / torch.sqrt(token_counts[nonempty])
        return example_weights[:, None].expand_as(valid_positions)[valid_positions].to(dtype=torch.float32)

    @staticmethod
    def _weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
        if weights is None:
            return values.mean()
        weights = weights.to(device=values.device, dtype=values.dtype)
        return torch.dot(values, weights) / weights.sum().clamp_min(1.0)

    @staticmethod
    def _weighted_per_example(
        values: Tensor,
        weights: Tensor | None,
        example_indices: Tensor,
        batch_size: int,
    ) -> Tensor:
        values = values.float()
        if weights is None:
            weights = torch.ones_like(values)
        else:
            weights = weights.to(device=values.device, dtype=values.dtype)
        loss_sum = torch.zeros(batch_size, device=values.device, dtype=torch.float32)
        weight_sum = torch.zeros(batch_size, device=values.device, dtype=torch.float32)
        loss_sum.scatter_add_(0, example_indices, values * weights)
        weight_sum.scatter_add_(0, example_indices, weights)
        global_weight_sum = weight_sum.sum().clamp_min(1.0)
        return loss_sum * float(batch_size) / global_weight_sum

    def _discrete_loss_from_backbone_outputs(
        self,
        batch: dict[str, Tensor],
        outputs: Any,
        reduction: str = "mean",
    ) -> tuple[Tensor, Tensor | None]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}. Expected 'mean' or 'none'.")
        labels = batch.get("labels")
        if labels is None:
            raise RuntimeError("MolmoAct2 discrete training requires labels.")
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise RuntimeError("MolmoAct2 backbone did not return last_hidden_state.")

        ignore_index = -100
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
        valid_positions = shift_labels != ignore_index
        if not bool(valid_positions.any()):
            raise RuntimeError("MolmoAct2 discrete training labels contain no valid action tokens.")

        hidden_size = hidden_states.shape[-1]
        selected_hidden = hidden_states.reshape(-1, hidden_size)[valid_positions.reshape(-1)]
        selected_labels = shift_labels.reshape(-1)[valid_positions.reshape(-1)].to(
            device=hidden_states.device
        )
        logits = F.linear(selected_hidden, self.model.lm_head.weight).float()
        log_z = logits.logsumexp(dim=-1)
        target_logits = logits.gather(dim=-1, index=selected_labels[:, None]).squeeze(-1)
        token_ce_loss = log_z - target_logits
        token_weights = self._discrete_token_weights(valid_positions)
        if reduction == "none":
            example_indices = valid_positions.nonzero(as_tuple=False)[:, 0].to(device=hidden_states.device)
            ce_loss = self._weighted_per_example(
                token_ce_loss,
                token_weights,
                example_indices,
                int(labels.shape[0]),
            )
        else:
            ce_loss = self._weighted_mean(token_ce_loss, token_weights)
        if not self.config.softmax_auxiliary_loss:
            return ce_loss, None

        if reduction == "none":
            z_loss = self.config.softmax_auxiliary_loss_scale * self._weighted_per_example(
                log_z.pow(2),
                token_weights,
                example_indices,
                int(labels.shape[0]),
            )
        else:
            z_loss = self.config.softmax_auxiliary_loss_scale * self._weighted_mean(
                log_z.pow(2), token_weights
            )
        return ce_loss, z_loss

    @staticmethod
    def _extract_discrete_token_bins(
        generated_ids: list[int],
        start_token_id: int,
        end_token_id: int,
        token_id_to_bin: dict[int, int],
    ) -> list[int]:
        start_idx = None
        end_idx = None
        for idx, token_id in enumerate(generated_ids):
            if token_id == start_token_id:
                start_idx = idx
                break
        if start_idx is not None:
            for idx in range(start_idx + 1, len(generated_ids)):
                if generated_ids[idx] == end_token_id:
                    end_idx = idx
                    break
        span_start = 0 if start_idx is None else start_idx + 1
        span_end = len(generated_ids) if end_idx is None else end_idx
        return [
            int(token_id_to_bin[token_id])
            for token_id in generated_ids[span_start:span_end]
            if token_id in token_id_to_bin
        ]

    def _action_token_id_to_bin(self) -> dict[int, int]:
        method = getattr(self.model, "_action_token_id_to_bin", None)
        if callable(method):
            return dict(method())
        start = getattr(self.model.config, "action_token_start_id", None)
        num_tokens = int(getattr(self.model.config, "num_action_tokens", 0) or 0)
        if start is None or num_tokens <= 0:
            return {}
        return {int(start) + idx: idx for idx in range(num_tokens)}

    def _require_discrete_eos_token_id(self) -> int:
        method = getattr(self.model, "_require_eos_token_id", None)
        if callable(method):
            return int(method())
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if eos_token_id is None and getattr(self.model, "generation_config", None) is not None:
            eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is None:
            raise RuntimeError("Discrete action generation requires eos_token_id in the checkpoint config.")
        return int(eos_token_id)

    def _discrete_generation_max_steps(self) -> int:
        if self.config.discrete_generation_max_steps is not None:
            return int(self.config.discrete_generation_max_steps)
        return max(1, self._generation_action_horizon() * 16)

    def _continue_discrete_generation_from_output(
        self,
        initial_output: Any,
        *,
        past_key_values: Any | None,
        attention_mask: Tensor | None,
        end_token_id: int,
        max_steps: int,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        consume_generation_tokens = getattr(self.model, "_consume_generation_tokens", None)
        ar_decode_step = getattr(self.model, "_run_ar_decode_step", None)
        if ar_decode_step is None:
            ar_decode_step = getattr(self.model, "_run_depth_decode_step", None)
        if attention_bias is None and not callable(consume_generation_tokens):
            raise RuntimeError("MolmoAct2 checkpoint does not expose discrete token generation helpers.")
        if attention_bias is not None and not callable(ar_decode_step):
            raise RuntimeError("MolmoAct2 checkpoint does not expose graph-backed AR decode helpers.")

        generated_tokens: list[Tensor] = []
        current_output = initial_output
        current_past_key_values = past_key_values
        current_attention_mask = attention_mask
        hit_end = False
        for _ in range(int(max_steps)):
            next_token = torch.argmax(current_output.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token)
            if bool((next_token == int(end_token_id)).all()):
                hit_end = True
                break
            if attention_bias is None:
                current_output, current_attention_mask = consume_generation_tokens(
                    next_token,
                    past_key_values=current_past_key_values,
                    attention_mask=current_attention_mask,
                )
                current_past_key_values = current_output.past_key_values
            else:
                last_hidden, current_past_key_values = ar_decode_step(
                    next_token,
                    past_key_values=current_past_key_values,
                    attention_bias=attention_bias,
                )
                current_output = types.SimpleNamespace(
                    logits=self.model.lm_head(last_hidden),
                    past_key_values=current_past_key_values,
                )
        if not generated_tokens:
            raise RuntimeError("Discrete continuation generated no tokens.")
        if not hit_end:
            raise RuntimeError(
                f"Discrete continuation did not emit end token {int(end_token_id)} within {int(max_steps)} steps."
            )
        return torch.stack(generated_tokens, dim=1)

    def _make_discrete_ar_graph_decode_inputs(
        self,
        model_inputs: dict[str, Tensor],
        *,
        max_steps: int,
    ) -> tuple[Any | None, Tensor | None]:
        if not bool(getattr(self.config, "enable_inference_cuda_graph", False)):
            return None, None
        if self.training or self.model.training:
            return None, None
        ar_decode_step = getattr(self.model, "_run_ar_decode_step", None)
        if ar_decode_step is None:
            ar_decode_step = getattr(self.model, "_run_depth_decode_step", None)
        make_attention_bias = getattr(self.model, "_make_depth_decode_attention_bias", None)
        if not callable(ar_decode_step) or not callable(make_attention_bias):
            return None, None

        make_static_cache = getattr(self.model, "_make_ar_decode_static_cache", None)
        if callable(make_static_cache):
            static_cache = make_static_cache(model_inputs, max_steps=max_steps)
        else:
            graph_manager = getattr(self.model, "depth_decode_cuda_graph_manager", None)
            make_manager_static_cache = getattr(graph_manager, "make_static_cache", None)
            if not callable(make_manager_static_cache):
                return None, None
            prompt_len = int(model_inputs["input_ids"].shape[1])
            static_cache = make_manager_static_cache(max_cache_len=prompt_len + max(1, int(max_steps)))

        attention_bias = make_attention_bias(model_inputs, static_cache)
        return static_cache, attention_bias

    def _decode_discrete_action_chunk(self, generated_token_ids: Tensor, *, action_dim: int) -> Tensor:
        if (
            getattr(self.model.config, "action_start_token_id", None) is None
            or getattr(self.model.config, "action_end_token_id", None) is None
        ):
            raise RuntimeError("Discrete action generation requires <action_start>/<action_end> token IDs.")
        token_id_to_bin = self._action_token_id_to_bin()
        if not token_id_to_bin:
            raise RuntimeError(
                "Discrete action generation requires indexed action tokens in the checkpoint config."
            )

        action_tokenizer = self._load_discrete_action_tokenizer()
        if generated_token_ids.ndim == 1:
            generated_token_ids = generated_token_ids.unsqueeze(0)
        if generated_token_ids.ndim == 3:
            generated_token_ids = generated_token_ids[:, 0, :]
        if generated_token_ids.ndim != 2:
            raise ValueError(f"Unexpected generated token tensor shape {tuple(generated_token_ids.shape)}.")

        chunks: list[Tensor] = []
        for token_row in generated_token_ids:
            generated_ids = [int(token_id) for token_id in token_row.detach().cpu().tolist()]
            discrete_token_ids = self._extract_discrete_token_bins(
                generated_ids,
                int(self.model.config.action_start_token_id),
                int(self.model.config.action_end_token_id),
                token_id_to_bin,
            )
            if not discrete_token_ids:
                raise RuntimeError(
                    "Model generated no decodable action tokens between <action_start>/<action_end>."
                )
            try:
                decoded = action_tokenizer.decode(
                    [discrete_token_ids],
                    time_horizon=self._generation_action_horizon(),
                    action_dim=int(action_dim),
                )
            except TypeError:
                decoded = action_tokenizer.decode([discrete_token_ids])
            action_chunk = np.asarray(decoded, dtype=np.float32)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[None, :]
            elif action_chunk.ndim == 3:
                if int(action_chunk.shape[0]) != 1:
                    action_chunk = action_chunk.reshape(action_chunk.shape[-2], action_chunk.shape[-1])
                else:
                    action_chunk = action_chunk[0]
            elif action_chunk.ndim > 3:
                action_chunk = action_chunk.reshape(action_chunk.shape[-2], action_chunk.shape[-1])
            if action_chunk.ndim != 2:
                raise RuntimeError(f"Decoded action chunk has unexpected shape {action_chunk.shape}.")
            chunks.append(torch.as_tensor(action_chunk, device=token_row.device, dtype=torch.float32))
        return torch.stack(chunks, dim=0)

    def _generate_discrete_actions_from_inputs(
        self,
        *,
        model_inputs: dict[str, Tensor],
        action_dim: int,
    ) -> Tensor:
        model_inputs = self._drop_trivial_attention_mask(model_inputs)
        max_steps = self._discrete_generation_max_steps()
        static_cache, attention_bias = self._make_discrete_ar_graph_decode_inputs(
            model_inputs,
            max_steps=max_steps,
        )
        prefill_kwargs: dict[str, Any] = {}
        if static_cache is not None:
            prefill_kwargs["past_key_values"] = static_cache
        prefill_output = self.model(
            **model_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            **prefill_kwargs,
        )
        generated_token_ids = self._continue_discrete_generation_from_output(
            prefill_output,
            past_key_values=prefill_output.past_key_values,
            attention_mask=model_inputs.get("attention_mask"),
            end_token_id=self._require_discrete_eos_token_id(),
            max_steps=max_steps,
            attention_bias=attention_bias,
        )
        return self._decode_discrete_action_chunk(generated_token_ids, action_dim=action_dim)

    def _generate_actions_from_inputs_with_rtc(
        self,
        *,
        model_inputs: dict[str, Tensor],
        action_dim_is_pad: Tensor | None,
        num_steps: int | None,
        generator: torch.Generator | None,
        inference_delay: int | None,
        prev_chunk_left_over: Tensor | None,
        execution_horizon: int | None,
    ) -> Tensor:
        backbone = self._backbone()
        action_expert = self._action_expert()
        outputs = backbone(
            **model_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        encoder_kv_states = backbone._extract_kv_states(outputs.past_key_values)
        encoder_attention_mask = self._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        depth_gate, depth_mask = backbone._depth_gate_from_condition(
            input_ids=model_inputs.get("input_ids"),
            encoder_attention_mask=encoder_attention_mask,
            layer_kv_states=encoder_kv_states,
        )
        encoder_kv_states = backbone._apply_depth_gate_to_layer_kv_states(
            encoder_kv_states,
            depth_mask,
            depth_gate,
        )

        steps = int(num_steps or backbone.config.flow_matching_num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {steps}.")
        source_tensor = encoder_kv_states[0][0]
        batch_size = int(source_tensor.shape[0])
        device = source_tensor.device
        trajectory = torch.randn(
            batch_size,
            self._generation_action_horizon(),
            int(backbone.config.max_action_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        if self.config.mask_action_dim_padding:
            trajectory = self._mask_action_dim_tensor(trajectory, action_dim_is_pad)

        action_context = action_expert.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            state_embeddings=None,
            batch_size=batch_size,
            seq_len=trajectory.shape[1],
            device=device,
            dtype=trajectory.dtype,
        )
        flow_timesteps = [
            torch.full((batch_size,), idx / steps, device=device, dtype=trajectory.dtype)
            for idx in range(steps)
        ]
        modulation_cache = action_expert.get_or_prepare_modulation_cache(
            flow_timesteps,
            cache_key=(steps, batch_size, device, trajectory.dtype),
        )

        dt = 1.0 / steps
        mask_enabled = self.config.mask_action_dim_padding
        for idx, flow_timestep in enumerate(flow_timesteps):
            modulation = modulation_cache[idx]

            def denoise_step(input_trajectory: Tensor, step_modulation=modulation) -> Tensor:
                velocity = action_expert.forward_with_context(
                    input_trajectory,
                    step_modulation.conditioning,
                    context=action_context,
                    modulation=step_modulation,
                )
                if mask_enabled:
                    velocity = self._mask_action_dim_tensor(velocity, action_dim_is_pad)
                return velocity

            if self._rtc_enabled():
                if self.rtc_processor is None:
                    raise RuntimeError("RTC is enabled but rtc_processor is not initialized.")

                def rtc_denoise_step(input_trajectory: Tensor) -> Tensor:
                    return -denoise_step(input_trajectory)

                rtc_time = 1.0 - float(flow_timestep[0].item())
                rtc_velocity = self.rtc_processor.denoise_step(
                    x_t=trajectory,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=int(inference_delay or 0),
                    time=rtc_time,
                    original_denoise_step_partial=rtc_denoise_step,
                    execution_horizon=execution_horizon,
                )
                velocity = -rtc_velocity
            else:
                velocity = denoise_step(trajectory)

            trajectory = trajectory + dt * velocity
            if mask_enabled:
                trajectory = self._mask_action_dim_tensor(trajectory, action_dim_is_pad)
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=float(flow_timestep[0].item()), x_t=trajectory, v_t=velocity)

        return trajectory

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}. Expected 'mean' or 'none'.")
        model_inputs = self._model_inputs(batch)
        losses: list[Tensor] = []
        metrics: dict[str, Any] = {}

        if self.config.action_mode == "discrete":
            outputs = self._backbone()(
                **model_inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            discrete_ce_loss, discrete_z_loss = self._discrete_loss_from_backbone_outputs(
                batch, outputs, reduction=reduction
            )
            discrete_loss = (
                discrete_ce_loss if discrete_z_loss is None else discrete_ce_loss + discrete_z_loss
            )
            losses.append(discrete_loss)
            metrics["discrete_ce_loss"] = discrete_ce_loss.detach().float().mean().item()
            if discrete_z_loss is not None:
                metrics["discrete_z_loss"] = discrete_z_loss.detach().float().mean().item()

        elif self.config.action_mode == "continuous":
            flow_loss, _ = self._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                reduction=reduction,
            )
            losses.append(flow_loss)
            metrics["action_flow_loss"] = flow_loss.detach().float().mean().item()

        else:
            flow_loss, hidden_states = self._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                reduction=reduction,
            )
            outputs = types.SimpleNamespace(last_hidden_state=hidden_states)
            discrete_ce_loss, discrete_z_loss = self._discrete_loss_from_backbone_outputs(
                batch, outputs, reduction=reduction
            )
            discrete_loss = (
                discrete_ce_loss if discrete_z_loss is None else discrete_ce_loss + discrete_z_loss
            )
            losses.append(discrete_loss)
            metrics["discrete_ce_loss"] = discrete_ce_loss.detach().float().mean().item()
            if discrete_z_loss is not None:
                metrics["discrete_z_loss"] = discrete_z_loss.detach().float().mean().item()
            losses.append(flow_loss)
            metrics["action_flow_loss"] = flow_loss.detach().float().mean().item()

        loss = torch.stack(losses).sum(dim=0)
        metrics["loss"] = loss.detach().float().mean().item()
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if "action_mode" in kwargs:
            raise TypeError(
                "MolmoAct2 predict_action_chunk got unexpected keyword argument 'action_mode'; "
                "use 'inference_action_mode'."
            )
        model_inputs = self._model_inputs(batch)
        inference_action_mode = self._resolve_inference_action_mode(kwargs.get("inference_action_mode"))
        num_steps = kwargs.get("num_steps", getattr(self.config, "num_inference_steps", None))
        generator = kwargs.get("generator")
        model_dtype = _torch_dtype(self.config.model_dtype)
        device = next(self.parameters()).device
        batch_size = int(next(iter(model_inputs.values())).shape[0])
        if generator is None:
            generator = self._rollout_generator_for_inputs(
                batch,
                batch_size=batch_size,
                device=device,
            )
        action_dim = self._output_action_dim(batch)
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=model_dtype)
            if device.type in {"cuda", "cpu"} and model_dtype in {torch.bfloat16, torch.float16}
            else nullcontext()
        )
        with autocast_context:
            if inference_action_mode == "discrete":
                if self._rtc_enabled():
                    raise ValueError("RTC is only supported for continuous MolmoAct2 inference.")
                actions = self._generate_discrete_actions_from_inputs(
                    model_inputs=model_inputs,
                    action_dim=action_dim,
                )
            elif self._rtc_enabled():
                actions = self._generate_actions_from_inputs_with_rtc(
                    model_inputs=model_inputs,
                    action_dim_is_pad=batch.get("action_dim_is_pad"),
                    num_steps=num_steps,
                    generator=generator,
                    inference_delay=kwargs.get("inference_delay"),
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                actions = self._backbone().generate_actions_from_inputs(
                    **model_inputs,
                    action_dim_is_pad=batch.get("action_dim_is_pad"),
                    action_horizon=self._generation_action_horizon(),
                    num_steps=num_steps,
                    generator=generator,
                )
        return actions[:, : self.config.n_action_steps, :action_dim].to(dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if self._rtc_enabled():
            raise AssertionError("RTC is not supported for select_action, use it with predict_action_chunk")
        batch_size = int(next(iter(self._model_inputs(batch).values())).shape[0])
        actions: list[Tensor] = []
        for batch_idx in range(batch_size):
            queue = self._action_queues[batch_idx]
            if not queue:
                chunk = self.predict_action_chunk(batch, **kwargs)
                for step in torch.unbind(chunk[batch_idx], dim=0):
                    queue.append(step)
            if not queue:
                raise RuntimeError("MolmoAct2 produced an empty action chunk.")
            actions.append(queue.popleft())
        return torch.stack(actions, dim=0)

    def _get_default_peft_targets(self) -> dict[str, Any]:
        target_modules = self._lora_target_modules(prefix=r"model\.model")
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
            "r": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "bias": self.config.lora_bias,
        }

    def _get_inner_peft_targets(self) -> dict[str, Any]:
        target_modules = self._lora_target_modules(prefix="model")
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
            "r": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "bias": self.config.lora_bias,
        }

    def _lora_target_modules(self, *, prefix: str) -> str:
        vlm_linear_leaves = "w1|w2|w3|wq|wk|wv|wo|att_proj|attn_out|ff_proj|ff_out|patch_embedding"
        target_modules = rf"{prefix}\.(transformer|vision_backbone)\.(?:.*\.)?({vlm_linear_leaves})$"
        if self.config.enable_lora_action_expert:
            action_expert_linear_paths = (
                r"time_embed\.(1|3)|"
                r"action_embed|context_k_proj|context_v_proj|"
                r"blocks\.\d+\.self_attn\.(qkv|out_proj)|"
                r"blocks\.\d+\.cross_attn\.(q_proj|out_proj)|"
                r"blocks\.\d+\.mlp\.(up_proj|gate_proj|down_proj)|"
                r"blocks\.\d+\.modulation\.linear|"
                r"final_layer\.(modulation\.linear|linear)"
            )
            target_modules = (
                f"({target_modules}|"
                rf"{prefix}\.action_expert\.({action_expert_linear_paths})$)"
            )
        return target_modules

    def _build_inner_lora_config(self):
        require_package("peft", extra="molmoact2")
        from peft import LoraConfig

        return LoraConfig(**self._get_inner_peft_targets())

    def _apply_lora_adapters(self) -> None:
        require_package("peft", extra="molmoact2")
        from peft import get_peft_model

        peft_config = self._build_inner_lora_config()
        self._validate_peft_config(peft_config)

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model = get_peft_model(self.model, peft_config)
        if not self.config.enable_lora_action_expert:
            self._unfreeze_action_expert_parameters()
        self.train(self.training)

    def _validate_peft_config(self, peft_config) -> None:
        del peft_config
        if not self.config.checkpoint_path:
            raise ValueError("MolmoAct2 LoRA fine-tuning requires `policy.checkpoint_path`.")
