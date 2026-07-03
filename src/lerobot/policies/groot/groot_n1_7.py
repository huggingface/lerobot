#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

import logging
from contextlib import suppress
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from torch import nn
from torch.distributions import Beta

from lerobot.utils.import_utils import _transformers_available, require_package

from .action_head.cross_attention_dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from .configuration_groot import N1_7_DEFAULT_IMAGE_CROP_SIZE, N1_7_DEFAULT_IMAGE_TARGET_SIZE

if TYPE_CHECKING or _transformers_available:
    from transformers import (
        AutoConfig,
        AutoModel,
        PretrainedConfig,
        PreTrainedModel,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
    )
    from transformers.feature_extraction_utils import BatchFeature
else:
    AutoConfig = None
    AutoModel = None
    PretrainedConfig = object
    PreTrainedModel = object
    BatchFeature = None
    Qwen3VLConfig = None
    Qwen3VLForConditionalGeneration = None

try:
    import tree
except ImportError:
    tree = None

logger = logging.getLogger(__name__)


def _tie_unused_qwen_lm_head(model: nn.Module) -> None:
    """Restore the TF4 weight tie so the unused LM head stays frozen and is omitted on save."""
    lm_head = getattr(model, "lm_head", None)
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if lm_head is None or not callable(get_input_embeddings):
        return
    input_embeddings = get_input_embeddings()
    embedding_weight = getattr(input_embeddings, "weight", None)
    if embedding_weight is None:
        return
    lm_head.weight = embedding_weight


GR00T_N1_7_DEFAULTS: dict[str, Any] = {
    "model_dtype": "bfloat16",
    "dtype": "bfloat16",
    "model_name": "nvidia/Cosmos-Reason2-2B",
    "backbone_model_type": "qwen",
    "model_revision": None,
    "tune_top_llm_layers": 0,
    "backbone_embedding_dim": 2048,
    "tune_llm": False,
    "tune_visual": False,
    "select_layer": 16,
    "reproject_vision": False,
    "use_flash_attention": False,
    "load_bf16": False,
    "backbone_trainable_params_fp32": True,
    "image_crop_size": N1_7_DEFAULT_IMAGE_CROP_SIZE,
    "image_target_size": N1_7_DEFAULT_IMAGE_TARGET_SIZE,
    "shortest_image_edge": None,
    "crop_fraction": None,
    "random_rotation_angle": None,
    "color_jitter_params": None,
    "use_albumentations_transforms": True,
    "extra_augmentation_config": None,
    "formalize_language": True,
    "apply_sincos_state_encoding": False,
    "use_percentiles": True,
    "use_relative_action": False,
    "max_state_dim": 132,
    "max_action_dim": 132,
    "action_horizon": 40,
    "hidden_size": 1024,
    "input_embedding_dim": 1536,
    "state_history_length": 1,
    "add_pos_embed": True,
    "attn_dropout": 0.2,
    "use_vlln": True,
    "max_seq_len": 1024,
    "use_alternate_vl_dit": True,
    "attend_text_every_n_blocks": 2,
    "diffusion_model_cfg": {
        "positional_embeddings": None,
        "num_layers": 32,
        "num_attention_heads": 32,
        "attention_head_dim": 48,
        "norm_type": "ada_norm",
        "dropout": 0.2,
        "final_dropout": True,
        "output_dim": 1024,
        "interleave_self_attention": True,
    },
    "vl_self_attention_cfg": {
        "positional_embeddings": None,
        "num_layers": 4,
        "num_attention_heads": 32,
        "attention_head_dim": 64,
        "dropout": 0.2,
        "final_dropout": True,
    },
    "num_inference_timesteps": 4,
    "noise_beta_alpha": 1.5,
    "noise_beta_beta": 1.0,
    "noise_s": 0.999,
    "num_timestep_buckets": 1000,
    "tune_projector": True,
    "tune_diffusion_model": True,
    "tune_vlln": True,
    "state_dropout_prob": 0.2,
    "exclude_state": False,
    "use_mean_std": False,
    "max_num_embodiments": 32,
    "rtc_ramp_rate": 6.0,
}


class GR00TN17Config(PretrainedConfig):
    """Configuration for NVIDIA GR00T N1.7.

    N1.7 uses the Cosmos-Reason2-2B / Qwen3-VL backbone and a multi-embodiment
    flow-matching action head. This mirrors the public N1.7 checkpoint config
    while keeping it local to LeRobot and independent from the external
    Isaac-GR00T ``gr00t`` Python package.
    """

    model_type = "Gr00tN1d7"

    _defaults = GR00T_N1_7_DEFAULTS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in GR00T_N1_7_DEFAULTS.items():
            setattr(self, key, deepcopy(kwargs.pop(key, value)))
        for key, value in kwargs.items():
            setattr(self, key, value)


class CategorySpecificLinear(nn.Module):
    """Linear layer with category-specific weights for multi-embodiment support."""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_w = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with category-specific weights."""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of shape ``(B, T, D)`` for timestep tensors ``(B, T)``.

    The frequency scalar is intentionally created on CPU and then broadcast with
    the device-local arange result. That mirrors Isaac-GR00T's N1.7 timestep
    embedding and avoids tiny dtype/device construction differences in parity
    tests.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class MultiEmbodimentActionEncoder(nn.Module):
    """Action encoder with category-specific projections and sinusoidal time encoding."""

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.dim() != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("Expected `timesteps` to have shape (B,).")
        timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
        action_emb = self.W1(actions, cat_ids)
        time_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        x = swish(self.W2(torch.cat([action_emb, time_emb], dim=-1), cat_ids))
        return self.W3(x, cat_ids)


class Qwen3Backbone(nn.Module):
    """Cosmos-Reason2/Qwen3-VL backbone used by GR00T N1.7.

    The public checkpoint stores the action head in the GR00T checkpoint but
    uses a Hugging Face Qwen3-VL-compatible backbone interface. This wrapper
    keeps the nested HF module layout compatible across transformer versions
    and exposes the hidden states consumed by the action head.
    """

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict[str, Any] | None = None,
        load_pretrained_weights: bool = True,
    ):
        require_package("transformers", extra="groot")
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is required for GR00T N1.7. "
                "Install a transformers version with Qwen3-VL support."
            )
        super().__init__()
        transformers_loading_kwargs = transformers_loading_kwargs or {"trust_remote_code": True}

        extra_kwargs: dict[str, Any] = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning("flash_attn is not installed. Falling back to SDPA attention.")
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        if load_pretrained_weights:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                **extra_kwargs,
                **transformers_loading_kwargs,
            ).eval()
        else:
            self.model = self._from_backbone_config(
                model_name=model_name,
                model_kwargs=extra_kwargs,
                config_kwargs=transformers_loading_kwargs,
            ).eval()

        _tie_unused_qwen_lm_head(self.model)
        while len(self.language_model.layers) > select_layer:
            self.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            for parameter in self.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(torch.float32)

    def set_trainable_parameters(
        self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int = 0
    ) -> None:
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for parameter in self.parameters():
            parameter.requires_grad = True
        if not tune_llm:
            self.language_model.requires_grad_(False)
        if not tune_visual:
            self.visual.requires_grad_(False)
        if tune_top_llm_layers > 0:
            for layer in self.language_model.layers[-tune_top_llm_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def set_frozen_modules_to_eval_mode(self) -> None:
        if self.training:
            if self.language_model and not self.tune_llm:
                self.language_model.eval()
            if self.visual and not self.tune_visual:
                self.visual.eval()

    @property
    def language_model(self) -> nn.Module:
        return getattr(self.model, "model", self.model).language_model

    @property
    def visual(self) -> nn.Module:
        return getattr(self.model, "model", self.model).visual

    def _from_backbone_config(
        self,
        *,
        model_name: str,
        model_kwargs: dict[str, Any],
        config_kwargs: dict[str, Any],
    ) -> nn.Module:
        if _is_cosmos_reason2_backbone(model_name):
            backbone_config = _cosmos_reason2_qwen3_vl_config()
        else:
            backbone_config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        return Qwen3VLForConditionalGeneration._from_config(backbone_config, **model_kwargs)

    def prepare_input(self, batch: dict[str, Any]) -> BatchFeature:
        return BatchFeature(data=batch)

    def _ensure_mm_token_type_ids(self, model_input: dict[str, torch.Tensor]) -> None:
        if "mm_token_type_ids" in model_input:
            return
        if "image_grid_thw" not in model_input and "video_grid_thw" not in model_input:
            return

        input_ids = model_input.get("input_ids")
        if input_ids is None:
            return

        mm_token_type_ids = torch.zeros(input_ids.shape, dtype=torch.int32, device=input_ids.device)
        image_token_id = getattr(self.model.config, "image_token_id", None)
        video_token_id = getattr(self.model.config, "video_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2

        model_input["mm_token_type_ids"] = mm_token_type_ids

    def _ensure_legacy_qwen3_position_ids(self, model_input: dict[str, torch.Tensor]) -> None:
        """Restore the Qwen3-VL text position ids used by older Transformers releases.

        Transformers 5.x computes 3-row multimodal RoPE ids for Qwen3-VL and then
        drops text position ids before calling text-layer flash attention. GR00T
        N1.7 was aligned against the older Transformers path, where a fourth text
        position row is forwarded alongside the temporal/height/width rows. Adding
        the row here preserves the newer multimodal position computation while
        keeping flash attention on the legacy code path.
        """

        if "position_ids" in model_input:
            return

        qwen3_model = getattr(self.model, "model", self.model)
        compute_3d_position_ids = getattr(qwen3_model, "compute_3d_position_ids", None)
        if compute_3d_position_ids is None:
            return

        position_ids = compute_3d_position_ids(
            input_ids=model_input.get("input_ids"),
            image_grid_thw=model_input.get("image_grid_thw"),
            video_grid_thw=model_input.get("video_grid_thw"),
            inputs_embeds=None,
            attention_mask=model_input.get("attention_mask"),
            past_key_values=None,
            mm_token_type_ids=model_input.get("mm_token_type_ids"),
        )
        if position_ids.ndim == 3 and position_ids.shape[0] == 3:
            position_ids = torch.cat([position_ids[:1], position_ids], dim=0)

        model_input["position_ids"] = position_ids

    def _last_decoder_layer_output(self, model_input: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the pre-final-norm decoder output consumed by the N1.7 action head.

        Older Transformers releases exposed this tensor as ``hidden_states[-1]``.
        Newer releases expose the post-final-norm tensor there instead. Capturing
        the last decoder layer output directly keeps the N1.7 action head input
        stable across Transformers versions.
        """

        captured: dict[str, torch.Tensor] = {}

        def capture_output(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                captured["features"] = output
            elif isinstance(output, (tuple, list)) and output:
                captured["features"] = output[0]
            elif hasattr(output, "last_hidden_state"):
                captured["features"] = output.last_hidden_state

        hook = self.language_model.layers[-1].register_forward_hook(capture_output)
        try:
            outputs = self.model(**model_input, output_hidden_states=True)
        finally:
            hook.remove()

        return captured.get("features", outputs.hidden_states[-1])

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        optional_keys = ["mm_token_type_ids", "pixel_values_videos", "video_grid_thw"]
        model_input = {key: vl_input[key] for key in keys_to_use}
        model_input.update({key: vl_input[key] for key in optional_keys if key in vl_input})
        self._ensure_mm_token_type_ids(model_input)
        self._ensure_legacy_qwen3_position_ids(model_input)
        features = self._last_decoder_layer_output(model_input)
        image_mask = model_input["input_ids"] == self.model.config.image_token_id
        attention_mask = model_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )


class GR00TN17ActionHead(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self, config: GR00TN17Config):
        require_package("diffusers", extra="groot")
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.state_dropout_prob = config.state_dropout_prob
        self._noise_beta_alpha = config.noise_beta_alpha
        self._noise_beta_beta = config.noise_beta_beta
        self._beta_dist = None
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model, config.tune_vlln)

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ) -> None:
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for parameter in self.parameters():
            parameter.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self) -> None:
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._beta_dist is None:
            beta_alpha = torch.tensor(self._noise_beta_alpha, device="cpu", dtype=torch.float32)
            beta_beta = torch.tensor(self._noise_beta_beta, device="cpu", dtype=torch.float32)
            self._beta_dist = Beta(beta_alpha, beta_beta, validate_args=False)
        sample = self._beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (1 - sample) * self.config.noise_s

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = self.vlln(backbone_output["backbone_features"])
        backbone_output["backbone_features"] = self.vl_self_attention(backbone_features)
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        embodiment_id = action_input.embodiment_id

        if action_input.state.shape[1] != self.config.state_history_length:
            raise ValueError("state history length does not match GR00T N1.7 config.")
        state = action_input.state.view(action_input.state.shape[0], 1, -1)
        state_features = self.state_encoder(state, embodiment_id)

        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device) < self.state_dropout_prob
            )
            state_features = state_features * (1 - do_dropout[:, None, None].to(dtype=state_features.dtype))

        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((state_features, action_features), dim=1)
        if self.config.use_alternate_vl_dit:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        return BatchFeature(
            data={
                "loss": loss,
                "action_loss": action_loss,
                "action_mask": action_mask,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    def _encode_features(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        state = action_input.state
        if state.shape[1] != self.config.state_history_length:
            raise ValueError("state history length does not match GR00T N1.7 config.")
        state = state.view(state.shape[0], 1, -1)
        state_features = self.state_encoder(state, action_input.embodiment_id)
        return BatchFeature(
            data={"backbone_features": backbone_output.backbone_features, "state_features": state_features}
        )

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )
        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            if options is None:
                raise ValueError("RTC options are required when action is provided to get_action.")
            action_horizon_before_padding = options["action_horizon"]
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)
            vel_strength[:, options["rtc_frozen_steps"] : options["rtc_overlap_steps"], :] = ramp[1:-1][
                None, :, None
            ].to(device)

        for t_step in range(self.num_inference_timesteps):
            t_cont = t_step / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = torch.cat((state_features, action_features), dim=1)

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)
            actions = actions + dt * pred[:, -self.action_horizon :] * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict[str, Any]) -> BatchFeature:
        return BatchFeature(data=batch)


def _is_cosmos_reason2_backbone(model_name: str) -> bool:
    return str(model_name).rstrip("/") == "nvidia/Cosmos-Reason2-2B"


def _cosmos_reason2_qwen3_vl_config() -> PretrainedConfig:
    """Hard-coded copy of the nvidia/Cosmos-Reason2-2B config.json (a Qwen3-VL-2B-Instruct layout)."""

    return Qwen3VLConfig(
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=True,
        text_config={
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "dtype": "bfloat16",
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
            "rope_theta": 5000000,
            "tie_word_embeddings": True,
            "use_cache": True,
            "vocab_size": 151936,
        },
        vision_config={
            "deepstack_visual_indexes": [5, 11, 17],
            "depth": 24,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1024,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "model_type": "qwen3_vl",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    )


def get_backbone_cls(config: GR00TN17Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        return Qwen3Backbone
    if config.backbone_model_type == "qwen":
        logger.warning(
            "Unrecognized GR00T N1.7 backbone model name '%s'; assuming a Qwen3-VL-compatible "
            "backbone because backbone_model_type='qwen'.",
            config.model_name,
        )
        return Qwen3Backbone
    raise ValueError(f"Unsupported GR00T N1.7 backbone model: {config.model_name}")


class GR00TN17(PreTrainedModel):
    """GR00T N1.7 model with a Cosmos-Reason2/Qwen3-VL backbone."""

    config_class = GR00TN17Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: GR00TN17Config,
        transformers_loading_kwargs: dict[str, Any] | None = None,
        load_backbone_weights: bool = True,
    ):
        _register_with_transformers()
        super().__init__(config)
        transformers_loading_kwargs = transformers_loading_kwargs or {"trust_remote_code": True}
        self.config = config
        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
            load_pretrained_weights=load_backbone_weights,
        )
        self.action_head = GR00TN17ActionHead(config)
        self.post_init()

    def prepare_input(self, inputs: dict[str, Any]) -> tuple[BatchFeature, BatchFeature]:
        require_package("dm-tree", extra="groot", import_name="tree")
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_dtype(x):
            if not isinstance(x, torch.Tensor):
                return x
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            return x.to(self.device)

        return (
            tree.map_structure(to_device_with_dtype, backbone_inputs),
            tree.map_structure(to_device_with_dtype, action_inputs),
        )

    def forward(self, inputs: dict[str, Any]) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        return self.action_head(backbone_outputs, action_inputs)

    def get_action(self, inputs: dict[str, Any], options: dict[str, Any] | None = None) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        return self.action_head.get_action(backbone_outputs, action_inputs, options)

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        tune_vlln = kwargs.pop("tune_vlln", True)
        transformers_loading_kwargs = kwargs.pop("transformers_loading_kwargs", None) or {
            "trust_remote_code": True
        }
        load_backbone_weights = kwargs.pop("load_backbone_weights", False)
        for key in ("cache_dir", "local_files_only", "token"):
            if key in kwargs:
                transformers_loading_kwargs.setdefault(key, kwargs[key])

        try:
            local_model_path = snapshot_download(
                pretrained_model_name_or_path,
                repo_type="model",
                revision=kwargs.get("revision"),
                cache_dir=kwargs.get("cache_dir"),
                local_files_only=kwargs.get("local_files_only", False),
                token=kwargs.get("token"),
            )
        except (HFValidationError, RepositoryNotFoundError):
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(
            local_model_path,
            transformers_loading_kwargs=transformers_loading_kwargs,
            load_backbone_weights=load_backbone_weights,
            **kwargs,
        )
        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual,
            tune_llm=tune_llm,
            tune_top_llm_layers=pretrained_model.config.tune_top_llm_layers,
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            tune_vlln=tune_vlln,
        )
        return pretrained_model


def _register_with_transformers() -> None:
    """Register GR00T N1.7 with transformers' Auto* factories.

    Idempotent: ``register(..., exist_ok=True)`` makes repeat calls no-ops (with a fallback that
    suppresses the already-registered error on transformers builds whose ``register()`` predates
    ``exist_ok``), so no run-once guard is needed.
    """
    if AutoConfig is None or AutoModel is None:
        return
    try:
        AutoConfig.register(GR00TN17Config.model_type, GR00TN17Config, exist_ok=True)
    except TypeError:
        with suppress(ValueError):
            AutoConfig.register(GR00TN17Config.model_type, GR00TN17Config)
    try:
        AutoModel.register(GR00TN17Config, GR00TN17, exist_ok=True)
    except TypeError:
        with suppress(ValueError):
            AutoModel.register(GR00TN17Config, GR00TN17)
