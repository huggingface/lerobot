#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub.constants import CONFIG_NAME

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

ACTION_MODE_CHOICES = {"absolute", "relative"}
_LEGACY_CONFIG_LOAD_KEYS = {
    "ar_action_max_steps",
    "embodiment_spec_prob",
    "goalbox_text_prob",
    "gradient_checkpointing",
    "history_pad_token_id",
    "image_aug_policy",
    "image_aug_prob",
    "image_aug_strict",
    "knowledge_insulation",
    "norm_stats_output_path",
    "optimize_ae_only",
    "state_text_prob",
}


def resolve_dm05_action_mode(config) -> str:
    """Return the DM05 training action target protocol."""

    mode = str(getattr(config, "action_mode", "absolute") or "absolute")
    if mode not in ACTION_MODE_CHOICES:
        raise ValueError("DM05 action_mode must be one of {'absolute', 'relative'}")
    return mode


def _sanitize_dm05_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    norm_stats_output_path = config.get("norm_stats_output_path")
    legacy_gradient_checkpointing = config.get("gradient_checkpointing")
    payload = {key: value for key, value in config.items() if key not in _LEGACY_CONFIG_LOAD_KEYS}
    if payload.get("norm_stats_path") is None and norm_stats_output_path is not None:
        payload["norm_stats_path"] = norm_stats_output_path
    if legacy_gradient_checkpointing is not None:
        if payload.get("vlm_gradient_checkpointing") is None:
            payload["vlm_gradient_checkpointing"] = bool(legacy_gradient_checkpointing)
        if payload.get("ae_gradient_checkpointing") is None:
            payload["ae_gradient_checkpointing"] = bool(legacy_gradient_checkpointing)
    core_config = payload.get("core_config")
    if isinstance(core_config, dict):
        payload["core_config"] = {
            key: value for key, value in core_config.items() if key != "knowledge_insulation"
        }
    return payload


@PreTrainedConfig.register_subclass("dm05")
@dataclass
class DM05Config(PreTrainedConfig):
    """LeRobot policy config for DM05.

    Saved DM05 LeRobot checkpoints use one standard ``config.json``. LeRobot reads
    the ``type=dm05`` policy fields, while DM05 keeps the core HF config as an
    opaque ``core_config`` payload in the same file.
    """

    # Full DM05 core HF config payload for self-contained LeRobot checkpoints.
    # Keeping this as one dict avoids maintaining a duplicate list of future core
    # config fields in the LeRobot adapter config.
    core_config: dict | None = None

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int | None = None
    drop_n_last_frames: int = 1

    max_state_dim: int = 32
    max_action_dim: int = 32
    image_resolution: tuple[int, int] = (448, 448)
    empty_cameras: int = 0

    pretrained_name_or_path: str | None = "Dexmal/DM05"
    # Internal processor source. Normal training should rely on the checkpoint.
    processor_name_or_path: str | None = None
    trust_remote_code: bool = True

    image_keys: list[str] | None = None

    n_bins: int = 256
    tokenizer_max_length: int | None = 1024

    diffusion_steps: int = 10
    compile_suffix: str = "auto"
    compile_suffix_pad_length: int | None = 1024
    compile_suffix_warmup_steps: int = 0
    # DM05 has separate attention backends for LLM/Vision and Action Expert blocks.
    # Kept explicit to avoid surprising coupling to Hugging Face generic policy knobs.
    llm_attn_implementation: str = "eager"
    vision_attn_implementation: str = "sdpa"
    action_attn_implementation: str = "sdpa"
    # Optional integration for Liger kernels. Default false because the dependency
    # is optional and may be unavailable on some training environments.
    use_liger_kernel: bool = False

    dtype: str = "bfloat16"
    vlm_gradient_checkpointing: bool | None = None
    ae_gradient_checkpointing: bool | None = None
    ae_gradient_checkpointing_layers: int | None = None
    freeze_vlm_embedding: bool = True

    # DM05 uses its own normalizer in the model-side conversion pipeline.
    # This preserves the original SFT order: build action chunk -> compute relative
    # action -> normalize state/action -> tokenize -> pad.
    norm_stats_path: str | None = None
    norm_stats_overwrite: bool = False
    action_mode: str = "absolute"
    use_absolute_action: bool = False
    action_target_offset: int = 0
    norm_non_delta_indices: tuple[int, ...] | None = None
    norm_max_samples: int | None = None
    norm_fallback_max_samples: int | None = 20000
    norm_clip: bool = False
    norm_use_quantiles: bool = True
    norm_stats_wait_seconds: int = 3600

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 50000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()
        if self.vlm_gradient_checkpointing is None:
            self.vlm_gradient_checkpointing = True
        if self.ae_gradient_checkpointing is None:
            self.ae_gradient_checkpointing = True
        if self.ae_gradient_checkpointing_layers is None:
            self.ae_gradient_checkpointing_layers = 1
        self.norm_use_quantiles = True
        self.norm_clip = False
        self.optimizer_lr = 2.5e-5
        self.optimizer_betas = (0.9, 0.95)
        self.optimizer_weight_decay = 1e-10
        self.scheduler_warmup_steps = 1000
        self.scheduler_decay_steps = 50000
        self.scheduler_decay_lr = self.optimizer_lr * 0.1
        resolve_dm05_action_mode(self)
        if self.n_action_steps is None:
            self.n_action_steps = self.chunk_size
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than "
                f"chunk_size ({self.chunk_size})."
            )

        if self.llm_attn_implementation not in {
            "auto",
            "eager",
            "flash_attention_2",
            "sdpa",
            "flex_attention",
        }:
            raise ValueError(
                "llm_attn_implementation must be one of {auto, eager, flash_attention_2, sdpa, flex_attention}."
            )
        if self.vision_attn_implementation not in {"auto", "eager", "flash_attention_2", "sdpa"}:
            raise ValueError(
                "vision_attn_implementation must be one of {auto, eager, flash_attention_2, sdpa}."
            )
        if self.action_attn_implementation not in {"auto", "eager", "sdpa", "flex_attention"}:
            raise ValueError("action_attn_implementation must be one of {auto, eager, sdpa, flex_attention}.")
        if self.compile_suffix not in {"auto", "on", "off"}:
            raise ValueError("compile_suffix must be one of {'auto', 'on', 'off'}.")
        if self.compile_suffix_pad_length is not None and self.compile_suffix_pad_length <= 0:
            raise ValueError("compile_suffix_pad_length must be positive or None")
        if self.compile_suffix_warmup_steps < 0:
            raise ValueError("compile_suffix_warmup_steps must be non-negative")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"dtype must be 'bfloat16' or 'float32', got {self.dtype!r}")
        if self.ae_gradient_checkpointing_layers is not None and self.ae_gradient_checkpointing_layers < 1:
            raise ValueError("ae_gradient_checkpointing_layers must be >= 1 or None")
        if self.tokenizer_max_length is not None and self.tokenizer_max_length <= 0:
            raise ValueError("tokenizer_max_length must be positive or None")
        if self.action_target_offset < 0:
            raise ValueError("action_target_offset must be non-negative")
        if self.norm_max_samples is not None and self.norm_max_samples <= 1:
            raise ValueError("norm_max_samples must be greater than 1 or None")
        if self.norm_fallback_max_samples is not None and self.norm_fallback_max_samples <= 1:
            raise ValueError("norm_fallback_max_samples must be greater than 1 or None")
        if self.norm_stats_wait_seconds <= 0:
            raise ValueError("norm_stats_wait_seconds must be positive")

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}

        for i in range(self.empty_cameras):
            self.input_features.setdefault(
                f"{OBS_IMAGES}.empty_camera_{i}",
                PolicyFeature(type=FeatureType.VISUAL, shape=(3, *self.image_resolution)),
            )

        self.input_features.setdefault(
            OBS_STATE,
            PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,)),
        )
        self.output_features.setdefault(
            ACTION,
            PolicyFeature(type=FeatureType.ACTION, shape=(self.max_action_dim,)),
        )

        state_shape = self.input_features[OBS_STATE].shape
        if state_shape and state_shape[-1] > self.max_state_dim:
            raise ValueError(f"State dimension {state_shape[-1]} exceeds max_state_dim {self.max_state_dim}.")

        action_shape = self.output_features[ACTION].shape
        if action_shape and action_shape[-1] > self.max_action_dim:
            raise ValueError(
                f"Action dimension {action_shape[-1]} exceeds max_action_dim {self.max_action_dim}."
            )

    def _save_pretrained(self, save_directory: Path) -> None:
        buffer = StringIO()
        with draccus.config_type("json"):
            draccus.dump(self, buffer, indent=4)
        payload = json.loads(buffer.getvalue())
        payload = _sanitize_dm05_config_payload(payload)
        (save_directory / CONFIG_NAME).write_text(json.dumps(payload, indent=4) + "\n")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        start = int(self.action_target_offset)
        return list(range(start, start + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
