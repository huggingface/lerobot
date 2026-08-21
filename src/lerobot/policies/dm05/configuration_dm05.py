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

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features

from .constants import DM05_STATE_BINS
from .utils import flatten_feature_names

_BASE_STATE_DIM = 14
_BASE_ACTION_DIM = 14


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
    add_state: bool = True

    pretrained_name_or_path: str | None = "Dexmal/DM05"
    # Internal processor source. Normal training should rely on the checkpoint.
    processor_name_or_path: str | None = None
    image_keys: list[str] | None = None
    license: str | None = "gemma"

    n_bins: int = 256
    tokenizer_max_length: int | None = 1024

    diffusion_steps: int = 10
    compile_model: bool = False
    compile_suffix_pad_length: int | None = 1024
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

    # Learn stored actions by default. Datasets containing absolute joint targets
    # can opt into OpenDM-style state-relative targets in the processor pipeline.
    use_relative_actions: bool = False
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None
    norm_clip: bool = True
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
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
        """Validate constructor defaults and normalize derived values."""
        super().__post_init__()
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be non-negative, got {self.drop_n_last_frames}")
        if self.vlm_gradient_checkpointing is None:
            self.vlm_gradient_checkpointing = True
        if self.ae_gradient_checkpointing is None:
            self.ae_gradient_checkpointing = True
        if self.ae_gradient_checkpointing_layers is None:
            self.ae_gradient_checkpointing_layers = 1
        if self.n_action_steps is None:
            self.n_action_steps = self.chunk_size
        if self.n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive, got {self.n_action_steps}")
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
        if self.compile_suffix_pad_length is not None and self.compile_suffix_pad_length <= 0:
            raise ValueError("compile_suffix_pad_length must be positive or None")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"dtype must be 'bfloat16' or 'float32', got {self.dtype!r}")
        if self.ae_gradient_checkpointing_layers is not None and self.ae_gradient_checkpointing_layers < 1:
            raise ValueError("ae_gradient_checkpointing_layers must be >= 1 or None")
        if self.tokenizer_max_length is not None and self.tokenizer_max_length <= 0:
            raise ValueError("tokenizer_max_length must be positive or None")
        if self.n_bins != DM05_STATE_BINS:
            raise ValueError(f"DM05 uses exactly {DM05_STATE_BINS} state bins, got {self.n_bins}")
        if self.diffusion_steps <= 0:
            raise ValueError(f"diffusion_steps must be positive, got {self.diffusion_steps}")

    def _validate_core_action_dim(self, core_action_dim: int | None) -> None:
        """Require the LeRobot adapter action dimension to match the core model."""
        if core_action_dim is not None and self.max_action_dim != int(core_action_dim):
            raise ValueError(
                f"max_action_dim {self.max_action_dim} must match DM05 core action_dim {core_action_dim}."
            )

    def validate_features(self) -> None:
        """Populate default DM05 features and enforce the adapter contract."""
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
            PolicyFeature(type=FeatureType.STATE, shape=(_BASE_STATE_DIM,)),
        )
        self.output_features.setdefault(
            ACTION,
            PolicyFeature(type=FeatureType.ACTION, shape=(_BASE_ACTION_DIM,)),
        )

        state_shape = self.input_features[OBS_STATE].shape
        if state_shape and state_shape[-1] > self.max_state_dim:
            raise ValueError(f"State dimension {state_shape[-1]} exceeds max_state_dim {self.max_state_dim}.")

        action_shape = self.output_features[ACTION].shape
        if action_shape and action_shape[-1] > self.max_action_dim:
            raise ValueError(
                f"Action dimension {action_shape[-1]} exceeds max_action_dim {self.max_action_dim}."
            )
        core_action_dim = self.core_config.get("action_dim") if isinstance(self.core_config, dict) else None
        self._validate_core_action_dim(core_action_dim)

        if self.use_relative_actions:
            if state_shape[-1] != action_shape[-1]:
                raise ValueError(
                    "DM05 relative actions require equal state/action dimensions, got "
                    f"{state_shape[-1]} and {action_shape[-1]}."
                )
            if self.normalization_mapping.get("ACTION") == NormalizationMode.MEAN_STD:
                raise ValueError("DM05 relative actions do not support MEAN_STD action normalization.")

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        """Use the training dataset contract instead of the base checkpoint contract."""
        policy_features = dataset_to_policy_features(features)
        self.output_features = {
            key: feature for key, feature in policy_features.items() if feature.type is FeatureType.ACTION
        }
        self.input_features = {
            key: feature for key, feature in policy_features.items() if key not in self.output_features
        }
        self.action_feature_names = flatten_feature_names(features.get(ACTION, {}).get("names"))

    def get_optimizer_preset(self) -> AdamWConfig:
        """Build the standard DM05 AdamW optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """Build the standard DM05 cosine-decay scheduler configuration."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        """DM05 does not use observation delta features."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return the action timestep offsets used for chunked training."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """DM05 does not use reward delta features."""
        return None
