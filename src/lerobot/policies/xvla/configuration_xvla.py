#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from .configuration_florence2 import Florence2Config


@PreTrainedConfig.register_subclass("xvla")
@dataclass
class XVLAConfig(PreTrainedConfig):
    """
    Configuration class for the XVLA (Extended Vision-Language-Action) policy so it can
    plug into the LeRobot training stack.

    The config mirrors the knobs exposed in the original XVLA repository but also
    declares the input/output feature contract required by LeRobot.
    """

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32
    num_actions: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Florence2 backbone and tokenizer configuration
    florence_config: dict[str, Any] | Florence2Config = field(default_factory=dict)
    tokenizer_name: str = "facebook/bart-large"
    tokenizer_max_length: int = 64
    tokenizer_padding_side: str = "right"
    pad_language_to: str = "max_length"

    # Transformer head
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_domains: int = 30
    len_soft_prompts: int = 32
    dim_time: int = 32
    max_len_seq: int = 512
    use_hetero_proj: bool = False

    # Action & proprioception
    action_mode: str = "ee6d"
    num_denoising_steps: int = 10
    use_proprio: bool = True
    max_state_dim: int = 32
    domain_feature_key: str | None = None

    # Vision preprocessing
    resize_imgs_with_padding: tuple[int, int] | None = (518, 518)
    num_image_views: int | None = None
    empty_cameras: int = 0

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.num_actions <= 0:
            raise ValueError("`num_actions` must be strictly positive.")
        if self.chunk_size != self.num_actions:
            self.chunk_size = self.num_actions
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= `chunk_size` ({self.chunk_size})."
            )
        if isinstance(self.florence_config, Florence2Config):
            self.florence_config = self.florence_config.to_dict()
        if self.num_image_views is not None and self.num_image_views <= 0:
            raise ValueError("`num_image_views` must be > 0 when specified.")
        self._florence_config_obj: Florence2Config | None = None

    def get_florence_config(self) -> Florence2Config:
        """
        Build (and cache) the Florence2 transformer config that should back the VLM.
        """
        if self._florence_config_obj is None:
            if isinstance(self.florence_config, Florence2Config):
                self._florence_config_obj = self.florence_config
            else:
                # TODO: jadechoghari: provide default way, and do not hardcode
                # Ensure vision_config and text_config are provided with defaults if not specified
                config_dict = dict(self.florence_config)
                if "vision_config" not in config_dict or config_dict["vision_config"] is None:
                    # Provide default vision config
                    config_dict["vision_config"] = {
                        "model_type": "davit",
                        "drop_path_rate": 0.1,
                        "patch_size": [14, 7, 7, 7],
                        "patch_stride": [4, 2, 2, 2],
                        "patch_padding": [3, 1, 1, 1],
                        "patch_prenorm": [False, True, True, True],
                        "dim_embed": [256, 512, 1024, 2048],
                        "num_heads": [8, 16, 32, 64],
                        "num_groups": [8, 16, 32, 64],
                        "depths": [1, 1, 9, 1],
                        "window_size": 12,
                        "projection_dim": 1024,
                        "visual_temporal_embedding": {"type": "COSINE", "max_temporal_embeddings": 100},
                        "image_pos_embed": {"type": "learned_abs_2d", "max_pos_embeddings": 50},
                        "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
                    }
                if "text_config" not in config_dict or config_dict["text_config"] is None:
                    # Provide default text config
                    config_dict["text_config"] = {
                        "model_type": "florence2_language",
                        "vocab_size": 51289,
                        "d_model": 1024,
                        "encoder_layers": 12,
                        "decoder_layers": 12,
                        "encoder_attention_heads": 16,
                        "decoder_attention_heads": 16,
                        "encoder_ffn_dim": 4096,
                        "decoder_ffn_dim": 4096,
                    }
                self._florence_config_obj = Florence2Config(**config_dict)
        return self._florence_config_obj

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("XVLA requires at least one visual feature in the inputs.")
        if self.use_proprio and self.robot_state_feature is None:
            raise ValueError("`use_proprio=True` requires a proprioceptive state feature.")
        if self.num_image_views is None:
            self.num_image_views = len(self.image_features) + self.empty_cameras
        else:
            self.num_image_views = max(self.num_image_views, len(self.image_features) + self.empty_cameras)

        if self.empty_cameras > 0:
            height, width = (480, 640)
            if self.resize_imgs_with_padding is not None:
                height, width = self.resize_imgs_with_padding
            for idx in range(self.empty_cameras):
                key = f"{OBS_IMAGES}.empty_camera_{idx}"
                if key not in self.input_features:
                    self.input_features[key] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, height, width),
                    )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None
