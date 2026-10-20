# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("longwam")
@dataclass
class LongWAMConfig(PreTrainedConfig):
    """Long-WAM architecture and sampling; backbone configs are saved in native checkpoints."""

    past_obs_steps: int = 48
    video_stride: int = 4
    future_video_steps: int = 32
    action_horizon: int = 32
    inference_action_horizon: int | None = None
    n_action_steps: int = 10
    denoising: str = "idm"
    num_inference_steps: int = 20
    imagine_sigma: float = 0.9
    imagine_steps: int = 4
    flow_shift: float = 5.0
    inference_seed: int | None = 1
    camera_keys: tuple[str, ...] = ("observation.images.image", "observation.images.wrist_image")
    camera_size: tuple[int, int] = (224, 224)
    action_dim: int = 7
    proprio_dim: int = 8
    dtype: torch.dtype | None = torch.bfloat16
    tokenizer_name: str = "google/umt5-xxl"
    tokenizer_max_length: int = 128
    prompt_template: str = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )
    # Populated by the native checkpoint. Construction never downloads backbone weights.
    video_config: dict[str, Any] = field(default_factory=dict)
    vae_config: dict[str, Any] = field(default_factory=dict)
    text_encoder_config: dict[str, Any] = field(default_factory=dict)
    action_hidden_dim: int = 1024
    action_ffn_dim: int = 4096
    use_gradient_checkpointing: bool = True
    toggle_action_dimensions: list[int] = field(default_factory=list)
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.01
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.video_stride != 4:
            raise ValueError("Long-WAM requires video_stride=4")
        if self.past_obs_steps < 0 or self.past_obs_steps % 16:
            raise ValueError("past_obs_steps must be a nonnegative multiple of 16")
        if self.future_video_steps <= 0 or self.future_video_steps % 16:
            raise ValueError("future_video_steps must be a positive multiple of 16")
        if self.action_horizon <= 0 or not 0 < self.n_action_steps <= self.prediction_steps:
            raise ValueError("Require action_horizon > 0 and 0 < n_action_steps <= prediction_steps")
        if self.denoising not in {"idm", "cod"}:
            raise ValueError("denoising must be idm or cod")
        if not 0 <= self.imagine_sigma <= 1 or (self.denoising == "cod" and self.imagine_sigma != 0):
            raise ValueError("imagine_sigma must be in [0,1], and zero for COD")
        if min(self.num_inference_steps, self.imagine_steps, self.flow_shift) <= 0:
            raise ValueError("Denoising steps and flow_shift must be positive")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be nonempty and unique, in checkpoint order")
        if any(x <= 0 or x % 16 for x in self.camera_size):
            raise ValueError("camera_size dimensions must be positive multiples of 16")
        self.input_features = self.input_features or {
            **{key: PolicyFeature(FeatureType.VISUAL, (3, *self.camera_size)) for key in self.camera_keys},
            "observation.state": PolicyFeature(FeatureType.STATE, (self.proprio_dim,)),
        }
        self.output_features = self.output_features or {
            "action": PolicyFeature(FeatureType.ACTION, (self.action_dim,))
        }
        self.validate_features()

    @property
    def prediction_steps(self) -> int:
        return self.action_horizon if self.inference_action_horizon is None else self.inference_action_horizon

    @property
    def observation_delta_indices(self) -> list[int]:
        # Decode only frames consumed by the VAE, not every intervening control tick.
        return list(range(-self.past_obs_steps, self.future_video_steps + 1, self.video_stride))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if set(self.image_features) != set(self.camera_keys):
            raise ValueError("input_features must contain exactly camera_keys")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (self.proprio_dim,):
            raise ValueError("observation.state does not match proprio_dim")
        if self.action_feature is None or self.action_feature.shape != (self.action_dim,):
            raise ValueError("action does not match action_dim")
        if any(len(feature.shape) != 3 or feature.shape[0] != 3 for feature in self.image_features.values()):
            raise ValueError("Long-WAM expects RGB cameras [3,H,W]")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr, betas=self.optimizer_betas, weight_decay=self.optimizer_weight_decay
        )

    def get_scheduler_preset(self) -> None:
        return None
