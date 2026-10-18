# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable Long-WAM architecture, temporal sampling, and execution settings."""

from dataclasses import dataclass, field

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("longwam")
@dataclass
class LongWAMConfig(PreTrainedConfig):
    push_to_hub: bool = False
    # History is in raw control ticks, not VAE latents or sampled video frames.
    past_obs_steps: int = 48
    video_stride: int = 4
    future_video_steps: int = 32
    action_horizon: int = 32
    inference_action_horizon: int | None = None
    n_action_steps: int = 10
    execution: str = "sync"
    trigger_stride: int | None = None
    inference_timeout: float = 600.0
    denoising: str = "idm"
    num_inference_steps: int = 20
    imagine_sigma: float = 0.9
    imagine_steps: int = 4
    inference_seed: int | None = 1
    camera_keys: tuple[str, ...] = ("observation.images.image", "observation.images.wrist_image")
    camera_size: tuple[int, int] = (224, 224)
    inference_image_preprocessing: str = "pil"
    streaming_vae: bool = False
    action_dim: int = 7
    proprio_dim: int = 8
    dtype: torch.dtype | None = torch.bfloat16
    tokenizer_max_len: int = 128
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    video_init_repo: str = "Efficient-Large-Model/LongLive2.0-Robot-S"
    video_init_revision: str = "9f23acf1a619afe69d866f105c17209880851894"
    video_init_filename: str = "model.pt"
    # An explicit local mapped ActionDiT file is needed for fresh LIBERO training.
    # Fine-tuning a complete Long-WAM policy uses its action weights instead.
    action_init_path: str | None = None
    action_init: str = "mapped"
    load_text_encoder: bool = True
    use_gradient_checkpointing: bool = True
    prompt_template: str = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )
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
            raise ValueError("The released Long-WAM architecture requires video_stride=4")
        if self.past_obs_steps < 0 or self.past_obs_steps % 16:
            raise ValueError("past_obs_steps must be a nonnegative multiple of 16")
        if self.future_video_steps <= 0 or self.future_video_steps % 16:
            raise ValueError("future_video_steps must be a positive multiple of 16")
        if self.action_horizon <= 0 or not 0 < self.n_action_steps <= self.prediction_steps:
            raise ValueError("Require 0 < n_action_steps <= inference action horizon")
        if self.execution not in {"sync", "async"}:
            raise ValueError("execution must be sync or async")
        if self.execution == "async":
            if self.trigger_stride is None:
                self.trigger_stride = (self.n_action_steps + 1) // 2
            if not self.n_action_steps / 2 <= self.trigger_stride < self.n_action_steps:
                raise ValueError("Async execution requires R/2 <= trigger_stride < R")
        elif self.trigger_stride is not None:
            raise ValueError("trigger_stride is only valid for async execution")
        if self.denoising not in {"idm", "cod"}:
            raise ValueError("denoising must be idm or cod")
        if not 0 <= self.imagine_sigma <= 1:
            raise ValueError("imagine_sigma must be in [0, 1]")
        if self.denoising == "cod" and self.imagine_sigma != 0:
            raise ValueError("COD requires imagine_sigma=0")
        if min(self.num_inference_steps, self.imagine_steps, self.inference_timeout) <= 0:
            raise ValueError("Inference steps and timeout must be positive")
        if self.inference_image_preprocessing not in {"pil", "tensor"}:
            raise ValueError("inference_image_preprocessing must be pil or tensor")
        if self.action_init not in {"mapped", "random"}:
            raise ValueError("action_init must be mapped or random")
        if self.dtype not in {torch.float32, torch.bfloat16, torch.float16}:
            raise ValueError("Unsupported Long-WAM dtype")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be nonempty and unique, in checkpoint order")
        if any(x <= 0 or x % 16 for x in self.camera_size):
            raise ValueError("camera_size dimensions must be positive multiples of 16")
        self.input_features: dict[str, PolicyFeature] = self.input_features or {
            **{key: PolicyFeature(FeatureType.VISUAL, (3, *self.camera_size)) for key in self.camera_keys},
            "observation.state": PolicyFeature(FeatureType.STATE, (self.proprio_dim,)),
        }
        self.output_features: dict[str, PolicyFeature] = self.output_features or {
            "action": PolicyFeature(FeatureType.ACTION, (self.action_dim,))
        }
        self.validate_features()

    @property
    def prediction_steps(self) -> int:
        return self.action_horizon if self.inference_action_horizon is None else self.inference_action_horizon

    @property
    def observation_delta_indices(self) -> list[int]:
        # Training receives past + current + supervised future, at raw frame rate.
        # select_action receives ONLY the current frame and owns the history buffer.
        return list(range(-self.past_obs_steps, self.future_video_steps + 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if set(self.image_features) != set(self.camera_keys):
            raise ValueError(
                "input_features must contain exactly camera_keys; specify checkpoint camera order"
            )
        if self.robot_state_feature is None or self.robot_state_feature.shape != (self.proprio_dim,):
            raise ValueError("observation.state does not match proprio_dim")
        if self.action_feature is None or self.action_feature.shape != (self.action_dim,):
            raise ValueError("action feature does not match action_dim")
        for feature in self.image_features.values():
            if len(feature.shape) != 3 or feature.shape[0] != 3:
                raise ValueError("Long-WAM expects RGB camera features [3,H,W]")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr, betas=self.optimizer_betas, weight_decay=self.optimizer_weight_decay
        )

    def get_scheduler_preset(self) -> None:
        return None
