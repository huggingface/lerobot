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

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("lawam")
@dataclass
class LaWAMConfig(PreTrainedConfig):
    """Configuration for the LaWAM policy adapter.

    The policy carries the LaWAM architecture in-tree and exposes it through
    LeRobot SFT/eval interfaces.
    """

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    num_video_frames: int = 2

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    base_vlm: str = "Qwen/Qwen3-VL-2B-Instruct"
    lam_ckpt_path: str | None = None
    lam_yaml_path: str | None = None
    hf_cache_dir: str | None = None

    lawam_checkpoint_path: str | None = None
    lawam_dataset_stats_path: str | None = None
    lawam_unnorm_key: str | None = None

    primary_image_features: list[str] | None = None
    wrist_image_features: list[str] | None = None
    default_task: str = "Execute the robot action."
    action_hz: float = 20.0
    embodiment_id: int = 25

    enable_primary_video_aug: bool = False
    enable_primary_random_resized_crop: bool = False
    guidance_scale: float | None = None
    num_inference_steps: int | None = None

    latent_action_placeholder_token: str = "<ACT_PH>"
    num_action_queries: int = 8
    flow_action_num_queries: int = 8
    perceptual_weight: float = 0.1
    lam_encoder_distill_weight: float = 0.1
    enable_loss_distill: bool = True
    future_prediction: bool = True
    detach_future_feature: bool = True
    repeated_diffusion_steps: int = 2

    flow_action_dim: int = 32
    flow_state_dim: int = 32
    flow_hidden_dim: int = 1024
    flow_num_layers: int = 16
    flow_attention_heads: int = 16
    flow_vlm_dim: int = 2048
    flow_vision_dim: int = 768
    flow_num_vision_tokens: int = 256
    flow_num_target_vision_tokens: int = -1
    flow_use_state: bool = False
    flow_num_embodiments: int = 32
    flow_horizon_sec: float = 0.4
    flow_cfg_drop_prob: float = 0.0
    flow_cfg_guidance_scale: float = 1.0
    flow_num_inference_steps: int = 10
    flow_num_timestep_buckets: int = 1000
    flow_interleave_self_attention: bool = True
    flow_use_alternate_vldit: bool = True
    flow_attend_text_every_n_blocks: int = 2
    flow_noise_beta_alpha: float = 1.5
    flow_noise_beta_beta: float = 1.0
    flow_noise_s: float = 0.999
    flow_token_independent_noise: bool = False
    flow_use_action_positional_embeddings: bool = True

    clip_normalized_actions: bool = True
    pre_snap_gripper_action: bool = True
    binarize_gripper_action: bool = True
    gripper_dim: int = 6
    gripper_threshold: float = 0.5

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_500
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 5e-7

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("`n_action_steps` must be <= `chunk_size`.")
        if self.num_video_frames < 1:
            raise ValueError("`num_video_frames` must be >= 1.")
        if self.action_hz <= 0:
            raise ValueError("`action_hz` must be > 0.")

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("LaWAM requires at least one visual input feature.")
        if self.action_feature is None:
            raise ValueError("LaWAM requires an action output feature.")

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
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.num_video_frames))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
