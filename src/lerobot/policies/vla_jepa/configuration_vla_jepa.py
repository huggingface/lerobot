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


@PreTrainedConfig.register_subclass("vla_jepa")
@dataclass
class VLAJEPAConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 7
    n_action_steps: int = 7

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    qwen_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    jepa_encoder_name: str = "facebook/vjepa2-vitl-fpc64-256"
    freeze_qwen: bool = False
    enable_world_model: bool = True
    # Enables cross-embodiment transfer: when fine-tuning a pretrained model on a robot with a
    # different action or state dimensionality, the input/output projection layers must be
    # re-initialised from scratch while the rest of the network keeps its pretrained weights.
    # List the key prefixes that are allowed to have shape mismatches; anything else raises an error.
    # e.g. ["model.action_model.action_encoder", "model.action_model.state_encoder"]
    reinit_modules: list[str] | None = None

    tokenizer_padding_side: str = "left"
    prompt_template: str = "Your task is {instruction}. Infer the temporal dynamics from frames {actions} and produce the corresponding policy actions {e_actions}."
    special_action_token: str = "<|action_{}|>"
    embodied_action_token: str = "<|embodied_action|>"

    action_dim: int = 7
    state_dim: int = 8

    num_action_tokens_per_timestep: int = 8
    num_embodied_action_tokens_per_instruction: int = 32
    num_inference_timesteps: int = 4

    action_hidden_size: int = 1024
    action_model_type: str = "DiT-B"
    action_num_layers: int = 16
    action_num_heads: int | None = None
    action_attention_head_dim: int | None = None
    action_dropout: float = 0.2
    action_num_timestep_buckets: int = 1000
    action_noise_beta_alpha: float = 1.5
    action_noise_beta_beta: float = 1.0
    action_noise_s: float = 0.999
    num_target_vision_tokens: int = 32
    action_max_seq_len: int = 1024

    # total video frames loaded per sample
    num_video_frames: int = 8
    predictor_depth: int = 12
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.0
    world_model_loss_weight: float = 0.1
    jepa_tubelet_size: int = 2  # must match the encoder (e.g. 2 for vjepa2-vitl-fpc64-256)
    repeated_diffusion_steps: int = 8  # independent noise draws per batch item (CogACT-style)

    resize_images_to: tuple[int, int] | None = None
    binarize_gripper_action: bool = True
    pre_snap_gripper_action: bool = True
    clip_normalized_actions: bool = True
    gripper_dim: int = 6
    gripper_threshold: float = 0.5
    torch_dtype: str = "bfloat16"

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.freeze_qwen and self.enable_world_model:
            # freezing qwen backbone makes world model training irrelevant since no grad flows
            self.enable_world_model = False
        if self.n_action_steps > self.chunk_size:
            raise ValueError("`n_action_steps` must be <= `chunk_size`.")
        if self.num_video_frames < 2 * self.jepa_tubelet_size:
            raise ValueError(
                f"`video_horizon` ({self.num_video_frames}) must be >= 2 * `jepa_tubelet_size` "
                f"({self.jepa_tubelet_size}) to have at least one context and one GT temporal position."
            )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("VLAJEPA requires at least one visual input feature.")
        if self.action_feature is None:
            raise ValueError("VLAJEPA requires an action output feature.")
        self.action_dim = self.action_feature.shape[0]
        if self.robot_state_feature is not None:
            self.state_dim = self.robot_state_feature.shape[0]

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
        # load video_horizon frames starting from current timestep: [t, t+1, ..., t+video_horizon-1]
        # matches original repo's observation_indices=list(range(video_horizon))
        return list(range(self.num_video_frames))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
