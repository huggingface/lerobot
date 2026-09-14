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

"""Configuration surface for the LeRobot LaWAM policy adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig


@LRSchedulerConfig.register_subclass("lawam_cosine_with_min_lr")
@dataclass
class LaWAMCosineWithMinLRSchedulerConfig(LRSchedulerConfig):
    """Linear warmup followed by the cosine schedule used by upstream LaWAM."""

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        del num_training_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < self.num_warmup_steps:
                return current_step / max(1, self.num_warmup_steps)

            # LeRobot's shared PI-style scheduler uses the absolute
            # `current_step / num_decay_steps` after warmup. Upstream LaWAM
            # instead starts cosine progress at the end of warmup.
            progress = (current_step - self.num_warmup_steps) / max(
                1, self.num_decay_steps - self.num_warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)
            min_lr_ratio = self.decay_lr / self.peak_lr
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine * (1.0 - min_lr_ratio) + min_lr_ratio

        return LambdaLR(optimizer, lr_lambda, -1)


@PreTrainedConfig.register_subclass("lawam")
@dataclass
class LaWAMConfig(PreTrainedConfig):
    """Configuration for the LaWAM policy adapter.

    The policy carries the LaWAM architecture in-tree and exposes it through
    LeRobot SFT/eval interfaces.
    """

    n_obs_steps: int = 1
    chunk_size: int = 50
    action_horizon: int = 8
    n_action_steps: int | None = None
    num_video_frames: int = 2

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    base_vlm: str = "Qwen/Qwen3-VL-2B-Instruct"

    lam_dim: int = 1024
    lam_num_heads: int = 16
    lam_ffn_expansion_factor: int = 4
    lam_enc_layers: int = 24
    lam_code_dim: int = 32
    lam_max_state_dim: int = 14
    lam_num_queries: int = 1
    lam_dec_layers: int = 12
    lam_dropout: float = 0.0
    lam_vq_layer_norm: bool = True
    lam_norm_latents: bool = True
    lam_norm_latents_type: str = "ln"
    lam_enc_modal_mask: bool = True
    lam_latent_layer_to_use: int = -2
    lam_num_embodiments: int = 32
    lam_image_hw: tuple[int, int] = (256, 256)
    lam_patch_size: int = 16
    lam_decoder_last_ln: bool = True
    dinov3_hidden_size: int = 768
    dinov3_intermediate_size: int = 3072
    dinov3_num_hidden_layers: int = 12
    dinov3_num_attention_heads: int = 12
    dinov3_num_register_tokens: int = 4

    primary_image_features: list[str] | None = None
    wrist_image_features: list[str] | None = None
    lam_image_feature: str | None = None
    default_task: str = "Execute the robot action."
    embodiment_id: int = 25

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

    clip_normalized_actions: bool = False
    pre_snap_gripper_action: bool = False
    binarize_gripper_action: bool = False
    gripper_dim: int = 6
    gripper_threshold: float = 0.5

    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    freeze_embedding: bool = True
    unfreeze_vision_merger: bool = True
    unfreeze_lam_decoder: bool = True
    keep_llm_first_n_layers: int | None = 16
    unfreeze_llm_last_n_layers: int | None = -1

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_500
    scheduler_decay_steps: int = 25_000
    scheduler_decay_lr: float = 5e-7

    def __post_init__(self) -> None:
        """Validate portable model identifiers and temporal settings."""
        super().__post_init__()
        try:
            validate_repo_id(self.base_vlm)
        except HFValidationError as exc:
            raise ValueError("`base_vlm` must be a portable Hugging Face model ID.") from exc
        if not 1 <= self.action_horizon <= self.chunk_size:
            raise ValueError("`action_horizon` must be in [1, chunk_size].")
        if self.n_action_steps is None:
            self.n_action_steps = self.action_horizon
        elif not 1 <= self.n_action_steps <= self.action_horizon:
            raise ValueError("`n_action_steps` must be in [1, action_horizon].")
        if self.num_video_frames < 1:
            raise ValueError("`num_video_frames` must be >= 1.")

    def validate_features(self) -> None:
        """Ensure the policy has the visual inputs and action output it requires."""
        if not self.image_features:
            raise ValueError("LaWAM requires at least one visual input feature.")
        if self.action_feature is None:
            raise ValueError("LaWAM requires an action output feature.")

        image_keys = list(self.image_features)
        primary = None if self.primary_image_features is None else list(self.primary_image_features)
        wrist = None if self.wrist_image_features is None else list(self.wrist_image_features)
        if primary is None and wrist is None:
            if len(image_keys) != 1:
                raise ValueError(
                    "LaWAM requires explicit `primary_image_features` and `wrist_image_features` "
                    "when more than one visual feature is configured."
                )
            primary, wrist = image_keys, []
        elif primary is None:
            primary = [key for key in image_keys if key not in wrist]
        elif wrist is None:
            wrist = [key for key in image_keys if key not in primary]

        if primary is None or wrist is None:
            raise RuntimeError("LaWAM camera roles were not resolved.")
        configured = primary + wrist
        unknown = sorted(set(configured) - set(image_keys))
        duplicates = sorted({key for key in configured if configured.count(key) > 1})
        unassigned = sorted(set(image_keys) - set(configured))
        if unknown:
            raise ValueError(f"LaWAM camera features are not policy inputs: {unknown}.")
        if duplicates:
            raise ValueError(f"LaWAM camera features must have exactly one role: {duplicates}.")
        if unassigned:
            raise ValueError(f"LaWAM camera features are missing a primary/wrist role: {unassigned}.")
        if not primary:
            raise ValueError("LaWAM requires at least one primary image feature.")

        lam_image_feature = self.lam_image_feature or primary[0]
        if lam_image_feature not in primary:
            raise ValueError("`lam_image_feature` must be one of `primary_image_features`.")
        if self.action_feature.shape[0] > self.flow_action_dim:
            raise ValueError("LaWAM action feature width cannot exceed `flow_action_dim`.")
        if self.robot_state_feature is not None and self.robot_state_feature.shape[0] > self.flow_state_dim:
            raise ValueError("LaWAM state feature width cannot exceed `flow_state_dim`.")
        if not 0 <= self.embodiment_id < min(self.lam_num_embodiments, self.flow_num_embodiments):
            raise ValueError("`embodiment_id` is outside the configured embodiment tables.")

        self.primary_image_features = primary
        self.wrist_image_features = wrist
        self.lam_image_feature = lam_image_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        """Build the default AdamW optimizer configuration for LaWAM."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LaWAMCosineWithMinLRSchedulerConfig:
        """Build the upstream-compatible LaWAM learning-rate schedule."""
        return LaWAMCosineWithMinLRSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        """Return the current timestep for non-visual observations."""
        return [0]

    @property
    def image_observation_delta_indices(self) -> list[int]:
        """Return current and future frames required by the LAM teacher."""
        if self.num_video_frames == 1:
            return [0]
        last_action_index = self.action_horizon - 1
        return [
            round(frame_idx * last_action_index / (self.num_video_frames - 1))
            for frame_idx in range(self.num_video_frames)
        ]

    @property
    def state_observation_delta_indices(self) -> list[int]:
        """Return only the current robot state."""
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        """Return action indices covering the configured prediction horizon."""
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        """Indicate that LaWAM does not request reward history from datasets."""
        return None
