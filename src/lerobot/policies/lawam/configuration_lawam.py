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

from dataclasses import dataclass, field

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

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
    base_vlm_path: str | None = None
    hf_cache_dir: str | None = None

    lam_dim: int = 1024
    lam_num_heads: int = 16
    lam_ffn_expansion_factor: int = 4
    lam_enc_layers: int = 24
    lam_codebook_size: int = 32
    lam_code_dim: int = 32
    lam_max_state_dim: int = 14
    lam_num_queries: int = 1
    lam_dec_layers: int = 12
    lam_dropout: float = 0.0
    lam_vq_type: str = "vae"
    lam_vq_layer_norm: bool = True
    lam_norm_latents: bool = True
    lam_norm_latents_type: str = "ln"
    lam_enc_add_state: bool = False
    lam_enc_modal_mask: bool = True
    lam_latent_layer_to_use: int = -2
    lam_multi_input: bool = False
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
    default_task: str = "Execute the robot action."
    action_hz: float | None = None
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
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 5e-7

    def __post_init__(self) -> None:
        """Validate portable model identifiers and temporal settings."""
        super().__post_init__()
        try:
            validate_repo_id(self.base_vlm)
        except HFValidationError as exc:
            raise ValueError(
                "`base_vlm` must be a portable Hugging Face model ID. "
                "Use `base_vlm_path` for a local Qwen directory."
            ) from exc
        if self.n_action_steps is not None and self.n_action_steps > self.chunk_size:
            raise ValueError("`n_action_steps` must be <= `chunk_size`.")
        if self.num_video_frames < 1:
            raise ValueError("`num_video_frames` must be >= 1.")
        if self.action_hz is not None and self.action_hz <= 0:
            raise ValueError("`action_hz` must be > 0.")
        if self.flow_horizon_sec <= 0:
            raise ValueError("`flow_horizon_sec` must be > 0.")

    def resolve_dataset_metadata(self, dataset_meta) -> None:
        """Resolve action frequency and horizon from dataset metadata."""
        if self.action_hz is None:
            dataset_fps = getattr(dataset_meta, "fps", None)
            if dataset_fps is None:
                raise ValueError(
                    "LaWAM requires `policy.action_hz` when dataset metadata is unavailable. "
                    "Training resolves it automatically from the dataset FPS."
                )
            self.action_hz = float(dataset_fps)

        effective_horizon = self.effective_action_horizon
        if self.n_action_steps is None:
            self.n_action_steps = effective_horizon
        elif self.n_action_steps > effective_horizon:
            raise ValueError(
                "`n_action_steps` cannot exceed the flow horizon: "
                f"got {self.n_action_steps}, but floor(flow_horizon_sec * action_hz) "
                f"limits this policy to {effective_horizon} steps."
            )

    def resolve_runtime_config(self, dataset_meta=None) -> None:
        """Resolve settings required before constructing the policy runtime."""
        if self.action_hz is None and dataset_meta is None:
            raise ValueError(
                "LaWAM requires `policy.action_hz` when dataset metadata is unavailable. "
                "Training resolves it automatically from the dataset FPS."
            )
        self.resolve_dataset_metadata(dataset_meta)

    @property
    def base_vlm_source(self) -> str:
        """Return the local VLM override or the portable Hub model ID."""
        return self.base_vlm_path or self.base_vlm

    @property
    def effective_action_horizon(self) -> int:
        """Return the action count supported by the configured time horizon."""
        if self.action_hz is None:
            raise ValueError("`action_hz` must be resolved before computing the action horizon.")
        horizon = min(self.chunk_size, int(self.flow_horizon_sec * self.action_hz))
        if horizon < 1:
            raise ValueError(
                "`flow_horizon_sec * action_hz` must cover at least one action step, "
                f"got {self.flow_horizon_sec} * {self.action_hz}."
            )
        return horizon

    def validate_features(self) -> None:
        """Ensure the policy has the visual inputs and action output it requires."""
        if not self.image_features:
            raise ValueError("LaWAM requires at least one visual input feature.")
        if self.action_feature is None:
            raise ValueError("LaWAM requires an action output feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        """Build the default AdamW optimizer configuration for LaWAM."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Build the default cosine decay scheduler configuration for LaWAM."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        """Return the frame indices used to construct each visual observation."""
        return list(range(self.num_video_frames))

    @property
    def action_delta_indices(self) -> list[int]:
        """Return action indices covering the effective prediction horizon."""
        horizon = self.chunk_size if self.action_hz is None else self.effective_action_horizon
        return list(range(horizon))

    @property
    def reward_delta_indices(self) -> None:
        """Indicate that LaWAM does not request reward history from datasets."""
        return None
