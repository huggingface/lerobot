#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# Shared cluster paths — load the policy directly without specifying --policy.path.
FASTWAM_CHECKPOINT_PATH = "/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint"
FASTWAM_WAN22_WEIGHTS_PATH = "/storage/project/r-agarg35-0/shared/awm/fastwam_wan22_weights"


@PreTrainedConfig.register_subclass("fastwam")
@dataclass
class FastWAMConfig(PreTrainedConfig):
    """Configuration for the FastWAM policy.

    FastWAM is a diffusion-based VLA built on Wan2.2-TI2V-5B that predicts robot actions
    via flow-matching. It uses a Mixture-of-Transformers (MoT) architecture where a video
    DiT imagines future frames and an action DiT denoises actions conditioned on those frames.

    Architecture values match fastwam.yaml from the original FastWAM repository.
    For LIBERO: 2 cameras (agentview + eye_in_hand) concatenated horizontally → 224×448 input.
    """

    # ---- Model variant ----
    model_variant: str = "fastwam"  # "fastwam" | "fastwam_joint" | "fastwam_idm"

    # ---- Temporal / chunking ----
    n_obs_steps: int = 1
    chunk_size: int = 32      # action_horizon passed to infer_action
    n_action_steps: int = 10  # replan_steps in original sim_libero.yaml

    # ---- Image / cameras ----
    image_size: tuple[int, int] = (224, 224)  # per-camera (H, W)
    num_cameras: int = 2                       # LIBERO: agentview + eye_in_hand

    # ---- Action / state dims (LIBERO defaults) ----
    action_dim: int = 7   # 6D eef delta pose + 1D gripper
    state_dim: int = 8    # 3D eef pos + 3D axis-angle + 2D gripper qpos
    max_state_dim: int = 8
    max_action_dim: int = 7
    tokenizer_max_length: int = 128

    # ---- Inference ----
    num_inference_steps: int = 20

    # ---- Temporal (video DiT) ----
    num_video_frames: int = 33          # total frames in video sequence (VAE latent T)
    action_video_freq_ratio: int = 4    # action tokens per video frame

    # ---- Pretrained weights (Wan2.2) ----
    wan22_pretrained_path: str = "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    load_wan22_weights: bool = True     # compat alias (always True at inference)
    load_text_encoder: bool = True
    redirect_common_files: bool = True  # use DiffSynth safetensors mirrors
    action_dit_pretrained_path: str | None = None

    # ---- dtype ----
    dtype: str = "bfloat16"

    # ---- Video DiT architecture (from fastwam.yaml — do not change) ----
    video_dit_has_image_input: bool = False
    video_dit_patch_size: tuple[int, int, int] = (1, 2, 2)
    video_dit_in_dim: int = 48
    video_dit_hidden_dim: int = 3072
    video_dit_ffn_dim: int = 14336
    video_dit_freq_dim: int = 256
    video_dit_text_dim: int = 4096
    video_dit_out_dim: int = 48
    video_dit_num_heads: int = 24
    video_dit_attn_head_dim: int = 128
    video_dit_num_layers: int = 30
    video_dit_eps: float = 1e-6
    video_dit_seperated_timestep: bool = True
    video_dit_require_clip_embedding: bool = False
    video_dit_require_vae_embedding: bool = False
    video_dit_fuse_vae_embedding_in_latents: bool = True
    video_dit_use_gradient_checkpointing: bool = False
    video_dit_video_attention_mask_mode: str = "first_frame_causal"
    video_dit_action_conditioned: bool = False  # "uncond" = no action conditioning in video DiT

    # ---- Action DiT architecture ----
    action_dit_hidden_dim: int = 1024
    action_dit_ffn_dim: int = 4096
    action_dit_num_heads: int = 24
    action_dit_attn_head_dim: int = 128
    action_dit_num_layers: int = 30
    action_dit_freq_dim: int = 256
    action_dit_eps: float = 1e-6
    action_dit_use_gradient_checkpointing: bool = False

    # ---- MoT ----
    mot_checkpoint_mixed_attn: bool = False  # set True only for training

    # ---- Schedulers ----
    video_train_shift: float = 5.0
    video_infer_shift: float = 5.0
    video_num_train_timesteps: int = 1000
    action_train_shift: float = 5.0
    action_infer_shift: float = 5.0
    action_num_train_timesteps: int = 1000

    # ---- Loss weights ----
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0

    # ---- Normalization ----
    # MIN_MAX for state+action (FastWAM training uses min/max norm).
    # IDENTITY for visual (VAE normalises images internally to [-1,1]).
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # ---- Freezing ----
    freeze_vae: bool = True
    freeze_text_encoder: bool = True
    freeze_video_dit: bool = False

    # ---- Optimizer ----
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-2
    optimizer_grad_clip_norm: float = 1.0

    # ---- LR scheduler ----
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 20_000
    scheduler_decay_lr: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.model_variant not in ("fastwam", "fastwam_joint", "fastwam_idm"):
            raise ValueError(
                f"Invalid model_variant '{self.model_variant}'. "
                "Must be one of: 'fastwam', 'fastwam_joint', 'fastwam_idm'."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"Invalid dtype '{self.dtype}'.")
        if self.n_obs_steps > 1:
            T = self.n_obs_steps
            if T % 4 != 1:
                raise ValueError(
                    f"n_obs_steps={T} must satisfy T % 4 == 1 (e.g. 5, 9, 33) for FastWAM video training."
                )
            if self.chunk_size % (T - 1) != 0:
                raise ValueError(
                    f"chunk_size={self.chunk_size} must be divisible by n_obs_steps-1={T - 1}."
                )

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}

        for i in range(self.num_cameras):
            key = f"{OBS_IMAGES}.image" if i == 0 else f"{OBS_IMAGES}.image{i + 1}"
            if key not in self.input_features:
                self.input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, *self.image_size),
                )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.state_dim,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
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
        # With n_obs_steps > 1 the dataset returns T past frames as video for training.
        # T must satisfy T % 4 == 1 and action_horizon % (T-1) == 0.
        # Default n_obs_steps=1 → None (single frame, tiled to T=5 in _prepare_video_for_training).
        if self.n_obs_steps <= 1:
            return None
        return list(range(self.n_obs_steps))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_video_dit_config(self) -> dict:
        return {
            "has_image_input": self.video_dit_has_image_input,
            "patch_size": list(self.video_dit_patch_size),
            "in_dim": self.video_dit_in_dim,
            "hidden_dim": self.video_dit_hidden_dim,
            "ffn_dim": self.video_dit_ffn_dim,
            "freq_dim": self.video_dit_freq_dim,
            "text_dim": self.video_dit_text_dim,
            "out_dim": self.video_dit_out_dim,
            "num_heads": self.video_dit_num_heads,
            "attn_head_dim": self.video_dit_attn_head_dim,
            "num_layers": self.video_dit_num_layers,
            "eps": self.video_dit_eps,
            "seperated_timestep": self.video_dit_seperated_timestep,
            "require_clip_embedding": self.video_dit_require_clip_embedding,
            "require_vae_embedding": self.video_dit_require_vae_embedding,
            "fuse_vae_embedding_in_latents": self.video_dit_fuse_vae_embedding_in_latents,
            "use_gradient_checkpointing": self.video_dit_use_gradient_checkpointing,
            "video_attention_mask_mode": self.video_dit_video_attention_mask_mode,
            "action_conditioned": self.video_dit_action_conditioned,
            "action_dim": self.action_dim,
            "action_group_causal_mask_mode": "group_diagonal",
        }

    def get_action_dit_config(self) -> dict:
        return {
            "action_dim": self.action_dim,
            "hidden_dim": self.action_dit_hidden_dim,
            "ffn_dim": self.action_dit_ffn_dim,
            "num_heads": self.action_dit_num_heads,
            "attn_head_dim": self.action_dit_attn_head_dim,
            "num_layers": self.action_dit_num_layers,
            "text_dim": self.video_dit_text_dim,
            "freq_dim": self.action_dit_freq_dim,
            "eps": self.action_dit_eps,
            "use_gradient_checkpointing": self.action_dit_use_gradient_checkpointing,
        }
