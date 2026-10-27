#!/usr/bin/env python

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

from dataclasses import dataclass, field

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION

# The Wan2.2 VAE compresses 4 video frames into one latent frame (the first latent frame holds
# only the first video frame).
VAE_TEMPORAL_DOWNSAMPLE = 4
VAE_SPATIAL_DOWNSAMPLE = 16

ROBOTWIN_ACTION_CHANNELS = [*range(0, 7), 28, *range(7, 14), 29]


@PreTrainedConfig.register_subclass("zero_wam")
@dataclass
class ZeroWAMConfig(PreTrainedConfig):
    """Configuration for Zero-WAM, an in-context video-action world model.

    Zero-WAM predicts future video latents and robot actions with a mixture-of-transformers built on
    Wan2.2. A task is specified by a language instruction, a human demonstration video (in-context
    learning, ICL), or both. Defaults follow the released ``zero-wam-pretrain`` checkpoint and the
    upstream training recipe.
    """

    # Transformer architecture (``WanICLTransformer3DModel``)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    action_dim: int = 30
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 14336
    num_layers: int = 30
    cross_attn_norm: bool = True
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    action_inner_dim: int = 3072
    action_ffn_dim: int = 14336
    # Multi-chunk prediction heads (training only; ``False`` drops them for a smaller checkpoint).
    enable_mcp: bool = True
    num_mcp_modules: int = 4
    mcp_blocks_per_group: int = 1
    mcp_hidden_collect_layers: tuple[int, ...] = (3, 11, 19, 29)
    # "flex" compiles flex attention (CUDA); "sdpa" materializes dense masks (CPU, debugging).
    attention_backend: str = "flex"
    gradient_checkpointing: bool = True

    # Frozen Wan2.2 VAE, UMT5-XXL text encoder and tokenizer, loaded from these sub-folders.
    frozen_components_path: str = "robbyant-research/zero-wam-pretrain"
    frozen_components_revision: str | None = None
    # Transformer weights. Use "float32" for training with --accelerator.mixed_precision=bf16 (fp32
    # master weights, bf16 compute, as upstream); "bfloat16" for inference.
    dtype: torch.dtype | None = torch.bfloat16
    # The frozen VAE and text encoder always run in this dtype.
    frozen_dtype: torch.dtype = torch.bfloat16
    # UMT5-XXL (~11 GB) runs once per instruction; "cpu" keeps it off the GPU.
    text_encoder_device: str = "cpu"
    max_sequence_length: int = 512
    # Optional ``{prompt: [tokens, text_dim]}`` file of precomputed UMT5 embeddings (see
    # examples/zero_wam/precompute_text_embeddings.py); UMT5 then loads only for unseen prompts.
    text_embedding_cache_path: str | None = None

    # Cameras: each is resized to (height, width), encoded separately and concatenated on width.
    obs_cam_keys: list[str] = field(
        default_factory=lambda: [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
    )
    height: int = 224
    width: int = 288

    # Temporal layout. Video frames are taken every ``video_frame_stride`` dataset frames; one
    # latent frame therefore spans ``4 * video_frame_stride`` actions (``action_per_frame``).
    video_frame_stride: int = 4
    # Latent frames predicted per chunk at inference.
    frame_chunk_size: int = 2
    # Attention window in chunk-id units (video and action ids alternate per chunk).
    attn_window: int = 64

    # Model action channels used by this robot, in the order of the dataset's action vector.
    # RoboTwin: left EEF(7), left gripper, right EEF(7), right gripper. Joint robots use 14-27.
    used_action_channel_ids: list[int] = field(default_factory=lambda: list(ROBOTWIN_ACTION_CHANNELS))
    # Normalized actions are clipped to this range, as upstream.
    action_clip: float = 2.0

    # Inference
    num_inference_steps: int = 50
    action_num_inference_steps: int = 50
    snr_shift: float = 5.0
    action_snr_shift: float = 1.0
    # Target text CFG: <0 replaces the instruction with the empty prompt (upstream's human-video
    # evaluation setting), (0, 1] conditions on the instruction without CFG, >1 applies CFG.
    guidance_scale: float = 1.0
    # CFG between the human-video conditioned and unconditioned branches (needs an ICL video).
    icl_guidance_scale: float = 5.0
    # Human demonstration prompt: a precomputed latent ``.pth`` (preferred) or an mp4 path.
    icl_latent_path: str | None = None
    icl_video_path: str | None = None
    icl_height: int = 320
    icl_width: int = 480
    icl_fps: int = 12
    icl_rope_h: int = 24

    # Training (upstream ``zerowam_train_config``)
    # Latent frames per training clip; the dataset provides 4*(n-1)+1 video frames per camera.
    train_num_latent_frames: int = 11
    # 0 samples the chunk size uniformly from [1, max_frame_chunk_size] per step.
    train_frame_chunk_size: int = 0
    max_frame_chunk_size: int = 4
    # 0 samples the attention window uniformly from [4, max_attn_window] per step.
    train_attn_window: int = 0
    max_attn_window: int = 64
    noisy_img_prob: float = 0.5
    noisy_cond_min_timestep_bd: float = 0.0
    noisy_cond_max_timestep_bd: float = 1.0
    video_loss_reweight: bool = True
    action_loss_reweight: bool = False
    drop_icl: float = 0.1
    droptext_target: float = 0.4
    mcp_snr_shift: float = 10.0
    mcp_loss_weights: tuple[float, ...] = (0.5, 0.25, 0.15, 0.1)
    future_chunk_stride: int = 2
    # JSONL of {"episode_index", "start_frame", "end_frame", "latent_path"} human-video pairings
    # used as ICL context during training. Samples without a pairing train text-only.
    icl_pairs_path: str | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # Optimizer / scheduler (upstream: AdamW, linear warmup then constant)
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 200

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float32, float16 or bfloat16")
        if self.frozen_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("frozen_dtype must be float32, float16 or bfloat16")
        if not self.obs_cam_keys or len(set(self.obs_cam_keys)) != len(self.obs_cam_keys):
            raise ValueError("obs_cam_keys must contain unique camera keys")
        for name in (
            "video_frame_stride",
            "frame_chunk_size",
            "max_frame_chunk_size",
            "num_inference_steps",
            "action_num_inference_steps",
            "max_sequence_length",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("height", "width", "icl_height", "icl_width"):
            if getattr(self, name) <= 0 or getattr(self, name) % (VAE_SPATIAL_DOWNSAMPLE * 2):
                raise ValueError(f"{name} must be a positive multiple of 32")
        if self.patch_size != (1, 2, 2):
            raise ValueError("The released Zero-WAM layout requires patch_size=(1, 2, 2)")
        if self.icl_latent_path and self.icl_video_path:
            raise ValueError("Set only one of icl_latent_path and icl_video_path")
        if self.attention_backend not in ("flex", "sdpa"):
            raise ValueError(f"attention_backend must be 'flex' or 'sdpa', got {self.attention_backend!r}")
        if self.train_num_latent_frames < 2:
            raise ValueError("train_num_latent_frames must be at least 2")
        if len(self.mcp_loss_weights) != self.num_mcp_modules:
            raise ValueError("mcp_loss_weights must have one entry per MCP module")
        if not self.used_action_channel_ids or any(
            c < 0 or c >= self.action_dim for c in self.used_action_channel_ids
        ):
            raise ValueError(f"used_action_channel_ids must lie in [0, {self.action_dim})")
        if len(set(self.used_action_channel_ids)) != len(self.used_action_channel_ids):
            raise ValueError("used_action_channel_ids must be unique")

    @property
    def action_per_frame(self) -> int:
        return VAE_TEMPORAL_DOWNSAMPLE * self.video_frame_stride

    @property
    def chunk_size(self) -> int:
        return self.frame_chunk_size * self.action_per_frame

    @property
    def n_action_steps(self) -> int:
        return self.chunk_size

    def validate_features(self) -> None:
        if self.input_features is None or self.output_features is None:
            raise ValueError("Zero-WAM requires resolved `input_features` and `output_features`.")
        missing = [k for k in self.obs_cam_keys if k not in self.input_features]
        if missing:
            raise ValueError(f"Camera keys {missing} are not among the policy input features.")
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(len(self.used_action_channel_ids),)
            )
        n_action = self.output_features[ACTION].shape[0]
        if n_action != len(self.used_action_channel_ids):
            raise ValueError(
                f"The action has {n_action} dimensions but {len(self.used_action_channel_ids)} "
                "used_action_channel_ids are configured."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def observation_delta_indices(self) -> list[int]:
        num_video_frames = VAE_TEMPORAL_DOWNSAMPLE * (self.train_num_latent_frames - 1) + 1
        return [i * self.video_frame_stride for i in range(num_video_frames)]

    @property
    def action_delta_indices(self) -> list[int]:
        # The first latent frame is the conditioning frame; its action slots are zero history.
        return list(range((self.train_num_latent_frames - 1) * self.action_per_frame))

    @property
    def reward_delta_indices(self) -> None:
        return None
