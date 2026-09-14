# Copyright 2026 The OpenEAI team and The HuggingFace Inc. team. All rights reserved.
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

"""OpenEAI VLA policy configuration."""

from dataclasses import dataclass, field

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("openeai")
@dataclass
class OpenEAIVLAConfig(PreTrainedConfig):
    """Configuration for OpenEAI-VLA policy.

    Architecture: DiT action head + Qwen3-VL vision backbone + linear flow matching.

    Architecture parameters (should match the pretrained model):
        qwen_path: Qwen3-VL model path (HF Hub or local)
        qwen_dim: Qwen3 hidden dimension
        hidden_dim: DiT transformer hidden dimension
        n_layers: Number of DiT transformer blocks
        num_heads: Number of attention heads in DiT
        ff_ratio: FFN hidden dim / hidden_dim ratio
        denoise_steps: Number of flow matching inference steps
        img_seq_len: Number of image tokens from Qwen3 vision encoder
        feat_length: Number of feat-query tokens appended to context

    Flow matching parameters:
        time_sampling_beta_alpha: Beta distribution alpha for time sampling
        time_sampling_beta_beta: Beta distribution beta for time sampling
        time_sampling_scale: Scale factor for time sampling
        time_sampling_offset: Offset for time sampling

    LeRobot compatibility parameters:
        n_obs_steps: Number of observation steps (always 1 for OpenEAI)
        chunk_size: Action chunk length
        n_action_steps: Number of action steps to execute per inference
        normalization_mapping: How to normalize STATE/VISUAL/ACTION features
        max_state_dim: Maximum state vector dimension (padded)
        max_action_dim: Maximum action vector dimension (padded)
        image_resolution: Target image size for vision encoder
        empty_cameras: Number of empty cameras to add
    """

    # ---- Qwen3-VL backbone ----
    qwen_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    qwen_dim: int = 2560
    backbone_dtype: str = "bfloat16"

    # ---- DiT head ----
    hidden_dim: int = 1664
    n_layers: int = 18
    num_heads: int = 32
    ff_ratio: float = 2.67

    # ---- Flow matching ----
    denoise_steps: int = 10

    # ---- Context / feat query ----
    img_seq_len: int = 64
    feat_length: int = 20

    # ---- Flow matching time sampling ----
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001

    # ---- LeRobot compat ----
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    max_state_dim: int = 32
    max_action_dim: int = 32
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    empty_cameras: int = 0

    # ---- Training hyperparams ----
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    # ---- Language ----
    tokenizer_max_length: int = 128
    pad_language_to: str = "longest"

    # ---- Freeze backbone for finetuning ----
    freeze_backbone: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"OpenEAI-VLA only supports n_obs_steps=1, got {self.n_obs_steps}")
        if self.pad_language_to not in {"max_length", "longest"}:
            raise ValueError(f"pad_language_to must be 'max_length' or 'longest', got {self.pad_language_to}")
        if isinstance(self.image_resolution, list):
            self.image_resolution = tuple(self.image_resolution)
        valid_dtypes = {"bfloat16", "float16", "float32"}
        if self.backbone_dtype not in valid_dtypes:
            raise ValueError(f"backbone_dtype must be one of {valid_dtypes}, got {self.backbone_dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features from dataset metadata."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature

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
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def backbone_torch_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.backbone_dtype]
