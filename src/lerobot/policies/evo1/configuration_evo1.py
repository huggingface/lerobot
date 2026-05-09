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

import math
from dataclasses import dataclass, field

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@LRSchedulerConfig.register_subclass("evo1_exact")
@dataclass
class Evo1SchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step: int) -> float:
            if current_step < self.num_warmup_steps:
                return current_step / max(1, self.num_warmup_steps)
            progress = (current_step - self.num_warmup_steps) / max(
                1, num_training_steps - self.num_warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@PreTrainedConfig.register_subclass("evo1")
@dataclass
class Evo1Config(PreTrainedConfig):
    training_stage: str = "stage1"
    use_amp: bool = True

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 24
    max_action_dim: int = 24
    max_views: int = 3
    image_resolution: tuple[int, int] = (448, 448)
    empty_cameras: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    vlm_model_name: str = "OpenGVLab/InternVL3-1B"
    vlm_num_layers: int | None = 14
    vlm_dtype: str = "bfloat16"
    use_flash_attn: bool = True
    action_head: str = "flowmatching"
    embed_dim: int = 896
    hidden_dim: int = 1024
    state_hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.0
    num_inference_timesteps: int = 32
    num_categories: int = 1
    return_cls_only: bool = False
    enable_gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False

    finetune_vlm: bool | None = None
    finetune_language_model: bool | None = None
    finetune_vision_model: bool | None = None
    finetune_action_head: bool | None = None

    task_field: str = "task"
    embodiment_id_field: str | None = None
    default_embodiment_id: int = 0

    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 300
    drop_last: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.training_stage not in {"stage1", "stage2"}:
            raise ValueError(
                f"Unsupported EVO1 training_stage '{self.training_stage}', expected 'stage1' or 'stage2'"
            )

        if self.training_stage == "stage1":
            if self.finetune_vlm is None:
                self.finetune_vlm = False
            if self.finetune_language_model is None:
                self.finetune_language_model = False
            if self.finetune_vision_model is None:
                self.finetune_vision_model = False
            if self.finetune_action_head is None:
                self.finetune_action_head = True
        elif self.training_stage == "stage2":
            has_explicit_branch_flags = any(
                flag is not None for flag in (self.finetune_language_model, self.finetune_vision_model)
            )
            if not has_explicit_branch_flags:
                if self.finetune_vlm is None:
                    self.finetune_vlm = True
                if self.finetune_language_model is None:
                    self.finetune_language_model = True
                if self.finetune_vision_model is None:
                    self.finetune_vision_model = True
            elif self.finetune_vlm is None:
                self.finetune_vlm = bool(self.finetune_language_model or self.finetune_vision_model)
            if self.finetune_action_head is None:
                self.finetune_action_head = True

        if self.finetune_vlm is None:
            self.finetune_vlm = False
        if self.finetune_language_model is None:
            self.finetune_language_model = False
        if self.finetune_vision_model is None:
            self.finetune_vision_model = False
        if self.finetune_action_head is None:
            self.finetune_action_head = False

        branch_vlm = self.finetune_language_model or self.finetune_vision_model
        if self.finetune_vlm != branch_vlm:
            raise ValueError(
                "Inconsistent EVO1 finetune config: "
                f"finetune_vlm={self.finetune_vlm} but "
                f"(finetune_language_model or finetune_vision_model)={branch_vlm}. "
                "When branch-level flags are used, finetune_vlm must match their effective union."
            )

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}

        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            if key not in self.input_features:
                self.input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, *self.image_resolution),
                )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return Evo1SchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
