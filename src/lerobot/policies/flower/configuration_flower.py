#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any, Dict, Tuple, Union, Optional
from pathlib import Path
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig, AdamWConfig, MultiAdamWConfig, MultiAdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.optim.schedulers import TriStageLRSchedulerConfig, MultiTriStageLRSchedulerPtConfig


@PreTrainedConfig.register_subclass("flower")
@dataclass
class FlowerConfig(PreTrainedConfig):
    # Inputs / output structure.
    n_obs_steps: int = 1  # num_latest_obs
    horizon: int = 16  # pred action
    n_action_steps: int = 8  # exec action  deployed_action_steps

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 0 # horizon - n_action_steps - n_obs_steps + 1

    # Inference
    num_inference_steps: int | None = 4  # num_sampling_steps=4

    # Loss computation
    do_mask_loss_for_padding: bool = True

    # Training presets
    training_stage: str = 'pretrain'

    # flower:
    # VLM Configuration
    vlm_path: Path = 'microsoft/Florence-2-large'
    freeze_florence: bool = False
    freeze_vision_tower: bool = False
    freeze_embeddings_only: bool = True
    vlm_prompt_style: str = 'default'
    token_dropout: float = 0.1  # Added token dropout parameter

    # Model Structure
    data_frequency: int = 3 # 30Hz
    device: str | None = 'cuda'
    mixed_precision: str | None = 'bf16'
    # pretraining stuff
    load_pretrained: bool = False
    pretrained_model_path: Path | None = None

    # Model flags
    action_type_adaln: bool = True
    use_causal_attention: bool = True   
    use_cross_attn: bool = True
    use_adaln_cond: bool = False
    use_readout_token: bool = False
    use_proprio: bool = True
    return_act_chunk: bool = False

    # DiT Configuration
    sampling_type: str = 'uniform'
    dit_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1

    # RoPE Configuration
    use_rope: bool = True
    use_nope: bool = False
    query_seq_len: int = 100
    rope_theta: float = 32.0

    resize_h: int = 224
    resize_w: int = 224

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        # Initialize model flags and configurations
        if self.vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
        if self.sampling_type not in ['ln', 'pi_zero', 'loglogistic', 'uniform', 'stratified']:
            raise ValueError(f"Invalid sampling type: {self.sampling_type}")
        
        self.use_readout_token = self.use_readout_token and self.use_adaln_cond
        self.use_rope = self.use_rope and not self.use_nope
        self.use_nope = self.use_nope and not self.use_rope
        self.return_act_chunk = False

        # Initialize model dimensions
        if self.dit_dim % self.n_heads != 0:
            raise ValueError(f"dit_dim ({self.dit_dim}) must be divisible by n_heads ({self.n_heads})")

        if self.training_stage == 'pretrain':
            self.learning_rate_dit: float = 1e-4
            self.learning_rate_vlm: float = 2e-5
            self.beta_dit: tuple[float, float] = (0.9, 0.95)
            self.beta_vlm: tuple[float, float] = (0.9, 0.99)
            self.weight_decay: dict[str, float] = {"transformer_weight_decay": 0.1, "vlm_weight_decay": 1e-9}
            self.dit_lr_scheduler: dict[str, float | str | int] = {"init_lr_scale": 0.1, "final_lr_scale": 0.1, "phase_ratio": "(0.01, 0.39, 0.6)", "total_steps": 2400000}
            self.vlm_lr_scheduler: dict[str, float | str | int] = {"init_lr_scale": 0.01, "final_lr_scale": 0.1, "phase_ratio": "(0.1, 0.3, 0.6)", "total_steps": 2400000}
        else:
            self.optimizer_weight_decay: float = 0.05
            self.optimizer_betas: tuple[float, float] = (0.9, 0.95)
            self.optimizer_eps: float = 1e-8

            self.init_lr: float = 2e-5
            self.init_lr_scale: float = 0.1
            self.final_lr_scale: float = 0.5
            self.total_steps: int = 50000
            self.phase_ratio: str = "(0.05, 0.1, 0.85)"
            self.lr: float = 2e-5

    def get_optimizer_preset(self) -> AdamWConfig | MultiAdamWConfig:
        if self.training_stage == 'pretrain':
            return MultiAdamWConfig(
            optimizer_groups={"vlm": {"lr": self.learning_rate_vlm, "betas": self.beta_vlm},
            "dit": {"lr": self.learning_rate_dit, "betas": self.beta_dit}}
        )
        else:
            return AdamWConfig(
                lr=self.lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )

    def get_scheduler_preset(self) -> MultiTriStageLRSchedulerPtConfig | TriStageLRSchedulerConfig:
        if self.training_stage == 'pretrain':        
            scheduler_groups = {"vlm": self.vlm_lr_scheduler, "dit": self.dit_lr_scheduler}
            return MultiTriStageLRSchedulerPtConfig(scheduler_groups=scheduler_groups)
        else:
            configs = {"lr_scheduler": {
                "init_lr": self.init_lr,
                "init_lr_scale": self.init_lr_scale,
                "final_lr_scale": self.final_lr_scale,
                "total_steps": self.total_steps,
                "phase_ratio": self.phase_ratio,
                "lr": self.lr,
                }}
            return TriStageLRSchedulerConfig(configs=configs)

    def validate_features(self) -> None:
        pass

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
