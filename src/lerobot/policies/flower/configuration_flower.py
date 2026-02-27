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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig, AdamWConfig, MultiAdamWConfig, MultiAdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.optim.schedulers import TriStageLRSchedulerConfig, MultiTriStageLRSchedulerPtConfig


@PreTrainedConfig.register_subclass("flower")
@dataclass
class FlowerConfig(PreTrainedConfig):
    # Inputs / output structure.
    n_obs_steps: int = 2  # num_latest_obs flower只支持1
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
    num_inference_steps: int | None = 4  # num_sampling_steps=4 # flower使用rectified flow，只需要4步

    # Loss computation
    do_mask_loss_for_padding: bool = True

    # Training presets
    # # Optimizer Configuration
    # optimizer_lr: float = 1e-4
    
    # optimizer_betas=(0.9, 0.95)
    # optimizer_eps=1e-8

    # # sft:
    # optimizer_weight_decay=0.05
    # optimizer_betas=(0.9, 0.95)
    # optimizer_eps=1e-8

    # init_lr=2e-5
    # init_lr_scale=0.1
    # final_lr_scale=0.5
    # total_steps=50000  # 经过total_steps，lr从init_lr_scale*init_lr到final_lr_scale*init_lr
    # phase_ratio="(0.05, 0.1, 0.85)"
    # lr=2e-5

    # pret:
    learning_rate_dit: float = 1e-4
    learning_rate_vlm: float = 2e-5
    beta_dit = (0.9, 0.95)
    beta_vlm = (0.9, 0.99)

    weight_decay = {"transformer_weight_decay": 0.1, "vlm_weight_decay": 1e-9}
    dit_lr_scheduler = {"init_lr_scale": 0.1, "final_lr_scale": 0.1, "phase_ratio": "(0.01, 0.39, 0.6)", "total_steps": 2400000}
    vlm_lr_scheduler = {"init_lr_scale": 0.01, "final_lr_scale": 0.1, "phase_ratio": "(0.1, 0.3, 0.6)", "total_steps": 2400000}

    # flower:
    # VLM Configuration
    vlm_path='/mnt/data/share/models/Florence-2-large'
    freeze_florence=False
    freeze_vision_tower=False
    freeze_embeddings_only=True
    vlm_prompt_style='default'
    token_dropout=0.1  # Added token dropout parameter

    # Model Structure
    data_frequency: int = 3 # 30Hz. 要用fps代替吗？
    device = 'cuda'
    mixed_precision = 'bf16'
    # pretraining stuff
    load_pretrained=False
    pretrained_model_path='/mnt/data_ssd/share/models/flower_vla_pret/360000_model_weights.pt'

    # Model flags
    action_type_adaln=True
    use_causal_attention=True   
    use_cross_attn=True
    use_adaln_cond=False
    use_readout_token=False
    use_proprio=True
    return_act_chunk=False

    # DiT Configuration
    sampling_type='uniform'
    dit_dim=1024
    n_heads=16
    n_layers=12
    attn_pdrop=0.1
    resid_pdrop=0.1
    mlp_pdrop=0.1

    # RoPE Configuration
    use_rope=True
    use_nope=False
    query_seq_len=100
    rope_theta=32.0

    resize_h=224
    resize_w=224
    cams='observation.images.top'

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

    # def get_optimizer_preset(self) -> AdamWConfig:
    #     return AdamWConfig(
    #         lr=self.lr,
    #         betas=self.optimizer_betas,
    #         eps=self.optimizer_eps,
    #         weight_decay=self.optimizer_weight_decay,
    #     )

    def get_optimizer_preset(self):
        return MultiAdamWConfig(
            optimizer_groups={"vlm": {"lr": self.learning_rate_vlm, "betas": self.beta_vlm},
            "dit": {"lr": self.learning_rate_dit, "betas": self.beta_dit}}
        )

    def get_scheduler_preset(self):        
        scheduler_groups = {"vlm": self.vlm_lr_scheduler, "dit": self.dit_lr_scheduler}
        return MultiTriStageLRSchedulerPtConfig(scheduler_groups=scheduler_groups)

    # def get_scheduler_preset(self) -> TriStageLRSchedulerConfig:        
    #     configs = {"lr_scheduler": {
    #         "init_lr": self.init_lr,
    #         "init_lr_scale": self.init_lr_scale,
    #         "final_lr_scale": self.final_lr_scale,
    #         "total_steps": self.total_steps,
    #         "phase_ratio": self.phase_ratio,
    #         "lr": self.lr,
    #     }}
    #     return TriStageLRSchedulerConfig(configs=configs)

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
