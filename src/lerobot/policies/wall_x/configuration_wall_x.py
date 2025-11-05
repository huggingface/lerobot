# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("wall_x")
@dataclass
class WallXConfig(PreTrainedConfig):
    """
    Configuration class for Wall-X policy.

    Wall-X is based on Qwen2.5-VL with action prediction capabilities using flow matching.
    It supports cross-embodiment robotic control through unified action representations.
    """
    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 32  # action_horizon in wall-x
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Action dimension - wall-x uses hardcoded 20
    max_action_dim: int = 20
    max_state_dim: int = 20  # For proprioception

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] | None = None  # wall-x uses Qwen processor

    # Tokenizer
    tokenizer_max_length: int = 256

    # Model architecture
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    load_vlm_weights: bool = True

    # Vision config
    vision_config: dict = field(default_factory=lambda: {
        "depth": 32,
        "hidden_size": 3584,
        "hidden_act": "silu",
        "intermediate_size": 3420,
        "num_heads": 16,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "window_size": 112,
        "out_hidden_size": 3584,
    })

    # Language model config
    hidden_size: int = 3584  # 8192 for 7B model
    intermediate_size: int = 18944  # 29568 for 7B model
    num_hidden_layers: int = 36  # 80 for 7B model
    num_attention_heads: int = 28  # 64 for 7B model
    num_key_value_heads: int = 4  # 8 for 7B model
    vocab_size: int = 152064

    # Action prediction mode: "flow" or "fast"
    prediction_mode: str = "flow"

    # Flow matching parameters
    noise_scheduler: dict = field(default_factory=lambda: {
        "beta_alpha": 1.5,  # Beta distribution concentration1
        "beta_beta": 1.0,   # Beta distribution concentration0
        "s": 0.999,         # Scaling factor for time
    })

    # Decoding parameters
    num_inference_timesteps: int = 10  # Number of ODE solver steps
    ode_solver_method: str = "euler"  # ODE solver method

    # Degrees of freedom configuration - example for bimanual robot
    dof_config: dict = field(default_factory=lambda: {
        "left_ee_pos": 3,
        "left_ee_rot": 3,
        "left_gripper": 1,
        "right_ee_pos": 3,
        "right_ee_rot": 3,
        "right_gripper": 1,
    })

    # Proprioception configuration (mirrors dof_config)
    agent_pos_config: dict = field(default_factory=lambda: {
        "left_ee_pos": 3,
        "left_ee_rot": 3,
        "left_gripper": 1,
        "right_ee_pos": 3,
        "right_ee_rot": 3,
        "right_gripper": 1,
    })

    # MoE configuration
    num_experts: int = 4
    attention_moe: bool = False
    mlp_moe: bool = False

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False  # wall-x trains more components
    train_action_head: bool = True

    # Cache
    use_cache: bool = True

    # Training presets
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-6

    # Dataset-specific normalization statistics
    # Maps dataset names to {min, delta} for action normalization
    action_statistics: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        """Input validation"""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.prediction_mode not in ["flow", "fast"]:
            raise ValueError(
                f"prediction_mode must be 'flow' or 'fast', got {self.prediction_mode}"
            )

        # Validate dof_config total doesn't exceed max_action_dim
        total_dof = sum(self.dof_config.values())
        if total_dof > self.max_action_dim:
            raise ValueError(
                f"Total DOF ({total_dof}) exceeds max_action_dim ({self.max_action_dim})"
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
