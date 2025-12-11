# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""
VLA-0 Configuration

VLA-0 is a Vision-Language-Action model that directly represents actions as text tokens.
It leverages pretrained VLMs (like Qwen2.5-VL) without architectural modifications.

Reference: https://github.com/NVlabs/vla0
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)


@PreTrainedConfig.register_subclass("vla0")
@dataclass
class VLA0Config(PreTrainedConfig):
    """
    Configuration class for VLA-0 policy.

    VLA-0 represents robot actions directly as discretized text tokens,
    allowing the use of pretrained VLMs without architectural changes.
    """

    # Model dtype
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 1  # VLA-0 predicts single-step actions by default
    n_action_steps: int = 1

    # Action discretization
    num_bins_actions: int = 1000  # Number of bins for action discretization (0 to num_bins_actions)
    action_dim: int = 7  # Default action dimension (e.g., 6 DoF + gripper)

    # Action bounds for normalization (will be set from dataset stats)
    action_min: list[float] = field(default_factory=lambda: [-1.0] * 7)
    action_max: list[float] = field(default_factory=lambda: [1.0] * 7)

    # Model configuration
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    load_vlm_weights: bool = True

    # Image configuration (from official vla0.yaml)
    image_size: int = 224  # Target image size after preprocessing
    num_cameras: int = 2  # Number of camera inputs (official default: 2)

    # LoRA configuration
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # QLoRA configuration
    use_qlora: bool = False

    # Flash Attention
    use_flash_attn: bool = False

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.IDENTITY,  # VLA-0 handles action normalization internally
        }
    )

    # Training configuration (from official vla0.yaml)
    optimizer_lr: float = 5e-6
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 0.0  # Official: disabled (0.0)

    scheduler_warmup_steps: int = 100
    scheduler_decay_steps: int = 10000
    scheduler_decay_lr: float = 1e-6

    # Generation parameters
    max_new_tokens: int = 64  # Maximum tokens to generate for action prediction
    temperature: float = 0.0  # Use greedy decoding by default
    do_sample: bool = False

    # Task prompt template
    task_prompt_template: str = "What action should the robot take to {task}?"

    # Tokenizer settings
    tokenizer_max_length: int = 256

    # History length (number of past observations to use)
    history_length: int = 1

    def __post_init__(self):
        super().__post_init__()

        # Validate dtype
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than "
                f"chunk_size ({self.chunk_size})."
            )

        # Ensure action bounds match action dimension
        if len(self.action_min) != self.action_dim:
            self.action_min = [-1.0] * self.action_dim
        if len(self.action_max) != self.action_dim:
            self.action_max = [1.0] * self.action_dim

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        # Set up output features for action
        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            )
            self.output_features["action"] = action_feature
        else:
            # Update action_dim and bounds from output_features (set by dataset)
            action_shape = self.output_features["action"].shape
            actual_action_dim = action_shape[0] if action_shape else self.action_dim
            if actual_action_dim != self.action_dim:
                self.action_dim = actual_action_dim
                # Update action bounds to match actual dimension
                self.action_min = [-1.0] * self.action_dim
                self.action_max = [1.0] * self.action_dim

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
