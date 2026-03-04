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
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("wall_x")
@dataclass
class WallXConfig(PreTrainedConfig):
    """
    Configuration class for Wall-X policy.

    Wall-X is based on Qwen2.5-VL with action prediction capabilities using flow matching.
    It supports cross-embodiment robotic control through unified action representations.

    This config supports multi-modal learning with vision, language, and action data.
    """

    # ==================== Input / Output Structure ====================
    n_obs_steps: int = 1
    chunk_size: int = 32  # action_horizon in wall-x
    n_action_steps: int = 32

    # Action dimension - wall-x uses 20
    max_action_dim: int = 20
    max_state_dim: int = 20  # For proprioception

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ==================== Action Prediction ====================
    # Pretrained model paths
    pretrained_name_or_path: str = "x-square-robot/wall-oss-flow"

    # Tokenizer settings
    action_tokenizer_path: str | None = "physical-intelligence/fast"

    # Action prediction mode: "diffusion" or "fast"
    prediction_mode: str = "diffusion"

    # Attention Implementation, options: "eager", "flash_attention_2", "sdpa"
    # NOTE: flash-attn==2.7.4.post1 is required for flash_attention_2 implementation
    attn_implementation: str = "eager"

    # ==================== Optimizer Presets ====================
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self):
        super().__post_init__()

        # Input validation
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.prediction_mode not in ["diffusion", "fast"]:
            raise ValueError(f"prediction_mode must be 'diffusion' or 'fast', got {self.prediction_mode}")

        # Assign use_fast_tokenizer based on prediction_mode
        if self.prediction_mode == "fast":
            self.use_fast_tokenizer = True
        elif self.prediction_mode == "diffusion":
            self.use_fast_tokenizer = False
            self.action_tokenizer_path = None  # disable action tokenizer for diffusion mode
        else:
            raise ValueError(f"prediction_mode must be 'diffusion' or 'fast', got {self.prediction_mode}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "Wall-X policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature
        else:
            state_shape = self.input_features[OBS_STATE].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature
        else:
            action_shape = self.output_features[ACTION].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
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
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
