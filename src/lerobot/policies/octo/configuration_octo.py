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
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)


@PreTrainedConfig.register_subclass("octo")
@dataclass
class OctoConfig(PreTrainedConfig):
    # Model architecture
    model_name: str = "octo-base"  # "octo-base" or "octo-small"
    token_embedding_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 10  # max_horizon in octo
    n_action_steps: int = 4  # action_horizon in octo

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    push_to_hub: bool = False

    # Image preprocessing
    resize_primary_image: tuple[int, int] = (256, 256)
    resize_wrist_image: tuple[int, int] = (128, 128)

    # Language model
    language_model_name: str = "t5-base"
    language_max_length: int = 16
    freeze_language_encoder: bool = True

    # Transformer settings
    repeat_task_tokens: bool = True
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    add_position_embedding: bool = False

    # Diffusion settings
    diffusion_steps: int = 20
    n_diffusion_samples: int = 1
    max_action: float = 5.0
    loss_type: str = "mse"
    action_dim: int = 7
    time_dim: int = 32
    num_blocks: int = 3
    hidden_dim: int = 256
    use_layer_norm: bool = True

    # Finetuning settings
    freeze_transformer: bool = False
    freeze_vision_encoder: bool = True
    train_action_head_only: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        # Set architecture parameters based on model_name
        if self.model_name == "octo-base":
            self.token_embedding_size = 768
            self.num_layers = 12
            self.num_heads = 12
            self.mlp_dim = 3072
        elif self.model_name == "octo-small":
            self.token_embedding_size = 384
            self.num_layers = 12
            self.num_heads = 6
            self.mlp_dim = 1536
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        # Input validation
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
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

    def validate_features(self) -> None:
        """Validate that the input and output features are correctly configured."""
        # For Octo, we don't need to add any additional features like SmolVLA does
        # The features are already defined in the base class
        pass
