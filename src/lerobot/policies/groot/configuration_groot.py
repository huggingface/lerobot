#!/usr/bin/env python

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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("groot")
@dataclass
class GrootConfig(PreTrainedConfig):
    """Configuration for Groot policy wrapper."""

    # Basic policy settings
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Dimension settings (must match pretrained GR00T model expectations)
    # Maximum state dimension. Shorter states will be zero-padded.
    max_state_dim: int = 64

    # Maximum action dimension. Shorter actions will be zero-padded.
    max_action_dim: int = 32

    # Normalization (start with identity, adjust as needed)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Image preprocessing (adjust to match Groot's expected input)
    image_size: tuple[int, int] = (224, 224)

    # Groot-specific model parameters (from groot_finetune_script.py)

    # Path or HuggingFace model ID for the base Groot model
    base_model_path: str = "nvidia/GR00T-N1.5-3B"

    # HF repo ID (or local path) that hosts vocab.json and merges.txt for Eagle tokenizer.
    tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5"

    # Embodiment tag to use for training (e.g. 'new_embodiment', 'gr1')
    embodiment_tag: str = "new_embodiment"

    # Fine-tuning control arguments

    # Whether to fine-tune the llm backbone
    tune_llm: bool = False

    # Whether to fine-tune the vision tower
    tune_visual: bool = False

    # Whether to fine-tune the projector
    tune_projector: bool = True

    # Whether to fine-tune the diffusion model
    tune_diffusion_model: bool = True

    # LoRA parameters (from groot_finetune_script.py)
    # Rank for the LORA model. If 0, no LORA will be used.
    lora_rank: int = 0

    # Alpha value for the LORA model
    lora_alpha: int = 16

    # Dropout rate for the LORA model
    lora_dropout: float = 0.1

    # Whether to use the full model for LORA
    lora_full_model: bool = False

    # Training parameters (matching groot_finetune_script.py)
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    use_bf16: bool = True

    # Dataset parameters
    # Video backend to use for training ('decord' or 'torchvision_av')
    video_backend: str = "decord"

    # Whether to balance dataset weights in mixture datasets
    balance_dataset_weights: bool = True

    # Whether to sample trajectories weighted by their length
    balance_trajectory_weights: bool = True

    # Optional dataset paths for delegating training to Isaac-GR00T runner
    dataset_paths: list[str] | None = None
    output_dir: str = "./tmp/gr00t"
    save_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 32
    dataloader_num_workers: int = 8
    report_to: str = "wandb"
    resume: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )

        # groot_repo_path is now optional since we ported the components
        # No validation needed

    def validate_features(self) -> None:
        """Validate and set up input/output features for Groot."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "Groot policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
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
                shape=(self.max_action_dim,),
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
        """Return optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Return scheduler configuration."""
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=int(10000 * self.warmup_ratio),  # 5% warmup by default
            num_decay_steps=10000,  # Adjust based on training steps
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
        )

    @property
    def observation_delta_indices(self) -> None:
        """Return indices for delta observations (None for Groot)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return indices for delta actions."""
        return list(range(min(self.chunk_size, 16)))

    @property
    def reward_delta_indices(self) -> None:
        """Return indices for delta rewards (None for Groot)."""
        return None
