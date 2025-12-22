#!/usr/bin/env python

# Copyright 2025 Nvidia and The HuggingFace Inc. team. All rights reserved.
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
Configuration for Groot N1.6 policy.

Key differences from N1.5:
- Uses AlternateVLDiT instead of standard DiT
- 32 DiT layers (vs 16 in N1.5)
- Unfrozen top 4 VLM layers instead of 4-layer post-VLM adapter
- State-relative action chunks instead of absolute joint angles
- Uses Cosmos-Reason-2B variant backbone (Eagle-Block2A-2B-v2)
- CategorySpecificMLP and MultiEmbodimentActionEncoder modules
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("gr00t_n1d6")
@dataclass
class Gr00tN1d6Config(PreTrainedConfig):
    """Configuration for Groot N1.6 policy wrapper.

    This configuration extends the base PreTrainedConfig to support the Groot N1.6
    architecture, which introduces several improvements over N1.5 including
    AlternateVLDiT, more DiT layers, and unfrozen VLM top layers.
    """

    # Basic policy settings
    n_obs_steps: int = 1
    chunk_size: int = 40  # max_action_horizon in N1.6 (vs 50 in N1.5)
    n_action_steps: int = 40

    # Dimension settings (must match pretrained GR00T model expectations)
    # Maximum state dimension. Shorter states will be zero-padded.
    max_state_dim: int = 29  # Default from N1.6 config

    # Maximum action dimension. Shorter actions will be zero-padded.
    max_action_dim: int = 29  # Default from N1.6 config

    # Normalization (start with identity, adjust as needed)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Image preprocessing
    image_size: tuple[int, int] = (224, 224)

    # =========================================================================
    # Groot N1.6 Backbone Configuration
    # =========================================================================

    # Path or HuggingFace model ID for the base Groot model
    base_model_path: str = "nvidia/GR00T-N1.6-3B"

    # HF repo ID that hosts vocab.json and merges.txt for Eagle tokenizer
    tokenizer_assets_repo: str = "nvidia/Eagle-Block2A-2B-v2"

    # Backbone model type (eagle for N1.6)
    backbone_model_type: str = "eagle"

    # Model revision (optional)
    model_revision: str | None = None

    # Backbone embedding dimension (project_to_dim)
    backbone_embedding_dim: int = 2048

    # Vision layer to select features from
    select_layer: int = 16

    # Whether to reproject vision features
    reproject_vision: bool = False

    # Whether to use flash attention
    use_flash_attention: bool = True

    # Load model in BF16
    load_bf16: bool = True

    # Use Eagle collator for any-res image handling
    eagle_collator: bool = False

    # Keep trainable backbone params in FP32
    backbone_trainable_params_fp32: bool = True

    # Embodiment tag to use for training (e.g. 'new_embodiment', 'gr1')
    embodiment_tag: str = "new_embodiment"

    # =========================================================================
    # N1.6 Specific: VLM Layer Tuning
    # =========================================================================

    # Number of top LLM layers to tune (new in N1.6, replaces post-VLM adapter)
    tune_top_llm_layers: int = 4

    # =========================================================================
    # Fine-tuning control arguments
    # =========================================================================

    # Whether to fine-tune the llm backbone
    tune_llm: bool = False

    # Whether to fine-tune the vision tower
    tune_visual: bool = False

    # Whether to fine-tune the projector
    tune_projector: bool = True

    # Whether to fine-tune the diffusion model
    tune_diffusion_model: bool = True

    # Whether to fine-tune the VLLN (new in N1.6)
    tune_vlln: bool = True

    # =========================================================================
    # Image Processing Parameters
    # =========================================================================

    # Image crop and target sizes (optional, for preprocessing)
    image_crop_size: tuple[int, int] | None = None
    image_target_size: tuple[int, int] | None = None

    # Shortest edge to resize to before cropping
    shortest_image_edge: int | None = 256

    # Fraction of image to keep when center cropping
    crop_fraction: float | None = 0.95

    # Random rotation angle for augmentation
    random_rotation_angle: int | None = None

    # Color jitter parameters
    color_jitter_params: dict[str, float] | None = None

    # Use albumentations for image transforms (new in N1.6)
    use_albumentations_transforms: bool = True

    # Formalize language prompts
    formalize_language: bool = True

    # =========================================================================
    # LoRA parameters
    # =========================================================================

    # Rank for the LORA model. If 0, no LORA will be used.
    lora_rank: int = 0

    # Alpha value for the LORA model
    lora_alpha: int = 16

    # Dropout rate for the LORA model
    lora_dropout: float = 0.1

    # Whether to use the full model for LORA
    lora_full_model: bool = False

    # =========================================================================
    # Action Head Configuration
    # =========================================================================

    # Action horizon (chunk size for action prediction)
    action_horizon: int = 16

    # Hidden size for action head
    hidden_size: int = 1024

    # Input embedding dimension
    input_embedding_dim: int = 1536

    # Add positional embeddings
    add_pos_embed: bool = True

    # Attention dropout
    attn_dropout: float = 0.2

    # Use VLLN (Vision-Language Layer Norm)
    use_vlln: bool = True

    # Maximum sequence length
    max_seq_len: int = 1024

    # =========================================================================
    # N1.6 Specific: AlternateVLDiT Configuration
    # =========================================================================

    # Use AlternateVLDiT instead of standard DiT (key N1.6 feature)
    use_alternate_vl_dit: bool = True

    # Attend to text every N blocks in AlternateVLDiT
    attend_text_every_n_blocks: int = 2

    # Diffusion model configuration with 32 layers (main difference from N1.5)
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,  # 32 layers instead of 16 in N1.5
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # =========================================================================
    # Flow Matching Parameters
    # =========================================================================

    # Number of inference timesteps for flow matching
    num_inference_timesteps: int = 4

    # Beta distribution parameters for noise
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0

    # Noise scale parameter
    noise_s: float = 0.999

    # Number of timestep buckets
    num_timestep_buckets: int = 1000

    # =========================================================================
    # State Augmentation Parameters (new in N1.6)
    # =========================================================================

    # State dropout probability
    state_dropout_prob: float = 0.0

    # Scale for additive Gaussian noise on state features
    state_additive_noise_scale: float = 0.0

    # Apply sin/cos encoding to state (per-embodiment)
    apply_sincos_state_encoding: bool = False

    # Use relative actions instead of absolute (new in N1.6)
    use_relative_action: bool = False

    # =========================================================================
    # Multi-Embodiment Parameters
    # =========================================================================

    # Maximum number of embodiments supported
    max_num_embodiments: int = 32

    # =========================================================================
    # Training parameters
    # =========================================================================

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    use_bf16: bool = True

    # =========================================================================
    # Dataset parameters
    # =========================================================================

    # Video backend to use for training ('decord' or 'torchvision_av')
    video_backend: str = "decord"

    # Whether to balance dataset weights in mixture datasets
    balance_dataset_weights: bool = True

    # Whether to sample trajectories weighted by their length
    balance_trajectory_weights: bool = True

    # Optional dataset paths for delegating training
    dataset_paths: list[str] | None = None
    output_dir: str = "./tmp/gr00t_n16"
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

        # Validate tune_top_llm_layers
        if self.tune_top_llm_layers < 0:
            raise ValueError(
                f"tune_top_llm_layers ({self.tune_top_llm_layers}) must be non-negative"
            )

        # Validate action_horizon vs chunk_size
        if self.action_horizon > self.chunk_size:
            raise ValueError(
                f"action_horizon ({self.action_horizon}) cannot exceed chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features for Groot N1.6."""
        image_features = [
            key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL
        ]
        if not image_features:
            raise ValueError(
                "Groot N1.6 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature
        else:
            state_shape = self.input_features["observation.state"].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature
        else:
            action_shape = self.output_features["action"].shape
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
        """Return indices for delta observations (None for Groot N1.6)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return indices for delta actions."""
        return list(range(min(self.chunk_size, self.action_horizon)))

    @property
    def reward_delta_indices(self) -> None:
        """Return indices for delta rewards (None for Groot N1.6)."""
        return None

