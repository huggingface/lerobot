#!/usr/bin/env python

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
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("sarm")
@dataclass
class SARMConfig(PreTrainedConfig):
    """Configuration class for SARM (Stage-Aware Reward Modeling).

    Supports three annotation modes:

    1. single_stage (default): No annotations needed. Uses the episode's task description
       as a single stage covering the entire episode. Progress is computed as linear
       interpolation from 0 to 1 over the episode. Best for simple tasks or when
       annotations are not available.

    2. dense_only: Uses dense (fine-grained) annotations from VLM, with an auto-generated
       single sparse "task" stage covering the full episode. The dense head learns detailed
       subtask progression while sparse provides overall task completion.

    3. dual: Full dual-head mode with both sparse (high-level) and dense (fine-grained)
       annotations from VLM. Both heads are trained on their respective annotations.

    The annotation_mode determines how sparse_temporal_proportions and dense_temporal_proportions
    are loaded/generated during model initialization.
    """

    # Annotation mode: "single_stage", "dense_only", or "dual"
    annotation_mode: str = "single_stage"

    # CLIP params
    image_dim: int = 512
    text_dim: int = 512
    
    # Temporal sampling params (Reference: config/sarm.yaml model section)
    n_obs_steps: int = 8  # Number of observation history steps
    frame_gap: int = 30  # Frame gap between frames (at 30 fps = 1 second)
    max_rewind_steps: int = 4  # Maximum rewind steps for temporal augmentation
    
    # Total frames = 1 + n_obs_steps + max_rewind_steps (computed in property)
    # During training with rewind: [obs_frames] + [rewind_frames]
    # During inference: [obs_frames] only

    # Architecture params
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 8
    max_state_dim: int = 32
    
    # Drop last frame to avoid boundary issues with backward-looking sampling
    drop_n_last_frames: int = 1
    
    # Augmentation params (Reference: rm_lerobot_dataset.py)
    rewind_probability: float = 0.8  # Probability of applying rewind augmentation
    language_perturbation_probability: float = 0.2  # Probability of perturbing task string

    # Sparse annotations (high-level stages)
    # For single_stage/dense_only: auto-set to dataset task with proportion [1.0]
    # For dual: loaded from annotations
    num_sparse_stages: int = 1  # Number of sparse stages (auto-updated from annotations)
    sparse_subtask_names: list | None = None  # List of sparse subtask names
    sparse_temporal_proportions: list | None = None  # Temporal proportions for sparse stages

    # Dense annotations (fine-grained stages)
    # Only used when annotation_mode is "dense_only" or "dual"
    num_dense_stages: int | None = None  # Number of dense stages
    dense_subtask_names: list | None = None  # List of dense subtask names
    dense_temporal_proportions: list | None = None  # Temporal proportions for dense stages

    # Training params
    batch_size: int = 64
    clip_batch_size: int = 64
    dropout: float = 0.1
    stage_loss_weight: float = 1.0  # Weight for stage classification loss when using subtask annotations

    pretrained_model_path: str | None = None
    device: str | None = None
    image_key: str = "observation.images.top"  # Key for image used from the dataset
    state_key: str = "observation.state"

    # Populated by the processor (video_features, state_features, text_features)
    input_features: dict = field(default_factory=lambda: {})

    # Output features (updated dynamically in __post_init__ based on annotation_mode)
    output_features: dict = field(
        default_factory=lambda: {
            "stage": PolicyFeature(shape=(9, 5), type=FeatureType.REWARD),
            "progress": PolicyFeature(shape=(9, 1), type=FeatureType.REWARD),
        }
    )

    # Inference mode for dual heads: "sparse", "dense", or "both"
    dual_inference_mode: str = "sparse"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "LANGUAGE": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()

        # Validate annotation_mode
        if self.annotation_mode not in ["single_stage", "dense_only", "dual"]:
            raise ValueError(
                f"annotation_mode must be 'single_stage', 'dense_only', or 'dual', got {self.annotation_mode}"
            )

        # Configure based on annotation_mode
        if self.annotation_mode == "single_stage":
            # Single stage mode: no annotations needed
            # Use task description as stage name, full episode as one stage
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ["task"]
            self.sparse_temporal_proportions = [1.0]
            # Clear dense settings
            self.num_dense_stages = None
            self.dense_subtask_names = None
            self.dense_temporal_proportions = None

        elif self.annotation_mode == "dense_only":
            # Dense-only mode: auto-generate single sparse stage, use dense from annotations
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ["task"]
            self.sparse_temporal_proportions = [1.0]
            # Dense will be loaded from annotations by the model

        self.input_features = {}
        self.output_features = {}

        # Add the image_key as VISUAL
        if self.image_key:
            self.input_features[self.image_key] = PolicyFeature(shape=(480, 640, 3), type=FeatureType.VISUAL)

        # Add state_key as STATE
        self.input_features[self.state_key] = PolicyFeature(
            shape=(self.max_state_dim,),  # Single frame state, temporal sampling handles sequence
            type=FeatureType.STATE,
        )

        # Update output features based on annotation_mode
        if self.annotation_mode in ["dense_only", "dual"]:
            # Dual head mode: separate outputs for sparse and dense
            self.output_features["sparse_stage"] = PolicyFeature(
                shape=(self.num_frames, self.num_sparse_stages), type=FeatureType.REWARD
            )
            self.output_features["sparse_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )
            dense_stages = self.num_dense_stages or self.num_sparse_stages
            self.output_features["dense_stage"] = PolicyFeature(
                shape=(self.num_frames, dense_stages), type=FeatureType.REWARD
            )
            self.output_features["dense_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )
        else:
            # Single head mode: sparse only
            self.output_features["sparse_stage"] = PolicyFeature(
                shape=(self.num_frames, self.num_sparse_stages), type=FeatureType.REWARD
            )
            self.output_features["sparse_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )

        # Validate configuration
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        
        # Validate rewind steps
        if self.max_rewind_steps >= self.n_obs_steps:
            raise ValueError(
                f"max_rewind_steps ({self.max_rewind_steps}) must be less than n_obs_steps ({self.n_obs_steps})"
            )

        # Validate num_sparse_stages
        if self.num_sparse_stages < 1:
            raise ValueError(f"num_sparse_stages must be at least 1, got {self.num_sparse_stages}")

        # Validate dual mode configuration
        if self.annotation_mode in ["dense_only", "dual"]:
            if self.dual_inference_mode not in ["sparse", "dense", "both"]:
                raise ValueError(
                    f"dual_inference_mode must be 'sparse', 'dense', or 'both', got {self.dual_inference_mode}"
                )
            if self.num_dense_stages is not None and self.num_dense_stages < 2:
                raise ValueError(f"num_dense_stages must be at least 2, got {self.num_dense_stages}")

    def get_optimizer_preset(self) -> AdamWConfig:
        """Get default optimizer configuration for SARM training."""
        return AdamWConfig(
            lr=5e-5,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Get default learning rate scheduler configuration."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=5e-5,
            decay_lr=5e-6,
            num_warmup_steps=500,
            num_decay_steps=50000,
        )

    def validate_features(self) -> None:
        """Validate input and output features."""
        pass

    @property
    def uses_dual_heads(self) -> bool:
        """Whether the model uses dual heads (dense_only or dual annotation modes)."""
        return self.annotation_mode in ["dense_only", "dual"]

    @property
    def num_frames(self) -> int:
        """Total number of frames in sequence.
        
        Reference: rm_lerobot_dataset.py
        
        For training: 1 + n_obs_steps + max_rewind_steps
        The sequence is: [obs_frames (n_obs_steps + 1)] + [rewind_frames (max_rewind_steps)]
        """
        return 1 + self.n_obs_steps + self.max_rewind_steps

    @property
    def max_length(self) -> int:
        """Maximum sequence length (same as num_frames)."""
        return self.num_frames

    @property
    def observation_delta_indices(self) -> list[int]:
        """Load frames for SARM with backward-looking sampling + rewind placeholders.
        
        Reference: rm_lerobot_dataset.py get_frame_indices()
        
        The model uses backward-looking frame sequence ending at target:
        - Pattern: [idx - n_obs_steps*gap, ..., idx-2*gap, idx-gap, idx]
        - Plus placeholders for rewind frames (filled by processor during augmentation)
        
        Returns:
            Delta indices: 9 observation frames + 4 rewind placeholders (total 13)
            Example with n_obs_steps=8, gap=30, max_rewind_steps=4:
            Obs: [-240, -210, -180, -150, -120, -90, -60, -30, 0]
            Rewind placeholders: [-30, -60, -90, -120] (reversed direction)
        """
        # Backward-looking pattern: sequence ends at target frame (0)
        # [idx - n_obs_steps*gap, ..., idx - gap, idx]
        obs_deltas = [-self.frame_gap * i for i in range(self.n_obs_steps, -1, -1)]
        
        # Rewind placeholders: frames going backwards from current position
        # These are placeholders that the processor will replace during augmentation
        # When rewind is applied, we sample frames from [idx - rewind_step*gap, ..., idx - gap]
        # reversed to simulate going backward
        rewind_deltas = [-self.frame_gap * (i + 1) for i in range(self.max_rewind_steps)]
        
        return obs_deltas + rewind_deltas

    @property
    def action_delta_indices(self) -> None:
        """SARM is a reward model, not an action policy."""
        return None

    @property
    def reward_delta_indices(self) -> None:
        """SARM doesn't use delta rewards."""
        return None
