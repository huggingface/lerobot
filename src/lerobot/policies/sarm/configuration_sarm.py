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
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("sarm")
@dataclass
class SARMConfig(PreTrainedConfig):
    """Configuration class for SARM (Stage-Aware Reward Modeling)"""
    
    # CLIP params
    image_dim: int = 512 
    text_dim: int = 512
    num_frames: int = 9  # 1 initial + 8 consecutive frames
    frame_gap: int = 30  # Frame gap between frames (at 30 fps = 1 second)
    
    # Architecture params
    hidden_dim: int = 768  
    num_heads: int = 12  
    num_layers: int = 8  
    max_state_dim: int = 32
    max_length: int = num_frames  # Maximum video sequence length (matches num_frames)
    use_temporal_sampler: bool = True  # Always enable temporal sequence loading
    
    # Dual sparse/dense head configuration (per SARM paper: twin MLP-based output heads)
    # When dual_sparse_dense=False: only sparse head is used (single head mode)
    # When dual_sparse_dense=True: both sparse and dense heads are used (dual head mode)
    dual_sparse_dense: bool = False
    
    # Sparse annotations (high-level stages)
    # Used in both single mode and dual mode
    num_sparse_stages: int = 5  # Number of sparse stages (auto-updated from annotations)
    sparse_subtask_names: list | None = None  # List of sparse subtask names
    sparse_temporal_proportions: list | None = None  # Temporal proportions for sparse stages
    
    # Dense annotations (fine-grained stages)
    # Only used when dual_sparse_dense=True
    num_dense_stages: int | None = None  # Number of dense stages
    dense_subtask_names: list | None = None  # List of dense subtask names
    dense_temporal_proportions: list | None = None  # Temporal proportions for dense stages
    
    # Training params
    batch_size: int = 64
    clip_batch_size: int = 64  # Batch size for CLIP encoding
    dropout: float = 0.1
    stage_loss_weight: float = 1.0  # Weight for stage classification loss when using subtask annotations
    
    pretrained_model_path: str | None = None
    device: str | None = None
    
    # Processor settings
    image_key: str = "observation.images.top"  # Key for image used from the dataset

    # State key in the dataset (for normalization)
    state_key: str = "observation.state"
    
    # Populated by the processor (video_features, state_features, text_features)
    input_features: dict = field(default_factory=lambda: {})
    
    # Output features (updated dynamically in __post_init__ based on dual_sparse_dense)
    output_features: dict = field(default_factory=lambda: {
        "stage": PolicyFeature(shape=(9, 5), type=FeatureType.REWARD),
        "progress": PolicyFeature(shape=(9, 1), type=FeatureType.REWARD),
    })
    
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

        # Add the image_key as VISUAL
        if self.image_key:
            self.input_features[self.image_key] = PolicyFeature(
                shape=(480, 640, 3),
                type=FeatureType.VISUAL
            )
        
        # Add state_key as STATE
        self.input_features[self.state_key] = PolicyFeature(
            shape=(self.max_state_dim,),  # Single frame state, temporal sampling handles sequence
            type=FeatureType.STATE
        )
        
        # Update output features based on dual_sparse_dense mode
        if self.dual_sparse_dense:
            # Dual head mode: separate outputs for sparse and dense
            self.output_features["sparse_stage"] = PolicyFeature(
                shape=(self.num_frames, self.num_sparse_stages), 
                type=FeatureType.REWARD
            )
            self.output_features["sparse_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), 
                type=FeatureType.REWARD
            )
            dense_stages = self.num_dense_stages or self.num_sparse_stages
            self.output_features["dense_stage"] = PolicyFeature(
                shape=(self.num_frames, dense_stages), 
                type=FeatureType.REWARD
            )
            self.output_features["dense_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), 
                type=FeatureType.REWARD
            )
        else:
            # Single head mode: sparse only
            self.output_features["sparse_stage"] = PolicyFeature(
                shape=(self.num_frames, self.num_sparse_stages), 
                type=FeatureType.REWARD
            )
            self.output_features["sparse_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), 
                type=FeatureType.REWARD
            )
        
        # Validate configuration
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        
        if self.max_length != self.num_frames:
            raise ValueError(
                f"max_length ({self.max_length}) must equal num_frames ({self.num_frames})"
            )
        
        if self.num_sparse_stages < 2:
            raise ValueError(f"num_sparse_stages must be at least 2, got {self.num_sparse_stages}")
        
        # Validate dual mode configuration
        if self.dual_sparse_dense:
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
    def observation_delta_indices(self) -> list[int]:
        """Load frames for SARM temporal sampling.
        
        Per SARM paper (Section A.4), the model uses 9 frames:
        - Frame 0: Initial frame of the episode
        - Frames 1-8: 8 consecutive frames with frame_gap spacing ending at current frame
        
        The first delta uses a large negative offset (-1_000_000) that will be clamped
        to the episode start (frame 0) by the dataset loader. This ensures we always
        get the initial frame regardless of the current position in the episode.
        
        Returns:
            9 delta indices: [-1_000_000, -(7*gap), -(6*gap), ..., -gap, 0]
        """
        initial_frame_delta = -1_000_000
        
        num_consecutive = self.num_frames - 1 # 9 - 1 = 8
        consecutive_deltas = list(range(-self.frame_gap * (num_consecutive - 1), 1, self.frame_gap)) # [-210, -180, -150, -120, -90, -60, -30, 0]
        return [initial_frame_delta] + consecutive_deltas
    
    @property
    def action_delta_indices(self) -> None:
        """SARM is a reward model, not an action policy."""
        return None
    
    @property
    def reward_delta_indices(self) -> None:
        """SARM doesn't use delta rewards."""
        return None

