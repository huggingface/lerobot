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
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("rewind")
@dataclass
class ReWiNDConfig(PreTrainedConfig):
    """Configuration class for ReWiND Reward Model.
    
    ReWiND (Reward from Video and Natural language Descriptions) is a reward model
    that computes task completion/progress rewards from video observations and 
    language task descriptions.
    """
    
    # Model architecture parameters
    video_dim: int = 768  # DINO embedding dimension
    text_dim: int = 384  # MiniLM embedding dimension
    hidden_dim: int = 512  
    num_heads: int = 8  
    num_layers: int = 4  
    
    # Temporal parameters
    max_length: int = 32  # Maximum video sequence length, ORIGINAL: 16!
    subsample_video: bool = True  # Whether to pad/subsample videos to max_length
    use_temporal_sampler: bool = True  # Always enable temporal sequence loading
    sequence_stride: int = 1  # Stride between frames when using temporal sampler
    rewind_ratio: float = 0.8  # Probability of applying rewind augmentation (original: 0.8)
    
    # Training parameters
    batch_size: int = 64 
    dino_batch_size: int = 64  
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    
    # Model loading
    pretrained_model_path: str | None = None 
    
    # Device settings
    device: str | None = None 
    
    # Dropout
    dropout: float = 0.1  # Dropout rate for transformer
    
    # Processor settings (for automatic preprocessing)
    image_key: str = "observation.images.top"  # Key for images in dataset
    task_description: str = "perform the task"  # Default task description (used if no task field in data)
    encode_on_the_fly: bool = True  # Encode images/text during training
    use_dataset_task: bool = True  # Use task descriptions from dataset (per-episode)
    
    # Features (required by PreTrainedPolicy)
    input_features: dict = field(default_factory=lambda: {
        "video_features": {"shape": [768], "dtype": "float32"},
        "text_features": {"shape": [384], "dtype": "float32"}
    })
    output_features: dict = field(default_factory=lambda: {
        "progress": {"shape": [1], "dtype": "float32"}
    })
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate configuration
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
    
    def get_optimizer_preset(self) -> AdamWConfig:
        """Get default optimizer configuration for ReWiND training."""
        return AdamWConfig(
            lr=1e-4, 
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Get default learning rate scheduler configuration."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=1e-4,  
            decay_lr=1e-5, 
            num_warmup_steps=1000,
            num_decay_steps=100000, 
        )
    
    def validate_features(self) -> None:
        pass
    
    @property
    def observation_delta_indices(self) -> list[int]:
        """Load all frames from episode start up to current frame.
        
        The sampler yields a random end point in each episode.
        This property tells the dataset to load all frames from -(end_idx - start_idx) to 0.
        
        Since we don't know the exact window size in advance, we load up to max_length frames.
        The dataset will automatically clamp to episode boundaries.
        
        Returns:
            Indices for loading history: [-31, -30, ..., -1, 0] for max_length=32
        """
        # Load the last max_length frames (or up to episode start)
        return list(range(-(self.max_length - 1), 1))
    
    @property
    def action_delta_indices(self) -> None:
        """ReWiND is a reward model, not an action policy."""
        return None
    
    @property
    def reward_delta_indices(self) -> None:
        """ReWiND doesn't use delta rewards."""
        return None

