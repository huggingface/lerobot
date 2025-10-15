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

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig


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
    max_length: int = 32  # Maximum video sequence length
    subsample_video: bool = True  # Whether to pad/subsample videos to max_length
    
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

