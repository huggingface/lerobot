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

"""
Real Time Chunking (RTC) and Bidirectional Decoding (BID) configuration classes.

Based on:
- Real Time Chunking: https://www.physicalintelligence.company/research/real_time_chunking
"""

from dataclasses import dataclass

from lerobot.configs.types import RTCAttentionSchedule


@dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting problem,
    strategically handling overlapping timesteps between action chunks using prefix attention.
    """

    # Infrastructure
    enabled: bool = False

    # Core RTC settings
    # Todo change to exp
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR

    # This parameter is used to clip the guidance weight
    # In the original RTC it's a hyperparameter that can be tuned to balance the smoothness of the transitions and the reactivity of the policy.
    # The original paper had value of 5.0, during the implementation it was found that this parameter is not needed and can be replaced with the number of steps.
    # Check the following paper - https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html
    # num of steps could be used as clipping parameter without requirements on hyperparameters tuning
    # If user doesn't provide this parameter, than the number of flow matching steps will be used as max guidance weight
    max_guidance_weight: float | None = None
    execution_horizon: int = 10

    # This parameter is used to clip the variance of the prior distribution
    # Check the following paper - https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html
    # The value could be in range of [0, 1], if it's 1.0, than the behavior is the same as the original RTC
    sigma_d: float = 1.0

    full_trajectory_alignment: bool = False

    # Debug settings
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """Validate RTC configuration parameters."""
        if self.max_guidance_weight is not None and self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")
