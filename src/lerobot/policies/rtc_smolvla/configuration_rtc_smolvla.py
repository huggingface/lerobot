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
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("rtc_smolvla")
@dataclass
class RTCSmolVLAConfig(SmolVLAConfig):
    """
    Configuration for Real-Time Chunking (RTC) SmolVLA Policy.

    This configuration extends SmolVLAConfig with RTC-specific parameters
    for guided inference as described in "Real-Time Execution of Action Chunking Flow Policies"
    (https://arxiv.org/pdf/2506.07339).
    """

    # RTC-specific hyperparameters
    beta: float = 5.0  # Guidance weight (β in the paper)
    inference_steps: int = (
        4  # Number of inference steps at which to start next inference for real-time chunking
    )
    s_min: int = 1  # Minimum execution horizon before starting next inference
    delay_buffer_size: int = 10  # Size of the delay buffer Q for tracking inference delays
    initial_delay: int = 4  # Initial delay estimate (in control ticks)
