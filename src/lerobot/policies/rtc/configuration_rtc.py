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

from lerobot.configs import RTCAttentionSchedule


def validate_trained_rtc_horizon(
    execution_horizon: int, prediction_steps: int, training_max_delay: int
) -> None:
    """Apply the shared conservative admission bounds for trained RTC.

    The supplied prefix must cover the checkpoint's maximum conditioned delay.
    Keep at least that many prediction steps outside the configured prefix, as
    required by the existing local rollout contract. ``execution_horizon`` is
    prefix capacity, not measured playback or the number of committed actions;
    these configuration bounds do not guarantee timely successor inference.
    """
    if training_max_delay <= 0:
        raise ValueError("Trained RTC requires a checkpoint with rtc_training_max_delay > 0.")
    if prediction_steps <= training_max_delay:
        raise ValueError("Trained RTC prediction_steps must exceed rtc_training_max_delay.")
    if execution_horizon < training_max_delay:
        raise ValueError(
            f"Trained RTC execution_horizon ({execution_horizon}) must be at least the checkpoint's "
            f"maximum conditioned delay rtc_training_max_delay ({training_max_delay})."
        )
    if execution_horizon > prediction_steps - training_max_delay:
        raise ValueError(
            f"Trained RTC execution_horizon ({execution_horizon}) must be at most "
            f"prediction_steps - rtc_training_max_delay ({prediction_steps} - {training_max_delay} = "
            f"{prediction_steps - training_max_delay}) under the conservative admission contract."
        )


@dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting problem,
    strategically handling overlapping timesteps between action chunks using prefix attention.
    """

    # Infrastructure
    enabled: bool = True

    # ``guided`` is the original inference-time Jacobian guidance. ``trained``
    # hard-inpaints a prefix and requires a compatible training-time RTC checkpoint.
    mode: str = "guided"

    # Core RTC settings
    # Todo change to exp
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

    # Debug settings
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """Validate RTC configuration parameters."""
        if self.mode not in {"guided", "trained"}:
            raise ValueError(f"mode must be 'guided' or 'trained', got {self.mode!r}")
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")
