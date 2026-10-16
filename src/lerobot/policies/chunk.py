# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Policy declarations for direct, current-observation chunk prediction."""

from dataclasses import dataclass
from enum import StrEnum

from lerobot.transport.wire.features import FeatureSpec as FeatureSpec


class ExecutionMode(StrEnum):
    """An explicitly negotiated action execution mode."""

    CHUNK = "chunk"
    RTC_GUIDED = "rtc_guided"
    RTC_TRAINED = "rtc_trained"


@dataclass(frozen=True)
class ChunkPolicySpec:
    """Policy author declaration; inheriting the default is an explicit contract.

    The current observation suffices, direct chunk calls perform all preparation,
    outputs are [B, prediction_steps, A], and canonical processors accept full chunks.
    Mutable policy/processor state belongs to the exclusive session, never the network.
    """

    prediction_steps: int
    execution_steps: int
    modes: tuple[ExecutionMode, ...] = (ExecutionMode.CHUNK,)
    current_observation_only: bool = True
    retains_session_state: bool = True
    training_max_delay: int = 0

    def __post_init__(self) -> None:
        """Validate the policy's declared prediction and execution lengths."""
        if not 0 < self.execution_steps <= self.prediction_steps:
            raise ValueError("Execution length must be positive and no longer than prediction length.")
