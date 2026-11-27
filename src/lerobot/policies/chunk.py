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

    A policy that conditions on past frames sets ``current_observation_only=False``
    and declares ``observation_history``: for each policy input feature, the frame
    offsets in policy ticks (``0`` is the current frame, ``-1`` the one before) in
    the order the policy stacks them along dimension 1, as its training
    ``observation_delta_indices`` do. Policies that keep temporal state of their
    own leave it empty and need a custom runner.
    """

    prediction_steps: int
    execution_steps: int
    modes: tuple[ExecutionMode, ...] = (ExecutionMode.CHUNK,)
    current_observation_only: bool = True
    retains_session_state: bool = True
    training_max_delay: int = 0
    observation_history: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        """Validate the policy's declared lengths and observation history."""
        if not 0 < self.execution_steps <= self.prediction_steps:
            raise ValueError("Execution length must be positive and no longer than prediction length.")
        if not self.observation_history:
            return
        if self.current_observation_only:
            raise ValueError("Declared observation history requires current_observation_only=False.")
        names = [name for name, _ in self.observation_history]
        if len(set(names)) != len(names) or not all(isinstance(name, str) and name for name in names):
            raise ValueError("Observation history must name each feature once.")
        for name, offsets in self.observation_history:
            if (
                not offsets
                or len(set(offsets)) != len(offsets)
                or any(type(offset) is not int or offset > 0 for offset in offsets)
            ):
                raise ValueError(f"Observation history offsets for {name} must be distinct integers <= 0.")
