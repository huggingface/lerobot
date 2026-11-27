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

"""How a policy's ``predict_action_chunk`` is driven."""

from dataclasses import dataclass
from enum import StrEnum


class ExecutionMode(StrEnum):
    """An explicitly negotiated action execution mode."""

    CHUNK = "chunk"
    RTC_GUIDED = "rtc_guided"
    RTC_TRAINED = "rtc_trained"


@dataclass(frozen=True)
class ChunkPolicySpec:
    """How a policy's ``predict_action_chunk`` is driven; the default derives it from the config.

    ``predict_action_chunk`` receives batches in the training layout. What it receives is not
    declared here but by the config's ``inference_deltas``: every declared observation arrives
    as ``[B, T, ...]`` with ``{feature}_is_pad``, past executed commands as
    ``observation.action``.

    Attributes:
        prediction_steps: Actions a call returns, ``[B, prediction_steps, A]``.
        execution_steps: Actions executed from each chunk before the next call.
        modes: Execution modes the policy supports.
        retains_session_state: Whether calls keep state between them (cleared by ``reset``).
        training_max_delay: Largest inference delay the policy was trained for (trained RTC).
        complete_execution: Every chunk executes fully before the next call, which observes the
            result of its last action, for policies whose state follows the executed actions.
            Such a policy may return fewer than ``prediction_steps`` actions (e.g. a shorter
            first chunk).
    """

    prediction_steps: int
    execution_steps: int
    modes: tuple[ExecutionMode, ...] = (ExecutionMode.CHUNK,)
    retains_session_state: bool = True
    training_max_delay: int = 0
    complete_execution: bool = False

    def __post_init__(self) -> None:
        """Validate lengths and complete execution."""
        if not 0 < self.execution_steps <= self.prediction_steps:
            raise ValueError("Execution length must be positive and no longer than prediction length.")
        if self.complete_execution and (
            tuple(self.modes) != (ExecutionMode.CHUNK,) or self.execution_steps != self.prediction_steps
        ):
            raise ValueError("Complete execution runs whole plain chunks.")

    def accepts(self, steps: int) -> bool:
        """Whether a call may return a chunk of ``steps`` actions."""
        if self.complete_execution:
            return 0 < steps <= self.prediction_steps
        return steps == self.prediction_steps

    def steps_to_execute(self, steps: int) -> int:
        """How many actions of a returned chunk of ``steps`` actions execute in plain chunk mode."""
        return steps if self.complete_execution else min(self.execution_steps, steps)
