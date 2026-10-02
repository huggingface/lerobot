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

"""The small world-I/O contract shared by real robots and simulators."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


def snapshot_value(value: Any) -> Any:
    """Copy nested canonical values so provider-owned buffers cannot mutate history."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_value(item) for item in value)
    copy = getattr(value, "copy", None)
    if callable(copy):
        try:
            return copy()
        except TypeError:
            pass
    return deepcopy(value)


@dataclass(frozen=True)
class Outcome:
    """Provider-independent task outcome for one transition."""

    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    success: bool | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Result of applying one action to a real or simulated endpoint.

    ``observation`` and ``applied_action`` are always in the canonical LeRobot
    feature space. Outcome fields are optional because a simulator usually
    provides them directly while a real robot may need a human or reward model.
    """

    observation: dict[str, Any]
    applied_action: torch.Tensor
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    success: bool | None = None
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TaskEndpoint(Protocol):
    """One scalar task endpoint.

    Implementations own provider-specific reset, observation, action and outcome
    conversion. The interaction runtime therefore has no simulator or robot branch.
    """

    @property
    def action_names(self) -> tuple[str, ...] | None:
        """Canonical action order, or ``None`` only for an anonymous endpoint."""
        ...

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        """Start an episode and return its first canonical observation."""
        ...

    def start_step(self) -> None:
        """Mark the start of one endpoint control cycle."""
        ...

    def step(self, action: torch.Tensor) -> StepResult:
        """Apply one canonical action and return the resulting transition."""
        ...

    def stop(self) -> None:
        """Put the endpoint in its episode-safe idle state."""
        ...

    def close(self) -> None:
        """Release resources owned by the endpoint."""
        ...


class StepPacer(Protocol):
    """Start-to-start control pacing used by real endpoints."""

    def tick(self) -> None:
        """Mark the start of a control cycle."""
        ...

    def wait(self) -> None:
        """Wait until that control cycle's deadline."""
        ...

    def cancel_cycle(self) -> None:
        """Abandon a started cycle without carrying its deadline forward."""
        ...
