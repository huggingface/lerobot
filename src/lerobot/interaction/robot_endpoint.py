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

"""Adapter from LeRobot's public ``Robot`` API to ``TaskEndpoint``."""

from collections.abc import Callable
from typing import Any, Protocol

import torch

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from .endpoint import Outcome, StepPacer, StepResult, snapshot_value


class RobotEndpointAdapter(Protocol):
    """Conversion between robot-native and canonical task values."""

    @property
    def action_names(self) -> tuple[str, ...]:
        """Canonical action dimension order."""
        ...

    def reset(self) -> None:
        """Reset episode-scoped processor state."""
        ...

    def to_canonical_observation(self, observation: RobotObservation) -> dict[str, Any]:
        """Map a raw robot observation to canonical LeRobot features."""
        ...

    def to_native_action(self, action: torch.Tensor, observation: RobotObservation) -> RobotAction:
        """Map a canonical action to a hardware command."""
        ...

    def to_canonical_applied_action(
        self, action: RobotAction, requested_action: torch.Tensor
    ) -> torch.Tensor:
        """Map the command actually sent by the robot back to canonical space."""
        ...


class ProcessorRobotAdapter:
    """Use current robot processors and dataset feature ordering at the boundary."""

    def __init__(
        self,
        *,
        dataset_features: dict[str, dict[str, Any]],
        action_names: list[str] | tuple[str, ...],
        observation_processor: Callable[[RobotObservation], RobotObservation],
        action_processor: Callable[[tuple[RobotAction, RobotObservation]], RobotAction],
        applied_action_processor: Callable[[RobotAction, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Bind current robot processors and declared action ordering."""
        self.dataset_features = dataset_features
        self.action_names = tuple(action_names)
        dataset_action_names = dataset_features.get(ACTION, {}).get("names")
        if dataset_action_names is not None and tuple(dataset_action_names) != self.action_names:
            raise ValueError(
                "Robot adapter action order differs from dataset features: "
                f"adapter={self.action_names!r}, dataset={tuple(dataset_action_names)!r}"
            )
        self.observation_processor = observation_processor
        self.action_processor = action_processor
        self.applied_action_processor = applied_action_processor

    def reset(self) -> None:
        """Reset stateful robot processors once per episode."""
        seen: set[int] = set()
        for processor in (self.observation_processor, self.action_processor):
            if id(processor) in seen:
                continue
            seen.add(id(processor))
            reset = getattr(processor, "reset", None)
            if reset is not None:
                reset()

    def to_canonical_observation(self, observation: RobotObservation) -> dict[str, Any]:
        """Process a robot observation into a canonical dataset frame."""
        processed = self.observation_processor(observation)
        return build_dataset_frame(self.dataset_features, processed, prefix=OBS_STR)

    def to_native_action(self, action: torch.Tensor, observation: RobotObservation) -> RobotAction:
        """Name canonical dimensions and apply the current robot action processor."""
        flat_action = action.detach().cpu()
        if flat_action.ndim == 2 and flat_action.shape[0] == 1:
            flat_action = flat_action[0]
        if flat_action.ndim != 1 or len(flat_action) != len(self.action_names):
            raise ValueError(
                f"Canonical action shape {tuple(action.shape)} does not match "
                f"{len(self.action_names)} declared action names"
            )
        canonical_dict = {name: float(flat_action[index]) for index, name in enumerate(self.action_names)}
        return self.action_processor((canonical_dict, observation))

    def to_canonical_applied_action(
        self, action: RobotAction, requested_action: torch.Tensor
    ) -> torch.Tensor:
        """Map the robot-reported command back to canonical action order."""
        if self.applied_action_processor is not None:
            return self.applied_action_processor(action, requested_action).detach().cpu()
        missing = [name for name in self.action_names if name not in action]
        if missing:
            raise ValueError(
                "Applied robot action cannot be mapped to canonical space because it is "
                f"missing keys {missing}. Pass applied_action_processor for actions that "
                "change coordinate systems."
            )
        return torch.as_tensor([action[name] for name in self.action_names])


OutcomeSource = Callable[[dict[str, Any], torch.Tensor, dict[str, Any]], Outcome]


class RobotEndpoint:
    """One real robot behind the shared endpoint contract.

    The optional ``outcome_source`` is where a reward classifier, human label or
    task-specific success detector plugs in. Without one, the mandatory runtime
    horizon still ends the episode safely.
    """

    def __init__(
        self,
        robot: Robot,
        *,
        adapter: RobotEndpointAdapter,
        stop_fn: Callable[[], None],
        pacer: StepPacer,
        reset_fn: Callable[[int | None], None] | None = None,
        outcome_source: OutcomeSource | None = None,
        close_robot: bool = False,
    ) -> None:
        """Bind robot I/O, a deadline pacer and an idempotent safe-stop callback."""
        self.robot = robot
        self.adapter = adapter
        try:
            self._action_names = tuple(adapter.action_names)
        except AttributeError as error:
            raise TypeError("RobotEndpointAdapter must declare canonical action_names") from error
        if not self._action_names:
            raise ValueError("RobotEndpointAdapter.action_names must not be empty")
        self.reset_fn = reset_fn
        self.pacer = pacer
        self.outcome_source = outcome_source
        self.stop_fn = stop_fn
        self.close_robot = close_robot
        self._last_raw_observation: RobotObservation | None = None
        self._last_observation: dict[str, Any] | None = None

    @property
    def action_names(self) -> tuple[str, ...]:
        """Return the endpoint's canonical action order."""
        return self._action_names

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        """Reset processors and task, then read the first observation."""
        self.pacer.cancel_cycle()
        self.adapter.reset()
        if self.outcome_source is not None:
            reset_outcome = getattr(self.outcome_source, "reset", None)
            if reset_outcome is not None:
                reset_outcome()
        if self.reset_fn is not None:
            self.reset_fn(seed)
        raw_observation = self.robot.get_observation()
        observation = snapshot_value(self.adapter.to_canonical_observation(raw_observation))
        self._last_raw_observation = raw_observation
        self._last_observation = observation
        return observation

    def start_step(self) -> None:
        """Start the real control-cycle deadline before policy inference."""
        self.pacer.tick()

    def step(self, action: torch.Tensor) -> StepResult:
        """Convert, send, wait, observe and evaluate one hardware transition."""
        if self._last_raw_observation is None or self._last_observation is None:
            raise RuntimeError("Call reset() before stepping a RobotEndpoint")

        native_action = self.adapter.to_native_action(action, self._last_raw_observation)
        sent_action = self.robot.send_action(native_action)
        applied_action = self.adapter.to_canonical_applied_action(sent_action, action)
        self.pacer.wait()

        raw_observation = self.robot.get_observation()
        observation = snapshot_value(self.adapter.to_canonical_observation(raw_observation))
        outcome = (
            self.outcome_source(self._last_observation, applied_action, observation)
            if self.outcome_source is not None
            else Outcome()
        )
        self._last_raw_observation = raw_observation
        self._last_observation = observation
        return StepResult(
            observation=observation,
            applied_action=applied_action,
            reward=outcome.reward,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
            success=outcome.success,
            info=outcome.info,
        )

    def stop(self) -> None:
        """Run the configured safe-idle command at every episode boundary."""
        try:
            self.stop_fn()
        finally:
            # ``start_step`` runs before inference. If inference fails, no matching
            # ``wait`` occurs and the old deadline must not leak into the next run.
            self.pacer.cancel_cycle()

    def close(self) -> None:
        """Disconnect only when endpoint ownership was explicitly requested."""
        self.stop()
        if self.close_robot and self.robot.is_connected:
            self.robot.disconnect()
