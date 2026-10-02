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

"""One small scalar interaction loop for real robots and simulation."""

import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import torch

from .endpoint import StepResult, TaskEndpoint, snapshot_value
from .record import EpisodeRecorder, NullRecorder, StepRecord


@runtime_checkable
class ActionProvider(Protocol):
    """Episode-scoped action interface implemented by ``SyncInferenceEngine``."""

    def reset(self) -> None:
        """Reset policy and processor state."""
        ...

    @property
    def action_names(self) -> tuple[str, ...]:
        """Canonical order of action dimensions produced by the policy."""
        ...

    def get_action(self, observation: dict[str, Any] | None) -> torch.Tensor | None:
        """Return one canonical action for a canonical observation."""
        ...


@dataclass(frozen=True)
class EpisodeResult:
    """Summary returned by ``run_episode``."""

    episode_id: str
    num_steps: int
    terminated: bool
    truncated: bool
    success: bool | None
    total_reward: float | None


def _scalar_action(action: torch.Tensor) -> torch.Tensor:
    action = action.detach().cpu()
    if action.ndim == 2 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 1:
        raise ValueError(
            "The scalar interaction runtime expects action shape (action_dim,) or "
            f"(1, action_dim), got {tuple(action.shape)}"
        )
    return action.clone()


def _validate_action_names(endpoint: TaskEndpoint, action_provider: ActionProvider) -> None:
    endpoint_names = getattr(endpoint, "action_names", None)
    provider_names = getattr(action_provider, "action_names", None)
    if endpoint_names is None:
        if provider_names:
            raise ValueError("A named action provider requires the endpoint to declare the same action_names")
        return
    if provider_names is None:
        raise ValueError("A named endpoint requires the action provider to declare action_names")
    if tuple(endpoint_names) != tuple(provider_names):
        raise ValueError(
            "Policy and endpoint action order differ: "
            f"policy={tuple(provider_names)!r}, endpoint={tuple(endpoint_names)!r}"
        )


def run_episode(
    *,
    endpoint: TaskEndpoint,
    action_provider: ActionProvider,
    max_steps: int,
    recorder: EpisodeRecorder | None = None,
    seed: int | None = None,
    episode_id: str | None = None,
    actor_id: str | None = None,
) -> EpisodeResult:
    """Run one batch-size-one episode through the same path on hardware and in sim.

    ``max_steps`` is mandatory so a missing real-world outcome source can never run
    hardware forever. Provider-native termination is always honored and the final
    transition is marked truncated when this shared horizon is reached.

    The caller owns ``endpoint.close()`` and inference-engine process lifecycle. This
    function resets episode state and always calls ``endpoint.stop()`` before returning.
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")

    recorder = recorder or NullRecorder()
    episode_id = episode_id or str(uuid.uuid4())
    _validate_action_names(endpoint, action_provider)
    num_steps = 0
    final_result: StepResult | None = None
    reward_sum = 0.0
    rewards_complete = True
    episode_success: bool | None = None
    recorder_started = False
    try:
        action_provider.reset()
        observation = endpoint.reset(seed=seed)
        recorder.start_episode(episode_id)
        recorder_started = True
        for step_index in range(max_steps):
            endpoint.start_step()
            policy_started_at = time.perf_counter()
            action = action_provider.get_action(observation)
            if action is None:
                raise RuntimeError(
                    "The scalar runtime requires one action per step; the action provider returned None"
                )
            action = _scalar_action(action)
            action_ready_at = time.perf_counter()
            recorded_observation = snapshot_value(observation)
            result = endpoint.step(action)
            observed_at = time.perf_counter()

            if step_index + 1 == max_steps and not (result.terminated or result.truncated):
                result = replace(
                    result,
                    truncated=True,
                    info={**result.info, "termination_reason": "max_steps"},
                )

            recorder.add(
                StepRecord(
                    observation=recorded_observation,
                    next_observation=snapshot_value(result.observation),
                    action=action,
                    applied_action=_scalar_action(result.applied_action),
                    reward=result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    success=result.success,
                    info=snapshot_value(result.info),
                    episode_id=episode_id,
                    step_index=step_index,
                    actor_id=actor_id,
                    timestamps={
                        "policy_started_at": policy_started_at,
                        "action_ready_at": action_ready_at,
                        "next_observation_at": observed_at,
                    },
                )
            )
            num_steps += 1
            final_result = result
            if result.reward is not None:
                reward_sum += result.reward
            else:
                rewards_complete = False
            if result.success is not None:
                episode_success = bool(episode_success) or result.success
            if result.terminated or result.truncated:
                break
            observation = result.observation
    except BaseException as error:
        try:
            endpoint.stop()
        except BaseException as stop_error:
            error.add_note(f"Endpoint also failed to stop safely: {stop_error!r}")
        if recorder_started:
            try:
                recorder.abort_episode(episode_id)
            except BaseException as abort_error:
                error.add_note(f"Recorder also failed to abort the episode: {abort_error!r}")
        raise
    else:
        try:
            endpoint.stop()
            recorder.end_episode(episode_id)
        except BaseException as error:
            try:
                recorder.abort_episode(episode_id)
            except BaseException as abort_error:
                error.add_note(f"Recorder also failed to abort the episode: {abort_error!r}")
            raise

    if final_result is None:  # pragma: no cover - max_steps validation makes this unreachable
        raise RuntimeError("Episode completed without applying an action")
    return EpisodeResult(
        episode_id=episode_id,
        num_steps=num_steps,
        terminated=final_result.terminated,
        truncated=final_result.truncated,
        success=episode_success,
        total_reward=reward_sum if rewards_complete else None,
    )
