#!/usr/bin/env python

# Copyright 2026 Gangelia and The HuggingFace Inc. team. All rights reserved.
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

"""Action-hold fault injection for LeRobot evaluation rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.logging import FaultEventLogger


@dataclass
class _EnvFaultState:
    episode_step: int = 0
    prev_action: np.ndarray | None = None
    remaining: int = 0
    activated: bool = False
    # Cached Bernoulli decision for this episode (None until trigger is reached).
    will_activate: bool | None = None
    episode_id: int | None = None
    # Sticky done within a vec-env batch: pass through only (no re-trigger).
    finished: bool = False


class ActionHoldFault:
    """Repeat the previous valid action for a configured duration at a trigger step.

    Integration point: call :meth:`apply` on the final postprocessed numpy action
    batch immediately before ``env.step(...)``. Do not modify policy internals.

    Per-environment state is maintained so vectorized evaluation never reuses
    another environment's previous action.
    """

    def __init__(
        self,
        config: FaultInjectionConfig,
        num_envs: int,
        event_logger: FaultEventLogger | None = None,
    ):
        if config.type != "action_hold":
            raise ValueError(f"ActionHoldFault requires type='action_hold', got {config.type!r}.")
        config.validate(num_envs=num_envs)
        self.config = config
        self.num_envs = num_envs
        self.event_logger = event_logger
        self._selected = set(range(num_envs)) if config.env_ids is None else set(config.env_ids)
        seed = 0 if config.seed is None else int(config.seed)
        # One Generator for all envs; draws happen in env-index order at trigger time.
        self._rng = np.random.default_rng(seed)
        self._states = [_EnvFaultState() for _ in range(num_envs)]

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def reset(
        self,
        env_ids: list[int] | None = None,
        episode_ids: list[int] | dict[int, int] | None = None,
    ) -> None:
        """Clear episode-specific state for the given environments (or all)."""
        indices = range(self.num_envs) if env_ids is None else env_ids
        for i in indices:
            if i < 0 or i >= self.num_envs:
                raise ValueError(f"env_id {i} out of range for num_envs={self.num_envs}.")
            ep_id = None
            if isinstance(episode_ids, dict):
                ep_id = episode_ids.get(i)
            elif isinstance(episode_ids, list):
                # Interpreted as aligned with ``indices`` when lengths match, else by env index.
                if len(episode_ids) == len(list(indices)):
                    ep_id = episode_ids[list(indices).index(i)]
                elif i < len(episode_ids):
                    ep_id = episode_ids[i]
            self._states[i] = _EnvFaultState(episode_id=ep_id)

    def notify_dones(self, dones: np.ndarray) -> None:
        """Mark finished envs so mid-batch tail steps cannot re-trigger a fault."""
        dones = np.asarray(dones, dtype=bool)
        if dones.shape != (self.num_envs,):
            raise ValueError(f"dones must have shape ({self.num_envs},), got {dones.shape}.")
        for i, done in enumerate(dones):
            if done:
                ep_id = self._states[i].episode_id
                self._states[i] = _EnvFaultState(episode_id=ep_id, finished=True)

    def apply(
        self,
        actions: np.ndarray,
        episode_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Return actions to execute, possibly holding previous actions.

        Args:
            actions: Proposed actions with shape ``(num_envs, action_dim)``.
            episode_ids: Optional evaluation episode ids aligned with env indices.

        Returns:
            Executed action array. When disabled, returns ``actions`` unchanged
            (same object). When enabled, returns a new array and never mutates
            the input in place.
        """
        if not self.config.enabled:
            return actions

        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected actions with shape ({self.num_envs}, action_dim), got {actions.shape}."
            )

        executed = actions.copy()

        for env_idx in range(self.num_envs):
            if episode_ids is not None:
                self._states[env_idx].episode_id = episode_ids[env_idx]

            if env_idx not in self._selected:
                # Still track previous action / step for selected-env consistency if
                # selection changes mid-run is not supported; unselected envs pass through.
                self._pass_through(env_idx, executed, actions)
                continue

            state = self._states[env_idx]
            if state.finished:
                # Done sticky in lerobot_eval batch rollouts: do not re-arm faults.
                executed[env_idx] = actions[env_idx]
                continue

            if state.remaining > 0:
                if state.prev_action is None:
                    raise RuntimeError(
                        f"ActionHoldFault env {env_idx}: remaining={state.remaining} but prev_action is None."
                    )
                held = state.prev_action.copy()
                proposed = actions[env_idx].copy()
                executed[env_idx] = held
                state.remaining -= 1
                status = "active" if state.remaining > 0 else "completed"
                self._log_event(
                    env_idx=env_idx,
                    status=status,
                    proposed=proposed,
                    held=held,
                )
                state.episode_step += 1
                continue

            # Not currently holding: maybe activate at trigger, else pass through.
            if (
                state.episode_step == self.config.trigger_step
                and not state.activated
                and state.will_activate is None
            ):
                state.will_activate = bool(self._rng.random() < self.config.probability)

            if state.episode_step == self.config.trigger_step and not state.activated and state.will_activate:
                if state.prev_action is None:
                    raise RuntimeError(
                        f"ActionHoldFault cannot activate at trigger_step="
                        f"{self.config.trigger_step} for env {env_idx}: no previous valid "
                        "action exists. Choose trigger_step >= 1 and ensure at least one "
                        "action was executed before the trigger."
                    )
                state.activated = True
                state.remaining = self.config.duration
                held = state.prev_action.copy()
                proposed = actions[env_idx].copy()
                executed[env_idx] = held
                state.remaining -= 1
                status = "activated" if state.remaining > 0 else "completed"
                self._log_event(
                    env_idx=env_idx,
                    status=status,
                    proposed=proposed,
                    held=held,
                )
                state.episode_step += 1
                continue

            self._pass_through(env_idx, executed, actions)

        return executed

    def _pass_through(self, env_idx: int, executed: np.ndarray, actions: np.ndarray) -> None:
        state = self._states[env_idx]
        executed[env_idx] = actions[env_idx]
        state.prev_action = actions[env_idx].copy()
        state.episode_step += 1

    def _log_event(
        self,
        *,
        env_idx: int,
        status: str,
        proposed: np.ndarray,
        held: np.ndarray,
    ) -> None:
        if self.event_logger is None:
            return
        state = self._states[env_idx]
        self.event_logger.log(
            {
                "event": "action_hold",
                "status": status,
                "fault_type": self.config.type,
                "evaluation_episode_id": state.episode_id,
                "vector_env_id": env_idx,
                "episode_step": state.episode_step,
                "trigger_step": self.config.trigger_step,
                "duration": self.config.duration,
                "probability": self.config.probability,
                "seed": self.config.seed,
                "remaining_after": state.remaining,
                "proposed_action": proposed.astype(float).tolist(),
                "executed_held_action": held.astype(float).tolist(),
            }
        )


def make_fault_injector(
    config: FaultInjectionConfig | None,
    num_envs: int,
    log_path: str | None = None,
) -> ActionHoldFault | None:
    """Build an injector from config, or ``None`` when disabled / unset."""
    if config is None or not config.enabled:
        return None
    config.validate(num_envs=num_envs)
    path = log_path if log_path is not None else config.log_path
    logger = FaultEventLogger(path) if path is not None else FaultEventLogger(None)
    if config.type == "action_hold":
        return ActionHoldFault(config=config, num_envs=num_envs, event_logger=logger)
    raise ValueError(f"Unsupported fault type {config.type!r}.")
