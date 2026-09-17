"""Inference-time action execution: turns a policy's per-chunk `plan_fn()` output into one action
per environment step. Deliberately independent of any model/checkpoint weights (see
`ActionExecutor`'s docstring) -- generation (`SafeDiffVLAPolicy.plan_action_chunk` /
`legacy.modeling_legacy_diffusion.LegacySafeDiffVLAPolicy.plan_action_chunk`) and execution
strategy (queueing, temporal ensembling, completion-gated replanning) are separate concerns that
can vary independently: the same trained checkpoint can be evaluated open-loop for the full
`action_horizon`, replanned every few steps, or temporally ensembled, without retraining.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import Tensor

from .state_predictor import completion_gap


class ExecutorConfig(Protocol):
    """The subset of `SafeDiffVLAConfig` / `legacy.LegacySafeDiffVLAConfig` fields `ActionExecutor`
    reads. Both configs satisfy this structurally (duck typing) -- no shared base class needed."""

    action_horizon: int
    execute_horizon: int
    use_temporal_ensembling: bool
    temporal_ensemble_coeff: float
    use_completion_gate: bool
    completion_threshold: float
    max_replan_retries: int


class ActionExecutor:
    """Owns all per-episode execution state (action queue, temporal-ensembling buffer, completion
    gate) and the policy for turning a fresh `plan_fn()` chunk into the action for *this* step.
    Holds no model parameters of its own.
    """

    def __init__(self, config: ExecutorConfig) -> None:
        self._action_horizon = config.action_horizon
        self._execute_horizon = config.execute_horizon
        self._use_temporal_ensembling = config.use_temporal_ensembling
        self._temporal_ensemble_coeff = config.temporal_ensemble_coeff
        self._use_completion_gate = config.use_completion_gate
        self._completion_threshold = config.completion_threshold
        self._max_replan_retries = config.max_replan_retries
        self.reset()

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self._execute_horizon)
        # Holds up to `action_horizon` past chunk predictions, oldest first, for temporal
        # ensembling: the k-th most recently appended chunk was queried k steps ago, so its
        # prediction for "now" lives at its own index k (`_ensembled_action` below).
        self._ensemble_buffer: deque[Tensor] = deque(maxlen=self._action_horizon)
        # Subgoal state predicted as of the last *fully committed* chunk, and the gap to it
        # measured at that same moment (see `select_action`'s completion gate). Both stay None
        # forever for architectures/policies with no subgoal signal.
        self._pending_target_state: Tensor | None = None
        self._last_gap: Tensor | None = None
        self._replan_retries = 0

    def _ensembled_action(self, chunk: Tensor) -> Tensor:
        """Blend "now"-predictions from every buffered chunk with exponential-decay weights.

        `chunk` (this step's fresh prediction) is pushed last, so iterating the buffer newest
        -> oldest via `reversed()` lines up positional age with the offset each chunk holds its
        prediction for "now" at: age 0 is `chunk` itself (offset 0), age 1 is last step's chunk
        (offset 1, since it was queried one step ago), and so on.
        """
        self._ensemble_buffer.append(chunk)
        predictions, weights = [], []
        for age, past_chunk in enumerate(reversed(self._ensemble_buffer)):
            predictions.append(past_chunk[:, age])
            weights.append(math.exp(-self._temporal_ensemble_coeff * age))
        weights_t = torch.tensor(weights, device=chunk.device, dtype=chunk.dtype)
        weights_t /= weights_t.sum()
        return (torch.stack(predictions, dim=0) * weights_t[:, None, None]).sum(dim=0)

    def select_action(
        self, current_state: Tensor, plan_fn: Callable[[], tuple[Tensor, dict[str, Any]]]
    ) -> Tensor:
        """`current_state`: `[B, state_dim]`, the current (t=0) state, used only for the
        completion gate. `plan_fn`: calls the policy's `plan_action_chunk` (or
        `predict_action_chunk`, for the temporal-ensembling path) and returns its
        `(actions, metrics)` pair."""
        if self._use_temporal_ensembling:
            actions, _ = plan_fn()
            return self._ensembled_action(actions)
        if not self._action_queue:
            # A subgoal can legitimately be many chunks away, so "not yet arrived after one
            # execute_horizon" is the normal case, not a problem: gating on that (as an earlier
            # version of this method did) made the gate fire on almost every commit, collapsing
            # execution into a near-permanent single-step replan loop and producing visibly jerky
            # motion. What actually signals trouble is the gap *growing* since the last check: the
            # last chunk moved away from the target it was aiming for, while still being
            # meaningfully far from it.
            gap = (
                completion_gap(self._pending_target_state, current_state)
                if self._pending_target_state is not None
                else None
            )
            diverging = (
                gap is not None
                and self._last_gap is not None
                and bool((gap > self._last_gap).any())
                and bool((gap > self._completion_threshold).any())
            )
            chunk, metrics = plan_fn()
            # Policies with no subgoal signal never gate: there's nothing to measure progress
            # against, so always commit a fresh chunk.
            has_subgoal = self._use_completion_gate and "predicted_subgoal_state" in metrics
            if has_subgoal and diverging and self._replan_retries < self._max_replan_retries:
                # Take one corrective step towards the *same* still-pending target and reassess
                # on the very next call, instead of silently moving on to whatever the model
                # proposes next.
                self._action_queue.extend(chunk.transpose(0, 1)[:1])
                self._replan_retries += 1
                self._last_gap = gap
            else:
                if has_subgoal:
                    self._pending_target_state = metrics["predicted_subgoal_state"]
                    self._last_gap = completion_gap(self._pending_target_state, current_state)
                self._action_queue.extend(chunk.transpose(0, 1)[: self._execute_horizon])
                self._replan_retries = 0
        return self._action_queue.popleft()
