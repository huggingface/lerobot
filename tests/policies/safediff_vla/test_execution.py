"""Unit tests for `ActionExecutor`, decoupled from any policy via a fake `plan_fn` -- proves
execution strategy (queueing, temporal ensembling, completion-gated replanning) works
independently of which model generated the action chunk."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lerobot.policies.safediff_vla.execution import ActionExecutor

ACTION_DIM = 3
HORIZON = 4


@dataclass
class FakeExecutorConfig:
    action_horizon: int = HORIZON
    execute_horizon: int = 2
    use_temporal_ensembling: bool = False
    temporal_ensemble_coeff: float = 0.01
    use_completion_gate: bool = True
    completion_threshold: float = 0.25
    max_replan_retries: int = 4


def make_chunk(batch_size: int = 2, fill: float = 1.0) -> torch.Tensor:
    return torch.full((batch_size, HORIZON, ACTION_DIM), fill)


def make_plan_fn(chunk: torch.Tensor, metrics: dict | None = None):
    calls = {"count": 0}

    def plan_fn():
        calls["count"] += 1
        return chunk, (metrics or {})

    return plan_fn, calls


def test_select_action_queue_drains_and_replans() -> None:
    executor = ActionExecutor(FakeExecutorConfig(execute_horizon=2))
    chunk = make_chunk()
    plan_fn, calls = make_plan_fn(chunk)
    current_state = torch.zeros(2, 5)

    first = executor.select_action(current_state, plan_fn)
    assert torch.equal(first, chunk[:, 0])
    assert len(executor._action_queue) == 1
    assert calls["count"] == 1

    second = executor.select_action(current_state, plan_fn)
    assert torch.equal(second, chunk[:, 1])
    assert len(executor._action_queue) == 0
    assert calls["count"] == 1  # queue not empty on the second call, so plan_fn wasn't re-invoked

    executor.select_action(current_state, plan_fn)
    assert calls["count"] == 2  # queue was empty -- fresh chunk requested


def test_reset_clears_all_state() -> None:
    executor = ActionExecutor(FakeExecutorConfig())
    chunk = make_chunk()
    plan_fn, _ = make_plan_fn(chunk, metrics={"predicted_subgoal_state": torch.ones(2, 5)})
    current_state = torch.zeros(2, 5)
    executor.select_action(current_state, plan_fn)

    executor.reset()
    assert len(executor._action_queue) == 0
    assert len(executor._ensemble_buffer) == 0
    assert executor._pending_target_state is None
    assert executor._last_gap is None
    assert executor._replan_retries == 0


def test_temporal_ensembling_bypasses_queue_and_blends_predictions() -> None:
    executor = ActionExecutor(FakeExecutorConfig(use_temporal_ensembling=True, execute_horizon=1))
    chunk = make_chunk(fill=2.0)
    plan_fn, calls = make_plan_fn(chunk)
    current_state = torch.zeros(2, 5)

    action = executor.select_action(current_state, plan_fn)
    assert calls["count"] == 1
    assert len(executor._action_queue) == 0  # queue never used on this path
    # A single buffered chunk: the ensembled action is exactly that chunk's own "now" prediction.
    assert torch.allclose(action, chunk[:, 0])

    executor.select_action(current_state, plan_fn)
    assert calls["count"] == 2  # temporal ensembling replans every step


def test_completion_gate_does_not_trigger_when_far_but_not_diverging() -> None:
    """Regression test for the original (timescale-mismatched) gate design: it flagged
    "not complete" whenever the gap merely still exceeded `completion_threshold`, which fired on
    almost every commit. With a fixed (never-changing) plan, the measured gap can't grow between
    checks, so the gate must commit a fresh full-length chunk even with `completion_threshold=0.0`
    (i.e. nowhere near "arrived")."""
    executor = ActionExecutor(FakeExecutorConfig(completion_threshold=0.0, max_replan_retries=2))
    chunk = make_chunk()
    plan_fn, _ = make_plan_fn(chunk, metrics={"predicted_subgoal_state": torch.ones(2, 5)})
    current_state = torch.zeros(2, 5)

    executor.select_action(current_state, plan_fn)
    while executor._action_queue:
        executor.select_action(current_state, plan_fn)
    executor.select_action(current_state, plan_fn)
    assert executor._replan_retries == 0
    assert len(executor._action_queue) == executor._execute_horizon - 1


def test_completion_gate_triggers_single_step_replan_when_gap_grows() -> None:
    executor = ActionExecutor(FakeExecutorConfig(completion_threshold=0.0, max_replan_retries=2))
    chunk = make_chunk()
    plan_fn, _ = make_plan_fn(chunk, metrics={"predicted_subgoal_state": torch.ones(2, 5)})
    current_state = torch.zeros(2, 5)

    executor.select_action(current_state, plan_fn)
    while executor._action_queue:
        executor.select_action(current_state, plan_fn)
    # Force the next check to look like the gap grew since the last one.
    executor._last_gap = torch.zeros_like(executor._last_gap)

    executor.select_action(current_state, plan_fn)
    assert executor._replan_retries == 1
    assert len(executor._action_queue) == 0

    executor._last_gap = torch.zeros_like(executor._last_gap)
    executor.select_action(current_state, plan_fn)
    assert executor._replan_retries == 2
    assert len(executor._action_queue) == 0

    # Retry budget exhausted: this call must fall back to a fresh full-length commit regardless.
    executor._last_gap = torch.zeros_like(executor._last_gap)
    executor.select_action(current_state, plan_fn)
    assert executor._replan_retries == 0
    assert len(executor._action_queue) == executor._execute_horizon - 1


def test_use_completion_gate_false_disables_gating_even_with_subgoal() -> None:
    """With the gate off, `_pending_target_state` is never even populated (nothing to gate with),
    and every chunk boundary commits a fresh full-length chunk regardless."""
    executor = ActionExecutor(
        FakeExecutorConfig(completion_threshold=0.0, max_replan_retries=2, use_completion_gate=False)
    )
    chunk = make_chunk()
    plan_fn, _ = make_plan_fn(chunk, metrics={"predicted_subgoal_state": torch.ones(2, 5)})
    current_state = torch.zeros(2, 5)

    for _ in range(3):
        executor.select_action(current_state, plan_fn)
        while executor._action_queue:
            executor.select_action(current_state, plan_fn)
        assert executor._replan_retries == 0
        assert executor._pending_target_state is None


def test_select_action_never_gates_without_subgoal_signal() -> None:
    """A `plan_fn` whose metrics never carry `predicted_subgoal_state` (mirrors `temporal_decoder`
    / `smolvla_nominal`) must never gate, regardless of `completion_threshold`."""
    executor = ActionExecutor(FakeExecutorConfig(completion_threshold=0.0, max_replan_retries=2))
    chunk = make_chunk()
    plan_fn, _ = make_plan_fn(chunk, metrics={})
    current_state = torch.zeros(2, 5)

    executor.select_action(current_state, plan_fn)
    while executor._action_queue:
        executor.select_action(current_state, plan_fn)
    executor.select_action(current_state, plan_fn)
    assert executor._replan_retries == 0
    assert len(executor._action_queue) == executor._execute_horizon - 1
