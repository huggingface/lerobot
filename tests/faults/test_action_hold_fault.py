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

"""Focused unit tests for ActionHoldFault evaluation-time injection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.faults import ActionHoldFault, FaultEventLogger, FaultInjectionConfig, make_fault_injector


def _cfg(**kwargs) -> FaultInjectionConfig:
    defaults = {
        "enabled": True,
        "type": "action_hold",
        "trigger_step": 3,
        "duration": 2,
        "probability": 1.0,
        "seed": 42,
        "env_ids": None,
        "log_path": None,
    }
    defaults.update(kwargs)
    return FaultInjectionConfig(**defaults)


def _action(batch: int, dim: int, fill: float) -> np.ndarray:
    return np.full((batch, dim), fill, dtype=np.float32)


def test_actions_before_trigger_unchanged():
    inj = ActionHoldFault(_cfg(trigger_step=3, duration=2), num_envs=1)
    for step, fill in enumerate([1.0, 2.0, 3.0]):
        proposed = _action(1, 4, fill)
        out = inj.apply(proposed)
        np.testing.assert_array_equal(out, proposed)
        assert step < 3


def test_holds_previous_action_during_fault():
    inj = ActionHoldFault(_cfg(trigger_step=2, duration=3), num_envs=1)
    outs = []
    for fill in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]:
        outs.append(inj.apply(_action(1, 2, fill)).copy())

    # steps 0,1 pass through; trigger at 2 holds action from step 1 (20) for 3 steps
    np.testing.assert_array_equal(outs[0], [[10, 10]])
    np.testing.assert_array_equal(outs[1], [[20, 20]])
    np.testing.assert_array_equal(outs[2], [[20, 20]])  # hold
    np.testing.assert_array_equal(outs[3], [[20, 20]])  # hold
    np.testing.assert_array_equal(outs[4], [[20, 20]])  # hold
    np.testing.assert_array_equal(outs[5], [[60, 60]])  # resume


def test_resumes_after_duration():
    inj = ActionHoldFault(_cfg(trigger_step=1, duration=1), num_envs=1)
    assert inj.apply(_action(1, 3, 1.0))[0, 0] == 1.0
    assert inj.apply(_action(1, 3, 2.0))[0, 0] == 1.0  # hold previous
    assert inj.apply(_action(1, 3, 3.0))[0, 0] == 3.0  # resume


def test_reset_clears_episode_state():
    inj = ActionHoldFault(_cfg(trigger_step=2, duration=5), num_envs=1)
    inj.apply(_action(1, 2, 1.0))
    inj.apply(_action(1, 2, 2.0))
    held = inj.apply(_action(1, 2, 99.0))
    np.testing.assert_array_equal(held, [[2, 2]])
    inj.reset()
    # New episode: before trigger again; no leftover hold.
    out = inj.apply(_action(1, 2, 7.0))
    np.testing.assert_array_equal(out, [[7, 7]])
    out = inj.apply(_action(1, 2, 8.0))
    np.testing.assert_array_equal(out, [[8, 8]])
    out = inj.apply(_action(1, 2, 9.0))
    np.testing.assert_array_equal(out, [[8, 8]])  # new episode hold of 8


def test_disabled_is_exact_noop():
    cfg = _cfg(enabled=False)
    assert make_fault_injector(cfg, num_envs=2) is None
    inj = ActionHoldFault(cfg, num_envs=1)
    proposed = _action(1, 4, 5.0)
    out = inj.apply(proposed)
    assert out is proposed
    assert inj.event_logger is None


def test_vector_envs_maintain_separate_state():
    inj = ActionHoldFault(_cfg(trigger_step=1, duration=2), num_envs=2)
    a0 = np.array([[1.0, 1.0], [10.0, 10.0]], dtype=np.float32)
    a1 = np.array([[2.0, 2.0], [20.0, 20.0]], dtype=np.float32)
    a2 = np.array([[3.0, 3.0], [30.0, 30.0]], dtype=np.float32)
    assert np.allclose(inj.apply(a0), a0)
    held = inj.apply(a1)
    # env0 holds 1.0, env1 holds 10.0 — never cross-contaminate
    np.testing.assert_array_equal(held[0], [1.0, 1.0])
    np.testing.assert_array_equal(held[1], [10.0, 10.0])
    held2 = inj.apply(a2)
    np.testing.assert_array_equal(held2[0], [1.0, 1.0])
    np.testing.assert_array_equal(held2[1], [10.0, 10.0])


def test_same_seed_reproducible_activation():
    def _activation_pattern(seed: int) -> list[bool]:
        inj = ActionHoldFault(
            _cfg(trigger_step=1, duration=1, probability=0.5, seed=seed),
            num_envs=1,
        )
        flags = []
        for ep in range(20):
            inj.reset(episode_ids=[ep])
            inj.apply(_action(1, 1, 1.0))
            out = inj.apply(_action(1, 1, 9.0))
            flags.append(bool(np.allclose(out, [[1.0]])))
        return flags

    assert _activation_pattern(123) == _activation_pattern(123)
    # Different seed should (almost surely) differ for p=0.5 over 20 episodes
    assert _activation_pattern(123) != _activation_pattern(999)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"trigger_step": 0}, "trigger_step"),
        ({"duration": 0}, "duration"),
        ({"probability": 1.5}, "probability"),
        ({"env_ids": []}, "env_ids"),
        ({"env_ids": [-1]}, "env_ids"),
        ({"env_ids": [0, 0]}, "duplicates"),
        ({"type": "drop_object"}, "Unsupported fault type"),
    ],
)
def test_invalid_config_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FaultInjectionConfig(enabled=True, **{"trigger_step": 2, "duration": 1, **kwargs})


def test_env_ids_out_of_range():
    cfg = _cfg(env_ids=[0, 2])
    with pytest.raises(ValueError, match="out of range"):
        ActionHoldFault(cfg, num_envs=2)


def test_input_not_mutated_in_place():
    inj = ActionHoldFault(_cfg(trigger_step=1, duration=2), num_envs=1)
    first = _action(1, 2, 1.0)
    second = _action(1, 2, 2.0)
    first_copy = first.copy()
    second_copy = second.copy()
    inj.apply(first)
    held = inj.apply(second)
    np.testing.assert_array_equal(first, first_copy)
    np.testing.assert_array_equal(second, second_copy)
    np.testing.assert_array_equal(held, [[1.0, 1.0]])
    assert held is not second


def test_logging_only_on_fault_activation(tmp_path: Path):
    log_path = tmp_path / "faults.jsonl"
    log_path.write_text("", encoding="utf-8")
    logger = FaultEventLogger(log_path)
    inj = ActionHoldFault(_cfg(trigger_step=2, duration=2), num_envs=1, event_logger=logger)
    for fill in [1.0, 2.0, 3.0, 4.0, 5.0]:
        inj.apply(_action(1, 1, fill), episode_ids=[7])
    logger.close()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2  # activated + completed (duration=2)
    events = [json.loads(line) for line in lines]
    assert events[0]["status"] == "activated"
    assert events[1]["status"] == "completed"
    assert events[0]["evaluation_episode_id"] == 7
    assert events[0]["vector_env_id"] == 0
    assert events[0]["episode_step"] == 2
    assert events[0]["proposed_action"] == [3.0]
    assert events[0]["executed_held_action"] == [2.0]
    assert events[0]["fault_type"] == "action_hold"


def test_event_logger_appends_across_open_close(tmp_path: Path):
    """Multi-task sequential evals share one log file via append mode."""
    log_path = tmp_path / "faults.jsonl"
    log_path.write_text("", encoding="utf-8")
    with FaultEventLogger(log_path) as logger:
        logger.log({"event": "a", "status": "activated"})
    with FaultEventLogger(log_path) as logger:
        logger.log({"event": "a", "status": "completed"})
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["status"] == "activated"
    assert json.loads(lines[1])["status"] == "completed"


def test_disabled_produces_no_log_file_events(tmp_path: Path):
    assert make_fault_injector(_cfg(enabled=False, log_path=tmp_path / "x.jsonl"), num_envs=1) is None
    assert not (tmp_path / "x.jsonl").exists()


def test_new_episode_can_trigger_independently():
    inj = ActionHoldFault(_cfg(trigger_step=1, duration=1), num_envs=1)
    inj.apply(_action(1, 1, 1.0), episode_ids=[0])
    inj.apply(_action(1, 1, 2.0), episode_ids=[0])  # hold
    inj.reset(episode_ids=[1])
    out = inj.apply(_action(1, 1, 5.0), episode_ids=[1])
    np.testing.assert_array_equal(out, [[5.0]])
    held = inj.apply(_action(1, 1, 6.0), episode_ids=[1])
    np.testing.assert_array_equal(held, [[5.0]])


def test_no_previous_action_raises_if_trigger_somehow_reached():
    # trigger_step validated >= 1, but force state to simulate missing prev_action.
    inj = ActionHoldFault(_cfg(trigger_step=1, duration=1), num_envs=1)
    inj._states[0].episode_step = 1
    inj._states[0].prev_action = None
    with pytest.raises(RuntimeError, match="no previous valid action"):
        inj.apply(_action(1, 1, 1.0))


def test_selected_env_ids_only():
    inj = ActionHoldFault(_cfg(trigger_step=1, duration=1, env_ids=[1]), num_envs=2)
    a0 = np.array([[1.0], [10.0]], dtype=np.float32)
    a1 = np.array([[2.0], [20.0]], dtype=np.float32)
    inj.apply(a0)
    out = inj.apply(a1)
    np.testing.assert_array_equal(out[0], [2.0])  # env0 not selected
    np.testing.assert_array_equal(out[1], [10.0])  # env1 held


def test_notify_dones_marks_finished_no_retrigger():
    inj = ActionHoldFault(_cfg(trigger_step=1, duration=5), num_envs=2)
    inj.apply(np.array([[1.0], [10.0]], dtype=np.float32))
    inj.apply(np.array([[2.0], [20.0]], dtype=np.float32))  # both holding
    inj.notify_dones(np.array([True, False]))
    out = inj.apply(np.array([[3.0], [30.0]], dtype=np.float32))
    # env0 finished: pass-through only; env1 still holding 10
    np.testing.assert_array_equal(out[0], [3.0])
    np.testing.assert_array_equal(out[1], [10.0])
    # Mid-batch tail steps must not re-arm a fault on the finished env.
    for fill in (4.0, 5.0, 6.0):
        out = inj.apply(np.array([[fill], [99.0]], dtype=np.float32))
        np.testing.assert_array_equal(out[0], [fill])
        np.testing.assert_array_equal(out[1], [10.0])


def test_probability_zero_never_holds():
    inj = ActionHoldFault(_cfg(trigger_step=1, duration=3, probability=0.0), num_envs=1)
    for fill in (1.0, 2.0, 3.0, 4.0):
        out = inj.apply(_action(1, 1, fill))
        np.testing.assert_array_equal(out, [[fill]])


def test_make_fault_injector_none_when_disabled():
    assert make_fault_injector(None, num_envs=1) is None
    assert make_fault_injector(_cfg(enabled=False), num_envs=1) is None


def test_eval_pipeline_default_fault_disabled():
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.configs import LiberoEnv

    cfg = EvalPipelineConfig(env=LiberoEnv(task="libero_object", task_ids=[0]))
    assert cfg.fault.enabled is False
    assert make_fault_injector(cfg.fault, num_envs=1) is None


def test_fault_off_action_path_is_identity():
    """Simulate the lerobot_eval guard: None injector leaves actions untouched."""
    proposed = _action(2, 7, 0.25)
    fault_injector = make_fault_injector(_cfg(enabled=False), num_envs=2)
    action_numpy = proposed
    if fault_injector is not None:
        action_numpy = fault_injector.apply(action_numpy)
    assert action_numpy is proposed
    np.testing.assert_array_equal(action_numpy, proposed)
