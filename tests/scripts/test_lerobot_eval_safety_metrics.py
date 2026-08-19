"""Tests for the LIBERO-Safety cost-signal plumbing added to lerobot_eval.py.

Split into two kinds, per the project's testing guidance for this feature:
  - `test_extract_batched_cost_*`: exercise `_extract_batched_cost` against a REAL
    `gymnasium.vector.SyncVectorEnv` (gymnasium's actual `_add_info` info-batching
    mechanism, not a mock of it) using a minimal fake env that stands in for the real
    LIBERO-Safety simulator. This verifies our extraction logic against real gymnasium
    behavior; it does not touch the LIBERO-Safety simulator itself.
  - `test_eval_policy_*`: exercise the full `eval_policy_all` -> `eval_one` -> `eval_policy`
    -> `rollout` aggregation chain with a fake env + trivial policy, verifying safety
    metrics survive every hop to the final returned dict (mirrors what lands in
    eval_info.json). Also a fake-env test, not a real-simulator test.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.scripts.lerobot_eval import _extract_batched_cost, eval_policy, eval_policy_all


class _NoOpPipeline:
    """Stand-in for PolicyProcessorPipeline: pass batches through unchanged."""

    def __call__(self, x):
        return x


class _ScriptedCostEnv(gym.Env):
    """Minimal LIBERO-Safety-like env: 2-step episodes, `info["cost"]` on every step,
    mirroring `BDDLBaseDomain.step()` (see lerobot.envs.libero.LiberoEnv.step, which passes
    the upstream `cost` dict straight through as `info["cost"]`)."""

    observation_space = gym.spaces.Dict(
        {"agent_pos": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, costs_per_step: list[dict[str, float]], task_description: str = "do a thing"):
        self._costs_per_step = costs_per_step
        self._t = 0
        self.task_description = task_description

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return {"agent_pos": np.zeros(2, dtype=np.float32)}, {"is_success": False, "cost": {}}

    def step(self, action):
        cost = self._costs_per_step[self._t] if self._t < len(self._costs_per_step) else {}
        self._t += 1
        terminated = self._t >= len(self._costs_per_step)
        is_success = terminated
        obs = {"agent_pos": np.full(2, float(self._t), dtype=np.float32)}
        return obs, float(is_success), terminated, False, {"is_success": is_success, "cost": cost}


def _make_vector_env(envs: list[gym.Env]) -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv(
        [lambda e=e: e for e in envs], autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
    )


def test_extract_batched_cost_mid_episode_step():
    """Real SyncVectorEnv.step() with no env finishing this step: `info["cost"]` is the
    recursively-batched dict-of-arrays gymnasium produces (verified against installed
    gymnasium==1.3.0's `VectorEnv._add_info`)."""
    env0 = _ScriptedCostEnv([{"checkgripperforce": 3.0}, {}])
    env1 = _ScriptedCostEnv([{}, {}])
    venv = _make_vector_env([env0, env1])
    venv.reset()
    _, _, _, _, info = venv.step(np.zeros((2, 1), dtype=np.float32))

    costs = _extract_batched_cost(info, num_envs=2)
    assert costs[0] == {"checkgripperforce": 3.0}
    assert costs[1] is None  # env1's cost dict was empty this step -> no entries recorded


def test_extract_batched_cost_prefers_final_info_on_terminal_step():
    """On the step an env terminates, gymnasium nests its terminal info under `final_info`
    (autoreset means the top-level slot for that env describes the *reset* env instead)."""
    env0 = _ScriptedCostEnv([{"checkgripperforce": 1.0}])  # terminates after 1 step
    env1 = _ScriptedCostEnv([{}, {}])
    venv = _make_vector_env([env0, env1])
    venv.reset()
    _, _, terminated, _, info = venv.step(np.zeros((2, 1), dtype=np.float32))

    assert terminated[0]
    costs = _extract_batched_cost(info, num_envs=2)
    assert costs[0] == {"checkgripperforce": 1.0}
    assert costs[1] is None


def test_extract_batched_cost_missing_key_returns_all_none():
    """Non-LIBERO-Safety envs never set `info["cost"]` at all — must never raise."""
    assert _extract_batched_cost({}, num_envs=3) == [None, None, None]
    assert _extract_batched_cost({"is_success": np.array([True, False])}, num_envs=2) == [None, None]


def test_extract_batched_cost_multiple_constraints():
    env0 = _ScriptedCostEnv([{"checkgripperforce": 2.0, "checkforce": 1.0}])
    venv = _make_vector_env([env0])
    venv.reset()
    _, _, _, _, info = venv.step(np.zeros((1, 1), dtype=np.float32))

    costs = _extract_batched_cost(info, num_envs=1)
    assert costs[0] == {"checkgripperforce": 2.0, "checkforce": 1.0}


class _ZeroPolicy(nn.Module):
    """Trivial policy: always outputs a zero action. Enough to drive `rollout()` without
    a real LIBERO-Safety checkpoint."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self._dummy = nn.Parameter(torch.zeros(1))

    def reset(self):
        pass

    def select_action(self, observation):
        batch_size = next(iter(observation.values())).shape[0] if observation else 1
        return torch.zeros(batch_size, self.action_dim)


# rollout()/eval_policy() only isinstance-check against PreTrainedPolicy; register _ZeroPolicy
# as a virtual subclass instead of inheriting the real ABC (which requires config_class/name and
# several abstract methods unrelated to what these tests exercise).
PreTrainedPolicy.register(_ZeroPolicy)


def test_eval_policy_surfaces_safety_metrics_end_to_end():
    """Full rollout() -> eval_policy() chain with a fake env: safety_cost/had_violation/
    violation_steps/constraint_violations must reach eval_policy()'s returned dict."""
    envs = [_ScriptedCostEnv([{"checkgripperforce": 1.0}, {"checkgripperforce": 1.0}, {}])]
    venv = _make_vector_env(envs)
    # _max_episode_steps is read via env.call(...) by rollout().
    venv.call = lambda name: [3] if name == "_max_episode_steps" else [None]

    policy = _ZeroPolicy(action_dim=1)
    noop = _NoOpPipeline()
    result = eval_policy(
        env=venv,
        policy=policy,
        env_preprocessor=noop,
        env_postprocessor=noop,
        preprocessor=noop,
        postprocessor=noop,
        n_episodes=1,
    )

    episode = result["per_episode"][0]
    assert episode["safety_cost"] == pytest.approx(2.0)
    assert episode["had_violation"] is True
    assert episode["violation_steps"] == 2
    assert episode["constraint_violations"] == {"checkgripperforce": pytest.approx(2.0)}
    assert result["aggregated"]["avg_safety_cost"] == pytest.approx(2.0)
    assert result["aggregated"]["pc_had_violation"] == pytest.approx(100.0)


def test_eval_policy_no_cost_key_when_env_reports_none():
    """A non-safety env (never sets info['cost']) must produce eval_policy() output
    identical in shape to before this feature — no safety_cost/aggregated keys at all."""

    class _PlainEnv(_ScriptedCostEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info.pop("cost", None)
            return obs, reward, terminated, truncated, info

        def reset(self, *, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            info.pop("cost", None)
            return obs, info

    envs = [_PlainEnv([{}, {}])]
    venv = _make_vector_env(envs)
    venv.call = lambda name: [2] if name == "_max_episode_steps" else [None]

    policy = _ZeroPolicy(action_dim=1)
    noop = _NoOpPipeline()
    result = eval_policy(
        env=venv,
        policy=policy,
        env_preprocessor=noop,
        env_postprocessor=noop,
        preprocessor=noop,
        postprocessor=noop,
        n_episodes=1,
    )

    assert "safety_cost" not in result["per_episode"][0]
    assert "avg_safety_cost" not in result["aggregated"]


def test_eval_policy_all_preserves_safety_metrics_through_multi_task_aggregation():
    """The outer multi-task orchestrator (eval_one/eval_policy_all, used by the real
    lerobot-eval CLI) must not drop safety fields extracted by eval_policy()."""
    envs = {
        "human_safety": {
            0: _make_vector_env([_ScriptedCostEnv([{"checkgripperforce": 4.0}])]),
        }
    }
    for group in envs.values():
        for vec in group.values():
            vec.call = lambda name: [1] if name == "_max_episode_steps" else [None]

    policy = _ZeroPolicy(action_dim=1)
    noop = _NoOpPipeline()
    result = eval_policy_all(
        envs=envs,
        policy=policy,
        env_preprocessor=noop,
        env_postprocessor=noop,
        preprocessor=noop,
        postprocessor=noop,
        n_episodes=1,
    )

    assert result["per_task"][0]["metrics"]["safety_costs"] == [pytest.approx(4.0)]
    assert result["per_group"]["human_safety"]["avg_safety_cost"] == pytest.approx(4.0)
    assert result["per_group"]["human_safety"]["constraint_violations"] == {
        "checkgripperforce": pytest.approx(4.0)
    }
    assert result["overall"]["avg_safety_cost"] == pytest.approx(4.0)
    assert result["overall"]["constraint_violations"] == {"checkgripperforce": pytest.approx(4.0)}
