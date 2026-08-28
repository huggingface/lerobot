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
"""Chunk-anchor pinning on the ``lerobot-eval`` path.

``rollout()`` drives the policy directly -- preprocessor, ``select_action``, postprocessor,
once per step -- so a relative-action chunk would be re-anchored to the current (moved)
state on every tick after the one that generated it, and the absolute targets would drift.
Every action of a chunk must resolve to ``r + S0``, the state observed when the chunk was
predicted.

These assertions fail loudly in both directions: without pinning the target drifts with the
arm, and with a second correction stacked on top (e.g. a policy carrying its own private
compensator) it overshoots by ``S0 - Sk``.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
    find_relative_action_step,
    pinned_relative_anchor,
)
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import ACTION, OBS_STATE

ACTION_DIM = 4
ACTION_NAMES = [f"joint_{i}.pos" for i in range(ACTION_DIM)]


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class _MovingEnv(gym.Env):
    """Reports a state that moves a fixed amount every step, so a drifting anchor is visible."""

    metadata = {"render_fps": 30}

    def __init__(self, s0: float, drift: float, max_steps: int):
        box = gym.spaces.Box(low=-1e4, high=1e4, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({"agent_pos": box})
        self.action_space = box
        self.s0 = s0
        self.drift = drift
        self._max_episode_steps = max_steps
        self._t = 0

    def _obs(self):
        # A distinct value per dim so a misshaped anchor cannot pass by symmetry.
        base = self.s0 + self.drift * self._t
        return {"agent_pos": np.array([base + i for i in range(ACTION_DIM)], dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {"is_success": False}

    def step(self, action):
        self._t += 1
        return self._obs(), 0.0, False, self._t >= self._max_episode_steps, {"is_success": False}

    def task_description(self):
        return "move the arm"


class _ChunkingRelativePolicy(nn.Module):
    """Serves a fixed chunk of relative offsets one action per call, refilling when drained.

    Mirrors the ``_action_queue`` shape shared by pi0/pi05/DM05: ``select_action`` predicts a
    whole chunk when the queue is empty, then pops from it without re-predicting.
    """

    def __init__(self, chunk_rel: torch.Tensor):
        super().__init__()
        self.chunk_rel = chunk_rel  # [n, action_dim]
        self._queue: deque[torch.Tensor] = deque()
        self.predict_calls = 0
        self.anchors_seen: list[torch.Tensor] = []

    def reset(self):
        self._queue.clear()

    def queued_action_count(self) -> int:
        return len(self._queue)

    def select_action(self, batch):
        if not self._queue:
            self.predict_calls += 1
            # What the real policies do implicitly: the chunk is generated against the state
            # in *this* batch, so record it to assert the anchor pinning matches it.
            self.anchors_seen.append(batch[OBS_STATE].clone())
            batch_size = batch[OBS_STATE].shape[0]
            for step_rel in self.chunk_rel:
                self._queue.append(step_rel.expand(batch_size, -1).clone())
        return self._queue.popleft()


def _relative_pipelines():
    """Minimal pre/post pipelines carrying the paired relative/absolute steps."""
    relative_step = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=[], action_names=list(ACTION_NAMES)
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    class _Pre:
        steps = [relative_step]

        def __call__(self, observation):
            # Run the relative step so it caches the anchor, then hand the batch through.
            relative_step(create_transition(observation={OBS_STATE: observation[OBS_STATE]}))
            return observation

        def reset(self):
            pass

    class _Post:
        def __call__(self, action):
            return absolute_step(create_transition(action=action))[TransitionKey.ACTION]

        def reset(self):
            pass

    return _Pre(), _Post(), relative_step


class _Identity:
    """Stand-in for the env pre/post processors, which are unrelated to anchoring."""

    def __call__(self, x):
        return x

    def reset(self):
        pass


def _make_env(num_envs: int, s0: float, drift: float, max_steps: int):
    return gym.vector.SyncVectorEnv(
        [lambda: _MovingEnv(s0=s0, drift=drift, max_steps=max_steps) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )


def _chunk(n: int) -> torch.Tensor:
    """A distinct offset per chunk step, so a wrong anchor cannot be masked by a flat chunk."""
    return torch.stack([torch.full((ACTION_DIM,), 0.1 * (i + 1)) for i in range(n)])


def _run_eval(policy, num_envs: int, s0: float, drift: float, max_steps: int):
    pre, post, relative_step = _relative_pipelines()
    env = _make_env(num_envs, s0=s0, drift=drift, max_steps=max_steps)
    try:
        data = rollout(
            env,
            policy,
            env_preprocessor=_Identity(),
            env_postprocessor=_Identity(),
            preprocessor=pre,
            postprocessor=post,
            seeds=[0] * num_envs,
        )
    finally:
        env.close()
    return data["action"], relative_step


def _expected_state(s0: float, drift: float, t: int) -> torch.Tensor:
    return torch.tensor([s0 + drift * t + i for i in range(ACTION_DIM)])


# --------------------------------------------------------------------------------------
# Eval loop
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_envs", [1, 3])
def test_eval_holds_anchor_across_chunk(num_envs):
    """Every action of a chunk resolves to ``r + S0``, for any batch size."""
    n = 4
    chunk_rel = _chunk(n)
    s0, drift = 1.0, 0.5
    policy = _ChunkingRelativePolicy(chunk_rel)

    actions, _ = _run_eval(policy, num_envs, s0=s0, drift=drift, max_steps=n)

    assert policy.predict_calls == 1, "a single chunk must cover the whole episode"
    assert actions.shape == (num_envs, n, ACTION_DIM)
    # S0 is the state at step 0 -- the one the chunk was generated against.
    anchor = _expected_state(s0, drift, 0)
    for tick in range(n):
        expected = anchor + chunk_rel[tick]
        for env_idx in range(num_envs):
            torch.testing.assert_close(actions[env_idx, tick], expected)


def test_eval_anchor_advances_on_chunk_refill():
    """When the queue drains, the next chunk anchors to the state at *that* step."""
    n = 3
    chunk_rel = _chunk(n)
    s0, drift = 1.0, 2.0
    policy = _ChunkingRelativePolicy(chunk_rel)

    actions, relative_step = _run_eval(policy, num_envs=1, s0=s0, drift=drift, max_steps=2 * n)

    assert policy.predict_calls == 2, "two chunks must be predicted across 2n steps"
    for chunk_idx in range(2):
        anchor = _expected_state(s0, drift, chunk_idx * n)
        # The policy saw exactly the state we expect to be anchored to.
        torch.testing.assert_close(policy.anchors_seen[chunk_idx], anchor.unsqueeze(0))
        for i in range(n):
            expected = anchor + chunk_rel[i]
            torch.testing.assert_close(actions[0, chunk_idx * n + i], expected)

    # After the loop the cached anchor is the second chunk's, not a held stale one.
    torch.testing.assert_close(relative_step.get_cached_state(), _expected_state(s0, drift, n).unsqueeze(0))


def test_eval_drifts_without_pinning():
    """Guard the guard: with pinning removed, the same setup drifts."""
    n = 4
    chunk_rel = _chunk(n)
    s0, drift = 1.0, 0.5
    policy = _ChunkingRelativePolicy(chunk_rel)
    pre, post, _ = _relative_pipelines()

    # Reproduce the eval inner loop verbatim, minus `pinned_relative_anchor`.
    outputs = []
    for tick in range(n):
        obs = {OBS_STATE: _expected_state(s0, drift, tick).unsqueeze(0)}
        outputs.append(post(policy.select_action(pre(obs))))

    anchor = _expected_state(s0, drift, 0)
    # Tick 0 is correct by construction; every later tick is off by exactly the arm's motion.
    torch.testing.assert_close(outputs[0][0], anchor + chunk_rel[0])
    for tick in range(1, n):
        moved = _expected_state(s0, drift, tick) - anchor
        torch.testing.assert_close(outputs[tick][0], anchor + chunk_rel[tick] + moved)
        assert not torch.allclose(outputs[tick][0], anchor + chunk_rel[tick])


def test_eval_reanchors_per_episode():
    """A second rollout must anchor to its own initial state, not the previous episode's."""
    n = 3
    chunk_rel = _chunk(n)
    policy = _ChunkingRelativePolicy(chunk_rel)

    _run_eval(policy, num_envs=1, s0=1.0, drift=1.0, max_steps=n)
    actions, _ = _run_eval(policy, num_envs=1, s0=100.0, drift=1.0, max_steps=n)

    anchor = _expected_state(100.0, 1.0, 0)
    for tick in range(n):
        torch.testing.assert_close(actions[0, tick], anchor + chunk_rel[tick])


# --------------------------------------------------------------------------------------
# Eval and the sync engine must agree
# --------------------------------------------------------------------------------------


def test_eval_and_sync_engine_agree_on_the_chunk():
    """Both callers of the shared helper resolve the same chunk to the same absolute targets."""
    from lerobot.rollout import SyncInferenceEngine

    n = 4
    chunk_rel = _chunk(n)
    s0, drift = 1.0, 0.5

    eval_actions, _ = _run_eval(_ChunkingRelativePolicy(chunk_rel), 1, s0=s0, drift=drift, max_steps=n)

    pre, post, _ = _relative_pipelines()
    sync_policy = _ChunkingRelativePolicy(chunk_rel)
    sync_policy.config = type("_Cfg", (), {"use_amp": False, "action_feature_names": list(ACTION_NAMES)})()
    engine = SyncInferenceEngine(
        policy=sync_policy,
        preprocessor=pre,
        postprocessor=post,
        dataset_features={ACTION: {"names": list(ACTION_NAMES)}},
        ordered_action_keys=list(ACTION_NAMES),
        task="test",
        device="cpu",
        robot_type="mock",
    )
    sync_actions = [
        engine.get_action({OBS_STATE: _expected_state(s0, drift, tick).numpy()}) for tick in range(n)
    ]

    assert sync_policy.predict_calls == 1
    for tick in range(n):
        torch.testing.assert_close(sync_actions[tick], eval_actions[0, tick])


# --------------------------------------------------------------------------------------
# Helper semantics
# --------------------------------------------------------------------------------------


def test_pinned_anchor_is_a_noop_without_a_relative_step():
    policy = _ChunkingRelativePolicy(_chunk(2))
    with pinned_relative_anchor(None, policy):
        pass  # must not raise, and must not consult the policy's queue


def test_pinned_anchor_ignores_a_disabled_step():
    """A disabled step still caches state; pinning it would freeze an anchor nobody reads."""
    step = RelativeActionsProcessorStep(enabled=False, action_names=list(ACTION_NAMES))
    policy = _ChunkingRelativePolicy(_chunk(2))
    policy._queue.append(torch.zeros(1, ACTION_DIM))

    with pinned_relative_anchor(step, policy):
        step(create_transition(observation={OBS_STATE: torch.ones(1, ACTION_DIM)}))

    torch.testing.assert_close(step.get_cached_state(), torch.ones(1, ACTION_DIM))


def test_pinned_anchor_lets_an_empty_queue_advance():
    """An empty queue means a fresh chunk: the anchor must follow the current state."""
    step = RelativeActionsProcessorStep(enabled=True, action_names=list(ACTION_NAMES))
    step.set_cached_state(torch.zeros(1, ACTION_DIM))
    policy = _ChunkingRelativePolicy(_chunk(2))  # queue is empty

    with pinned_relative_anchor(step, policy):
        step(create_transition(observation={OBS_STATE: torch.full((1, ACTION_DIM), 5.0)}))

    torch.testing.assert_close(step.get_cached_state(), torch.full((1, ACTION_DIM), 5.0))


def test_pinned_anchor_restores_on_exception():
    """A failed inference must not leave the anchor pointing at the moved state."""
    step = RelativeActionsProcessorStep(enabled=True, action_names=list(ACTION_NAMES))
    step.set_cached_state(torch.zeros(1, ACTION_DIM))
    policy = _ChunkingRelativePolicy(_chunk(2))
    policy._queue.append(torch.zeros(1, ACTION_DIM))

    with pytest.raises(RuntimeError), pinned_relative_anchor(step, policy):
        step(create_transition(observation={OBS_STATE: torch.full((1, ACTION_DIM), 5.0)}))
        raise RuntimeError("inference blew up")

    torch.testing.assert_close(step.get_cached_state(), torch.zeros(1, ACTION_DIM))


def test_find_relative_action_step_skips_disabled_and_missing():
    enabled = RelativeActionsProcessorStep(enabled=True)
    disabled = RelativeActionsProcessorStep(enabled=False)

    assert find_relative_action_step(type("_P", (), {"steps": [disabled, enabled]})()) is enabled
    assert find_relative_action_step(type("_P", (), {"steps": [disabled]})()) is None
    assert find_relative_action_step(type("_P", (), {})()) is None


def test_eval_wires_up_the_pinning_lookup():
    """The eval path finds the same step the engine does."""
    pre, _, relative_step = _relative_pipelines()
    assert find_relative_action_step(pre) is relative_step


def test_moving_env_actually_moves():
    """Sanity check on the fixture: a static env would make every assertion above vacuous."""
    env = _MovingEnv(s0=1.0, drift=0.5, max_steps=4)
    first, _ = env.reset(options={NEW_ROLLOUT_OPTION: True})
    second, *_ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert not np.allclose(first["agent_pos"], second["agent_pos"])
