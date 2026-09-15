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
"""``lerobot-eval`` binds the relative-action anchor to the policy's action queue.

``rollout()`` drives the policy directly -- preprocessor, ``select_action``, postprocessor,
once per step -- so a relative-action chunk would be re-anchored to the current (moved) state
on every tick after the one that generated it. The hold itself lives in
``RelativeActionsProcessorStep`` and is tested in ``tests/policies/test_relative_actions.py``;
what is worth testing here is that this path actually calls ``bind_relative_anchor``, end to
end through a real gym env. Both tests below fail if that call is removed.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
)
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import OBS_STATE

ACTION_DIM = 4
ACTION_NAMES = [f"joint_{i}.pos" for i in range(ACTION_DIM)]


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

    Mirrors the ``_action_queue`` shape shared by pi0/pi05/DM05, including the contract
    ``bind_relative_anchor`` relies on: ``queued_action_count`` reads the same queue
    ``select_action`` drains.
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
            # The chunk is generated against the state in *this* batch; record it so the test
            # can assert the anchor the postprocessor used matches it.
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


def _chunk(n: int) -> torch.Tensor:
    """A distinct offset per chunk step, so a wrong anchor cannot be masked by a flat chunk."""
    return torch.stack([torch.full((ACTION_DIM,), 0.1 * (i + 1)) for i in range(n)])


def _expected_state(s0: float, drift: float, t: int) -> torch.Tensor:
    return torch.tensor([s0 + drift * t + i for i in range(ACTION_DIM)])


def _run_eval(policy, num_envs: int, s0: float, drift: float, max_steps: int):
    env = gym.vector.SyncVectorEnv(
        [lambda: _MovingEnv(s0=s0, drift=drift, max_steps=max_steps) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    pre, post, relative_step = _relative_pipelines()
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
            torch.testing.assert_close(actions[0, chunk_idx * n + i], anchor + chunk_rel[i])

    # After the loop the cached anchor is the second chunk's, not a held stale one.
    torch.testing.assert_close(relative_step.get_cached_state(), _expected_state(s0, drift, n).unsqueeze(0))


def test_moving_env_actually_moves():
    """Sanity check on the fixture: a static env would make every assertion above vacuous."""
    env = _MovingEnv(s0=1.0, drift=0.5, max_steps=4)
    first, _ = env.reset(options={NEW_ROLLOUT_OPTION: True})
    second, *_ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert not np.allclose(first["agent_pos"], second["agent_pos"])
