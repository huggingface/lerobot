# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for per-episode metric accumulation in ``lerobot_eval.eval_policy``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from lerobot.policies import PreTrainedPolicy
from lerobot.scripts import lerobot_eval


class _FakePolicy(nn.Module):
    """Minimal policy stand-in.

    ``eval_policy`` only requires the object to pass the ``PreTrainedPolicy``
    isinstance check and to expose ``.eval()`` (inherited from ``nn.Module``).
    The rollout is mocked, so no forward pass is ever run.
    """


PreTrainedPolicy.register(_FakePolicy)


def _fake_rollout_data(num_envs: int, n_steps: int = 3) -> dict[str, torch.Tensor]:
    """Mimic the shape of ``rollout()``'s return value: tensors of (num_envs, n_steps)."""
    done = torch.zeros(num_envs, n_steps, dtype=torch.bool)
    done[:, -1] = True
    return {
        "done": done,
        "reward": torch.ones(num_envs, n_steps, dtype=torch.float32),
        "success": torch.zeros(num_envs, n_steps, dtype=torch.bool),
    }


def _run_eval(num_envs: int, n_episodes: int, start_seed: int | None) -> dict:
    env = MagicMock()
    env.num_envs = num_envs
    with patch.object(lerobot_eval, "rollout", return_value=_fake_rollout_data(num_envs)):
        return lerobot_eval.eval_policy(
            env=env,
            policy=_FakePolicy(),
            env_preprocessor=None,
            env_postprocessor=None,
            preprocessor=None,
            postprocessor=None,
            n_episodes=n_episodes,
            max_episodes_rendered=0,
            videos_dir=None,
            return_episode_data=False,
            start_seed=start_seed,
        )


@pytest.mark.parametrize(
    ("num_envs", "n_episodes"),
    [(2, 2), (3, 3), (2, 4)],  # single batch, single batch, two batches
)
def test_eval_policy_unseeded_multi_env_does_not_crash(num_envs: int, n_episodes: int):
    """With ``start_seed=None`` (the documented default) and ``num_envs > 1``,
    ``eval_policy`` must return one entry per episode instead of raising a
    ``ValueError`` from the ``strict=True`` zip over mismatched-length lists.
    """
    info = _run_eval(num_envs=num_envs, n_episodes=n_episodes, start_seed=None)

    assert len(info["per_episode"]) == n_episodes
    assert all(ep["seed"] is None for ep in info["per_episode"])


def test_eval_policy_seeded_multi_env_preserves_per_episode_seeds():
    """Regression guard: the seeded path keeps assigning incrementing seeds."""
    info = _run_eval(num_envs=2, n_episodes=2, start_seed=1000)

    assert len(info["per_episode"]) == 2
    assert [ep["seed"] for ep in info["per_episode"]] == [1000, 1001]
