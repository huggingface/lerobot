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

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts import lerobot_eval  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from tests.fixtures.dummy_checkpoint_policy import make_dummy_policy  # noqa: E402


def _rollout_batch(batch_size: int) -> dict:
    return {
        ACTION: torch.zeros(batch_size, 2, 1),
        "reward": torch.zeros(batch_size, 2),
        "success": torch.zeros(batch_size, 2, dtype=torch.bool),
        "done": torch.ones(batch_size, 2, dtype=torch.bool),
        OBS_STR: {
            "observation.state": torch.zeros(batch_size, 3, 1),
        },
    }


@pytest.mark.parametrize("n_episodes", range(1, 8))
def test_eval_policy_returns_exactly_requested_episode_data(monkeypatch, n_episodes):
    batch_size = 3
    n_batches = (n_episodes + batch_size - 1) // batch_size
    rollouts = iter([_rollout_batch(batch_size) for _ in range(n_batches)])

    monkeypatch.setattr(lerobot_eval, "rollout", lambda **_: next(rollouts))

    env = SimpleNamespace(
        num_envs=batch_size,
        unwrapped=SimpleNamespace(metadata={"render_fps": 10}),
    )
    policy = make_dummy_policy()

    info = lerobot_eval.eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=n_episodes,
        return_episode_data=True,
    )

    episode_ids = torch.unique(info["episodes"]["episode_index"]).tolist()
    assert episode_ids == list(range(n_episodes))
    assert len(info["per_episode"]) == n_episodes
