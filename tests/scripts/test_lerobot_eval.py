#!/usr/bin/env python

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

from unittest.mock import MagicMock

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.scripts import lerobot_eval
from lerobot.scripts.lerobot_eval import _get_episode_end_indices_and_mask


def test_episode_mask_excludes_steps_after_first_done():
    done = torch.tensor(
        [
            [False, True, True],
            [False, False, True],
        ]
    )

    done_indices, mask = _get_episode_end_indices_and_mask(done)

    torch.testing.assert_close(done_indices, torch.tensor([1, 2]))
    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [True, True, False],
                [True, True, True],
            ]
        ),
    )


def test_eval_policy_ignores_padded_steps_in_metrics(monkeypatch):
    rollout_data = {
        "done": torch.tensor(
            [
                [False, True, True],
                [False, False, True],
            ]
        ),
        "reward": torch.tensor(
            [
                [-2.0, -1.0, 0.0],
                [1.0, 2.0, 3.0],
            ]
        ),
        "success": torch.tensor(
            [
                [False, False, True],
                [False, False, True],
            ]
        ),
    }
    monkeypatch.setattr(lerobot_eval, "rollout", lambda **_: rollout_data)

    env = MagicMock()
    env.num_envs = 2
    policy = MagicMock(spec=PreTrainedPolicy)
    policy.training = True

    result = lerobot_eval.eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=2,
    )

    assert result["per_episode"][0] == {
        "episode_ix": 0,
        "sum_reward": -3.0,
        "max_reward": -1.0,
        "success": False,
        "seed": None,
    }
    assert result["per_episode"][1]["sum_reward"] == 6.0
    assert result["per_episode"][1]["max_reward"] == 3.0
    assert result["per_episode"][1]["success"] is True
