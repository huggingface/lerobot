# Copyright 2026 Daniel Wahnich.
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

"""Native CPU checks for saved episode boundaries and unchanged live inference."""

from functools import partial
from multiprocessing import get_all_start_methods

import gymnasium as gym
import numpy as np
import pytest
import torch

pytest.importorskip("datasets")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.factory import make_policy_processor_pipelines
from lerobot.scripts.lerobot_eval import eval_policy, rollout
from lerobot.utils.constants import ACTION, OBS_STATE


class _EpisodeEnv(gym.Env):
    """Actions set the state; autoreset positions differ from terminal positions."""

    metadata = {"render_fps": 1}
    task = "move"
    task_description = "Move each joint by one."
    _max_episode_steps = 4
    action_space = gym.spaces.Box(-10000, 10000, (6,), np.float32)

    def __init__(self, length, truncated, images):
        self.length = length
        self.truncated = truncated
        self.images = images
        spaces = {"agent_pos": self.action_space}
        if images:
            spaces["pixels"] = gym.spaces.Dict({"top": gym.spaces.Box(0, 255, (64, 96, 3), np.uint8)})
        self.observation_space = gym.spaces.Dict(spaces)
        self.base = 0
        self.episode = -1

    def _observation(self):
        observation = {"agent_pos": self.state.copy()}
        if self.images:
            observation["pixels"] = {"top": np.full((64, 96, 3), int(self.state[0]) % 256, np.uint8)}
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.base, self.episode = seed, 0
        else:
            self.episode += 1
        self.state = np.full(6, self.base + self.episode * 100, np.float32)
        self.tick = 0
        return self._observation(), {"is_success": False}

    def step(self, action):
        np.testing.assert_array_equal(action, self.state + 1)
        self.state = action.copy()
        self.tick += 1
        done = self.tick == self.length
        return (
            self._observation(),
            float(self.tick),
            done and not self.truncated,
            done and self.truncated,
            {"is_success": done},
        )


class _PolicyConfig(PreTrainedConfig):
    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError

    def get_scheduler_preset(self):
        raise NotImplementedError

    def validate_features(self):
        pass


class _StatePolicy(PreTrainedPolicy):
    config_class = _PolicyConfig
    name = "terminal_observation_fixture"

    def __init__(self):
        super().__init__(_PolicyConfig(device="cpu", push_to_hub=False))
        self.register_buffer("unit", torch.ones(6))
        self.reset()

    def get_optim_params(self):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def predict_action_chunk(self, batch, **kwargs):
        return (batch[OBS_STATE] + self.unit).unsqueeze(1)

    def reset(self):
        self.inputs = []

    def select_action(self, observation):
        self.inputs.append(
            {key: value.clone() for key, value in observation.items() if torch.is_tensor(value)}
        )
        return observation[OBS_STATE] + self.unit


def _processors():
    pre, post = make_policy_processor_pipelines([], [])
    return {
        "env_preprocessor": PolicyProcessorPipeline(steps=[]),
        "env_postprocessor": PolicyProcessorPipeline(steps=[]),
        "preprocessor": pre,
        "postprocessor": post,
    }


def _vector_env(vector, mode, truncated, images, lengths=(1, 3)):
    factories = [partial(_EpisodeEnv, length, truncated, images) for length in lengths]
    vector_cls = gym.vector.AsyncVectorEnv if vector == "async" else gym.vector.SyncVectorEnv
    context = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
    kwargs = {"context": context} if vector == "async" else {}
    # Exercise reused observation buffers too: recording must own the retained bytes.
    return vector_cls(factories, autoreset_mode=gym.vector.AutoresetMode(mode), copy=False, **kwargs)


@pytest.mark.parametrize("vector", ["sync", "async"])
@pytest.mark.parametrize("mode", ["SameStep", "NextStep"])
@pytest.mark.parametrize("truncated", [False, True], ids=["terminated", "truncated"])
@pytest.mark.parametrize("images", [False, True], ids=["state", "nested_rgb"])
def test_episode_data_retains_terminal_observations(vector, mode, truncated, images):
    env = _vector_env(vector, mode, truncated, images)
    try:
        result = eval_policy(
            env, _StatePolicy(), **_processors(), n_episodes=2, start_seed=41, return_episode_data=True
        )
    finally:
        env.close()
    episodes = result["episodes"]
    for episode, initial, length in ((0, 41, 1), (1, 42, 3)):
        mask = episodes["episode_index"] == episode
        expected = torch.arange(initial, initial + length + 1, dtype=torch.float32)
        torch.testing.assert_close(episodes[OBS_STATE][mask], expected[:, None].expand(-1, 6), rtol=0, atol=0)
        if images:
            torch.testing.assert_close(
                episodes["observation.images.top"][mask],
                (expected / 255)[:, None, None, None].expand(-1, 3, 64, 96),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("vector", ["sync", "async"])
@pytest.mark.parametrize("mode", ["SameStep", "NextStep"])
@pytest.mark.parametrize("truncated", [False, True], ids=["terminated", "truncated"])
@pytest.mark.parametrize("images", [False, True], ids=["state", "nested_rgb"])
def test_recording_preserves_live_inputs_and_transitions(vector, mode, truncated, images):
    env = _vector_env(vector, mode, truncated, images)
    policy = _StatePolicy()
    try:
        plain = rollout(env, policy, **_processors(), seeds=[41, 42])
        plain_inputs = policy.inputs
        recorded = rollout(env, policy, **_processors(), seeds=[41, 42], return_observations=True)
    finally:
        env.close()
    for key in (ACTION, "reward", "success", "done"):
        torch.testing.assert_close(plain[key], recorded[key], rtol=0, atol=0)
    for actual, expected in zip(policy.inputs, plain_inputs, strict=True):
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
    # The early finisher's reset observation must still reach live inference.
    expected_short = [41, 141, 241] if mode == "SameStep" else [41, 42, 141]
    observed = torch.stack([item[OBS_STATE][:, 0] for item in policy.inputs], dim=1)
    torch.testing.assert_close(
        observed, torch.tensor([expected_short, [42, 43, 44]], dtype=torch.float32), rtol=0, atol=0
    )


@pytest.mark.parametrize("vector", ["sync", "async"])
def test_recording_retains_last_observation_at_rollout_step_limit(vector):
    env = _vector_env(vector, "NextStep", False, False, lengths=(5, 6))
    try:
        result = rollout(env, _StatePolicy(), **_processors(), seeds=[41, 42], return_observations=True)
    finally:
        env.close()
    assert result[ACTION].shape == (2, 4, 6)
    assert result["done"][:, -1].all()
    for index, initial in enumerate((41, 42)):
        expected = torch.arange(initial, initial + 5, dtype=torch.float32)
        torch.testing.assert_close(
            result["observation"][OBS_STATE][index], expected[:, None].expand(-1, 6), rtol=0, atol=0
        )
