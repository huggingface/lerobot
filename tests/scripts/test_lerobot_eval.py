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

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("datasets")

from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.configs import PushtEnvConfig
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.flux3 import make_flux3_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline, ProcessorStep
from lerobot.processor.factory import make_policy_processor_pipelines
from lerobot.scripts import lerobot_eval
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.flux3.helpers import task_config

pytestmark = pytest.mark.dataset


class _TwoStepEnv(gym.Env):
    """Seed selects an episode with distinct images and measured robot positions."""

    task = "move"
    task_description = "move the robot"
    _max_episode_steps = 2
    observation_space = gym.spaces.Dict(
        {
            "pixels": gym.spaces.Dict({"top": gym.spaces.Box(0, 255, (64, 96, 3), np.uint8)}),
            "agent_pos": gym.spaces.Box(-10.0, 10.0, (6,), np.float32),
        }
    )
    action_space = gym.spaces.Box(-100.0, 100.0, (6,), np.float32)

    def _observation(self):
        return {
            "pixels": {"top": np.full((64, 96, 3), self.episode * 100 + self.tick, np.uint8)},
            "agent_pos": np.full(6, self.episode + self.tick * 0.1, np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode = seed
        self.tick = 0
        return self._observation(), {}

    def step(self, action):
        self.tick += 1
        return self._observation(), 0.0, self.tick == self._max_episode_steps, False, {}


class _RecordingPolicy(nn.Module):
    def reset(self):
        self.inputs = []

    def select_action(self, observation):
        self.inputs.append(
            {key: value.clone() for key, value in observation.items() if torch.is_tensor(value)}
        )
        # A nonzero normalized delta updates the FLUX3 processor's command anchor each tick.
        return torch.full((observation[OBS_STATE].shape[0], 6), 0.25)


class _AccumulatingStep(ProcessorStep):
    """Independent pre/post state: resetting one pipeline cannot mask a missing reset of the other."""

    def __init__(self, key):
        self.key = key
        self.reset()

    def reset(self):
        self.offset = 0

    def __call__(self, transition):
        self.offset += 1
        result = transition.copy()
        if self.key == OBS_STATE:
            observation = dict(result[TransitionKey.OBSERVATION])
            observation[OBS_STATE] = observation[OBS_STATE] + self.offset
            result[TransitionKey.OBSERVATION] = observation
        else:
            result[TransitionKey.ACTION] = result[TransitionKey.ACTION] + self.offset
        return result

    def transform_features(self, features):
        return features


@pytest.mark.parametrize("pipeline_kind", ["flux3", "independent", "stateless"])
def test_rollout_starts_each_episode_with_fresh_processor_state(pipeline_kind):
    def make_processors():
        if pipeline_kind == "flux3":
            return make_flux3_pre_post_processors(task_config())
        if pipeline_kind == "independent":
            return make_policy_processor_pipelines(
                [_AccumulatingStep(OBS_STATE)], [_AccumulatingStep(ACTION)]
            )
        return make_policy_processor_pipelines([], [])

    def run(env, policy, pre, post, seed):
        result = rollout(
            env,
            policy,
            env_preprocessor=PolicyProcessorPipeline(steps=[]),
            env_postprocessor=PolicyProcessorPipeline(steps=[]),
            preprocessor=pre,
            postprocessor=post,
            seeds=[seed],
        )
        return result[ACTION], policy.inputs

    env = gym.vector.SyncVectorEnv([_TwoStepEnv])
    try:
        policy = _RecordingPolicy()
        pre, post = make_processors()
        run(env, policy, pre, post, seed=0)
        # Episode A's last FLUX3 joint command is 1.0; B starts at a distinct measured position, 2.0.
        reused_actions, reused_inputs = run(env, policy, pre, post, seed=2)

        fresh_pre, fresh_post = make_processors()
        fresh_actions, fresh_inputs = run(env, _RecordingPolicy(), fresh_pre, fresh_post, seed=2)
    finally:
        env.close()

    # Compare every processed observation (including FLUX3 image/state/command histories)
    # and every absolute action sent to the environment, not just reset() call counts.
    assert len(reused_inputs) == len(fresh_inputs) == 2
    for reused, fresh in zip(reused_inputs, fresh_inputs, strict=True):
        assert reused.keys() == fresh.keys()
        for key in fresh:
            torch.testing.assert_close(reused[key], fresh[key], rtol=0, atol=0, msg=key)
    torch.testing.assert_close(reused_actions, fresh_actions, rtol=0, atol=0)
    if pipeline_kind == "flux3":
        # Delta 0.25 unnormalizes to 0.5 and must be added to B's initial measured position.
        torch.testing.assert_close(reused_actions[0, 0, :-1], torch.full((5,), 2.5), rtol=0, atol=0)


def test_rollout_reports_what_the_environment_measures():
    class _MeasuringEnv(_TwoStepEnv):
        def step(self, action):
            observation, reward, terminated, truncated, _ = super().step(action)
            return observation, reward, terminated, truncated, {"metrics": {"tick": float(self.tick)}}

    env = gym.vector.SyncVectorEnv([_MeasuringEnv])
    try:
        result = rollout(
            env,
            _RecordingPolicy(),
            env_preprocessor=PolicyProcessorPipeline(steps=[]),
            env_postprocessor=PolicyProcessorPipeline(steps=[]),
            preprocessor=make_policy_processor_pipelines([], [])[0],
            postprocessor=make_policy_processor_pipelines([], [])[1],
            seeds=[0],
        )
    finally:
        env.close()
    assert result["metrics"]["tick"].tolist() == [[1.0, 2.0]]


def test_rollout_ends_at_the_step_cap_below_the_env_limit():
    class _LongEnv(_TwoStepEnv):
        _max_episode_steps = 500

    env = gym.vector.SyncVectorEnv([_LongEnv])
    try:
        result = rollout(
            env,
            _RecordingPolicy(),
            env_preprocessor=PolicyProcessorPipeline(steps=[]),
            env_postprocessor=PolicyProcessorPipeline(steps=[]),
            preprocessor=make_policy_processor_pipelines([], [])[0],
            postprocessor=make_policy_processor_pipelines([], [])[1],
            seeds=[0],
            max_episode_steps=3,
        )
    finally:
        env.close()
    assert result[ACTION].shape[1] == 3
    assert result["done"][0].tolist() == [False, False, True]


@pytest.fixture
def restore_cudnn_flags():
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)


@pytest.mark.parametrize("cudnn_deterministic", [False, True])
def test_eval_main_sets_cudnn_determinism(monkeypatch, tmp_path, restore_cudnn_flags, cudnn_deterministic):
    class _StopBeforeEnvError(Exception):
        pass

    def _make_env(*args, **kwargs):
        raise _StopBeforeEnvError

    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    monkeypatch.setattr(lerobot_eval, "make_env", _make_env)
    cfg = EvalPipelineConfig(
        env=PushtEnvConfig(),
        policy=ACTConfig(device="cpu"),
        output_dir=tmp_path,
        cudnn_deterministic=cudnn_deterministic,
    )

    with pytest.raises(_StopBeforeEnvError):
        lerobot_eval.eval_main.__wrapped__(cfg)

    assert torch.backends.cudnn.benchmark is not cudnn_deterministic
    assert torch.backends.cudnn.deterministic is cudnn_deterministic
    assert torch.are_deterministic_algorithms_enabled() is cudnn_deterministic
