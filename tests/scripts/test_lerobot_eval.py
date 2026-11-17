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

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("datasets")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.flux3 import make_flux3_pre_post_processors
from lerobot.processor import (
    PolicyProcessorPipeline,
    ProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.factory import make_policy_processor_pipelines
from lerobot.scripts.lerobot_eval import eval_policy, rollout
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig, DummyCheckpointPolicy, make_dummy_policy
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


def test_eval_consent_is_the_env_own():
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.configs import PushtEnvConfig

    with pytest.raises(ValueError, match="--env.trust_remote_code=true"):
        EvalPipelineConfig(env=PushtEnvConfig(), trust_remote_code=True)


class DummyCountingEnv(gym.Env):
    metadata = {"render_fps": 10}

    def __init__(self, max_steps: int = 3):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )

    def _max_episode_steps(self) -> int:
        return self.max_steps

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = {"agent_pos": np.zeros(4, dtype=np.float32)}
        info = {"is_success": False}
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        reward = 1.0 if terminated else 0.0
        obs = {"agent_pos": np.ones(4, dtype=np.float32) * self.step_count}
        info = {"is_success": terminated}
        return obs, reward, terminated, truncated, info


def _action_postprocessor() -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline(
        [],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )


def _make_dummy_env_features() -> dict:
    return {
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }


def test_eval_policy_recording_multiple_sequential_batches(tmp_path: Path):
    """Verify that eval_policy with recording records all episodes across sequential batches (batch_size=1, n_episodes=2)."""
    vec_env = gym.vector.SyncVectorEnv([lambda: DummyCountingEnv(max_steps=3)])
    policy = make_dummy_policy()
    pipeline = PolicyProcessorPipeline([])
    post = _action_postprocessor()
    recording_dir = tmp_path / "recordings"

    info = eval_policy(
        env=vec_env,
        policy=policy,
        env_preprocessor=pipeline,
        env_postprocessor=post,
        preprocessor=pipeline,
        postprocessor=post,
        n_episodes=2,
        max_episodes_rendered=0,
        recording_dir=recording_dir,
        env_features=_make_dummy_env_features(),
    )

    assert len(info["per_episode"]) == 2
    assert recording_dir.exists()

    ds = LeRobotDataset("eval_recording", root=str(recording_dir))
    assert ds.meta.total_episodes == 2
    vec_env.close()


def test_eval_policy_recording_multi_env_multiple_batches(tmp_path: Path):
    """Verify that eval_policy with recording records all episodes across multi-env sequential batches."""
    vec_env = gym.vector.SyncVectorEnv([lambda: DummyCountingEnv(max_steps=3), lambda: DummyCountingEnv(max_steps=3)])
    policy = make_dummy_policy()
    pipeline = PolicyProcessorPipeline([])
    post = _action_postprocessor()
    recording_dir = tmp_path / "recordings_multi"

    info = eval_policy(
        env=vec_env,
        policy=policy,
        env_preprocessor=pipeline,
        env_postprocessor=post,
        preprocessor=pipeline,
        postprocessor=post,
        n_episodes=4,
        max_episodes_rendered=0,
        recording_dir=recording_dir,
        env_features=_make_dummy_env_features(),
    )

    assert len(info["per_episode"]) == 4
    for i in range(2):
        env_dir = recording_dir / f"env_{i}"
        assert env_dir.exists()
        ds = LeRobotDataset(f"eval_recording_env_{i}", root=str(env_dir))
        assert ds.meta.total_episodes == 2
    vec_env.close()


def test_rollout_standalone_recording(tmp_path: Path):
    """Verify that standalone rollout() with recording_dir initializes and finalizes dataset properly."""
    vec_env = gym.vector.SyncVectorEnv([lambda: DummyCountingEnv(max_steps=3)])
    policy = make_dummy_policy()
    pipeline = PolicyProcessorPipeline([])
    post = _action_postprocessor()
    recording_dir = tmp_path / "standalone_recording"

    rollout_data = rollout(
        env=vec_env,
        policy=policy,
        env_preprocessor=pipeline,
        env_postprocessor=post,
        preprocessor=pipeline,
        postprocessor=post,
        recording_dir=recording_dir,
        env_features=_make_dummy_env_features(),
    )

    assert "action" in rollout_data
    assert recording_dir.exists()
    ds = LeRobotDataset("eval_recording", root=str(recording_dir))
    assert ds.meta.total_episodes == 1
    vec_env.close()


def test_eval_policy_aborted_episode_buffer_cleared(tmp_path: Path):
    """Verify that interrupted/crashing rollouts discard partial episode buffers before finalizing."""

    class CrashingPolicy(DummyCheckpointPolicy):
        def __init__(self, config: DummyCheckpointConfig):
            super().__init__(config)
            self.call_count = 0

        def select_action(self, batch, **kwargs):
            self.call_count += 1
            if self.call_count > 4:  # crash mid-way through episode 2 (step 2)
                raise RuntimeError("Simulated mid-episode failure")
            return super().select_action(batch, **kwargs)

    vec_env = gym.vector.SyncVectorEnv([lambda: DummyCountingEnv(max_steps=3)])
    config = DummyCheckpointConfig(device="cpu")
    policy = CrashingPolicy(config)
    with torch.no_grad():
        policy.net.weight.fill_(0.5)

    pipeline = PolicyProcessorPipeline([])
    post = _action_postprocessor()
    recording_dir = tmp_path / "recordings_aborted"

    with pytest.raises(RuntimeError, match="Simulated mid-episode failure"):
        eval_policy(
            env=vec_env,
            policy=policy,
            env_preprocessor=pipeline,
            env_postprocessor=post,
            preprocessor=pipeline,
            postprocessor=post,
            n_episodes=3,
            max_episodes_rendered=0,
            recording_dir=recording_dir,
            env_features=_make_dummy_env_features(),
        )

    # Episode 1 completed (steps 1-3), episode 2 crashed at step 2 -> partial buffer cleared, only 1 full episode saved
    ds = LeRobotDataset("eval_recording", root=str(recording_dir))
    assert ds.meta.total_episodes == 1
    vec_env.close()

