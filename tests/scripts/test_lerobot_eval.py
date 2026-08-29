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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import PolicyProcessorPipeline
from lerobot.scripts.lerobot_eval import eval_policy, rollout
from tests.fixtures.dummy_checkpoint_policy import make_dummy_policy


class DummyCountingEnv(gym.Env):
    metadata = {"render_fps": 10}

    def __init__(self, max_steps: int = 3):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation.state": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )

    def _max_episode_steps(self) -> int:
        return self.max_steps

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = {"observation.state": np.zeros(4, dtype=np.float32)}
        info = {"is_success": False}
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        reward = 1.0 if terminated else 0.0
        obs = {"observation.state": np.ones(4, dtype=np.float32) * self.step_count}
        info = {"is_success": terminated}
        return obs, reward, terminated, truncated, info


def _make_dummy_env_features() -> dict:
    return {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }


def test_eval_policy_recording_multiple_sequential_batches(tmp_path: Path):
    """Verify that eval_policy with recording records all episodes across sequential batches (batch_size=1, n_episodes=2)."""
    vec_env = gym.vector.SyncVectorEnv([lambda: DummyCountingEnv(max_steps=3)])
    policy = make_dummy_policy()
    pipeline = PolicyProcessorPipeline([])
    recording_dir = tmp_path / "recordings"

    info = eval_policy(
        env=vec_env,
        policy=policy,
        env_preprocessor=pipeline,
        env_postprocessor=pipeline,
        preprocessor=pipeline,
        postprocessor=pipeline,
        n_episodes=2,
        max_episodes_rendered=0,
        recording_dir=recording_dir,
        env_features=_make_dummy_env_features(),
    )

    assert len(info["per_episode"]) == 2
    assert recording_dir.exists()

    ds = LeRobotDataset(root=str(recording_dir))
    assert ds.meta.total_episodes == 2
    vec_env.close()


def test_eval_policy_recording_multi_env_multiple_batches(tmp_path: Path):
    """Verify that eval_policy with recording records all episodes across multi-env sequential batches."""
    vec_env = gym.vector.SyncVectorEnv([lambda: DummyCountingEnv(max_steps=3), lambda: DummyCountingEnv(max_steps=3)])
    policy = make_dummy_policy()
    pipeline = PolicyProcessorPipeline([])
    recording_dir = tmp_path / "recordings_multi"

    info = eval_policy(
        env=vec_env,
        policy=policy,
        env_preprocessor=pipeline,
        env_postprocessor=pipeline,
        preprocessor=pipeline,
        postprocessor=pipeline,
        n_episodes=4,
        max_episodes_rendered=0,
        recording_dir=recording_dir,
        env_features=_make_dummy_env_features(),
    )

    assert len(info["per_episode"]) == 4
    for i in range(2):
        env_dir = recording_dir / f"env_{i}"
        assert env_dir.exists()
        ds = LeRobotDataset(root=str(env_dir))
        assert ds.meta.total_episodes == 2
    vec_env.close()


def test_rollout_standalone_recording(tmp_path: Path):
    """Verify that standalone rollout() with recording_dir initializes and finalizes dataset properly."""
    vec_env = gym.vector.SyncVectorEnv([lambda: DummyCountingEnv(max_steps=3)])
    policy = make_dummy_policy()
    pipeline = PolicyProcessorPipeline([])
    recording_dir = tmp_path / "standalone_recording"

    rollout_data = rollout(
        env=vec_env,
        policy=policy,
        env_preprocessor=pipeline,
        env_postprocessor=pipeline,
        preprocessor=pipeline,
        postprocessor=pipeline,
        recording_dir=recording_dir,
        env_features=_make_dummy_env_features(),
    )

    assert "action" in rollout_data
    assert recording_dir.exists()
    ds = LeRobotDataset(root=str(recording_dir))
    assert ds.meta.total_episodes == 1
    vec_env.close()
