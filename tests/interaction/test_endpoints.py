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

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from lerobot.envs.configs import LiberoEnv, PushtEnv
from lerobot.interaction import (
    FeatureMapGymAdapter,
    GymEndpoint,
    IdentityGymAdapter,
    ProcessorRobotAdapter,
    RobotEndpoint,
)
from tests.mocks.mock_robot import MockRobot, MockRobotConfig


class UnpacedTestPacer:
    def tick(self) -> None:
        pass

    def wait(self) -> None:
        pass

    def cancel_cycle(self) -> None:
        pass


def make_robot_adapter(
    n_motors: int,
    *,
    action_processor=lambda pair: pair[0],
) -> ProcessorRobotAdapter:
    action_names = [f"motor_{index}.pos" for index in range(1, n_motors + 1)]
    return ProcessorRobotAdapter(
        dataset_features={
            "observation.state": {
                "dtype": "float32",
                "shape": (n_motors,),
                "names": action_names,
            },
            "action": {"dtype": "float32", "shape": (n_motors,), "names": action_names},
        },
        action_names=action_names,
        observation_processor=lambda observation: observation,
        action_processor=action_processor,
    )


def test_robot_endpoint_reports_action_actually_sent_by_robot() -> None:
    robot = MockRobot(MockRobotConfig(n_motors=2, random_values=False, static_values=[1.0, 2.0]))
    robot.connect()
    observations_seen_by_action_processor: list[dict[str, Any]] = []

    def process_action(pair):
        action, observation = pair
        observations_seen_by_action_processor.append(observation)
        return action

    adapter = make_robot_adapter(2, action_processor=process_action)

    def send_clipped(action: dict[str, float]) -> dict[str, float]:
        return {key: min(value, 3.5) for key, value in action.items()}

    robot.send_action = send_clipped
    endpoint = RobotEndpoint(robot, adapter=adapter, stop_fn=lambda: None, pacer=UnpacedTestPacer())

    observation = endpoint.reset(seed=123)
    result = endpoint.step(torch.tensor([3.0, 4.0]))

    np.testing.assert_array_equal(observation["observation.state"], np.array([1.0, 2.0]))
    assert observations_seen_by_action_processor == [{"motor_1.pos": 1.0, "motor_2.pos": 2.0}]
    torch.testing.assert_close(result.applied_action, torch.tensor([3.0, 3.5]))
    np.testing.assert_array_equal(result.observation["observation.state"], np.array([1.0, 2.0]))
    endpoint.close()
    assert robot.is_connected
    robot.disconnect()


def test_robot_endpoint_preserves_one_dimensional_action() -> None:
    robot = MockRobot(MockRobotConfig(n_motors=1, random_values=False, static_values=[1.0]))
    robot.connect()
    endpoint = RobotEndpoint(
        robot, adapter=make_robot_adapter(1), stop_fn=lambda: None, pacer=UnpacedTestPacer()
    )
    endpoint.reset()

    result = endpoint.step(torch.tensor([[3.0]]))

    torch.testing.assert_close(result.applied_action, torch.tensor([3.0]))
    robot.disconnect()


class TinyGymEnv(gym.Env):
    def __init__(self) -> None:
        self.state = 0.0
        self.closed = False
        self.seeds: list[int | None] = []
        self.action_shapes: list[tuple[int, ...]] = []

    def reset(self, *, seed: int | None = None, options=None):
        del options
        self.seeds.append(seed)
        self.state = 0.0
        return {"observation.state": np.array([0.0, 0.0], dtype=np.float32)}, {"seed": seed}

    def step(self, action: np.ndarray):
        self.action_shapes.append(action.shape)
        self.state += float(action.sum())
        observation = {"observation.state": np.array([self.state, self.state], dtype=np.float32)}
        return observation, self.state, self.state >= 3.0, False, {"is_success": self.state >= 3.0}

    def close(self) -> None:
        self.closed = True


def test_gym_endpoint_returns_atomic_step_result() -> None:
    env = TinyGymEnv()
    endpoint = GymEndpoint(env, adapter=IdentityGymAdapter())

    with pytest.raises(RuntimeError, match="reset"):
        endpoint.step(torch.tensor([1.0, 2.0]))

    first = endpoint.reset(seed=7)
    result = endpoint.step(torch.tensor([[1.0, 2.0]]))

    assert first["observation.state"].tolist() == [0.0, 0.0]
    assert env.seeds == [7]
    assert env.action_shapes == [(2,)]
    assert endpoint.reset_info == {"seed": 7}
    assert result.observation["observation.state"].tolist() == [3.0, 3.0]
    torch.testing.assert_close(result.applied_action, torch.tensor([1.0, 2.0]))
    assert result.reward == 3.0
    assert result.terminated
    assert result.success is True
    assert result.info == {"is_success": True}

    endpoint.close()
    assert env.closed


def test_gym_endpoint_rejects_vector_environment() -> None:
    env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    try:
        with pytest.raises(TypeError, match="scalar environment"):
            GymEndpoint(env, adapter=IdentityGymAdapter())  # type: ignore[arg-type]
    finally:
        env.close()


def test_feature_map_adapter_uses_only_active_env_features() -> None:
    cfg = PushtEnv()
    adapter = FeatureMapGymAdapter.from_env_config(cfg)
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)

    canonical = adapter.to_canonical_observation(
        {"agent_pos": np.array([1.0, 2.0], dtype=np.float32), "pixels": pixels}
    )

    assert set(canonical) == {"observation.state", "observation.image"}
    assert "environment_state" not in adapter.features_map
    np.testing.assert_array_equal(canonical["observation.state"], np.array([1.0, 2.0]))
    assert canonical["observation.image"] is pixels


def test_feature_map_adapter_rejects_env_specific_processor_path() -> None:
    with pytest.raises(NotImplementedError, match="environment-specific processors"):
        FeatureMapGymAdapter.from_env_config(LiberoEnv())


def test_feature_map_adapter_maps_nested_observations() -> None:
    adapter = FeatureMapGymAdapter(
        {"agent_pos": "observation.state", "pixels/wrist": "observation.images.wrist"}
    )
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)

    canonical = adapter.to_canonical_observation(
        {"agent_pos": np.array([1.0, 2.0], dtype=np.float32), "pixels": {"wrist": pixels}}
    )

    assert set(canonical) == {"observation.state", "observation.images.wrist"}
    assert canonical["observation.images.wrist"] is pixels
