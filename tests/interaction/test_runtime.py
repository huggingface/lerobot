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
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from lerobot.interaction import (
    GymEndpoint,
    IdentityGymAdapter,
    ListRecorder,
    ProcessorRobotAdapter,
    RobotEndpoint,
    run_episode,
)
from lerobot.rollout.inference import SyncInferenceEngine


class SpyPipeline:
    def __init__(self) -> None:
        self.values: list[Any] = []
        self.reset_count = 0

    def __call__(self, value):
        if isinstance(value, dict):
            self.values.append(
                {
                    key: item.detach().clone() if isinstance(item, torch.Tensor) else item
                    for key, item in value.items()
                }
            )
        else:
            self.values.append(value.detach().clone())
        return value

    def reset(self) -> None:
        self.reset_count += 1


class DeterministicPolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_amp=False)
        self.observations: list[dict[str, Any]] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:
        self.observations.append(
            {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in observation.items()
            }
        )
        return torch.tensor([[1.0, 0.0]])


class StatefulRobot:
    def __init__(self, image: np.ndarray) -> None:
        self.state = 0.0
        self.image = image
        self.is_connected = True

    def get_observation(self) -> dict[str, Any]:
        return {"joint_1.pos": self.state, "joint_2.pos": self.state, "camera": self.image.copy()}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.state += sum(action.values())
        return action

    def disconnect(self) -> None:
        self.is_connected = False


class StatefulGym(gym.Env):
    def __init__(self, image: np.ndarray, *, terminate_at: float | None = None) -> None:
        self.state = 0.0
        self.image = image
        self.terminate_at = terminate_at

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "observation.state": np.array([self.state, self.state], dtype=np.float32),
            "observation.images.camera": self.image.copy(),
        }

    def reset(self, *, seed: int | None = None, options=None):
        del seed, options
        self.state = 0.0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        self.state += float(action.sum())
        terminated = self.terminate_at is not None and self.state >= self.terminate_at
        return self._observation(), self.state, terminated, False, {"is_success": terminated}


class CountingPacer:
    def __init__(self) -> None:
        self.ticks = 0
        self.waits = 0
        self.cancelled_cycles = 0

    def tick(self) -> None:
        self.ticks += 1

    def wait(self) -> None:
        self.waits += 1

    def cancel_cycle(self) -> None:
        self.cancelled_cycles += 1


def make_robot_endpoint(
    image: np.ndarray, *, stop_fn=lambda: None, pacer: CountingPacer | None = None
) -> RobotEndpoint:
    names = ["joint_1.pos", "joint_2.pos"]
    adapter = ProcessorRobotAdapter(
        dataset_features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": names},
            "observation.images.camera": {
                "dtype": "image",
                "shape": image.shape,
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": names},
        },
        action_names=names,
        observation_processor=lambda observation: observation,
        action_processor=lambda pair: pair[0],
    )
    return RobotEndpoint(  # type: ignore[arg-type]
        StatefulRobot(image),
        adapter=adapter,
        stop_fn=stop_fn,
        pacer=pacer or CountingPacer(),
    )


def make_sync_engine(policy: DeterministicPolicy, pre: SpyPipeline, post: SpyPipeline):
    return SyncInferenceEngine(
        policy=policy,  # type: ignore[arg-type]
        preprocessor=pre,  # type: ignore[arg-type]
        postprocessor=post,  # type: ignore[arg-type]
        dataset_features={},
        ordered_action_keys=["joint_1.pos", "joint_2.pos"],
        task="move",
        device="cpu",
        robot_type="test_robot",
    )


def test_real_and_sim_use_same_policy_processors_and_transition_path() -> None:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    policy = DeterministicPolicy()
    preprocessor = SpyPipeline()
    postprocessor = SpyPipeline()
    engine = make_sync_engine(policy, preprocessor, postprocessor)
    robot_stops: list[str] = []
    robot_pacer = CountingPacer()

    robot_records = ListRecorder()
    gym_records = ListRecorder()
    robot_result = run_episode(
        endpoint=make_robot_endpoint(
            image,
            stop_fn=lambda: robot_stops.append("stop"),
            pacer=robot_pacer,
        ),
        action_provider=engine,
        max_steps=2,
        recorder=robot_records,
        episode_id="robot-episode",
    )
    gym_result = run_episode(
        endpoint=GymEndpoint(
            StatefulGym(image),
            adapter=IdentityGymAdapter(),
            action_names=("joint_1.pos", "joint_2.pos"),
        ),
        action_provider=engine,
        max_steps=2,
        recorder=gym_records,
        episode_id="gym-episode",
    )

    assert policy.reset_count == preprocessor.reset_count == postprocessor.reset_count == 2
    assert robot_stops == ["stop"]
    assert robot_pacer.ticks == robot_pacer.waits == 2
    assert robot_pacer.cancelled_cycles == 2  # reset boundary, then stop boundary
    assert robot_result.num_steps == gym_result.num_steps == 2
    assert robot_result.truncated and gym_result.truncated
    assert len(policy.observations) == 4
    for robot_input, gym_input in zip(policy.observations[:2], policy.observations[2:], strict=True):
        torch.testing.assert_close(robot_input["observation.state"], gym_input["observation.state"])
        torch.testing.assert_close(
            robot_input["observation.images.camera"], gym_input["observation.images.camera"]
        )
        assert robot_input["observation.images.camera"].shape == (1, 3, 2, 3)
        assert robot_input["observation.images.camera"].dtype == torch.float32
        assert robot_input["observation.images.camera"].max() <= 1.0

    for robot_record, gym_record in zip(robot_records.records, gym_records.records, strict=True):
        assert robot_record.step_index == gym_record.step_index
        torch.testing.assert_close(robot_record.action, gym_record.action)
        torch.testing.assert_close(robot_record.applied_action, gym_record.applied_action)
        np.testing.assert_array_equal(
            robot_record.observation["observation.state"],
            gym_record.observation["observation.state"],
        )
        np.testing.assert_array_equal(
            robot_record.next_observation["observation.state"],
            gym_record.next_observation["observation.state"],
        )


def test_robot_endpoint_cancels_pacing_when_inference_fails() -> None:
    class FailingProvider:
        action_names = ("joint_1.pos", "joint_2.pos")

        def reset(self) -> None:
            pass

        def get_action(self, observation):
            del observation
            raise RuntimeError("inference failed")

    pacer = CountingPacer()
    endpoint = make_robot_endpoint(np.zeros((2, 3, 3), dtype=np.uint8), pacer=pacer)

    with pytest.raises(RuntimeError, match="inference failed"):
        run_episode(endpoint=endpoint, action_provider=FailingProvider(), max_steps=1)

    assert pacer.ticks == 1
    assert pacer.waits == 0
    assert pacer.cancelled_cycles == 2  # reset boundary, then aborted step


class ConstantActionProvider:
    action_names: tuple[str, ...] = ()

    def reset(self) -> None:
        pass

    def get_action(self, observation):
        del observation
        return torch.tensor([1.0, 0.0])


def test_native_gym_termination_stops_runner_without_extra_outcome_wiring() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = ListRecorder()

    result = run_episode(
        endpoint=GymEndpoint(StatefulGym(image, terminate_at=1.0), adapter=IdentityGymAdapter()),
        action_provider=ConstantActionProvider(),
        max_steps=10,
        recorder=recorder,
    )

    assert result.num_steps == 1
    assert result.terminated
    assert not result.truncated
    assert result.success is True
    assert result.total_reward == 1.0
    assert recorder.records[0].info == {"is_success": True}


class EarlySuccessGym(StatefulGym):
    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated, truncated, {**info, "is_success": self.state == 1.0}


def test_episode_success_is_accumulated_across_steps() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    result = run_episode(
        endpoint=GymEndpoint(EarlySuccessGym(image), adapter=IdentityGymAdapter()),
        action_provider=ConstantActionProvider(),
        max_steps=2,
    )

    assert result.success is True
    assert result.truncated


class ReusingObservationGym(StatefulGym):
    def __init__(self, image: np.ndarray) -> None:
        super().__init__(image)
        self.shared_state = np.zeros(2, dtype=np.float32)

    def _observation(self) -> dict[str, np.ndarray]:
        self.shared_state[:] = self.state
        return {
            "observation.state": self.shared_state,
            "observation.images.camera": self.image,
        }


def test_runner_snapshots_provider_owned_observation_buffers() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = ListRecorder()

    run_episode(
        endpoint=GymEndpoint(ReusingObservationGym(image), adapter=IdentityGymAdapter()),
        action_provider=ConstantActionProvider(),
        max_steps=2,
        recorder=recorder,
    )

    assert recorder.records[0].observation["observation.state"].tolist() == [0.0, 0.0]
    assert recorder.records[0].next_observation["observation.state"].tolist() == [1.0, 1.0]
    assert recorder.records[1].observation["observation.state"].tolist() == [1.0, 1.0]
    assert recorder.records[1].next_observation["observation.state"].tolist() == [2.0, 2.0]


class FailingEndpoint:
    def __init__(self) -> None:
        self.reset_called = False
        self.stop_calls = 0

    def reset(self, *, seed=None):
        del seed
        self.reset_called = True
        return {"observation.state": np.array([0.0], dtype=np.float32)}

    def start_step(self) -> None:
        pass

    def step(self, action):
        del action
        raise OSError("camera disconnected")

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        pass


def test_runner_requires_horizon_before_endpoint_io_and_aborts_partial_episode() -> None:
    endpoint = FailingEndpoint()
    provider = ConstantActionProvider()
    with pytest.raises(ValueError, match="positive"):
        run_episode(endpoint=endpoint, action_provider=provider, max_steps=0)
    assert not endpoint.reset_called
    assert endpoint.stop_calls == 0

    recorder = ListRecorder()
    with pytest.raises(OSError, match="camera disconnected"):
        run_episode(
            endpoint=endpoint,
            action_provider=provider,
            max_steps=2,
            recorder=recorder,
            episode_id="failed",
        )
    assert recorder.aborted_episode_ids == ["failed"]
    assert endpoint.stop_calls == 1


def test_action_order_mismatch_fails_before_endpoint_io() -> None:
    class NamedEndpoint(FailingEndpoint):
        action_names = ("shoulder.pos", "wrist.pos")

    class NamedProvider(ConstantActionProvider):
        action_names = ("wrist.pos", "shoulder.pos")

    endpoint = NamedEndpoint()
    with pytest.raises(ValueError, match="action order differ"):
        run_episode(endpoint=endpoint, action_provider=NamedProvider(), max_steps=1)
    assert not endpoint.reset_called


def test_named_endpoint_rejects_provider_without_action_names_before_io() -> None:
    class NamedEndpoint(FailingEndpoint):
        action_names = ("shoulder.pos", "wrist.pos")

    class UnnamedProvider:
        def reset(self) -> None:
            pass

        def get_action(self, observation):
            del observation
            return torch.tensor([0.0, 0.0])

    endpoint = NamedEndpoint()
    with pytest.raises(ValueError, match="declare action_names"):
        run_episode(endpoint=endpoint, action_provider=UnnamedProvider(), max_steps=1)  # type: ignore[arg-type]
    assert not endpoint.reset_called


def test_named_provider_rejects_anonymous_gym_endpoint_before_io() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    env = StatefulGym(image)

    class NamedProvider(ConstantActionProvider):
        action_names = ("joint_1.pos", "joint_2.pos")

    with pytest.raises(ValueError, match="endpoint to declare"):
        run_episode(
            endpoint=GymEndpoint(env, adapter=IdentityGymAdapter()),
            action_provider=NamedProvider(),
            max_steps=1,
        )
    assert env.state == 0.0
