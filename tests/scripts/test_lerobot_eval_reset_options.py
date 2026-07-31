from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

import lerobot.scripts.lerobot_eval as eval_module
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import ACTION


class _IdentityProcessor:
    def __call__(self, value: Any) -> Any:
        return value


class _FakePolicy(nn.Module):
    def reset(self) -> None:
        pass

    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:
        return torch.zeros((2, 1), dtype=torch.float32)


class _FakeVectorEnv:
    num_envs = 2

    def __init__(self) -> None:
        self.reset_options: dict[str, Any] | None = None

    def reset(
        self,
        *,
        seed: list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        self.reset_options = options
        observation = {"observation.state": np.zeros((self.num_envs, 1), dtype=np.float32)}
        return observation, {}

    def call(self, name: str):
        if name == "_max_episode_steps":
            return [1] * self.num_envs
        if name in {"task_description", "task"}:
            return [""] * self.num_envs
        raise AttributeError(name)

    def step(self, action: np.ndarray):
        observation = {"observation.state": np.zeros((self.num_envs, 1), dtype=np.float32)}
        reward = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.ones(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        info = {"is_success": np.zeros(self.num_envs, dtype=bool)}
        return observation, reward, terminated, truncated, info


def test_rollout_forwards_initial_state_indices_to_reset(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_module,
        "check_env_attributes_and_types",
        lambda env: None,
    )
    monkeypatch.setattr(
        eval_module,
        "preprocess_observation",
        lambda observation: observation,
    )

    env = _FakeVectorEnv()
    identity = _IdentityProcessor()

    rollout(
        env=env,
        policy=_FakePolicy(),
        env_preprocessor=identity,
        env_postprocessor=identity,
        preprocessor=identity,
        postprocessor=identity,
        initial_state_indices=[4, 5],
    )

    assert env.reset_options == {"initial_state_indices": [4, 5]}


def test_eval_policy_assigns_initial_state_indices_per_batch(monkeypatch) -> None:
    captured_initial_state_indices: list[list[int] | None] = []

    def fake_rollout(**kwargs):
        captured_initial_state_indices.append(kwargs.get("initial_state_indices"))
        return {
            ACTION: torch.zeros((2, 1, 1), dtype=torch.float32),
            "reward": torch.zeros((2, 1), dtype=torch.float32),
            "success": torch.zeros((2, 1), dtype=torch.bool),
            "done": torch.ones((2, 1), dtype=torch.bool),
        }

    monkeypatch.setattr(eval_module, "PreTrainedPolicy", nn.Module)
    monkeypatch.setattr(eval_module, "rollout", fake_rollout)

    class _FakeProgress:
        def __init__(self, count: int) -> None:
            self._values = range(count)

        def __iter__(self):
            return iter(self._values)

        def set_postfix(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setattr(
        eval_module,
        "trange",
        lambda count, **kwargs: _FakeProgress(count),
    )

    env = _FakeVectorEnv()
    identity = _IdentityProcessor()

    eval_module.eval_policy(
        env=env,
        policy=_FakePolicy(),
        env_preprocessor=identity,
        env_postprocessor=identity,
        preprocessor=identity,
        postprocessor=identity,
        n_episodes=4,
    )

    assert captured_initial_state_indices == [[0, 1], [2, 3]]
