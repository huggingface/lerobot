from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("libero")

from lerobot.envs.libero import LiberoEnv


class _FakeController:
    def __init__(self) -> None:
        self.use_delta: bool | None = None


class _FakeRobot:
    def __init__(self) -> None:
        self.controller = _FakeController()


class _FakeInnerEnv:
    def __init__(self) -> None:
        self.robots = [_FakeRobot()]
        self.selected_state: int | None = None

    def seed(self, seed: int | None) -> None:
        pass

    def reset(self) -> dict:
        return {}

    def set_init_state(self, state: np.integer) -> dict:
        self.selected_state = int(state)
        return {"selected_state": self.selected_state}


def _make_libero_env(episode_index: int, n_envs: int) -> tuple[LiberoEnv, _FakeInnerEnv]:
    env = LiberoEnv.__new__(LiberoEnv)
    gym.Env.__init__(env)

    inner_env = _FakeInnerEnv()

    env._env = inner_env
    env._ensure_env = lambda: None
    env._format_raw_obs = lambda raw_obs: raw_obs

    env.init_states = True
    env._init_states = np.arange(20)
    env.init_state_id = episode_index
    env._reset_stride = n_envs
    env.episode_index = episode_index
    env.num_steps_wait = 0
    env.control_mode = "relative"

    return env, inner_env


def test_reset_uses_explicit_initial_state_index_for_vector_slot() -> None:
    env, inner_env = _make_libero_env(episode_index=1, n_envs=2)

    env.reset(options={"initial_state_indices": [4, 5]})

    assert inner_env.selected_state == 5


def _collect_evaluated_initial_states(extra_resets_between_batches: int) -> list[int]:
    env, inner_env = _make_libero_env(episode_index=1, n_envs=2)
    evaluated_states: list[int] = []

    env.reset(options={"initial_state_indices": [0, 1]})
    assert inner_env.selected_state is not None
    evaluated_states.append(inner_env.selected_state)

    # Simulate additional resets caused by earlier policy termination or
    # vector-environment autoreset behavior.
    for _ in range(extra_resets_between_batches):
        env.reset()

    env.reset(options={"initial_state_indices": [2, 3]})
    assert inner_env.selected_state is not None
    evaluated_states.append(inner_env.selected_state)

    return evaluated_states


def test_explicit_initial_state_sequence_is_independent_of_extra_resets() -> None:
    sequence_without_extra_resets = _collect_evaluated_initial_states(extra_resets_between_batches=0)
    sequence_with_extra_resets = _collect_evaluated_initial_states(extra_resets_between_batches=2)

    assert sequence_without_extra_resets == [1, 3]
    assert sequence_with_extra_resets == sequence_without_extra_resets
