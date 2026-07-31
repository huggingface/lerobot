from __future__ import annotations

from collections.abc import Callable, Sequence
from unittest.mock import Mock, call

import pytest

from lerobot.envs import robocasa
from lerobot.envs.configs import RoboCasaEnv as RoboCasaEnvConfig


def _instantiate_envs(
    factories: Sequence[Callable[[], robocasa.RoboCasaEnv]],
) -> list[robocasa.RoboCasaEnv]:
    return [factory() for factory in factories]


def test_robocasa_config_uses_registered_horizon_by_default() -> None:
    assert RoboCasaEnvConfig().episode_length is None


def test_multi_task_envs_use_registered_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    horizons = {"CloseFridge": 900, "SearingMeat": 4350}
    get_task_horizon = Mock(side_effect=horizons.__getitem__)
    monkeypatch.setattr(robocasa, "_get_task_horizon", get_task_horizon)

    envs = robocasa.create_robocasa_envs(
        task="CloseFridge,SearingMeat",
        n_envs=1,
        env_cls=_instantiate_envs,
    )

    assert envs["CloseFridge"][0][0]._max_episode_steps == 900
    assert envs["SearingMeat"][0][0]._max_episode_steps == 4350
    assert get_task_horizon.call_args_list == [call("CloseFridge"), call("SearingMeat")]


def test_explicit_episode_length_overrides_registered_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    get_task_horizon = Mock()
    monkeypatch.setattr(robocasa, "_get_task_horizon", get_task_horizon)

    envs = robocasa.create_robocasa_envs(
        task="CloseFridge,SearingMeat",
        n_envs=1,
        env_cls=_instantiate_envs,
        episode_length=1234,
    )

    assert envs["CloseFridge"][0][0]._max_episode_steps == 1234
    assert envs["SearingMeat"][0][0]._max_episode_steps == 1234
    get_task_horizon.assert_not_called()
