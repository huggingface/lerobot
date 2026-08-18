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


def test_robocasa_config_defaults_task_prompt_to_env_id() -> None:
    cfg = RoboCasaEnvConfig()
    assert cfg.task_prompt == "env_id"
    assert cfg.gym_kwargs["task_prompt"] == "env_id"


def test_robocasa_config_rejects_unknown_task_prompt() -> None:
    with pytest.raises(ValueError, match="task_prompt"):
        RoboCasaEnvConfig(task_prompt="nope")


def test_task_description_defaults_to_env_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robocasa, "_get_task_horizon", Mock(return_value=100))
    env = robocasa.RoboCasaEnv(task="CloseFridge")
    assert env.task_prompt == "env_id"
    assert env.task_description == "CloseFridge"
    env._update_task_description({"lang": "Close the fridge doors."})
    assert env.task_description == "CloseFridge"


def test_task_description_language_uses_ep_meta_lang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robocasa, "_get_task_horizon", Mock(return_value=100))
    env = robocasa.RoboCasaEnv(task="CloseFridge", task_prompt="language")
    env._update_task_description({"lang": "Close the fridge doors."})
    assert env.task_description == "Close the fridge doors."


def test_task_description_language_falls_back_to_env_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robocasa, "_get_task_horizon", Mock(return_value=100))
    env = robocasa.RoboCasaEnv(task="CloseFridge", task_prompt="language")
    env._update_task_description({})
    assert env.task_description == "CloseFridge"


def test_unknown_task_prompt_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robocasa, "_get_task_horizon", Mock(return_value=100))
    with pytest.raises(ValueError, match="task_prompt"):
        robocasa.RoboCasaEnv(task="CloseFridge", task_prompt="nope")


def test_create_envs_honors_task_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robocasa, "_get_task_horizon", Mock(return_value=100))
    envs = robocasa.create_robocasa_envs(
        task="CloseFridge",
        n_envs=1,
        env_cls=_instantiate_envs,
        gym_kwargs={"task_prompt": "language"},
    )
    env = envs["CloseFridge"][0][0]
    assert env.task_prompt == "language"
    env._update_task_description({"lang": "Close the fridge doors."})
    assert env.task_description == "Close the fridge doors."
