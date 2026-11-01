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

from __future__ import annotations

from unittest.mock import Mock, call

import pytest

from lerobot.envs.configs import RoboCasaEnv as RoboCasaEnvConfig
from lerobot.sim.integrations import robocasa


def test_robocasa_config_uses_registered_horizon_by_default() -> None:
    assert RoboCasaEnvConfig().episode_length is None


def test_multi_task_envs_use_registered_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    horizons = {"CloseFridge": 900, "SearingMeat": 4350}
    get_task_horizon = Mock(side_effect=horizons.__getitem__)
    monkeypatch.setattr(robocasa, "_get_task_horizon", get_task_horizon)

    envs = robocasa.env_fns(task="CloseFridge,SearingMeat", n_envs=1)

    assert envs["CloseFridge"][0][0]()._max_episode_steps == 900
    assert envs["SearingMeat"][0][0]()._max_episode_steps == 4350
    assert get_task_horizon.call_args_list == [call("CloseFridge"), call("SearingMeat")]


def test_explicit_episode_length_overrides_registered_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    get_task_horizon = Mock()
    monkeypatch.setattr(robocasa, "_get_task_horizon", get_task_horizon)

    envs = robocasa.env_fns(task="CloseFridge,SearingMeat", n_envs=1, episode_length=1234)

    assert envs["CloseFridge"][0][0]()._max_episode_steps == 1234
    assert envs["SearingMeat"][0][0]()._max_episode_steps == 1234
    get_task_horizon.assert_not_called()
