#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Unit tests for the RoboCasa `task_prompt` option.

The released `lerobot/smolvla_robocasa` checkpoint was trained on CamelCase
task IDs (e.g. "CloseFridge") but the wrapper defaults to feeding RoboCasa's
natural-language instruction (e.g. "Close the fridge doors.") at eval. These
tests pin the prompt-selection behaviour. They construct the `RoboCasaEnv`
wrapper directly and mock the inner env, so they run without the full
`robocasa` installation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lerobot.envs.configs import RoboCasaEnv as RoboCasaEnvConfig
from lerobot.envs.robocasa import RoboCasaEnv

LANG = "Close the fridge doors."
TASK = "CloseFridge"


def test_resolve_task_description_task_id_uses_camelcase():
    env = RoboCasaEnv(task=TASK, task_prompt="task_id")
    assert env._resolve_task_description({"lang": LANG}) == TASK


def test_resolve_task_description_lang_uses_instruction():
    env = RoboCasaEnv(task=TASK, task_prompt="lang")
    assert env._resolve_task_description({"lang": LANG}) == LANG


def test_resolve_task_description_lang_falls_back_to_task():
    env = RoboCasaEnv(task=TASK, task_prompt="lang")
    assert env._resolve_task_description({}) == TASK


def test_wrapper_rejects_unknown_task_prompt():
    with pytest.raises(ValueError, match="task_prompt"):
        RoboCasaEnv(task=TASK, task_prompt="bogus")


@pytest.mark.parametrize(
    "task_prompt, expected",
    [("task_id", TASK), ("lang", LANG)],
)
def test_reset_sets_task_description_from_prompt(task_prompt, expected):
    """reset() must expose the prompt selected by `task_prompt`."""
    env = RoboCasaEnv(task=TASK, task_prompt=task_prompt)
    # Pre-populate `_env` so `_ensure_env()` short-circuits and no real
    # robocasa runtime is created.
    inner = MagicMock()
    inner.reset.return_value = ({}, {})
    inner.env.get_ep_meta.return_value = {"lang": LANG}
    env._env = inner

    env.reset(seed=0)
    assert env.task_description == expected


def test_config_default_prompt_is_lang():
    cfg = RoboCasaEnvConfig(task=TASK)
    assert cfg.task_prompt == "lang"
    assert cfg.gym_kwargs["task_prompt"] == "lang"


def test_config_exposes_task_prompt_in_gym_kwargs():
    cfg = RoboCasaEnvConfig(task=TASK, task_prompt="task_id")
    assert cfg.gym_kwargs["task_prompt"] == "task_id"


def test_config_rejects_unknown_task_prompt():
    with pytest.raises(ValueError, match="task_prompt"):
        RoboCasaEnvConfig(task=TASK, task_prompt="bad")
