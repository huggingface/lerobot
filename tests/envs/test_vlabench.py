#!/usr/bin/env python

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
"""Unit tests for the VLABench Gymnasium wrapper.

These tests stub out the VLABench and dm_control modules so they run without
MuJoCo, dm_control, the VLABench checkout or its mesh assets.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scipy")

from lerobot.envs.vlabench import VLABenchEnv  # noqa: E402


def _make_task(instruction: str | None) -> MagicMock:
    """Mimic `LM4ManipBaseTask`: no `task_description` / `language_instruction` attributes."""
    task = MagicMock(spec=["get_instruction"])
    task.get_instruction.return_value = instruction
    return task


def _stub_vlabench(monkeypatch: pytest.MonkeyPatch, task: MagicMock) -> None:
    inner_env = MagicMock()
    inner_env.task = task
    inner_env.get_robot_frame_position.return_value = [0.0, -0.4, 0.78]
    modules = {
        "VLABench": MagicMock(),
        "VLABench.robots": MagicMock(),
        "VLABench.tasks": MagicMock(),
        "VLABench.envs": MagicMock(load_env=MagicMock(return_value=inner_env)),
        "dm_control": MagicMock(),
        "dm_control.rl": MagicMock(),
        "dm_control.rl.control": MagicMock(PhysicsError=RuntimeError),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_task_description_uses_vlabench_instruction(monkeypatch):
    _stub_vlabench(monkeypatch, _make_task("Put the apple into the plate"))
    env = VLABenchEnv(task="select_fruit")

    env._ensure_env()

    assert env.task_description == "Put the apple into the plate"


@pytest.mark.parametrize("instruction", [None, ""])
def test_missing_instruction_raises(monkeypatch, instruction):
    _stub_vlabench(monkeypatch, _make_task(instruction))
    env = VLABenchEnv(task="select_fruit")

    with pytest.raises(ValueError, match="select_fruit' produced no language instruction"):
        env._ensure_env()


@pytest.mark.parametrize("instruction", [None, ""])
def test_missing_instruction_reset_can_be_retried(monkeypatch, instruction):
    _stub_vlabench(monkeypatch, _make_task(instruction))
    loader = sys.modules["VLABench.envs"].load_env
    inner = loader.return_value
    env = VLABenchEnv(task="select_fruit")
    get_obs = MagicMock(return_value={})
    monkeypatch.setattr(env, "_get_obs", get_obs)

    for _ in range(2):
        with pytest.raises(ValueError, match="produced no language instruction"):
            env.reset()

    assert env._env is None
    assert inner.close.call_count == 2
    assert loader.call_count == 2
    inner.reset.assert_not_called()
    get_obs.assert_not_called()


@pytest.mark.parametrize("instruction", [None, ""])
def test_missing_instruction_can_recover_with_a_new_environment(monkeypatch, instruction):
    _stub_vlabench(monkeypatch, _make_task(instruction))
    loader = sys.modules["VLABench.envs"].load_env
    rejected = loader.return_value
    valid = MagicMock()
    valid.task = _make_task("Put the pear into the plate")
    valid.get_robot_frame_position.return_value = [0.0, -0.4, 0.78]
    loader.side_effect = [rejected, valid]
    env = VLABenchEnv(task="select_fruit")
    monkeypatch.setattr(env, "_get_obs", lambda: {})

    with pytest.raises(ValueError, match="produced no language instruction"):
        env.reset()
    assert env.reset() == ({}, {"is_success": False})
    assert env._env is valid
    assert env.task_description == "Put the pear into the plate"
    rejected.close.assert_called_once()
    rejected.reset.assert_not_called()
    valid.reset.assert_called_once()
    valid.close.assert_not_called()
