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

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lerobot.envs import process_isolated
from lerobot.scripts import lerobot_eval


@dataclass
class _EvalSettings:
    batch_size: int = 1
    use_async_envs: bool = False
    process_isolated: bool = False


@dataclass
class _PolicySettings:
    device: str = "cpu"


@dataclass
class _EnvSettings:
    max_parallel_tasks: int = 1


@dataclass
class _EvalPipelineSettings:
    env: _EnvSettings = field(default_factory=_EnvSettings)
    eval: _EvalSettings = field(default_factory=_EvalSettings)
    policy: _PolicySettings = field(default_factory=_PolicySettings)
    output_dir: Path = Path("unused")
    seed: int = 0
    rename_map: dict[str, str] = field(default_factory=dict)
    trust_remote_code: bool = False


def test_eval_main_closes_envs_when_policy_creation_fails(monkeypatch):
    envs = {"suite": {0: MagicMock()}}
    close_envs = MagicMock()
    monkeypatch.setattr(lerobot_eval, "get_safe_torch_device", MagicMock(return_value=MagicMock(type="cpu")))
    monkeypatch.setattr(lerobot_eval, "set_seed", MagicMock())
    monkeypatch.setattr(lerobot_eval, "make_env", MagicMock(return_value=envs))
    monkeypatch.setattr(lerobot_eval, "make_policy", MagicMock(side_effect=RuntimeError("policy failed")))
    monkeypatch.setattr(lerobot_eval, "close_envs", close_envs)

    with pytest.raises(RuntimeError, match="policy failed"):
        lerobot_eval.eval_main.__wrapped__(_EvalPipelineSettings())

    close_envs.assert_called_once_with(envs)


def test_eval_main_closes_process_isolated_envs_when_policy_creation_fails(monkeypatch):
    envs = {"suite": {0: MagicMock()}}
    close_envs = MagicMock()
    cfg = _EvalPipelineSettings(eval=_EvalSettings(process_isolated=True))
    monkeypatch.setattr(lerobot_eval, "get_safe_torch_device", MagicMock(return_value=MagicMock(type="cpu")))
    monkeypatch.setattr(lerobot_eval, "set_seed", MagicMock())
    monkeypatch.setattr(process_isolated, "make_env_in_subprocess", MagicMock(return_value=envs))
    monkeypatch.setattr(process_isolated, "close_process_isolated_envs", close_envs)
    monkeypatch.setattr(lerobot_eval, "make_policy", MagicMock(side_effect=RuntimeError("policy failed")))

    with pytest.raises(RuntimeError, match="policy failed"):
        lerobot_eval.eval_main.__wrapped__(cfg)

    close_envs.assert_called_once_with(envs)
