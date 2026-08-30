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
"""Startup platform validation, from `TrainPipelineConfig.validate()`.

`lerobot_train` only reaches `make_env` from the eval branch inside the training
loop. Without a startup check, a run that can never work on this machine still
downloads the dataset, builds the policy and trains `env_eval_freq` steps before
the first rollout raises. Regression tests for #4388.
"""

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env_config


@pytest.fixture
def simulator_missing(monkeypatch):
    """Reproduce a non-Linux install, where the simulator is silently absent.

    `lerobot[libero]` carries a `sys_platform == 'linux'` marker on `hf-libero`,
    so the extra resolves cleanly and omits it. Patched rather than assumed, so
    these tests still mean something on Linux CI where LIBERO is installed.
    """
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "libero":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)


def make_cfg(**overrides) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


class TestPlatformValidation:
    def test_libero_without_simulator_fails_at_startup(self, simulator_missing):
        cfg = make_cfg(env=LiberoEnv(), env_eval_freq=20_000)

        with pytest.raises(ModuleNotFoundError, match="Linux") as exc_info:
            cfg.validate()

        message = str(exc_info.value)
        assert "hf-libero" in message
        assert "lerobot[libero]" in message

    def test_fires_before_other_validation_work(self, simulator_missing, monkeypatch):
        """The whole point of this check is *when* it fires.

        Everything downstream of the platform block must not have run yet:
        pretrained resolution here, and later the dataset download in
        `lerobot_train`. Blow up if resolution is reached, then assert we never
        got there.
        """

        def explode(self):
            raise AssertionError("pretrained resolution ran before the check")

        monkeypatch.setattr(TrainPipelineConfig, "_resolve_pretrained_from_cli", explode)
        cfg = make_cfg(env=LiberoEnv(), env_eval_freq=20_000)

        with pytest.raises(ModuleNotFoundError, match="Linux"):
            cfg.validate()

    def test_eval_disabled_skips_the_check(self, simulator_missing):
        """With in-training eval off, nothing constructs a simulator.

        The env config is then only used to build processors for rollouts that
        never happen, so training on LIBERO data on a non-Linux machine works
        today and must keep working. Reaching the policy check proves validation
        walked past the platform block rather than stopping there.
        """
        cfg = make_cfg(env=LiberoEnv(), env_eval_freq=0)

        with pytest.raises(ValueError, match="Neither policy nor reward_model"):
            cfg.validate()

    def test_no_env_skips_the_check(self, simulator_missing):
        cfg = make_cfg(env=None, env_eval_freq=20_000)

        with pytest.raises(ValueError, match="Neither policy nor reward_model"):
            cfg.validate()

    def test_env_without_constraint_passes(self, simulator_missing):
        cfg = make_cfg(env=make_env_config("aloha"), env_eval_freq=20_000)

        with pytest.raises(ValueError, match="Neither policy nor reward_model"):
            cfg.validate()
