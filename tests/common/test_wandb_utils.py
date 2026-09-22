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

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, sentinel

from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.default import WandBConfig


def test_wandb_config_console_defaults():
    cfg = WandBConfig()

    assert cfg.resume is None
    assert cfg.console == "wrap"
    assert cfg.console_multipart is False
    assert cfg.console_chunk_max_seconds == 0


def test_wandb_logger_forwards_resume_and_console_settings(monkeypatch, tmp_path):
    wandb = MagicMock()
    wandb.Settings.return_value = sentinel.settings
    wandb.run.id = "run-id"
    wandb.run.get_url.return_value = "https://wandb.ai/test/run-id"
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setenv("WANDB_SILENT", "False")

    wandb_cfg = WandBConfig(
        run_id="run-id",
        resume="allow",
        console="off",
        console_multipart=True,
        console_chunk_max_seconds=60,
    )
    cfg = SimpleNamespace(
        wandb=wandb_cfg,
        output_dir=tmp_path,
        job_name="test-run",
        env=None,
        policy=SimpleNamespace(type="test-policy"),
        is_reward_model_training=False,
        seed=42,
        dataset=None,
        resume=False,
        to_dict=lambda: {},
    )

    WandBLogger(cfg)

    wandb.Settings.assert_called_once_with(
        console="off",
        console_multipart=True,
        console_chunk_max_seconds=60,
    )
    assert wandb.init.call_args.kwargs["resume"] == "allow"
    assert wandb.init.call_args.kwargs["settings"] is sentinel.settings


def test_wandb_logger_resumes_with_checkpoint(monkeypatch, tmp_path):
    wandb = MagicMock()
    wandb.run.id = "run-id"
    wandb.run.get_url.return_value = "https://wandb.ai/test/run-id"
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setenv("WANDB_SILENT", "False")

    cfg = SimpleNamespace(
        wandb=WandBConfig(run_id="run-id"),
        output_dir=tmp_path,
        job_name="test-run",
        env=None,
        policy=SimpleNamespace(type="test-policy"),
        is_reward_model_training=False,
        seed=42,
        dataset=None,
        resume=True,
        to_dict=lambda: {},
    )

    WandBLogger(cfg)

    assert wandb.init.call_args.kwargs["resume"] == "must"
