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

import sys

import draccus
import pytest

# Importing lerobot_train eagerly pulls in lerobot.datasets, which needs the
# `dataset` extra. The base CI tier runs without it, so skip the whole module there.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.policies.act.configuration_act import (
    ACTConfig,  # noqa: E402, F401  (registers --policy.type act)
)
from lerobot.scripts.lerobot_train import _remote_target_in_argv, train  # noqa: E402


def _set_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["lerobot-train", *args])


def test_remote_target_detected_space_separated(monkeypatch):
    _set_argv(monkeypatch, "--policy.type", "act", "--job.target", "a10g-small")
    assert _remote_target_in_argv() is True


def test_remote_target_detected_equals(monkeypatch):
    _set_argv(monkeypatch, "--job.target=t4-small")
    assert _remote_target_in_argv() is True


def test_local_string_is_not_remote(monkeypatch):
    _set_argv(monkeypatch, "--job.target", "local")
    assert _remote_target_in_argv() is False


def test_no_target_is_not_remote(monkeypatch):
    _set_argv(monkeypatch, "--policy.type", "act")
    assert _remote_target_in_argv() is False


def test_train_dispatches_to_submit_when_remote(monkeypatch):
    """A remote --job.target short-circuits train() to the HF Jobs submitter."""
    import lerobot.scripts.lerobot_train as train_module

    captured = []
    monkeypatch.setattr(train_module, "submit_to_hf", lambda cfg: captured.append(cfg) or "submitted")
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--job.target", "a10g-small"],
    )
    # Returns the submitter's result and never enters the local training path.
    assert train(cfg) == "submitted"
    assert captured == [cfg]
