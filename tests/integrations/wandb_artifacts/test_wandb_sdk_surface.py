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
"""Network-free guard against the installed ``wandb`` package drifting out from under this
integration (``lerobot.integrations.wandb_artifacts``) within its pinned version range.

Full SDK mocking in the rest of this test package can't detect the real package changing shape;
this asserts the exact methods/parameters/properties ``store.py`` and ``cli.py`` call on directly
still exist. It never imports ``wandb.sdk.wandb_run`` implicitly triggering network setup, and
never calls ``wandb.init``.
"""

import inspect

import pytest

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")

import wandb


def _params(callable_obj) -> set[str]:
    return set(inspect.signature(callable_obj).parameters)


def test_wandb_init_accepts_expected_params():
    params = _params(wandb.init)
    assert {"entity", "project", "job_type", "mode"} <= params


def test_artifact_constructor_accepts_expected_params():
    params = _params(wandb.Artifact.__init__)
    assert {"name", "type", "metadata"} <= params


def test_artifact_add_dir_accepts_local_path():
    assert "local_path" in _params(wandb.Artifact.add_dir)


def test_artifact_download_accepts_root():
    assert "root" in _params(wandb.Artifact.download)


def test_artifact_wait_exists():
    assert hasattr(wandb.Artifact, "wait")


@pytest.mark.parametrize("attr", ["type", "version", "digest", "metadata", "qualified_name", "name"])
def test_artifact_exposes_expected_read_attributes(attr):
    assert hasattr(wandb.Artifact, attr)


def test_run_log_artifact_accepts_expected_params():
    params = _params(wandb.sdk.wandb_run.Run.log_artifact)
    assert {"artifact_or_path", "aliases"} <= params


def test_run_use_artifact_accepts_expected_params():
    # store.py calls use_artifact with a bare ref string and does its own type check afterwards
    # (rather than relying on the SDK's built-in `type=` mismatch check), so only assert the
    # parameter this integration actually passes.
    params = _params(wandb.sdk.wandb_run.Run.use_artifact)
    assert "artifact_or_name" in params


@pytest.mark.parametrize("attr", ["entity", "project", "finish"])
def test_run_exposes_expected_attributes(attr):
    assert hasattr(wandb.sdk.wandb_run.Run, attr)
