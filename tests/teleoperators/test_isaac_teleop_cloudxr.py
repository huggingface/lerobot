#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Tests for the CloudXR auto-launch lifecycle on ``IsaacTeleopTeleoperator``.

The ``isaacteleop`` package is an optional NVIDIA dependency that ``base.py``
imports lazily inside :meth:`connect` / :meth:`_ensure_cloudxr_runtime`. These
tests stub the exact three deferred modules with ``MagicMock`` so the lifecycle
can be exercised without the real runtime installed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lerobot.teleoperators.isaac_teleop.base import IsaacTeleopTeleoperator
from lerobot.teleoperators.isaac_teleop.config_isaac_teleop import IsaacTeleopConfig

# The three modules base.py defers imports from.
_DEFERRED_MODULES = (
    "isaacteleop",
    "isaacteleop.cloudxr",
    "isaacteleop.teleop_session_manager",
)

_SKIP_ENV_VAR = "LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH"


@pytest.fixture(autouse=True)
def _stub_isaacteleop(monkeypatch):
    """Install MagicMock stubs for the deferred isaacteleop modules.

    Saves and restores any pre-existing ``sys.modules`` entries (rather than
    blindly deleting), and pops/restores the skip env var so each test starts
    from a clean slate.
    """
    saved_modules = {name: sys.modules.get(name) for name in _DEFERRED_MODULES}
    for name in _DEFERRED_MODULES:
        sys.modules[name] = MagicMock(name=name)

    monkeypatch.delenv(_SKIP_ENV_VAR, raising=False)

    yield

    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class _FakeTeleop(IsaacTeleopTeleoperator):
    """Minimal concrete teleoperator stubbing the device-specific override points."""

    name = "fake_isaac_teleop"

    def _build_pipeline(self):
        return MagicMock(name="pipeline")

    @property
    def action_features(self):
        return {}

    @property
    def feedback_features(self):
        return {}

    def get_action(self):
        return {}


def _make_teleop(**config_kwargs) -> _FakeTeleop:
    return _FakeTeleop(IsaacTeleopConfig(**config_kwargs))


def _launcher_cls():
    return sys.modules["isaacteleop.cloudxr"].CloudXRLauncher


# ----------------------------------------------------------------------
# _ensure_cloudxr_runtime
# ----------------------------------------------------------------------


def test_ensure_skips_when_env_var_set(monkeypatch):
    monkeypatch.setenv(_SKIP_ENV_VAR, " 1 ")
    teleop = _make_teleop()

    teleop._ensure_cloudxr_runtime()

    _launcher_cls().assert_not_called()
    assert teleop._cloudxr_launcher is None


def test_ensure_launches_when_env_var_zero(monkeypatch):
    monkeypatch.setenv(_SKIP_ENV_VAR, "0")
    teleop = _make_teleop()

    teleop._ensure_cloudxr_runtime()

    _launcher_cls().assert_called_once()
    assert teleop._cloudxr_launcher is not None


def test_ensure_skips_when_auto_launch_false():
    teleop = _make_teleop(auto_launch_cloudxr=False)

    teleop._ensure_cloudxr_runtime()

    _launcher_cls().assert_not_called()
    assert teleop._cloudxr_launcher is None


def test_ensure_env_var_takes_precedence_over_auto_launch(monkeypatch):
    # auto_launch is True, but the env var opts out and wins.
    monkeypatch.setenv(_SKIP_ENV_VAR, "1")
    teleop = _make_teleop(auto_launch_cloudxr=True)

    teleop._ensure_cloudxr_runtime()

    _launcher_cls().assert_not_called()
    assert teleop._cloudxr_launcher is None


def test_ensure_is_idempotent():
    teleop = _make_teleop()

    teleop._ensure_cloudxr_runtime()
    teleop._ensure_cloudxr_runtime()

    _launcher_cls().assert_called_once()


def test_ensure_launches_with_expected_args(monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: Path("/home/fake"))
    teleop = _make_teleop()

    teleop._ensure_cloudxr_runtime()

    _launcher_cls().assert_called_once_with(
        install_dir=str(Path("/home/fake") / ".cloudxr"),
        env_config=None,
        accept_eula=False,
    )


# ----------------------------------------------------------------------
# connect / disconnect lifecycle
# ----------------------------------------------------------------------


def test_connect_launches_runtime_before_session():
    teleop = _make_teleop()
    calls = []

    launcher_cls = _launcher_cls()
    launcher_cls.side_effect = lambda **kwargs: calls.append("launcher") or MagicMock()

    session_cls = sys.modules["isaacteleop.teleop_session_manager"].TeleopSession
    session_cls.side_effect = lambda cfg: calls.append("session") or MagicMock()

    teleop.connect()

    assert calls == ["launcher", "session"]


def test_connect_stops_launcher_and_nulls_session_on_session_failure():
    teleop = _make_teleop()

    launcher_instance = MagicMock(name="launcher_instance")
    _launcher_cls().return_value = launcher_instance

    session_cls = sys.modules["isaacteleop.teleop_session_manager"].TeleopSession
    session_cls.side_effect = RuntimeError("session boom")

    with pytest.raises(RuntimeError, match="session boom"):
        teleop.connect()

    assert teleop._session is None
    launcher_instance.stop.assert_called_once()
    # Clean stop nulls the handle.
    assert teleop._cloudxr_launcher is None


def test_disconnect_clears_launcher_on_success():
    teleop = _make_teleop()
    launcher_instance = MagicMock(name="launcher_instance")
    teleop._cloudxr_launcher = launcher_instance

    teleop.disconnect()

    launcher_instance.stop.assert_called_once()
    assert teleop._cloudxr_launcher is None


def test_disconnect_retains_launcher_on_runtime_error(caplog):
    teleop = _make_teleop()
    launcher_instance = MagicMock(name="launcher_instance")
    launcher_instance.stop.side_effect = RuntimeError("cannot terminate")
    teleop._cloudxr_launcher = launcher_instance

    with caplog.at_level(logging.WARNING):
        teleop.disconnect()

    launcher_instance.stop.assert_called_once()
    # Handle is retained for atexit cleanup.
    assert teleop._cloudxr_launcher is launcher_instance
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_disconnect_without_launcher_is_noop():
    teleop = _make_teleop()
    assert teleop._cloudxr_launcher is None

    # Should not raise and should leave the handle unset.
    teleop.disconnect()

    assert teleop._cloudxr_launcher is None
