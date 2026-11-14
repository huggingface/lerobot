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

"""Unit tests for the Windows robosuite 1.4.0 compat shim.

The shim lives in ``lerobot.sim.integrations.libero`` and is covered end-to-end only on
Windows (robosuite 1.4.0 + hf-libero + mujoco installed). These tests lock down the
control-flow invariants that can be checked on any OS, so a regression in the gating
logic or the path-matching predicate is caught in CI.
"""

from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

from tests.utils import import_without_simulator


def _get_shim(monkeypatch: pytest.MonkeyPatch):
    libero = import_without_simulator("libero", monkeypatch)
    return libero._apply_robosuite_windows_compat


def test_shim_is_noop_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = _get_shim(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    shim()  # must return silently with no exception and no warning


def test_shim_is_noop_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = _get_shim(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    shim()


def test_shim_warns_when_robosuite_is_already_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = _get_shim(monkeypatch)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "robosuite", mock.MagicMock())
    with pytest.warns(RuntimeWarning, match="already imported"):
        shim()


def test_dll_redirect_predicate_canonicalizes_paths() -> None:
    """The redirect inside the shim keys on ``os.path.normcase(name) == wanted``.

    This test pins that predicate on representative inputs so a refactor cannot silently
    break case-insensitive path matching or allow non-target DLL loads to be redirected.
    """
    rs_dir = r"C:\site-packages\robosuite"
    wanted = os.path.normcase(os.path.join(rs_dir, "utils", "mujoco.dll"))

    assert os.path.normcase(os.path.join(rs_dir, "utils", "mujoco.dll")) == wanted
    assert os.path.normcase(os.path.join(rs_dir.upper(), "UTILS", "MUJOCO.DLL")) == wanted
    assert os.path.normcase(os.path.join(rs_dir, "utils", "other.dll")) != wanted
    assert os.path.normcase(os.path.join(r"C:\other\robosuite", "utils", "mujoco.dll")) != wanted
