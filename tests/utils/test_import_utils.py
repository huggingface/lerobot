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
"""Tests for the third-party plugin auto-discovery scan in `lerobot.utils.import_utils`."""

import importlib
import importlib.metadata

import pytest

from lerobot.utils.import_utils import register_third_party_plugins


class _FakeMetadata:
    """Minimal stand-in for `importlib.metadata.Distribution.metadata` (a `Message`)."""

    def __init__(self, name: str | None):
        self._name = name

    def get(self, key: str, default=None):
        if key == "Name":
            return self._name if self._name is not None else default
        return default


class _FakeDistribution:
    def __init__(self, name: str | None):
        self.metadata = _FakeMetadata(name)


@pytest.fixture
def plugin_scan(monkeypatch):
    """Return a `run(*dist_names)` helper that returns the imported module names."""

    def run(*dist_names: str | None) -> list[str]:
        imported: list[str] = []

        def fake_distributions():
            return iter([_FakeDistribution(name) for name in dist_names])

        def fake_import_module(name: str, package: str | None = None):
            imported.append(name)
            return object()

        monkeypatch.setattr(importlib.metadata, "distributions", fake_distributions)
        monkeypatch.setattr(importlib, "import_module", fake_import_module)
        register_third_party_plugins()
        return imported

    return run


def test_strategy_plugin_distributions_are_discovered(plugin_scan):
    assert plugin_scan("lerobot_strategy_fake") == ["lerobot_strategy_fake"]


@pytest.mark.parametrize("dist_name", ["numpy", "lerobot", "lerobot-strategy-fake", "lerobot_strategy"])
def test_unrelated_distributions_are_not_imported(dist_name, plugin_scan):
    assert plugin_scan(dist_name) == []
