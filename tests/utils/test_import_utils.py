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
"""Tests for the third-party plugin auto-discovery scan in `lerobot.utils.import_utils`.

Every plugin family (robots, cameras, teleoperators, policies, envs, rollout strategies)
relies on `register_third_party_plugins()` importing installed distributions whose name
starts with one of a hard-coded set of prefixes. These tests fake
`importlib.metadata.distributions()` so no package ever has to be installed.
"""

import importlib
import importlib.metadata
import logging

import pytest

from lerobot.utils.import_utils import register_third_party_plugins

# The prefix tuple is a function-local constant; pull it out of the code object so the
# tests below are driven by the real value instead of a copy that can silently drift.
EXPECTED_PREFIXES = (
    "lerobot_robot_",
    "lerobot_camera_",
    "lerobot_teleoperator_",
    "lerobot_policy_",
    "lerobot_env_",
    "lerobot_strategy_",
)


def _is_plugin_prefix(value: object) -> bool:
    return isinstance(value, str) and value.startswith("lerobot_")


def _real_prefixes() -> tuple[str, ...]:
    """Extract the `prefixes` tuple from `register_third_party_plugins`, or `()` if unreachable."""
    for const in register_third_party_plugins.__code__.co_consts:
        if isinstance(const, tuple) and const and all(_is_plugin_prefix(item) for item in const):
            return const
    return ()


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
    """Return a `run(*dist_names, failing=())` helper that returns the imported module names."""

    def run(*dist_names: str | None, failing: tuple[str, ...] = ()) -> list[str]:
        imported: list[str] = []

        def fake_distributions():
            return iter([_FakeDistribution(name) for name in dist_names])

        def fake_import_module(name: str, package: str | None = None):
            imported.append(name)
            if name in failing:
                raise ImportError(f"boom: {name}")
            return object()

        monkeypatch.setattr(importlib.metadata, "distributions", fake_distributions)
        monkeypatch.setattr(importlib, "import_module", fake_import_module)
        register_third_party_plugins()
        return imported

    return run


def test_prefix_tuple_is_pinned():
    """Adding a prefix without adding a test here must fail loudly."""
    prefixes = _real_prefixes()
    assert prefixes, "could not reach the `prefixes` tuple in register_third_party_plugins()"
    assert set(prefixes) == set(EXPECTED_PREFIXES)


def test_underscored_strategy_plugin_is_imported(plugin_scan):
    assert plugin_scan("lerobot_strategy_fake") == ["lerobot_strategy_fake"]


def test_hyphenated_distribution_is_skipped_with_a_warning(plugin_scan, caplog):
    """A hyphenated distribution matches no prefix, so it must at least say why."""
    with caplog.at_level(logging.WARNING):
        assert plugin_scan("lerobot-strategy-fake") == []

    assert "lerobot-strategy-fake" in caplog.text
    assert "must be named with underscores" in caplog.text
    # The message has to name the fix, not just the problem.
    assert "lerobot_strategy_fake" in caplog.text


@pytest.mark.parametrize("prefix", EXPECTED_PREFIXES)
def test_every_prefix_is_discovered(prefix, plugin_scan):
    dist_name = f"{prefix}fake"
    assert plugin_scan(dist_name) == [dist_name]


@pytest.mark.parametrize("prefix", EXPECTED_PREFIXES)
def test_every_prefix_is_discovered_only_with_underscores(prefix, plugin_scan):
    assert plugin_scan(f"{prefix.replace('_', '-')}fake") == []


@pytest.mark.parametrize("dist_name", ["numpy", "torch", "lerobot", "lerobot_strategy", "strategy_lerobot_"])
def test_unrelated_distributions_are_not_imported(dist_name, plugin_scan):
    assert plugin_scan(dist_name) == []


def test_distribution_without_name_metadata_is_skipped(plugin_scan):
    assert plugin_scan(None, "", "lerobot_strategy_fake") == ["lerobot_strategy_fake"]


def test_failing_plugin_is_swallowed_and_does_not_block_later_plugins(plugin_scan, caplog):
    """A broken plugin must not propagate; it surfaces later as a "type not registered" parse error."""
    with caplog.at_level(logging.ERROR):
        imported = plugin_scan(
            "lerobot_strategy_broken",
            "lerobot_strategy_fine",
            failing=("lerobot_strategy_broken",),
        )

    assert imported == ["lerobot_strategy_broken", "lerobot_strategy_fine"]
    assert "lerobot_strategy_broken" in caplog.text
    assert "boom: lerobot_strategy_broken" in caplog.text


def test_scan_returns_none_and_raises_nothing_with_no_distributions(plugin_scan):
    assert plugin_scan() == []
