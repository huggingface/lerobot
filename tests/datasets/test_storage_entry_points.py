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

"""Tests for entry-point registration of third-party storage backends (issue #4420)."""

import sys
import types
from importlib.metadata import EntryPoints

import pytest

from lerobot.datasets import storage


@pytest.fixture
def reset_registry(monkeypatch, tmp_path):
    """Snapshot the registry and entry-point latch; restore after the test."""
    saved_modules = dict(storage._DATASET_READER_MODULES)
    monkeypatch.setattr(storage, "_entry_points_loaded", False)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    storage._DATASET_READER_MODULES.clear()
    storage._DATASET_READER_MODULES.update(saved_modules)
    # latch restored by monkeypatch teardown only for attributes we patched;
    # reset it so later tests re-scan against the restored registry
    storage._entry_points_loaded = False


def _fake_entry_points(monkeypatch, eps):
    monkeypatch.setattr(storage, "entry_points", lambda group: EntryPoints(eps))


def _write_toy_backend(tmp_path, name="toy_backend"):
    module_name = f"_toy_backend_{name}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "class DATASET_READER:\n    def __init__(self, **kwargs):\n        self.kwargs = kwargs\n"
    )
    sys.modules.pop(module_name, None)
    return module_name


def test_entry_point_backend_is_discovered(tmp_path, monkeypatch, reset_registry):
    module_name = _write_toy_backend(tmp_path)
    _fake_entry_points(monkeypatch, [types.SimpleNamespace(name="toyfmt", value=module_name)])

    module = storage._reader_module("toyfmt")
    assert module.DATASET_READER is not None

    # A second resolution must not re-import or fail
    assert storage._reader_module("toyfmt") is module


def test_unknown_format_lists_registered_backends(monkeypatch, reset_registry):
    _fake_entry_points(monkeypatch, [types.SimpleNamespace(name="toyfmt", value="_does_not_matter")])
    with pytest.raises(ValueError, match="toyfmt"):
        storage.make_dataset_reader("nonesuch")


def test_conflicting_entry_point_is_skipped_with_warning(tmp_path, monkeypatch, reset_registry, caplog):
    module_name = _write_toy_backend(tmp_path)
    # "lance" is already registered as a built-in; the entry point must not override it
    _fake_entry_points(
        monkeypatch,
        [
            types.SimpleNamespace(name="lance", value=module_name),
            types.SimpleNamespace(name="toyfmt", value=module_name),
        ],
    )
    with caplog.at_level("WARNING", logger="lerobot.datasets.storage"):
        storage.load_entry_point_backends()

    assert "lerobot.datasets.lance_backend" in storage._DATASET_READER_MODULES["lance"]
    assert storage._DATASET_READER_MODULES["toyfmt"] == module_name
    assert any("lance" in record.message for record in caplog.records)


def test_default_format_cannot_be_shadowed(tmp_path, monkeypatch, reset_registry, caplog):
    module_name = _write_toy_backend(tmp_path)
    _fake_entry_points(monkeypatch, [types.SimpleNamespace(name="lerobot", value=module_name)])
    with caplog.at_level("WARNING", logger="lerobot.datasets.storage"):
        storage.load_entry_point_backends()

    assert "lerobot" not in storage._DATASET_READER_MODULES
    assert any("lerobot" in record.message for record in caplog.records)


def test_entry_point_backends_join_remote_root_probe(tmp_path, monkeypatch, reset_registry):
    """localize_remote_root probes registered backends, including entry-point ones."""
    (tmp_path / "_probe_module.py").write_text(
        "class DATASET_READER:\n    pass\n\n"
        "def localize_root(*args, **kwargs):\n"
        "    raise FileNotFoundError('toy says no')\n"
    )
    sys.modules.pop("_probe_module", None)
    _fake_entry_points(monkeypatch, [types.SimpleNamespace(name="toyfmt", value="_probe_module")])

    with pytest.raises(FileNotFoundError, match="toyfmt: toy says no"):
        storage.localize_remote_root("user/ds", "file:///nowhere")
