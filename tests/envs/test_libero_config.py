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
"""`ensure_libero_config` against a stand-in `libero` package, so it runs without the Linux-only simulator."""

import builtins
import importlib
import sys
import textwrap

import pytest
import yaml

from lerobot.envs.libero_config import default_libero_paths, ensure_libero_config, libero_config_file

# Reproduces the import-time behaviour of hf-libero's `libero/libero/__init__.py`: prompt on stdin
# when the config file is missing.
_PROMPTING_LIBERO_INIT = textwrap.dedent(
    """
    import os
    libero_config_path = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    config_file = os.path.join(libero_config_path, "config.yaml")
    if not os.path.exists(config_file):
        input("Do you want to specify a custom path for the dataset folder? (Y/N): ")
    """
)


@pytest.fixture
def fake_libero(tmp_path, monkeypatch):
    """A `libero` package on sys.path whose `libero.libero` prompts like hf-libero, and a temp config dir."""
    site = tmp_path / "site"
    pkg = site / "libero" / "libero"
    pkg.mkdir(parents=True)
    (site / "libero" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text(_PROMPTING_LIBERO_INIT)
    # A real `libero` may be installed (Linux CI with the libero extra) or already imported.
    for name in [m for m in sys.modules if m == "libero" or m.startswith("libero.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    config_dir = tmp_path / "libero_home"
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config_dir))
    yield pkg
    for name in [m for m in sys.modules if m == "libero" or m.startswith("libero.")]:
        sys.modules.pop(name, None)


def _no_stdin(*args, **kwargs):
    raise EOFError("stdin is not interactive")


def test_config_file_honours_libero_config_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    assert libero_config_file() == tmp_path / "config.yaml"


def test_writes_libero_defaults_when_missing(fake_libero):
    written = ensure_libero_config()

    assert written == libero_config_file()
    config = yaml.safe_load(written.read_text())
    assert config == default_libero_paths(fake_libero)
    # The keys LeRobot (init_states) and LIBERO's benchmark (bddl_files) read.
    assert config["init_states"] == str(fake_libero / "init_files")
    assert config["bddl_files"] == str(fake_libero / "bddl_files")
    assert config["datasets"] == str(fake_libero.parent / "datasets")


def test_libero_import_no_longer_prompts(fake_libero, monkeypatch):
    """The headless failure: without the config, importing LIBERO calls input() and dies with EOFError."""
    monkeypatch.setattr(builtins, "input", _no_stdin)
    with pytest.raises(EOFError):
        importlib.import_module("libero.libero")
    for name in [m for m in sys.modules if m == "libero" or m.startswith("libero.")]:
        monkeypatch.delitem(sys.modules, name)

    ensure_libero_config()
    importlib.import_module("libero.libero")  # no prompt


def test_existing_config_is_left_untouched(fake_libero):
    config_file = libero_config_file()
    config_file.parent.mkdir(parents=True)
    config_file.write_text("datasets: /my/custom/datasets\n")

    assert ensure_libero_config() is None
    assert config_file.read_text() == "datasets: /my/custom/datasets\n"


def test_namespace_libero_package_is_supported(fake_libero):
    """LIBERO-plus installs `libero` as a namespace package (no top-level __init__.py)."""
    (fake_libero.parent / "__init__.py").unlink()
    importlib.invalidate_caches()

    written = ensure_libero_config()

    assert written is not None
    assert yaml.safe_load(written.read_text())["init_states"] == str(fake_libero / "init_files")


def test_noop_when_libero_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_home"))
    monkeypatch.setattr("lerobot.envs.libero_config.importlib.util.find_spec", lambda name: None)

    assert ensure_libero_config() is None
    assert not (tmp_path / "libero_home").exists()
