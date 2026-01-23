#!/usr/bin/env python

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
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest

from lerobot.configs.parser import PluginLoadError, load_plugin, parse_plugin_args, wrap
from lerobot.envs.configs import EnvConfig


def create_plugin_code(*, base_class: str = "EnvConfig", plugin_name: str = "test_env") -> str:
    """Creates a dummy plugin module that implements its own EnvConfig subclass."""
    return f"""
from dataclasses import dataclass
from lerobot.envs.configs import {base_class}

@{base_class}.register_subclass("{plugin_name}")
@dataclass
class TestPluginConfig:
    value: int = 42
    """


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Creates a temporary plugin package structure."""
    plugin_pkg = tmp_path / "test_plugin"
    plugin_pkg.mkdir()
    (plugin_pkg / "__init__.py").touch()

    with open(plugin_pkg / "my_plugin.py", "w") as f:
        f.write(create_plugin_code())

    # Add tmp_path to Python path so we can import from it
    sys.path.insert(0, str(tmp_path))
    yield plugin_pkg
    sys.path.pop(0)


def test_parse_plugin_args():
    cli_args = [
        "--env.type=test",
        "--model.discover_packages_path=some.package",
        "--env.discover_packages_path=other.package",
    ]
    plugin_args = parse_plugin_args("discover_packages_path", cli_args)
    assert plugin_args == {
        "model.discover_packages_path": "some.package",
        "env.discover_packages_path": "other.package",
    }


def test_load_plugin_success(plugin_dir: Path):
    # Import should work and register the plugin with the real EnvConfig
    load_plugin("test_plugin")

    assert "test_env" in EnvConfig.get_known_choices()
    plugin_cls = EnvConfig.get_choice_class("test_env")
    plugin_instance = plugin_cls()
    assert plugin_instance.value == 42


def test_load_plugin_failure():
    with pytest.raises(PluginLoadError) as exc_info:
        load_plugin("nonexistent_plugin")
    assert "Failed to load plugin 'nonexistent_plugin'" in str(exc_info.value)


def test_wrap_with_plugin(plugin_dir: Path):
    @dataclass
    class Config:
        env: EnvConfig

    @wrap()
    def dummy_func(cfg: Config):
        return cfg

    # Test loading plugin via CLI args
    sys.argv = [
        "dummy_script.py",
        "--env.discover_packages_path=test_plugin",
        "--env.type=test_env",
    ]

    cfg = dummy_func()
    assert isinstance(cfg, Config)
    assert isinstance(cfg.env, EnvConfig.get_choice_class("test_env"))
    assert cfg.env.value == 42
