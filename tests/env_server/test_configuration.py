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

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from lerobot.env_server.configuration import load_config


def test_launcher_runs_without_installed_lerobot(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/run_sim.py"
    config = tmp_path / "sim.yaml"
    config.write_text("sim:\n  type: libero\n  task: libero_spatial\n")
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$SIM" "$SIM_CONFIG"\n')
    docker.chmod(0o700)
    result = subprocess.run(
        [sys.executable, "-I", str(script), str(config), "--set", "sim.task=libero_goal", "--no-build"],
        env={**os.environ, "PATH": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )
    backend, effective_config = result.stdout.strip().splitlines()
    assert backend == "libero"
    assert yaml.safe_load(Path(effective_config).read_text())["sim"]["task"] == "libero_goal"


def test_native_config_overrides_preserve_source(tmp_path):
    path = tmp_path / "sim.yaml"
    source = "sim:\n  type: libero\n  task: libero_spatial\n  kwargs:\n    init_states: true\n"
    path.write_text(source)
    data = load_config(path, ["sim.task=libero_goal", "sim.kwargs.init_states=false"])
    assert data["sim"]["task"] == "libero_goal"
    assert data["sim"]["kwargs"]["init_states"] is False
    assert path.read_text() == source
    for override in ["sim.typo=1", "sim.task.typo=1", "sim..task=foo", "sim.task"]:
        with pytest.raises(ValueError):
            load_config(path, [override])


def test_launcher_selects_backend_from_overridden_config(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/run_sim.py"
    spec = importlib.util.spec_from_file_location("run_sim", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "sim.yaml"
    config.write_text("sim:\n  type: libero\n  task: libero_spatial\n")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(
        sys, "argv", [str(script), str(config), "--set", "sim.task=libero_goal", "--no-build"]
    )
    module.main()
    command, options = calls[0]
    assert command[-4:] == ["up", "-d", "--wait", "sim"]
    assert options["env"]["SIM"] == "libero"
    mounted = Path(options["env"]["SIM_CONFIG"])
    assert yaml.safe_load(mounted.read_text())["sim"]["task"] == "libero_goal"
    assert yaml.safe_load(config.read_text())["sim"]["task"] == "libero_spatial"
    config.write_text("sim:\n  type: invalid-backend\n  task: libero_spatial\n")
    with pytest.raises(SystemExit):
        module.main()
    assert len(calls) == 1
