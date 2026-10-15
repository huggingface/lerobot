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
import types
from pathlib import Path

import pytest
import yaml

from lerobot.env_server.configuration import load_config


def test_launcher_runs_without_installed_lerobot(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/run_sim.py"
    config = tmp_path / "sim.yaml"
    config.write_text("sim:\n  type: libero\n  task: libero_spatial\n")
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/sh\n[ "$1" = image ] && exit 0\nprintf "%s\\n" "$SIM" "$SIM_CONFIG"\n')
    docker.chmod(0o700)
    result = subprocess.run(
        [sys.executable, "-I", str(script), str(config), "--set", "sim.task=libero_goal", "--build"],
        env={**os.environ, "PATH": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )
    backend, effective_config = result.stdout.strip().splitlines()
    assert backend == "libero"
    assert yaml.safe_load(Path(effective_config).read_text())["sim"]["task"] == "libero_goal"


def test_launcher_serves_an_envhub_repository(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/run_sim.py"
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/bin/sh\n[ "$1" = image ] && exit 0\n'
        'printf "%s\\n" "$SIM" "$ENVHUB_REPO" "$SIM_TAG" "$SIM_START_PERIOD" "$SIM_CONFIG"\n'
    )
    docker.chmod(0o700)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
            "--envhub",
            "User/My-Env@v1",
            "--set",
            "sim.fps=50",
            "--build",
        ],
        # Set, so the launcher does not fetch them from the Hub.
        env={**os.environ, "PATH": str(tmp_path), "ENVHUB_REQUIREMENTS": "", "ENVHUB_PACKAGES": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    backend, repo, tag, start_period, effective_config = result.stdout.strip().splitlines()
    assert (backend, repo, tag, start_period) == ("envhub", "User/My-Env@v1", "user-my-env-v1", "600s")
    sim = yaml.safe_load(Path(effective_config).read_text())["sim"]
    assert sim["kwargs"] == {"hub_path": "User/My-Env@v1", "trust_remote_code": True}
    assert sim["fps"] == 50


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
    image_exists = True

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0 if image_exists else 1)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [str(script), str(config), "--set", "sim.task=libero_goal"])
    module.main()
    # An existing image is reused; a missing one is built.
    assert calls[0][0] == ["docker", "image", "inspect", "lerobot-sim-libero:local"]
    command, options = calls[1]
    assert command[-4:] == ["up", "-d", "--wait", "sim"]
    assert options["env"]["SIM"] == "libero"
    mounted = Path(options["env"]["SIM_CONFIG"])
    assert yaml.safe_load(mounted.read_text())["sim"]["task"] == "libero_goal"
    assert yaml.safe_load(config.read_text())["sim"]["task"] == "libero_spatial"
    image_exists = False
    module.main()
    assert calls[-1][0][-5:] == ["up", "-d", "--wait", "--build", "sim"]
    config.write_text("sim:\n  type: invalid-backend\n  task: libero_spatial\n")
    with pytest.raises(SystemExit):
        module.main()
    assert len(calls) == 4
