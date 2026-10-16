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

import sys
import types
from pathlib import Path

import pytest
import yaml

from lerobot.env_server import launcher
from lerobot.env_server.configuration import load_config
from lerobot.scripts import lerobot_env_server


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


def test_docker_launch_selects_backend_from_overridden_config(tmp_path, monkeypatch):
    config = tmp_path / "sim.yaml"
    config.write_text("sim:\n  type: libero\n  task: libero_spatial\n")
    calls = []
    image_exists = True

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0 if image_exists else 1)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    argv = ["lerobot-env-server", "--docker", f"--config={config}", "--set", "sim.task=libero_goal"]
    monkeypatch.setattr(sys, "argv", argv)
    lerobot_env_server.main()
    # An existing image is reused; a missing one is built.
    assert calls[0][0] == ["docker", "image", "inspect", "lerobot-sim-libero:local"]
    command, options = calls[1]
    assert command[-4:] == ["up", "-d", "--wait", "sim"]
    assert options["env"]["SIM"] == "libero"
    mounted = Path(options["env"]["SIM_CONFIG"])
    assert yaml.safe_load(mounted.read_text())["sim"]["task"] == "libero_goal"
    assert yaml.safe_load(config.read_text())["sim"]["task"] == "libero_spatial"
    image_exists = False
    lerobot_env_server.main()
    assert calls[-1][0][-5:] == ["up", "-d", "--wait", "--build", "sim"]
    config.write_text("sim:\n  type: invalid-backend\n  task: libero_spatial\n")
    with pytest.raises(SystemExit):
        lerobot_env_server.main()
    assert len(calls) == 4


def test_docker_launch_serves_an_envhub_repository(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "env.py").write_text("")
    downloads = []

    def download(repo, revision=None):
        downloads.append((repo, revision))
        return str(snapshot)

    monkeypatch.setattr(launcher, "snapshot_download", download)
    monkeypatch.setattr(launcher.tempfile, "gettempdir", lambda: str(tmp_path))
    argv = ["lerobot-env-server", "--docker", "--envhub=User/My-Env@v1", "--set", "sim.fps=50", "--build"]
    monkeypatch.setattr(sys, "argv", argv)
    lerobot_env_server.main()
    command, options = calls[-1]
    assert command[-5:] == ["up", "-d", "--wait", "--build", "sim"]
    env = options["env"]
    assert (env["SIM"], env["SIM_TAG"], env["SIM_START_PERIOD"]) == ("envhub", "user-my-env-v1", "600s")
    # Downloaded on the host, then copied as plain files into the build context.
    assert downloads == [("User/My-Env", "v1")]
    assert (Path(env["ENVHUB_DIR"]) / "env.py").is_file() and Path(env["ENVHUB_DIR"]) != snapshot
    # The server loads the copy baked into the image.
    sim = yaml.safe_load(Path(env["SIM_CONFIG"]).read_text())["sim"]
    assert sim["kwargs"] == {"hub_path": launcher.ENVHUB_IMAGE_DIR, "trust_remote_code": True}
    assert sim["fps"] == 50


def test_docker_launch_builds_a_local_envhub_directory(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    (tmp_path / "custom.py").write_text("")
    argv = ["lerobot-env-server", "--docker", f"--envhub={tmp_path}:custom.py", "--build"]
    monkeypatch.setattr(sys, "argv", argv)
    lerobot_env_server.main()
    env = calls[-1][1]["env"]
    assert env["ENVHUB_DIR"] == str(tmp_path.resolve())
    assert env["SIM_TAG"] == f"local-{tmp_path.name.lower()}"
    hub_path = yaml.safe_load(Path(env["SIM_CONFIG"]).read_text())["sim"]["kwargs"]["hub_path"]
    assert hub_path == f"{launcher.ENVHUB_IMAGE_DIR}:custom.py"


def test_build_and_stop_require_docker(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lerobot-env-server", "--envhub=user/repo", "--build"])
    with pytest.raises(SystemExit):
        lerobot_env_server.main()


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).resolve().parents[2] / "docker/sims/configs").glob("*.yaml")), ids=str
)
def test_server_configs_take_their_conventions_from_the_simulator(path):
    from lerobot.sims.backend import BackendConfig
    from lerobot.sims.conventions import resolve

    sim = BackendConfig(**resolve(load_config(path, [])["sim"]))
    assert sim.semantics == f"{sim.type}-{sim.control.replace('_', '-')}-v1"
    assert sim.action_names and set(sim.gripper_indices) <= set(range(len(sim.action_names)))
