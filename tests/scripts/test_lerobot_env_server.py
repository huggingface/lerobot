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

"""lerobot-env-server: configuration overrides, Docker launch, and the shipped server configurations."""

import signal
import subprocess
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import draccus
import numpy as np
import pytest
import yaml

from lerobot.env_server.client import EnvClient
from lerobot.env_server.server import EnvServerConfig
from lerobot.envs.configs import EnvConfig, check_served
from lerobot.scripts import lerobot_env_server
from lerobot.scripts.lerobot_env_server import EnvServerLaunchConfig
from lerobot.sim.config import SimSettings
from tests.fixtures.sim_server import free_port
from tests.fixtures.toy_env import ToyEnvConfig
from tests.utils import resolve_without_simulator

ROOT = Path(__file__).resolve().parents[2]
SERVER_CONFIGS = sorted((ROOT / "docker/sim/configs").glob("*.yaml"))


def test_overrides_merge_into_the_config_and_leave_the_file_alone(tmp_path):
    path = tmp_path / "sim.yaml"
    source = (
        "deployment: libero\nsim:\n  type: libero\n  task: libero_spatial\n"
        "  kwargs:\n    init_states: true\n    camera_name: agentview_image\n"
    )
    path.write_text(source)
    args = ["--sim.task=libero_goal", "--sim.kwargs={init_states: false}"]
    cfg = draccus.parse(EnvServerLaunchConfig, config_path=str(path), args=args)
    assert cfg.sim.task == "libero_goal"
    assert cfg.sim.kwargs == {"init_states": False, "camera_name": "agentview_image"}
    # Left to the integration's conventions when the server starts.
    assert cfg.sim.fps is None and cfg.sim.control_mode is None
    assert path.read_text() == source
    with pytest.raises(SystemExit):
        draccus.parse(EnvServerLaunchConfig, config_path=str(path), args=["--sim.typo=1"])


def test_a_config_needs_a_simulator_and_a_deployment():
    for options in ({"sim": SimSettings(type="libero", task="libero_spatial")}, {"deployment": "libero"}):
        with pytest.raises(ValueError, match="required"):
            EnvServerConfig(**options)
    # The command names its own flags.
    with pytest.raises(ValueError, match="--config_path"):
        EnvServerLaunchConfig()


def test_docker_launch_selects_backend_from_overridden_config(tmp_path, monkeypatch):
    config = tmp_path / "sim.yaml"
    config.write_text("deployment: libero\nsim:\n  type: libero\n  task: libero_spatial\n")
    calls = []
    image_exists = True

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0 if image_exists else 1)

    monkeypatch.setattr(lerobot_env_server.subprocess, "run", run)
    monkeypatch.setattr(lerobot_env_server, "HF_LEROBOT_HOME", tmp_path)
    argv = ["lerobot-env-server", f"--config_path={config}", "--sim.task=libero_goal", "--docker=true"]
    monkeypatch.setattr(sys, "argv", argv)
    lerobot_env_server.main()
    # An existing image is reused; a missing one is built.
    assert calls[0][0] == ["docker", "image", "inspect", "lerobot-sim-libero:local"]
    command, options = calls[1]
    assert command[-4:] == ["up", "-d", "--wait", "sim"]
    assert options["env"]["SIM"] == "libero"
    mounted = yaml.safe_load(Path(options["env"]["SIM_CONFIG"]).read_text())
    assert mounted["sim"]["task"] == "libero_goal"
    # The container gets the server's config, not the launch options.
    assert "docker" not in mounted and "envhub" not in mounted
    assert yaml.safe_load(config.read_text())["sim"]["task"] == "libero_spatial"
    image_exists = False
    lerobot_env_server.main()
    assert calls[-1][0][-5:] == ["up", "-d", "--wait", "--build", "sim"]
    config.write_text("deployment: libero\nsim:\n  type: invalid-backend\n  task: libero_spatial\n")
    with pytest.raises(ValueError, match="Unsupported backend"):
        lerobot_env_server.main()
    assert len(calls) == 4


def test_docker_launch_serves_an_envhub_repository(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        lerobot_env_server.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "env.py").write_text("")
    downloads = []

    def download(repo, revision=None):
        downloads.append((repo, revision))
        return str(snapshot)

    monkeypatch.setattr(lerobot_env_server, "snapshot_download", download)
    monkeypatch.setattr(lerobot_env_server, "HF_LEROBOT_HOME", tmp_path)
    argv = [
        "lerobot-env-server",
        "--envhub=User/My-Env@v1",
        "--sim.fps=50",
        "--sim.kwargs={make_env_kwargs: {episode_length: 900}}",
        "--docker=true",
        "--build=true",
    ]
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
    mounted = yaml.safe_load(Path(env["SIM_CONFIG"]).read_text())
    assert mounted["deployment"] == "envhub"
    assert mounted["sim"]["kwargs"] == {
        "hub_path": lerobot_env_server.ENVHUB_IMAGE_DIR,
        "trust_remote_code": True,
        "make_env_kwargs": {"episode_length": 900},
    }
    assert mounted["sim"]["fps"] == 50


def test_docker_launch_builds_a_local_envhub_directory(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        lerobot_env_server.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    (tmp_path / "custom.py").write_text("")
    argv = ["lerobot-env-server", f"--envhub={tmp_path}:custom.py", "--docker=true", "--build=true"]
    monkeypatch.setattr(sys, "argv", argv)
    lerobot_env_server.main()
    env = calls[-1][1]["env"]
    assert env["ENVHUB_DIR"] == str(tmp_path.resolve())
    assert env["SIM_TAG"] == f"local-{tmp_path.name.lower()}"
    hub_path = yaml.safe_load(Path(env["SIM_CONFIG"]).read_text())["sim"]["kwargs"]["hub_path"]
    assert hub_path == f"{lerobot_env_server.ENVHUB_IMAGE_DIR}:custom.py"


def test_build_and_stop_require_docker():
    for options in ({"build": True}, {"stop": True}):
        with pytest.raises(ValueError, match="--docker"):
            EnvServerLaunchConfig(envhub="user/repo", **options)


@pytest.mark.parametrize("path", SERVER_CONFIGS, ids=str)
def test_server_configs_take_their_conventions_from_the_simulator(path, monkeypatch):
    # The image starts the server from this file, so it must parse as the launch config.
    cfg = draccus.parse(EnvServerLaunchConfig, config_path=str(path), args=[])
    sim = resolve_without_simulator(asdict(cfg.sim), monkeypatch)
    assert sim.semantics == f"{sim.type}-{sim.control_mode.replace('_', '-')}-v1"
    assert sim.action_names


@pytest.mark.parametrize("path", SERVER_CONFIGS, ids=str)
def test_server_configs_serve_what_their_env_options_describe(path, monkeypatch):
    # `--env.type=<sim> --env.task=<task> --env.server.endpoint=...` against the shipped image passes the client's check.
    sim = yaml.safe_load(path.read_text())["sim"]
    served = asdict(resolve_without_simulator(sim, monkeypatch))
    expected = EnvConfig.get_choice_class(sim["type"])(task=sim["task"]).sim_settings()
    check_served(served, expected, str(path))


def test_native_server_listens_on_localhost_unless_told_otherwise(tmp_path, monkeypatch):
    assert EnvServerConfig.__dataclass_fields__["zenoh"].default_factory().listen_endpoints == [
        "tcp/127.0.0.1:7448"
    ]
    configs = []

    class Server:
        def __init__(self, config):
            configs.append(config)

        def serve(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(lerobot_env_server, "EnvServer", Server)
    config = tmp_path / "toy.yaml"
    config.write_text("deployment: toy\nsim:\n  type: toy\n  task: reach\n  semantics: toy-delta-v1\n")
    argv = ["lerobot-env-server", f"--config_path={config}", "--zenoh.listen_endpoints=[tcp/0.0.0.0:7999]"]
    monkeypatch.setattr(sys, "argv", argv)
    lerobot_env_server.main()
    assert configs[0].zenoh.listen_endpoints == ["tcp/0.0.0.0:7999"]


def test_the_server_runs_in_its_own_process_and_stops_on_sigterm(tmp_path):
    # As in its Docker image: the CLI in its own process, stopped by `docker stop`.
    pytest.importorskip("zenoh")
    pytest.importorskip("msgpack")
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    config = tmp_path / "toy.yaml"
    config.write_text(yaml.safe_dump({"deployment": "toy", "sim": ToyEnvConfig().sim_settings()}))
    # Importing the toy fixture registers its simulator integration in the server process.
    code = "import tests.fixtures.toy_env\nfrom lerobot.scripts.lerobot_env_server import main\nmain()"
    args = [f"--config_path={config}", f"--zenoh.listen_endpoints=[{endpoint}]"]
    with (tmp_path / "server.log").open("w+") as log:
        server = subprocess.Popen([sys.executable, "-c", code, *args], cwd=ROOT, stdout=log, stderr=log)
        client = EnvClient(endpoint, "toy", handshake_timeout_s=1)
        try:
            deadline = time.monotonic() + 60
            while True:
                try:
                    client.describe()
                    break
                except (ConnectionError, TimeoutError):  # Not listening yet.
                    client.close()
                    if server.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        pytest.fail(f"lerobot-env-server did not start:\n{log.read()}")
            client.open(1, seeds=[0])
            assert client.request("step", actions=np.zeros((1, 2), np.float32)).elapsed_steps[0] == 1
        finally:
            client.close()
            server.send_signal(signal.SIGTERM)
            try:
                exit_code = server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                raise
        assert exit_code == 0
