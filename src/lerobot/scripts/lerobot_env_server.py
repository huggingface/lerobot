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

"""Run an env server from a YAML config or an EnvHub repository.

lerobot-env-server --config_path=docker/sim/configs/libero.yaml --docker=true   # in its Docker image, on :7448
lerobot-env-server --envhub=<user/repo[@revision]> --docker=true                 # serve an EnvHub repository
lerobot-env-server --config_path=docker/sim/configs/metaworld.yaml             # in the current Python environment

Every key of the config takes a nested override, e.g. `--sim.task=libero_10`.
"""

import hashlib
import os
import re
import shutil
import signal
import subprocess  # nosec B404
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import cast

import draccus
import yaml
from huggingface_hub import snapshot_download

import lerobot
from lerobot.env_server import EnvServer, EnvServerConfig
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

COMPOSE_FILE = "docker/sim/compose.server.yaml"
DOCKERFILE = "docker/Dockerfile.sim"
# Where the EnvHub stage of the Dockerfile copies the repository.
ENVHUB_IMAGE_DIR = "/envhub"


@dataclass
class EnvServerLaunchConfig(EnvServerConfig):
    """The server's config, plus an EnvHub repository to serve and whether to run in Docker.

    Args:
        envhub (`str`, *optional*):
            EnvHub repository to serve: `user/repo[@revision]`, or a local directory. Its `env.py` is trusted
            and run. Settings left unset come from its `LEROBOT_SIM`, else from the EnvHub defaults.
        docker (`bool`, *optional*, defaults to `False`):
            Run the server in its Docker image, built from this LeRobot checkout, on `localhost:7448`.
        build (`bool`, *optional*, defaults to `False`):
            With `docker`: rebuild the image. It is built when missing.
        stop (`bool`, *optional*, defaults to `False`):
            With `docker`: stop the server.
    """

    envhub: str | None = None
    docker: bool = False
    build: bool = False
    stop: bool = False

    def __post_init__(self) -> None:
        """Fill in the settings of an EnvHub repository, then check the server config."""
        if not self.sim.type and not self.envhub:
            raise ValueError("Pass --config_path=<server YAML> or --envhub=<user/repo>")
        if (self.build or self.stop) and not self.docker:
            raise ValueError("--build and --stop apply to --docker=true")
        if self.envhub:
            if self.sim.type not in ("", "envhub"):
                raise ValueError(f"--envhub serves an EnvHub repository, not sim.type={self.sim.type!r}")
            self.sim.type = "envhub"
            self.sim.kwargs = {
                "make_env_kwargs": {},
                **self.sim.kwargs,
                "hub_path": self.envhub,
                "trust_remote_code": True,
            }
            self.deployment = self.deployment or "envhub"
        super().__post_init__()


def envhub_context(hub_path: str, directory: Path) -> Path:
    """Return the Docker build context of an EnvHub repo: a local directory, or a Hub download.

    The download uses the host's Hugging Face login, so private repos build without giving Docker a token.
    """
    location = hub_path.split(":")[0]
    if Path(location).is_dir():
        return Path(location).resolve()
    repo, _, revision = location.partition("@")
    snapshot = snapshot_download(repo, revision=revision or None)
    context = directory / "envhub" / re.sub(r"[^a-z0-9_.-]+", "-", repo.lower())
    shutil.rmtree(context, ignore_errors=True)
    shutil.copytree(snapshot, context)
    return context


def run_in_docker(cfg: EnvServerLaunchConfig) -> None:
    """Start (or `stop`) the server in its Docker image on localhost, building the image if needed."""
    root = Path(lerobot.__file__).resolve().parents[2]
    if not (root / DOCKERFILE).is_file():
        raise RuntimeError(
            f"--docker=true builds simulator images from a LeRobot source checkout ({DOCKERFILE}); "
            "install LeRobot from a clone of https://github.com/huggingface/lerobot"
        )
    backend = cfg.sim.type
    # Each backend is a `<sim>_native` stage of the Dockerfile.
    dockerfile = (root / DOCKERFILE).read_text()
    supported = set(re.findall(r"^FROM \S+ AS (\w+)_native$", dockerfile, flags=re.MULTILINE))
    if backend not in supported:
        raise ValueError(f"Unsupported backend {backend!r}; choose from {sorted(supported)}")
    directory = HF_LEROBOT_HOME / "sim"
    directory.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SIM": backend}
    # The container reads the server's config; the launch options stay on this side.
    values = asdict(cfg)
    data = {item.name: values[item.name] for item in fields(EnvServerConfig)}
    if backend == "envhub":
        # The image contains the repository at ENVHUB_IMAGE_DIR, so each repository gets its own tag.
        hub_path = cfg.sim.kwargs["hub_path"]
        location = Path(hub_path.split(":")[0])
        name = f"local-{location.resolve().name}" if location.is_dir() else hub_path
        env.setdefault("SIM_TAG", re.sub(r"[^a-z0-9_.-]+", "-", name.lower()).lstrip("-.")[:128])
        # A repository's simulator may take minutes to start (e.g. Isaac Sim compiling shaders).
        env.setdefault("SIM_START_PERIOD", "600s")
        _, separator, file_path = hub_path.partition(":")
        data["sim"]["kwargs"]["hub_path"] = ENVHUB_IMAGE_DIR + separator + file_path
    # The container bind-mounts this file; keep it after startup so a restarted container can read it.
    content = yaml.safe_dump(data, sort_keys=False)
    config = directory / f"{hashlib.sha256(content.encode()).hexdigest()}.yaml"
    config.write_text(content)
    env["SIM_CONFIG"] = str(config)
    command = ["docker", "compose", "-f", str(root / COMPOSE_FILE)]
    if cfg.stop:
        command.append("down")
    else:
        command.extend(["up", "-d", "--wait"])
        image = f"lerobot-sim-{backend}:{env.get('SIM_TAG', 'local')}"
        inspect = ["docker", "image", "inspect", image]
        # Runs docker from PATH, like docker compose below.
        if cfg.build or subprocess.run(inspect, capture_output=True).returncode:  # nosec B603
            command.append("--build")
            if backend == "envhub":
                env["ENVHUB_DIR"] = str(envhub_context(hub_path, directory))
        command.append("sim")
    subprocess.run(command, env=env, check=True)  # nosec B603


# Parsed with draccus itself, not lerobot.configs.parser: the parser imports torch, which simulator images lack.
@draccus.wrap()
def serve(cfg: EnvServerLaunchConfig) -> None:
    """Serve the configured simulator until an operator stops it, here or in its Docker image."""
    if cfg.docker:
        run_in_docker(cfg)
        return
    init_logging()
    server = EnvServer(cfg)
    previous_handlers = {
        sig: signal.signal(sig, lambda *_: server.stop()) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        server.serve()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    """Parse the env server CLI and serve."""
    cast(Callable[[], None], serve)()


if __name__ == "__main__":
    main()
