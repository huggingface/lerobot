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

lerobot-env-server --docker --config=docker/sim/configs/libero.yaml   # in its Docker image, on :7448
lerobot-env-server --docker --envhub=<user/repo[@revision]>            # serve an EnvHub repository
lerobot-env-server --config=docker/sim/configs/metaworld.yaml         # in the current Python environment
"""

import argparse
import copy
import hashlib
import os
import re
import shutil
import signal
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download

import lerobot
from lerobot.env_server.server import EnvServer, ServerConfig
from lerobot.sim.config import SimConfig
from lerobot.transport.zenoh import ZenohConfig
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

COMPOSE_FILE = "docker/sim/compose.server.yaml"
DOCKERFILE = "docker/Dockerfile.sim"
# Where the EnvHub stage of the Dockerfile copies the repository.
ENVHUB_IMAGE_DIR = "/envhub"

# Base config to serve an EnvHub repository. Null settings come from the optional LEROBOT_SIM dict in the
# repository's env.py, else from defaults (fps 30, position control, generic component names). --set
# overrides both.
ENVHUB_CONFIG: dict[str, Any] = {
    "deployment": "envhub",
    "sim": {
        "type": "envhub",
        "task": "",
        "fps": None,
        "control_mode": None,
        "semantics": None,
        "kwargs": {"hub_path": "", "trust_remote_code": False, "make_env_kwargs": {}},
        "state_names": None,
        "action_names": None,
    },
}


def load_config(path: str | Path, overrides: list[str]) -> dict[str, Any]:
    """Read a YAML config and apply ``key=value`` overrides; typos and malformed keys raise ``ValueError``."""
    return apply_overrides(yaml.safe_load(Path(path).read_text()), overrides)


def envhub_config(repo: str, overrides: list[str], path: str | Path | None = None) -> dict[str, Any]:
    """Return the config to serve an EnvHub repo, based on ``path`` if given; this trusts its ``env.py``."""
    data = yaml.safe_load(Path(path).read_text()) if path else copy.deepcopy(ENVHUB_CONFIG)
    serve = [f"sim.kwargs.hub_path={repo}", "sim.kwargs.trust_remote_code=true"]
    return apply_overrides(data, [*serve, *overrides])


def apply_overrides(data: Any, overrides: list[str]) -> dict[str, Any]:
    """Set existing dotted keys of a config mapping in place. Values are parsed as YAML."""
    if not isinstance(data, dict):
        raise ValueError("Simulator configuration must be a YAML mapping")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"Expected a dotted key=value override, got {override!r}")
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown simulator override path: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown simulator override key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    return data


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


def run_in_docker(data: dict[str, Any], build: bool = False, stop: bool = False) -> None:
    """Start (or ``stop``) the server for ``data`` in its Docker image on localhost, building it if needed."""
    root = Path(lerobot.__file__).resolve().parents[2]
    if not (root / DOCKERFILE).is_file():
        raise RuntimeError(
            f"--docker builds simulator images from a LeRobot source checkout ({DOCKERFILE}); "
            "install LeRobot from a clone of https://github.com/huggingface/lerobot"
        )
    backend = data["sim"]["type"]
    # Each backend is a `<sim>_native` stage of the Dockerfile.
    dockerfile = (root / DOCKERFILE).read_text()
    supported = set(re.findall(r"^FROM \S+ AS (\w+)_native$", dockerfile, flags=re.MULTILINE))
    if backend not in supported:
        raise ValueError(f"Unsupported backend {backend!r}; choose from {sorted(supported)}")
    directory = HF_LEROBOT_HOME / "sim"
    directory.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SIM": backend}
    if backend == "envhub":
        # The image contains the repository at ENVHUB_IMAGE_DIR, so each repository gets its own tag.
        hub_path = data["sim"]["kwargs"]["hub_path"]
        location = Path(hub_path.split(":")[0])
        name = f"local-{location.resolve().name}" if location.is_dir() else hub_path
        env.setdefault("SIM_TAG", re.sub(r"[^a-z0-9_.-]+", "-", name.lower()).lstrip("-.")[:128])
        # A repository's simulator may take minutes to start (e.g. Isaac Sim compiling shaders).
        env.setdefault("SIM_START_PERIOD", "600s")
        _, separator, file_path = hub_path.partition(":")
        data = copy.deepcopy(data)
        data["sim"]["kwargs"]["hub_path"] = ENVHUB_IMAGE_DIR + separator + file_path
    # The container bind-mounts this file; keep it after startup so a restarted container can read it.
    content = yaml.safe_dump(data, sort_keys=False)
    config = directory / f"{hashlib.sha256(content.encode()).hexdigest()}.yaml"
    config.write_text(content)
    env["SIM_CONFIG"] = str(config)
    command = ["docker", "compose", "-f", str(root / COMPOSE_FILE)]
    if stop:
        command.append("down")
    else:
        command.extend(["up", "-d", "--wait"])
        image = f"lerobot-sim-{backend}:{env.get('SIM_TAG', 'local')}"
        inspect = ["docker", "image", "inspect", image]
        # Runs docker from PATH, like docker compose below.
        if build or subprocess.run(inspect, capture_output=True).returncode:  # nosec B603
            command.append("--build")
            if backend == "envhub":
                env["ENVHUB_DIR"] = str(envhub_context(hub_path, directory))
        command.append("sim")
    subprocess.run(command, env=env, check=True)  # nosec B603


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Simulator YAML; sim.type selects the backend and its Docker image")
    parser.add_argument(
        "--envhub",
        metavar="REPO",
        help="Serve an EnvHub repository (user/repo[@revision], or a local directory); trusts and runs its env.py",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Override an existing dotted YAML key"
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run the server in its Docker image (built from this LeRobot checkout), on localhost:7448",
    )
    parser.add_argument(
        "--build", action="store_true", help="With --docker: rebuild the image (it is built when missing)"
    )
    parser.add_argument("--stop", action="store_true", help="With --docker: stop the server")
    parser.add_argument(
        "--listen",
        metavar="ENDPOINT",
        help="Zenoh endpoint to listen on, e.g. tcp/0.0.0.0:7448 to accept other machines (default: localhost)",
    )
    args = parser.parse_args()
    if (args.build or args.stop) and not args.docker:
        parser.error("--build and --stop apply to --docker")
    if args.listen and args.docker:
        parser.error("--listen applies without --docker; the image publishes its port on localhost")
    if args.envhub:
        data = envhub_config(args.envhub, args.set, args.config)
    elif args.config:
        data = load_config(args.config, args.set)
    else:
        parser.error("Pass --config or --envhub")
    if args.docker:
        try:
            run_in_docker(data, build=args.build, stop=args.stop)
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        return
    init_logging()
    # Fill null settings from the simulator's conventions (for EnvHub, the repository's LEROBOT_SIM).
    data["sim"] = SimConfig.resolve(data["sim"])
    if args.listen:
        data.setdefault("zenoh", {})["listen_endpoints"] = [args.listen]
    if "zenoh" in data:
        data["zenoh"] = ZenohConfig(**data["zenoh"])
    server = EnvServer(ServerConfig(**data))
    signal.signal(signal.SIGTERM, lambda *_: server.stop.set())
    try:
        server.serve()
    except KeyboardInterrupt:
        server.stop.set()


if __name__ == "__main__":
    main()
