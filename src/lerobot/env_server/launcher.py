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

"""Run a simulator server in its Docker image, built from a LeRobot source checkout."""

import copy
import hashlib
import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download

import lerobot

COMPOSE_FILE = "docker/sims/compose.server.yaml"
DOCKERFILE = "docker/Dockerfile.sim"
# Where the EnvHub stage of the Dockerfile copies the repository.
ENVHUB_IMAGE_DIR = "/envhub"


def source_root() -> Path:
    """The LeRobot checkout this package is installed from, which holds the simulator Dockerfile."""
    root = Path(lerobot.__file__).resolve().parents[2]
    if not (root / DOCKERFILE).is_file():
        raise RuntimeError(
            f"--docker builds simulator images from a LeRobot source checkout ({DOCKERFILE}); "
            "install LeRobot from a clone of https://github.com/huggingface/lerobot"
        )
    return root


def envhub_context(hub_path: str, directory: Path) -> Path:
    """The build context holding an EnvHub repository: a local directory as is, else a Hub download.

    The download uses the host's Hugging Face login, so private repositories build without passing a token
    to Docker. The Hub cache links its files into a blob store outside the context: copy them as files
    under ``directory``.
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
    """Start the server for configuration ``data`` in its image on localhost (or stop it).

    The image is selected by ``sim.type`` and built when missing or when ``build`` is set.
    """
    root = source_root()
    backend = data["sim"]["type"]
    # Each backend is a `<sim>_native` stage of the Dockerfile.
    dockerfile = (root / DOCKERFILE).read_text()
    supported = set(re.findall(r"^FROM \S+ AS (\w+)_native$", dockerfile, flags=re.MULTILINE))
    if backend not in supported:
        raise ValueError(f"Unsupported backend {backend!r}; choose from {sorted(supported)}")
    directory = Path(tempfile.gettempdir()) / f"lerobot-sim-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    env = {**os.environ, "SIM": backend}
    if backend == "envhub":
        # The image bakes in the repository at ENVHUB_IMAGE_DIR, so each repository gets its own tag.
        hub_path = data["sim"]["kwargs"]["hub_path"]
        location = Path(hub_path.split(":")[0])
        name = f"local-{location.resolve().name}" if location.is_dir() else hub_path
        env.setdefault("SIM_TAG", re.sub(r"[^a-z0-9_.-]+", "-", name.lower()).lstrip("-.")[:128])
        # A repository's simulator may take minutes to start (e.g. Isaac Sim compiling shaders).
        env.setdefault("SIM_START_PERIOD", "600s")
        _, separator, file_path = hub_path.partition(":")
        data = copy.deepcopy(data)
        data["sim"]["kwargs"]["hub_path"] = ENVHUB_IMAGE_DIR + separator + file_path
    # Keep the bind-mounted effective configuration after startup, for container restarts.
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
        # Docker from PATH, as for docker compose below.
        if build or subprocess.run(inspect, capture_output=True).returncode:  # nosec B603
            command.append("--build")
            if backend == "envhub":
                env["ENVHUB_DIR"] = str(envhub_context(hub_path, directory))
        command.append("sim")
    subprocess.run(command, env=env, check=True)  # nosec B603
