# /// script
# requires-python = ">=3.12"
# dependencies = ["PyYAML>=6,<7"]
# ///

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

"""Start an isolated Docker simulator selected by its native configuration YAML.

uv run scripts/run_sim.py docker/sims/configs/libero.yaml
uv run scripts/run_sim.py --envhub <user/repo[@revision]>   # serve an EnvHub repository
"""

import argparse
import base64
import hashlib
import os
import re
import runpy
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml


def hf_token() -> str:
    """The Hugging Face token from HF_TOKEN or `hf auth login`, or "" without one."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    path = Path(os.environ.get("HF_TOKEN_PATH", hf_home / "token"))
    return path.read_text().strip() if path.is_file() else ""


def envhub_file(hub_path: str, name: str, token: str) -> str:
    """Base64 of a file at the repository root at the served revision, or "" when it has none."""
    repo, _, revision = hub_path.split(":")[0].partition("@")
    url = f"https://huggingface.co/{repo}/resolve/{revision or 'main'}/{name}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:  # nosec B310
            return base64.b64encode(response.read()).decode()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return ""
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, nargs="?", help="Simulator YAML; sim.type selects its Docker image"
    )
    parser.add_argument(
        "--envhub",
        metavar="REPO",
        help="Serve an EnvHub repository (user/repo[@revision]); trusts and runs its env.py",
    )
    parser.add_argument(
        "--build", action="store_true", help="Rebuild the simulator image (it is built when missing)"
    )
    parser.add_argument("--stop", action="store_true", help="Stop the simulator and remove its container")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Override an existing dotted YAML key"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.envhub:
        # Passing the repository is the explicit consent to run its remote code.
        args.config = args.config or root / "docker/sims/configs/envhub.yaml"
        args.set = [f"sim.kwargs.hub_path={args.envhub}", "sim.kwargs.trust_remote_code=true", *args.set]
    if args.config is None:
        parser.error("Pass a simulator YAML or --envhub <user/repo>")
    config = args.config.resolve(strict=True)
    # Load the shared torch-free reader from this checkout, without installing LeRobot.
    reader = runpy.run_path(str(root / "src/lerobot/env_server/configuration.py"))
    data = reader["load_config"](config, args.set)
    backend = data["sim"]["type"]
    # Each backend is a `<sim>_native` stage of docker/Dockerfile.sim.
    dockerfile = (root / "docker/Dockerfile.sim").read_text()
    supported = set(re.findall(r"^FROM \S+ AS (\w+)_native$", dockerfile, flags=re.MULTILINE))
    if backend not in supported:
        parser.error(f"Unsupported backend {backend!r}; choose from {sorted(supported)}")
    if args.set:
        # Keep the derived bind-mounted file after startup for container restarts.
        content = yaml.safe_dump(data, sort_keys=False)
        directory = Path(tempfile.gettempdir()) / f"lerobot-sim-{os.getuid()}"
        directory.mkdir(mode=0o700, exist_ok=True)
        config = directory / f"{hashlib.sha256(content.encode()).hexdigest()}.yaml"
        config.write_text(content)
    env = {**os.environ, "SIM": backend, "SIM_CONFIG": str(config)}
    if backend == "envhub":
        # The image bakes in the repository and its requirements, so each repository gets its own tag.
        repo = data["sim"]["kwargs"]["hub_path"]
        env["ENVHUB_REPO"] = repo
        env.setdefault("SIM_TAG", re.sub(r"[^a-z0-9_.-]+", "-", repo.lower())[:128])
        # A repository's simulator may take minutes to start (e.g. Isaac Sim compiling shaders).
        env.setdefault("SIM_START_PERIOD", "600s")
    command = ["docker", "compose", "-f", str(root / "docker/sims/compose.server.yaml")]
    if args.stop:
        command.append("down")
    else:
        command.extend(["up", "-d", "--wait"])
        image = f"lerobot-sim-{backend}:{env.get('SIM_TAG', 'local')}"
        inspect = ["docker", "image", "inspect", image]
        # Docker from PATH, as for docker compose below.
        if args.build or subprocess.run(inspect, capture_output=True).returncode:  # nosec B603
            command.append("--build")
            if backend == "envhub":
                # Private repositories: the build reads the token as a secret, never as a build argument.
                env["HF_TOKEN"] = hf_token()
                # Passed by content, so that the image layers installing them are reused across commits.
                for variable, name in (
                    ("ENVHUB_REQUIREMENTS", "requirements.txt"),
                    ("ENVHUB_PACKAGES", "packages.txt"),
                ):
                    if variable not in env:
                        env[variable] = envhub_file(repo, name, env["HF_TOKEN"])
        command.append("sim")
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
