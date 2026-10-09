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

"""Start an isolated Docker simulator selected by its native configuration YAML."""

import argparse
import hashlib
import os
import runpy
import subprocess
import tempfile
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Simulator YAML; sim.type selects its Docker image")
    parser.add_argument("--no-build", action="store_true", help="Reuse the existing simulator image")
    parser.add_argument("--stop", action="store_true", help="Stop the simulator and remove its container")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Override an existing dotted YAML key"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config.resolve(strict=True)
    # Load the shared torch-free reader from this checkout, without installing LeRobot.
    reader = runpy.run_path(str(root / "src/lerobot/env_server/configuration.py"))
    data = reader["load_config"](config, args.set)
    backend = data["sim"]["type"]
    supported = {
        path.name.removeprefix("Dockerfile.sim.") for path in (root / "docker").glob("Dockerfile.sim.*")
    }
    if backend not in supported:
        parser.error(f"Unsupported backend {backend!r}; choose from {sorted(supported)}")
    if args.set:
        # Keep the derived bind-mounted file after startup for container restarts.
        content = yaml.safe_dump(data, sort_keys=False)
        directory = Path(tempfile.gettempdir()) / f"lerobot-sim-{os.getuid()}"
        directory.mkdir(mode=0o700, exist_ok=True)
        config = directory / f"{hashlib.sha256(content.encode()).hexdigest()}.yaml"
        config.write_text(content)
    command = ["docker", "compose", "-f", str(root / "docker/sims/compose.server.yaml")]
    if args.stop:
        command.append("down")
    else:
        command.extend(["up", "-d", "--wait"])
        if not args.no_build:
            command.append("--build")
        command.append("sim")
    subprocess.run(command, env={**os.environ, "SIM": backend, "SIM_CONFIG": str(config)}, check=True)


if __name__ == "__main__":
    main()
