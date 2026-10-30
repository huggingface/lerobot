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

"""Print the CI benchmark matrix from benchmarks/benchmarks.yaml as JSON, with a sim_tag per entry."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def image_hash(root: Path, entry: dict) -> str:
    """Hash what the simulator image is built from; keep in sync with Dockerfile.sim's COPY lines."""
    digest = hashlib.sha256()
    paths = [
        root / entry["dockerfile"],
        root / entry["sim_config"],
        root / "scripts/ci/convert_libero_init_states.py",
        root / "scripts/ci/patch_libero_runtime.py",
        root / "scripts/ci/check_sim_parity.py",
        root / "pyproject.toml",
        root / "setup.py",
        root / "README.md",
        root / "MANIFEST.in",
        root / "src/lerobot/__init__.py",
        root / "src/lerobot/__version__.py",
        root / "src/lerobot/scripts/lerobot_env_server.py",
    ]
    for directory in (
        "src/lerobot/env_server",
        "src/lerobot/sim",
        "src/lerobot/transport",
        "src/lerobot/utils",
    ):
        paths.extend((root / directory).rglob("*.py"))
    paths.append(root / "src/lerobot/sim/integrations/metaworld_config.json")
    paths.extend(path for path in (root / "docker/sim").rglob("*") if path.is_file())
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(entry.get("build_args", {}), sort_keys=True).encode())
    return digest.hexdigest()[:24]


def with_defaults(entry: dict) -> dict:
    """Add default values for the fields a benchmarks.yaml entry leaves out."""
    name = entry["name"]
    defaults = {
        "policy": f"lerobot/smolvla_{name}",
        "sim_config": f"docker/sim/configs/{name}.yaml",
        # Maps simulator camera keys to the camera keys the checkpoint expects.
        "rename_map": {},
        "dockerfile": "docker/Dockerfile.sim",
        "policy_extras": ["remote", "evaluation", "smolvla", "training", "hardware"],
        "episodes": 1,
        "batch_size": 1,
        # CI checks that the pipeline runs end to end, not the success rate, so short episodes are enough.
        "max_episode_steps": 30,
        "train_smoke": False,
        # Run check_sim_parity.py; only deterministic simulators give the same steps locally and served.
        "parity": False,
    }
    merged = {**defaults, **entry}
    # Dockerfile.sim picks the simulator's build stage from SIM.
    merged["build_args"] = {"SIM": name, **entry.get("build_args", {})}
    return merged


def load_entries(root: Path) -> list[dict]:
    entries = [
        with_defaults(entry)
        for entry in yaml.safe_load((root / "benchmarks/benchmarks.yaml").read_text())["benchmarks"]
    ]
    if len({e["name"] for e in entries}) != len(entries):
        raise ValueError("Duplicate benchmark name")
    return [{**entry, "sim_tag": image_hash(root, entry)} for entry in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--name")
    args = parser.parse_args()
    entries = load_entries(args.root)
    if args.name:
        matches = [e for e in entries if e["name"] == args.name]
        if not matches:
            raise SystemExit(f"Unknown benchmark: {args.name}")
        print(json.dumps(matches[0]))
    else:
        print(json.dumps({"include": entries}))


if __name__ == "__main__":
    main()
