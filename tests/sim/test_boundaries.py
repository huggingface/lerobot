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

"""Dependency boundaries of the simulation code: no simulator names in shared code, no transport locally."""

import ast
import builtins
from pathlib import Path

import numpy as np

from tests.fixtures.toy_env import ToyEnvConfig


def test_shared_simulation_code_names_no_simulator() -> None:
    root = Path(__file__).parents[2] / "src" / "lerobot"
    simulators = {path.stem for path in (root / "sim" / "integrations").glob("*.py")} - {"__init__", "utils"}
    shared = [
        *(root / "sim").glob("*.py"),
        *(root / "env_server").glob("*.py"),
        *(root / "robots" / "sim_robot").glob("*.py"),
    ]
    named = [
        f"{path.relative_to(root)}: {node.value}"
        for path in shared
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and node.value in simulators
    ]
    assert named == []


def test_simulation_package_does_not_import_service_or_transport() -> None:
    package = Path(__file__).parents[2] / "src" / "lerobot" / "sim"
    forbidden: list[str] = []
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            forbidden.extend(
                f"{path.relative_to(package)}: {module}"
                for module in modules
                if module.startswith(("lerobot.env_server", "lerobot.transport"))
            )
    assert forbidden == []


def test_local_environment_does_not_import_deployment_code(monkeypatch) -> None:
    original_import = builtins.__import__

    def import_without_deployment(name, *args, **kwargs):
        if name.startswith(("lerobot.env_server", "lerobot.transport")):
            raise AssertionError(f"Local environment imported deployment module {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_deployment)
    env = ToyEnvConfig().create_envs(n_envs=1)["reach"][0]
    try:
        obs, info = env.reset(seed=3)
        assert obs["observation.state"].shape == (1, 2)
        assert info["elapsed_steps"].tolist() == [0]

        terminal = env.step(np.array([[2.0, 0.0]], dtype=np.float32))
        assert terminal[1].tolist() == [1.0]
        assert terminal[2].tolist() == [True]
        terminal_obs = terminal[0]["observation.state"].copy()

        frozen = env.step(np.array([[-1.0, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(frozen[0]["observation.state"], terminal_obs)
        assert frozen[4]["elapsed_steps"].tolist() == [1]

        _, reset_info = env.reset(seed=3)
        assert reset_info["elapsed_steps"].tolist() == [0]
    finally:
        env.close()
