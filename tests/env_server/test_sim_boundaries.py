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

"""Dependency and behavior boundaries for local simulation."""

import ast
import builtins
from pathlib import Path

import numpy as np

from tests.env_server.toy_env import ToyEnvConfig


def test_simulation_package_does_not_import_service_or_transport() -> None:
    sims = Path(__file__).parents[2] / "src" / "lerobot" / "sims"
    forbidden: list[str] = []
    for path in sims.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            forbidden.extend(
                f"{path.relative_to(sims)}: {module}"
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
        assert info["step"].tolist() == [0]

        terminal = env.step(np.array([[2.0, 0.0]], dtype=np.float32))
        assert terminal[1].tolist() == [1.0]
        assert terminal[2].tolist() == [True]
        terminal_obs = terminal[0]["observation.state"].copy()

        frozen = env.step(np.array([[-1.0, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(frozen[0]["observation.state"], terminal_obs)
        assert frozen[4]["step"].tolist() == [1]

        _, reset_info = env.reset(seed=3)
        assert reset_info["step"].tolist() == [0]
    finally:
        env.close()


def test_backend_delegates_native_observation_conventions(monkeypatch) -> None:
    from gymnasium import spaces

    from lerobot.sims import backend as backend_module
    from lerobot.sims.native import NativeIntegration

    class NativeEnv:
        observation_space = spaces.Dict({"native_state": spaces.Box(-1, 1, (1,), np.float32)})
        action_space = spaces.Box(-1, 1, (1,), np.float32)
        task = "pick"
        task_description = "Pick the object"
        _max_episode_steps = 2

        def reset(self, seed=None):
            return {"native_state": np.array([0.25], dtype=np.float32)}, {}

        def step(self, action):
            return {"native_state": np.asarray(action, dtype=np.float32)}, 0.0, False, False, {}

        def close(self):
            pass

    adapted = []

    def canonical_observation(obs):
        adapted.append(obs)
        return {"observation.state": np.asarray(obs["native_state"], dtype=np.float32)}

    def create_envs(task, n_envs, env_cls, **kwargs):
        return {task: {0: env_cls([NativeEnv] * n_envs)}}

    integration = NativeIntegration(create_envs, canonical_observation)
    monkeypatch.setattr(backend_module, "load_native_integration", lambda name: integration)
    backend = backend_module.Backend(
        backend_module.BackendConfig(type="new_engine", task="pick", semantics="new-engine-v1")
    )
    try:
        observation_space, _ = backend.single_spaces()
        assert observation_space["observation.state"].shape == (1,)
        result = backend.reset([7])
        np.testing.assert_array_equal(result.obs["observation.state"], [[0.25]])
        assert len(adapted) == 2
    finally:
        backend.close()
