#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import importlib
import os
from typing import Optional, Union
import gymnasium as gym
from huggingface_hub import hf_hub_download, snapshot_download

from lerobot.envs.configs import AlohaEnv, EnvConfig, LiberoEnv, PushtEnv, XarmEnv

# helper to safely load a python file as a module
def _load_module_from_path(path: str, module_name: Optional[str] = None):
    module_name = module_name or f"hub_env_{os.path.basename(path).replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

# helper to parse hub string (supports "user/repo", "user/repo@rev", optional path)
# examples:
#   "user/repo" -> will look for env.py at repo root
#   "user/repo@main:envs/my_env.py" -> explicit revision and path
def _parse_hub_uri(hub_uri: str):
    # very small parser: [repo_id][@revision][:path]
    # repo_id is required (user/repo or org/repo)
    revision = None
    file_path = "env.py"
    if "@" in hub_uri:
        repo_and_rev, *rest = hub_uri.split(":", 1)
        repo_id, rev = repo_and_rev.split("@", 1)
        revision = rev
        if rest:
            file_path = rest[0]
    else:
        repo_id, *rest = hub_uri.split(":", 1)
        if rest:
            file_path = rest[0]
    return repo_id, revision, file_path

def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "xarm":
        return XarmEnv(**kwargs)
    elif env_type == "libero":
        return LiberoEnv(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


def make_env(
    cfg: Union[EnvConfig, str], n_envs: int = 1, use_async_envs: bool = False, hub_cache_dir: Optional[str] = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Makes a gym vector environment according to the config or Hub reference.
    
    This function is the main entrypoint for creating environments in LeRobot. It supports two modes:
    1. **Local mode** – when `cfg` is an `EnvConfig` instance, it builds the environment from the
       locally registered environment types (e.g., `aloha`, `pusht`, `libero`).
    2. **Hub mode** – when `cfg` is a string (e.g., `"username/repo"` or `"username/repo@rev:envs/my_env.py"`),
       it downloads an `env.py` file from the Hugging Face Hub, dynamically imports it, and calls a
       `make_env(n_envs, use_async_envs)` function defined there.

    The returned object is always a dictionary mapping suite names to vectorized environments, which
    ensures a consistent interface across single-task and multi-task setups.
    
    Args:
        cfg (EnvConfig | str): Either an `EnvConfig` object describing the environment to build locally,
            or a Hugging Face Hub repository identifier (e.g. `"username/repo"`). In the latter case,
            the repo must include a Python file (usually `env.py`) exposing a function:
            
            ```python
            def make_env(n_envs: int = 1, use_async_envs: bool = False) -> dict | gym.Env | gym.vector.VectorEnv:
                ...
            ```
        n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
        use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
            False.

    Raises:
        ValueError: if n_envs < 1
        ModuleNotFoundError: If the requested env package is not installed
        AttributeError: If the hub module does not define a `make_env` function (or the function
            specified via `hub_uri_entry`).
        FileNotFoundError: If the requested `env.py` file cannot be found in the Hub repository.
        ImportError: If importing or executing the downloaded hub module fails due to missing
            dependencies or runtime errors.

    Returns:
        dict[str, dict[int, gym.vector.VectorEnv]]:
            A mapping from suite name to indexed vectorized environments.
            - For multi-task benchmarks (e.g., LIBERO): one entry per suite, and one vec env per task_id.
            - For single-task environments: a single suite entry (cfg.type) with task_id=0.
    Example:
        >>> # Local environment
        >>> envs = make_env(AlohaEnv(task="AlohaInsertion-v0"), n_envs=4)
        
        >>> # Hub environment (downloads env.py and calls make_env)
        >>> envs = make_env("username/my-robot-env", n_envs=8)
        
        >>> # Hub environment with custom entrypoint and cache path
        >>> envs = make_env(
        ...     "username/multi-env-repo@main:envs/pick_cube.py",
        ...     n_envs=4,
        ...     hub_uri_entry="make_env_pickcube",
        ...     hub_cache_dir="/raid/hub_cache"
        ... )
    """
    # if user passed a hub id string (e.g., "username/repo", "username/repo@main:env.py")
    # simplified: only support hub-provided `make_env`
    if isinstance(cfg, str):
        repo_id, revision, file_path = _parse_hub_uri(cfg)

        # try to download the single file; fallback to snapshot if needed
        try:
            local_file = hf_hub_download(repo_id=repo_id, filename=file_path, revision=revision)
        except Exception as e:
            snapshot_dir = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=hub_cache_dir)
            local_file = os.path.join(snapshot_dir, file_path)
            if not os.path.exists(local_file):
                raise FileNotFoundError(
                    f"Could not find {file_path} in repository {repo_id}@{revision or 'main'}"
                ) from e

        # import the downloaded module
        try:
            module = _load_module_from_path(local_file, module_name=f"hub_env_{repo_id.replace('/', '_')}")
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or str(e)
            raise ModuleNotFoundError(
                f"Hub env '{repo_id}:{file_path}' failed to import because the dependency "
                f"'{missing}' is not installed locally.\n\n"
                f"Suggested fixes:\n"
                f"  1) Install the missing package directly:    pip install {missing}\n"
                f"  2) Check the Hub repo for a requirements.txt or pyproject.toml and run:\n"
                f"       pip install -r requirements.txt\n"
                f"  3) If the repo documents an extras installation (e.g. `lerobot[foo]`), try:\n"
                f"       pip install \"lerobot[<extra>]\"\n\n"
                f"After installing the dependency, re-run your code. (Original error: {e})"
            ) from e
        except ImportError as e:
            # other import-time issues (e.g. incompatible package versions, syntax errors)
            raise ImportError(
                f"Failed to load hub env module '{repo_id}:{file_path}'. Import error: {e}\n\n"
                f"Check that the repository files are present and its dependencies are installed "
                f"(see requirements.txt / pyproject.toml in the repo)."
            ) from e

        # require a make_env entrypoint on the hub
        if not hasattr(module, "make_env"):
            raise AttributeError(
                f"The hub module {repo_id}:{file_path} must expose `make_env(n_envs=int, use_async_envs=bool)`."
            )
        entry_fn = getattr(module, "make_env")

        # call it
        result = entry_fn(n_envs=n_envs, use_async_envs=use_async_envs)

        # If the hub already returned the mapping we expect, return it directly
        if isinstance(result, dict):
            return result

        # If the hub returned a VectorEnv, wrap into mapping
        if isinstance(result, gym.vector.VectorEnv):
            suite_name = getattr(result, "spec", None) and getattr(result.spec, "id", "hub_env") or "hub_env"
            return {suite_name: {0: result}}

        # If the hub returned a single gym.Env, vectorize and return mapping
        if isinstance(result, gym.Env):
            # wrap into SyncVectorEnv of one env (consistent with local behavior)
            vec = gym.vector.SyncVectorEnv([lambda: result])
            suite_name = getattr(result, "spec", None) and getattr(result.spec, "id", "hub_env") or "hub_env"
            return {suite_name: {0: vec}}

        raise ValueError(
            "Hub `make_env` must return either a mapping {suite: {task_id: vec_env}}, "
            "a gym.vector.VectorEnv, or a single gym.Env."
        )

    # otherwise existing behavior: cfg is an EnvConfig (unchanged)
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    if "libero" in cfg.type:
        from lerobot.envs.libero import create_libero_envs

        if cfg.task is None:
            raise ValueError("LiberoEnv requires a task to be specified")

        return create_libero_envs(
            task=cfg.task,
            n_envs=n_envs,
            camera_name=cfg.camera_name,
            init_states=cfg.init_states,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
        )

    package_name = f"gym_{cfg.type}"
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
        raise e

    gym_handle = f"{package_name}/{cfg.task}"

    def _make_one():
        return gym.make(gym_handle, disable_env_checker=cfg.disable_env_checker, **(cfg.gym_kwargs or {}))

    vec = env_cls([_make_one for _ in range(n_envs)])

    # normalize to {suite: {task_id: vec_env}} for consistency
    suite_name = cfg.type  # e.g., "pusht", "aloha"
    return {suite_name: {0: vec}}
