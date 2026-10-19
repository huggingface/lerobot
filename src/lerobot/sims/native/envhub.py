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

"""EnvHub environments: the torch-free Hub loader and the simulator-server backend."""

import importlib.util
import logging
import os
from typing import Any

import gymnasium as gym
from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

# Simulator settings an EnvHub env.py may declare in a module-level LEROBOT_SIM dict, with the
# defaults used when neither the server configuration nor the repository sets them.
SIM_DEFAULTS: dict[str, Any] = {
    "fps": 30,
    "control": "position",
    "semantics": "envhub-v1",
    "state_names": [],
    "action_names": [],
    "gripper_indices": [],
    "robot_type": "",
}


# helper to safely load a python file as a module
def _load_module_from_path(path: str, module_name: str | None = None):
    module_name = module_name or f"hub_env_{os.path.basename(path).replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# helper to parse hub string (supports "user/repo", "user/repo@rev", optional path)
# examples:
#   "user/repo" -> will look for env.py at repo root
#   "user/repo@main:envs/my_env.py" -> explicit revision and path
def _parse_hub_url(hub_uri: str):
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


def _download_hub_file(
    cfg_str: str,
    trust_remote_code: bool,
    hub_cache_dir: str | None,
) -> tuple[str, str, str, str | None]:
    """
    Parse `cfg_str` (hub URL or local directory), enforce `trust_remote_code`, and return
    (repo_id, file_path, local_file, revision).
    """
    if not trust_remote_code:
        raise RuntimeError(
            f"Refusing to execute remote code from the Hub for '{cfg_str}'. "
            "Executing hub env modules runs arbitrary Python code from third-party repositories. "
            "If you trust this repo and understand the risks, call `make_env(..., trust_remote_code=True)` "
            "and prefer pinning to a specific revision: 'user/repo@<commit-hash>:env.py'."
        )

    # A local directory holding the repository's files, e.g. one baked into a simulator image.
    directory, _, local_path = cfg_str.partition(":")
    if os.path.isdir(directory):
        file_path = local_path or "env.py"
        return (
            os.path.basename(os.path.abspath(directory)),
            file_path,
            os.path.join(directory, file_path),
            None,
        )

    repo_id, revision, file_path = _parse_hub_url(cfg_str)

    try:
        local_file = hf_hub_download(
            repo_id=repo_id, filename=file_path, revision=revision, cache_dir=hub_cache_dir
        )
    except Exception as e:
        # fallback to snapshot download
        snapshot_dir = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=hub_cache_dir)
        local_file = os.path.join(snapshot_dir, file_path)
        if not os.path.exists(local_file):
            raise FileNotFoundError(
                f"Could not find {file_path} in repository {repo_id}@{revision or 'main'}"
            ) from e

    return repo_id, file_path, local_file, revision


def _import_hub_module(local_file: str, repo_id: str) -> Any:
    """
    Import the downloaded file as a module and surface helpful import error messages.
    """
    module_name = f"hub_env_{repo_id.replace('/', '_')}"
    try:
        module = _load_module_from_path(local_file, module_name=module_name)
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise ModuleNotFoundError(
            f"Hub env '{repo_id}:{os.path.basename(local_file)}' failed to import because the dependency "
            f"'{missing}' is not installed locally.\n\n"
        ) from e
    except ImportError as e:
        raise ImportError(
            f"Failed to load hub env module '{repo_id}:{os.path.basename(local_file)}'. Import error: {e}\n\n"
        ) from e
    return module


# Hub modules already executed in this process, keyed by file, modification time and size.
_MODULES: dict[tuple[str, int, int], Any] = {}


def _load_hub_module(hub_path: str, trust_remote_code: bool, hub_cache_dir: str | None) -> tuple[str, Any]:
    """Download and import a Hub env.py once per process.

    The server builds a Backend for its probe and for every session; executing env.py only once
    lets a repository keep process-wide state, such as a simulator app that can start only once.
    """
    repo_id, _, local_file, _ = _download_hub_file(hub_path, trust_remote_code, hub_cache_dir)
    stat = os.stat(local_file)
    key = (os.path.realpath(local_file), stat.st_mtime_ns, stat.st_size)
    if key not in _MODULES:
        _MODULES[key] = _import_hub_module(local_file, repo_id)
    return repo_id, _MODULES[key]


def _call_make_env(module: Any, n_envs: int, use_async_envs: bool, cfg: Any | None) -> Any:
    """
    Ensure module exposes make_env and call it.
    """
    if not hasattr(module, "make_env"):
        raise AttributeError(
            f"The hub module {getattr(module, '__name__', 'hub_module')} must expose `make_env(n_envs=int, use_async_envs=bool)`."
        )
    entry_fn = module.make_env
    # Only pass cfg if it's not None (i.e., when an EnvConfig was provided, not a string hub ID)
    if cfg is not None:
        return entry_fn(n_envs=n_envs, use_async_envs=use_async_envs, cfg=cfg)
    else:
        return entry_fn(n_envs=n_envs, use_async_envs=use_async_envs)


def _normalize_hub_result(result: Any) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """
    Normalize possible return types from hub `make_env` into the mapping:
      { suite_name: { task_id: vector_env } }
    Accepts:
      - dict (assumed already correct)
      - gym.vector.VectorEnv
      - gym.Env (will be wrapped into SyncVectorEnv)
    """
    if isinstance(result, dict):
        return result

    # VectorEnv: use its spec.id if available
    if isinstance(result, gym.vector.VectorEnv):
        suite_name = getattr(result, "spec", None) and getattr(result.spec, "id", None) or "hub_env"
        return {suite_name: {0: result}}

    # Single Env: wrap into SyncVectorEnv
    if isinstance(result, gym.Env):
        vec = gym.vector.SyncVectorEnv([lambda: result])
        suite_name = getattr(result, "spec", None) and getattr(result.spec, "id", None) or "hub_env"
        return {suite_name: {0: vec}}

    raise ValueError(
        "Hub `make_env` must return either a mapping {suite: {task_id: vec_env}}, "
        "a gym.vector.VectorEnv, or a single gym.Env."
    )


def _world_factories(vec: Any, n_envs: int) -> list:
    """Take the per-world factories of a built vector env, then close it: the server builds worlds lazily."""
    fns = list(getattr(vec, "env_fns", None) or getattr(vec, "_env_fns", None) or ())
    if len(fns) != n_envs:
        raise ValueError(f"Hub make_env must return vector envs built from {n_envs} per-world factories")
    vec.close()
    return fns


def create_envhub_envs(
    task: str,
    n_envs: int,
    env_cls: Any,
    hub_path: str,
    trust_remote_code: bool = False,
    hub_cache_dir: str | None = None,
) -> dict[str, dict[int, Any]]:
    """Serve the worlds of a Hub repository's ``make_env``.

    ``task`` is unused: suites and task ids come from the repository, and clients select them.
    """
    _, module = _load_hub_module(hub_path, trust_remote_code, hub_cache_dir)
    result = _call_make_env(module, n_envs=n_envs, use_async_envs=False, cfg=None)
    if isinstance(result, gym.Env):
        # A single env has no reusable factory, so each world calls make_env again.
        suite = getattr(result.spec, "id", None) if result.spec else None
        result.close()
        fns = [lambda: _call_make_env(module, n_envs=1, use_async_envs=False, cfg=None)] * n_envs
        return {suite or "hub_env": {0: env_cls(fns)}}
    return {
        suite: {task_id: env_cls(_world_factories(vec, n_envs)) for task_id, vec in tasks.items()}
        for suite, tasks in _normalize_hub_result(result).items()
    }


def resolve_envhub_sim(sim: dict[str, Any]) -> dict[str, Any]:
    """Fill the server settings left null with the repository's ``LEROBOT_SIM``, else the defaults.

    Explicit values (YAML or ``--set``) win; repositories without ``LEROBOT_SIM`` use the defaults.
    """
    missing = [key for key in SIM_DEFAULTS if sim.get(key) is None]
    if not missing:
        return sim
    kwargs = sim.get("kwargs", {})
    repo_id, module = _load_hub_module(
        kwargs["hub_path"], kwargs.get("trust_remote_code", False), kwargs.get("hub_cache_dir")
    )
    declared = getattr(module, "LEROBOT_SIM", {})
    unknown = set(declared) - set(SIM_DEFAULTS)
    if unknown:
        raise ValueError(
            f"{repo_id} LEROBOT_SIM has unknown keys {sorted(unknown)}; expected {list(SIM_DEFAULTS)}"
        )
    resolved = dict(sim)
    for key in missing:
        resolved[key] = declared.get(key, SIM_DEFAULTS[key])
    logger.info(
        "EnvHub %s settings: from LEROBOT_SIM %s, defaults %s",
        repo_id,
        sorted(key for key in missing if key in declared),
        sorted(key for key in missing if key not in declared),
    )
    return resolved
