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

"""EnvHub envs: load a Hub repository's ``env.py`` and build its envs."""

import importlib.util
import logging
import os
from collections.abc import Callable
from difflib import get_close_matches
from typing import Any

import gymnasium as gym
from huggingface_hub import hf_hub_download, snapshot_download

from .utils import canonical_observation

__all__ = ["canonical_observation", "conventions", "env_fns"]

logger = logging.getLogger(__name__)

# Settings an EnvHub env.py may set in a module-level LEROBOT_SIM dict. A key falls back to its default
# here when neither the sim config nor the repository sets it.
SIM_DEFAULTS: dict[str, Any] = {
    "fps": 30,
    "control_mode": "position",
    "semantics": "envhub-v1",
    "state_names": [],
    "action_names": [],
    "robot_type": "",
}


def _load_module_from_path(path: str, module_name: str | None = None):
    module_name = module_name or f"hub_env_{os.path.basename(path).replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_hub_url(hub_uri: str):
    """Split ``user/repo[@revision][:path]`` into repo id, revision and path (default ``env.py``)."""
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
    """Find (downloading if needed) the ``env.py`` ``cfg_str`` points to; needs ``trust_remote_code``."""
    if not trust_remote_code:
        raise RuntimeError(
            f"Refusing to execute remote code from the Hub for '{cfg_str}'. "
            "Executing hub env modules runs arbitrary Python code from third-party repositories. "
            "If you trust this repo and understand the risks, call `make_env(..., trust_remote_code=True)` "
            "and prefer pinning to a specific revision: 'user/repo@<commit-hash>:env.py'."
        )

    # A local copy of the repository, e.g. one baked into a simulator Docker image.
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


# Hub modules already run in this process, keyed by file path, mtime and size, so an edited file reloads.
_MODULES: dict[tuple[str, int, int], Any] = {}


def _load_hub_module(hub_path: str, trust_remote_code: bool, hub_cache_dir: str | None) -> tuple[str, Any]:
    """Import a Hub ``env.py`` at most once per process, so it can keep process-wide state."""
    repo_id, _, local_file, _ = _download_hub_file(hub_path, trust_remote_code, hub_cache_dir)
    stat = os.stat(local_file)
    key = (os.path.realpath(local_file), stat.st_mtime_ns, stat.st_size)
    if key not in _MODULES:
        _MODULES[key] = _import_hub_module(local_file, repo_id)
    return repo_id, _MODULES[key]


def _call_make_env(module: Any, n_envs: int, use_async_envs: bool, cfg: Any | None, **kwargs: Any) -> Any:
    """Call the module's ``make_env``, passing ``kwargs`` as repository-specific options."""
    if not hasattr(module, "make_env"):
        raise AttributeError(
            f"The hub module {getattr(module, '__name__', 'hub_module')} must expose `make_env(n_envs=int, use_async_envs=bool)`."
        )
    if cfg is not None:
        kwargs["cfg"] = cfg
    return module.make_env(n_envs=n_envs, use_async_envs=use_async_envs, **kwargs)


def _suite_name(env: Any) -> str:
    """Return the env's registered Gymnasium id, or ``hub_env`` if it has none."""
    return getattr(getattr(env, "spec", None), "id", None) or "hub_env"


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

    if isinstance(result, gym.vector.VectorEnv):
        return {_suite_name(result): {0: result}}

    if isinstance(result, gym.Env):
        return {_suite_name(result): {0: gym.vector.SyncVectorEnv([lambda: result])}}

    raise ValueError(
        "Hub `make_env` must return either a mapping {suite: {task_id: vec_env}}, "
        "a gym.vector.VectorEnv, or a single gym.Env."
    )


def _env_factories(vec: Any, n_envs: int) -> list[Callable[[], Any]]:
    """Return a vector env's per-env factories and close it; ``LocalVectorEnv`` builds the envs again."""
    try:
        if type(vec) not in {gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv}:
            raise ValueError(
                "The shared simulation path requires an unwrapped SyncVectorEnv or AsyncVectorEnv. "
                "Use --env.use_native_batch=true to keep provider-owned vector behavior locally."
            )
        fns = list(vec.env_fns)
        if len(fns) != n_envs:
            raise ValueError(f"Hub make_env must return {n_envs} per-environment factories")
        return fns
    finally:
        vec.close()


def env_fns(
    task: str,
    n_envs: int,
    hub_path: str,
    trust_remote_code: bool = False,
    hub_cache_dir: str | None = None,
    make_env_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[int, list[Callable[[], Any]]]]:
    """Return factories for a Hub repo's envs from ``make_env(**make_env_kwargs)``; ``task`` is unused."""
    _, module = _load_hub_module(hub_path, trust_remote_code, hub_cache_dir)
    options = make_env_kwargs or {}
    result = _call_make_env(module, n_envs=n_envs, use_async_envs=False, cfg=None, **options)
    if isinstance(result, gym.Env):
        # A plain env has no factory to reuse, so each factory calls make_env again.
        suite = _suite_name(result)
        result.close()
        fns = [lambda: _call_make_env(module, n_envs=1, use_async_envs=False, cfg=None, **options)] * n_envs
        return {suite: {0: fns}}
    return {
        suite: {task_id: _env_factories(vec, n_envs) for task_id, vec in tasks.items()}
        for suite, tasks in _normalize_hub_result(result).items()
    }


def conventions(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the repo's ``LEROBOT_SIM`` over ``SIM_DEFAULTS``; an unknown key raises, naming the closest."""
    repo_id, module = _load_hub_module(
        kwargs["hub_path"], kwargs.get("trust_remote_code", False), kwargs.get("hub_cache_dir")
    )
    declared = getattr(module, "LEROBOT_SIM", {})
    unknown = sorted(set(declared) - set(SIM_DEFAULTS))
    if unknown:
        keys = []
        for key in unknown:
            match = get_close_matches(key, SIM_DEFAULTS, n=1)
            keys.append(f"{key!r} (did you mean {match[0]!r}?)" if match else repr(key))
        raise ValueError(
            f"{repo_id} LEROBOT_SIM has unknown keys {', '.join(keys)}; expected {list(SIM_DEFAULTS)}"
        )
    logger.info("EnvHub %s declares %s in LEROBOT_SIM", repo_id, sorted(declared))
    return {**SIM_DEFAULTS, **declared}
