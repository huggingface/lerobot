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

"""EnvHub envs: load a Hub repository's `env.py` and build its envs."""

import importlib.util
import logging
import os
import warnings
from collections.abc import Callable
from difflib import get_close_matches
from functools import partial
from typing import Any

import gymnasium as gym
from huggingface_hub import snapshot_download

from .utils import canonical_observation

__all__ = ["canonical_observation", "env_fns", "make_envs", "sim_settings", "vector_env_fns"]

logger = logging.getLogger(__name__)

# Settings an EnvHub env.py's sim_settings may return. A key falls back to its default here when neither the
# sim config nor the repository sets it.
_DEFAULTS: dict[str, Any] = {
    "fps": 30,
    "control_mode": "position",
    "semantics": "envhub-v1",
    "state_names": [],
    "action_names": [],
    "robot_type": "",
    "features": {},
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
    """Split `user/repo[@revision][:path]` into repo id, revision and path (default `env.py`)."""
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


def _split_local_path(cfg_str: str) -> tuple[str, str]:
    """Split a local `directory[:file_path]` spec without splitting Windows drive letters."""
    if os.path.isdir(cfg_str):
        return cfg_str, ""
    drive, path = os.path.splitdrive(cfg_str)
    directory, _, local_path = path.partition(":")
    return drive + directory, local_path


def _download_hub_file(
    cfg_str: str,
    trust_remote_code: bool,
    hub_cache_dir: str | None,
) -> tuple[str, str, str, str | None]:
    """Find the `env.py` `cfg_str` points to, downloading its whole repository; needs `trust_remote_code`.

    The repository's other files (scenes, models, configs) sit next to `env.py`, at the same revision, as
    they do in a local directory and in a simulator image.
    """
    # A local copy of the repository, e.g. one baked into a simulator Docker image.
    directory, local_path = _split_local_path(cfg_str)
    if not trust_remote_code:
        # A local folder has no commit to pin.
        read = (
            "Read its env.py"
            if os.path.isdir(directory)
            else "Read its env.py, pin the commit you read (user/repo@<commit>)"
        )
        raise RuntimeError(
            f"Refusing to run the EnvHub environment '{cfg_str}': loading it runs its env.py in this process. {read}, "
            "then pass --env.trust_remote_code=true (--robot.env.trust_remote_code=true for a sim robot, "
            "HubEnvConfig(..., trust_remote_code=True) in Python), or serve it with lerobot-env-server, which "
            "runs it there."
        )
    if os.path.isdir(directory):
        file_path = local_path or "env.py"
        return (
            os.path.basename(os.path.abspath(directory)),
            file_path,
            os.path.join(directory, file_path),
            None,
        )

    repo_id, revision, file_path = _parse_hub_url(cfg_str)

    snapshot_dir = snapshot_download(
        repo_id=repo_id, repo_type="space", revision=revision, cache_dir=hub_cache_dir
    )
    local_file = os.path.join(snapshot_dir, file_path)
    if not os.path.isfile(local_file):
        raise FileNotFoundError(f"Could not find {file_path} in repository {repo_id}@{revision or 'main'}")

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
    """Import a Hub `env.py` at most once per process, so it can keep process-wide state."""
    repo_id, _, local_file, _ = _download_hub_file(hub_path, trust_remote_code, hub_cache_dir)
    stat = os.stat(local_file)
    key = (os.path.realpath(local_file), stat.st_mtime_ns, stat.st_size)
    if key not in _MODULES:
        _MODULES[key] = _import_hub_module(local_file, repo_id)
    return repo_id, _MODULES[key]


def _call_make_env(module: Any, n_envs: int, use_async_envs: bool, cfg: Any | None, **kwargs: Any) -> Any:
    """Call the module's `make_env`, passing `kwargs` as repository-specific options."""
    if not hasattr(module, "make_env"):
        raise AttributeError(
            f"The hub module {getattr(module, '__name__', 'hub_module')} must expose `make_env(n_envs=int, use_async_envs=bool)`."
        )
    if cfg is not None:
        kwargs["cfg"] = cfg
    return module.make_env(n_envs=n_envs, use_async_envs=use_async_envs, **kwargs)


def _suite_name(env: Any) -> str:
    """Return the env's registered Gymnasium id, or `hub_env` if it has none."""
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


def make_envs(
    hub_path: str,
    n_envs: int,
    use_async_envs: bool,
    trust_remote_code: bool = False,
    hub_cache_dir: str | None = None,
    cfg: Any | None = None,
    **options: Any,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Return the envs of a Hub repo's `make_env`, called with `options` and, if given, `cfg`."""
    _, module = _load_hub_module(hub_path, trust_remote_code, hub_cache_dir)
    return _normalize_hub_result(_call_make_env(module, n_envs, use_async_envs, cfg, **options))


def _env_factories(vec: Any, n_envs: int) -> list[Callable[[], Any]]:
    """Return a vector env's per-env factories and close it; `LocalVectorEnv` builds the envs again."""
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
    """Return factories for a Hub repo's envs from `make_env(**make_env_kwargs)`; `task` is unused."""
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


def vector_env_fns(
    task: str,
    n_envs: int,
    hub_path: str,
    trust_remote_code: bool = False,
    hub_cache_dir: str | None = None,
    make_env_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[int, Callable[[], Any]]] | None:
    """Return the repo's own `vector_env_fns(task, n_envs, **make_env_kwargs)`, or None if it has none.

    As for an in-tree integration, each factory builds one vector env of `n_envs` envs that does not
    autoreset, only when its task runs (e.g. a GPU simulator stepping every env at once).
    """
    _, module = _load_hub_module(hub_path, trust_remote_code, hub_cache_dir)
    make = getattr(module, "vector_env_fns", None)
    if make is None:
        return None
    options = make_env_kwargs or {}
    repo = (hub_path, trust_remote_code, hub_cache_dir, task, n_envs, options)
    return {
        suite: {task_id: partial(_build_repo_vector_env, *repo, suite, task_id) for task_id in tasks}
        for suite, tasks in make(task=task, n_envs=n_envs, **options).items()
    }


def _build_repo_vector_env(
    hub_path: str,
    trust_remote_code: bool,
    hub_cache_dir: str | None,
    task: str,
    n_envs: int,
    options: dict[str, Any],
    suite: str,
    task_id: int,
) -> Any:
    """Build one task's vector env from the repo's `vector_env_fns`, loading the repo in this process.

    A worker process can't receive the repo's own factories: `env.py` isn't importable there, so they would be
    pickled with its module state (e.g. a running simulator).
    """
    _, module = _load_hub_module(hub_path, trust_remote_code, hub_cache_dir)
    return module.vector_env_fns(task=task, n_envs=n_envs, **options)[suite][task_id]()


def sim_settings(
    task: str,
    hub_path: str,
    trust_remote_code: bool = False,
    hub_cache_dir: str | None = None,
    make_env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the repo's `sim_settings(task, **make_env_kwargs)` over the EnvHub defaults.

    An unknown key raises, naming the closest. A repo without `sim_settings` may set the dict as `LEROBOT_SIM`,
    as LeRobot 0.7 reads it.
    """
    repo_id, module = _load_hub_module(hub_path, trust_remote_code, hub_cache_dir)
    declare = getattr(module, "sim_settings", None)
    if declare is not None:
        declared = declare(task, **(make_env_kwargs or {}))
        if not isinstance(declared, dict):
            raise TypeError(
                f"{repo_id} sim_settings({task!r}) must return a dict, got {type(declared).__name__}"
            )
        source = "sim_settings"
    else:
        legacy = getattr(module, "LEROBOT_SIM", None)
        declared, source = legacy or {}, "LEROBOT_SIM"
        if legacy is not None:
            warnings.warn(
                f"{repo_id} sets LEROBOT_SIM, which is deprecated: define sim_settings(task, **make_env_kwargs) "
                "returning the same dict instead (https://huggingface.co/docs/lerobot/envhub#sim_settings).",
                FutureWarning,
                stacklevel=2,
            )
    unknown = sorted(set(declared) - set(_DEFAULTS))
    if unknown:
        keys = []
        for key in unknown:
            match = get_close_matches(key, _DEFAULTS, n=1)
            keys.append(f"{key!r} (did you mean {match[0]!r}?)" if match else repr(key))
        raise ValueError(f"{repo_id} {source} has unknown keys {', '.join(keys)}; expected {list(_DEFAULTS)}")
    # Called for each env built, so once per task.
    logger.debug("EnvHub %s declares %s for %r in %s", repo_id, sorted(declared), task, source)
    return {**_DEFAULTS, **declared}
