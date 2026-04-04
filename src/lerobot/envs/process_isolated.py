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
"""Process-isolated environment wrapper for low-VRAM GPUs.

Runs simulation environments in a subprocess with GPU access disabled,
so that the main process can use all available VRAM for policy inference.
The subprocess uses OSMesa (software rendering) instead of EGL (GPU rendering).

This solves the OOM issue on GPUs with <=8 GB VRAM where the PyTorch CUDA
context and the EGL rendering context compete for limited GPU memory.

Security note: IPC uses ``multiprocessing.Pipe`` (pickle-based). Only the
self-spawned subprocess communicates over the pipe; do not expose the pipe
file descriptor to untrusted processes.

Usage:
    lerobot-eval \\
        --policy.path=lerobot/smolvla_base \\
        --env.type=libero \\
        --eval.batch_size=1 \\
        --eval.process_isolated=true \\
        --policy.device=cuda
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import multiprocessing as mp
import multiprocessing.process
import os
import threading
import traceback
from collections.abc import Iterator
from multiprocessing.connection import Connection
from typing import Any

import gymnasium as gym
import numpy as np

from lerobot.envs.configs import EnvConfig

logger = logging.getLogger(__name__)

_SUBPROCESS_STARTUP_TIMEOUT_S = 120
_DEFAULT_RENDER_FPS = 30


def make_env_in_subprocess(
    env_cfg: EnvConfig | str,
    n_envs: int = 1,
    use_async_envs: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, dict[int, EnvProxy]]:
    """Create environments in a subprocess with GPU access disabled.

    This is a drop-in replacement for ``make_env`` that returns proxy objects
    implementing the same interface as ``gym.vector.VectorEnv``.

    The subprocess sets ``CUDA_VISIBLE_DEVICES=""`` and ``MUJOCO_GL=osmesa``
    so that no GPU memory is allocated by the environment or its rendering
    backend.

    Args:
        env_cfg: Environment configuration (same as ``make_env``).
        n_envs: Number of parallel environments per VectorEnv.
        use_async_envs: Whether to use AsyncVectorEnv inside the subprocess.
        trust_remote_code: Whether to trust remote code from the Hub.

    Returns:
        Nested dict mapping suite names to task IDs to EnvProxy objects.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    lock = threading.Lock()

    process = ctx.Process(
        target=_env_server_main,
        args=(child_conn, env_cfg, n_envs, use_async_envs, trust_remote_code),
        daemon=True,
    )
    process.start()
    child_conn.close()

    # Wait for the server to report ready with environment metadata.
    if not parent_conn.poll(timeout=_SUBPROCESS_STARTUP_TIMEOUT_S):
        exit_code = process.exitcode
        raise RuntimeError(
            f"Environment subprocess did not respond within {_SUBPROCESS_STARTUP_TIMEOUT_S}s "
            f"(exit code: {exit_code}). Check that OSMesa is installed and MUJOCO_GL=osmesa works."
        )

    msg = parent_conn.recv()
    if msg[0] == "error":
        process.join(timeout=5)
        raise RuntimeError(f"Environment subprocess failed to start: {msg[1]}")

    if msg[0] != "ready":
        raise RuntimeError(f"Unexpected message from subprocess: {msg}")

    env_metadata: dict[str, dict[int, dict]] = msg[1]

    # Build proxy objects.  All proxies share the same connection and lock.
    envs_dict: dict[str, dict[int, EnvProxy]] = {}
    for suite_name, tasks in env_metadata.items():
        envs_dict[suite_name] = {}
        for task_id_str, meta in tasks.items():
            task_id = int(task_id_str)
            envs_dict[suite_name][task_id] = EnvProxy(
                conn=parent_conn,
                lock=lock,
                suite_name=suite_name,
                task_id=task_id,
                metadata=meta,
                process=process,
            )

    # Ensure the subprocess is cleaned up if the main process exits unexpectedly.
    atexit.register(close_process_isolated_envs, envs_dict)

    return envs_dict


# ── Subprocess entry point ──────────────────────────────────────────────────


def _env_server_main(
    conn: Connection,
    env_cfg: EnvConfig | str,
    n_envs: int,
    use_async_envs: bool,
    trust_remote_code: bool,
) -> None:
    """Entry point for the environment subprocess."""
    # Disable GPU access so all rendering uses software (OSMesa).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_GL"] = "osmesa"

    try:
        from lerobot.envs.factory import make_env

        envs = make_env(
            env_cfg,
            n_envs=n_envs,
            use_async_envs=use_async_envs,
            trust_remote_code=trust_remote_code,
        )

        # Collect metadata for each VectorEnv so the parent can build proxies.
        metadata: dict[str, dict[str, dict]] = {}
        for suite_name, tasks in envs.items():
            metadata[suite_name] = {}
            for task_id, vec_env in tasks.items():
                metadata[suite_name][str(task_id)] = _collect_env_metadata(vec_env)

        conn.send(("ready", metadata))

    except Exception:
        logger.error("Environment subprocess failed during setup:\n%s", traceback.format_exc())
        conn.send(("error", traceback.format_exc()))
        conn.close()
        return

    # Serve requests until the parent closes the connection.
    try:
        _serve_requests(conn, envs)
    except (EOFError, BrokenPipeError):
        pass
    finally:
        for suite_tasks in envs.values():
            for vec_env in suite_tasks.values():
                with contextlib.suppress(Exception):
                    vec_env.close()
        conn.close()


def _collect_env_metadata(vec_env: gym.vector.VectorEnv) -> dict[str, Any]:
    """Extract serializable metadata from a VectorEnv."""
    has_task_description = False
    has_task = False
    task_descriptions: list[str] = []

    if isinstance(vec_env, gym.vector.SyncVectorEnv) and hasattr(vec_env.envs[0], "task_description"):
        has_task_description = True
        task_descriptions = [str(e.task_description) for e in vec_env.envs]
    elif isinstance(vec_env, gym.vector.SyncVectorEnv) and hasattr(vec_env.envs[0], "task"):
        has_task = True
        task_descriptions = [str(e.task) for e in vec_env.envs]

    try:
        max_episode_steps = vec_env.call("_max_episode_steps")[0] if vec_env.num_envs > 0 else 0
    except (AttributeError, IndexError):
        max_episode_steps = 0

    render_fps = _DEFAULT_RENDER_FPS
    if hasattr(vec_env, "unwrapped") and hasattr(vec_env.unwrapped, "metadata"):
        render_fps = vec_env.unwrapped.metadata.get("render_fps", _DEFAULT_RENDER_FPS)

    return {
        "num_envs": vec_env.num_envs,
        "has_task_description": has_task_description,
        "has_task": has_task,
        "task_descriptions": task_descriptions,
        "max_episode_steps": max_episode_steps,
        "render_fps": render_fps,
    }


def _serve_requests(conn: Connection, envs: dict[str, dict[int, gym.vector.VectorEnv]]) -> None:
    """Handle RPC-style requests from the parent process."""
    while True:
        cmd, payload = conn.recv()

        if cmd == "close_all":
            conn.send(("ok", None))
            return

        suite, task_id = payload["suite"], payload["task_id"]
        vec_env = envs[suite][task_id]

        try:
            if cmd == "reset":
                obs, info = vec_env.reset(seed=payload.get("seed"))
                conn.send(("ok", (obs, info)))

            elif cmd == "step":
                result = vec_env.step(payload["action"])
                conn.send(("ok", result))

            elif cmd == "call":
                result = vec_env.call(payload["method"])
                # Convert tuples to lists for consistency.
                if isinstance(result, tuple):
                    result = list(result)
                conn.send(("ok", result))

            elif cmd == "render_single":
                idx = payload["index"]
                if isinstance(vec_env, gym.vector.SyncVectorEnv):
                    frame = vec_env.envs[idx].render()
                else:
                    frame = vec_env.call("render")[idx]
                conn.send(("ok", frame))

            elif cmd == "close_env":
                vec_env.close()
                conn.send(("ok", None))

            else:
                conn.send(("error", f"Unknown command: {cmd}"))

        except Exception:
            tb = traceback.format_exc()
            logger.error("Subprocess error handling '%s': %s", cmd, tb)
            conn.send(("error", tb))


# ── Proxy objects ────────────────────────────────────────────────────────────


class _UnwrappedProxy:
    """Minimal proxy for vec_env.unwrapped.metadata access."""

    def __init__(self, render_fps: int):
        self.metadata = {"render_fps": render_fps}


class _EnvItemProxy:
    """Proxy for a single environment item (env.envs[i])."""

    def __init__(self, proxy: EnvProxy, index: int):
        self._proxy = proxy
        self._index = index

    @property
    def task_description(self) -> str:
        return self._proxy._task_descriptions[self._index]

    @property
    def task(self) -> str:
        return self._proxy._task_descriptions[self._index]

    def render(self) -> np.ndarray:
        """Render this single environment via IPC."""
        return self._proxy._render_single(self._index)


class _EnvsListProxy:
    """Proxy for the env.envs list, providing indexed access."""

    def __init__(self, proxy: EnvProxy):
        self._proxy = proxy
        self._items = [_EnvItemProxy(proxy, i) for i in range(proxy.num_envs)]

    def __getitem__(self, index: int) -> _EnvItemProxy:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[_EnvItemProxy]:
        return iter(self._items)


class EnvProxy:
    """Proxy for a gym.vector.VectorEnv running in a subprocess.

    Implements the subset of the VectorEnv interface used by the LeRobot
    evaluation pipeline (rollout, eval_policy, render_frame, add_envs_task).

    All IPC calls are guarded by a threading lock, so multiple proxies
    sharing the same pipe can be used safely from different threads (though
    calls will be serialized).
    """

    def __init__(
        self,
        conn: Connection,
        lock: threading.Lock,
        suite_name: str,
        task_id: int,
        metadata: dict,
        process: mp.process.BaseProcess,
    ):
        self._conn = conn
        self._lock = lock
        self._suite = suite_name
        self._task_id = task_id
        self._process = process

        self.num_envs: int = metadata["num_envs"]
        self._has_task_description: bool = metadata["has_task_description"]
        self._has_task: bool = metadata["has_task"]
        self._task_descriptions: list[str] = metadata["task_descriptions"]
        self._max_episode_steps: int = metadata["max_episode_steps"]
        self._render_fps: int = metadata["render_fps"]

        self._unwrapped = _UnwrappedProxy(self._render_fps)
        self._envs = _EnvsListProxy(self)

    def _send_recv(self, msg: tuple) -> Any:
        """Thread-safe send-then-recv on the shared pipe."""
        with self._lock:
            self._conn.send(msg)
            status, result = self._conn.recv()
        if status == "error":
            raise RuntimeError(f"Env subprocess error: {result}")
        return result

    # ── VectorEnv interface ──────────────────────────────────────────────

    def reset(self, seed: list[int] | range | None = None) -> tuple[dict, dict]:
        kwargs: dict[str, Any] = {"suite": self._suite, "task_id": self._task_id}
        if seed is not None:
            kwargs["seed"] = list(seed) if isinstance(seed, range) else seed
        return self._send_recv(("reset", kwargs))

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        return self._send_recv(("step", {"suite": self._suite, "task_id": self._task_id, "action": action}))

    def call(self, method_name: str) -> Any:
        # Fast path: avoid IPC for known cached values.
        if method_name == "_max_episode_steps":
            return [self._max_episode_steps] * self.num_envs
        if method_name == "task_description":
            return list(self._task_descriptions)
        if method_name == "task":
            return list(self._task_descriptions)

        return self._send_recv(
            ("call", {"suite": self._suite, "task_id": self._task_id, "method": method_name})
        )

    def close(self) -> None:
        with contextlib.suppress(EOFError, BrokenPipeError, OSError, RuntimeError):
            self._send_recv(("close_env", {"suite": self._suite, "task_id": self._task_id}))

    # ── Properties used by eval pipeline ─────────────────────────────────

    @property
    def envs(self) -> _EnvsListProxy:
        """Proxy for direct env access (used by render_frame and add_envs_task)."""
        return self._envs

    @property
    def unwrapped(self) -> _UnwrappedProxy:
        return self._unwrapped

    @property
    def connection(self) -> Connection:
        """The shared pipe connection (used by close_process_isolated_envs)."""
        return self._conn

    @property
    def subprocess(self) -> mp.process.BaseProcess:
        """The subprocess managing this environment."""
        return self._process

    # ── Rendering ────────────────────────────────────────────────────────

    def _render_single(self, index: int) -> np.ndarray:
        """Render a single environment by index (called by _EnvItemProxy.render)."""
        return self._send_recv(
            ("render_single", {"suite": self._suite, "task_id": self._task_id, "index": index})
        )


def close_process_isolated_envs(envs: dict[str, dict[int, EnvProxy]]) -> None:
    """Close all process-isolated environments and terminate the subprocess.

    Must be called when evaluation is complete.  Also registered via
    ``atexit`` as a safety net.
    """
    conn = None
    process = None
    for suite_tasks in envs.values():
        for proxy in suite_tasks.values():
            conn = proxy.connection
            process = proxy.subprocess
            break
        if conn is not None:
            break

    if conn is not None:
        try:
            conn.send(("close_all", {}))
            conn.recv()
        except (EOFError, BrokenPipeError, OSError):
            pass
        finally:
            conn.close()

    if process is not None and process.is_alive():
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
