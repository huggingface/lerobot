#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Docker runtime for lerobot-eval.

The policy stays on the host GPU; gym environments run inside Docker containers.
Each container runs `lerobot-eval-worker`, which calls back to a host HTTP inference
server for action chunks.

Architecture:
    host (GPU):
        1. Load policy + preprocessors from EvalPipelineConfig.
        2. Start ``policy_servers`` HTTP inference servers on consecutive ports.
        3. Spawn ``instance_count`` Docker containers, round-robin assigned to servers.
        4. Wait; collect per-task JSON written to the mounted output volume.
        5. Merge shards → aggregate → write eval_info.json.

    container (CPU only):
        1. make_env(cfg.env) → shard tasks by (instance_id, instance_count).
        2. For each task: run n_episodes, POST obs to /predict_chunk, step env.
        3. Write per-task JSON to /results/worker_{instance_id}.json.
"""

from __future__ import annotations

import json
import logging
import pickle  # nosec B403 — internal serialisation only
import platform
import subprocess  # nosec B404
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device

if TYPE_CHECKING:
    from lerobot.configs.eval import EvalPipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP inference server (host side)
# ---------------------------------------------------------------------------


class _PolicyInferenceHandler(BaseHTTPRequestHandler):
    """POST /predict_chunk → pickled numpy action chunk."""

    server: _InferenceServer

    def do_POST(self) -> None:
        if self.path != "/predict_chunk":
            self.send_error(404)
            return
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        payload: dict = pickle.loads(body)  # nosec B301
        obs_t: dict = payload["obs_t"]

        with self.server._lock:
            chunk_np = self.server._predict(obs_t)

        resp = pickle.dumps(chunk_np)  # nosec B301
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401
        pass  # suppress per-request logs


class _InferenceServer(HTTPServer):
    """Wraps the loaded policy behind a trivial HTTP interface."""

    def __init__(
        self,
        addr: tuple[str, int],
        policy: Any,
        env_preprocessor: Any,
        preprocessor: Any,
        postprocessor: Any,
    ) -> None:
        super().__init__(addr, _PolicyInferenceHandler)
        self._policy = policy
        self._env_preprocessor = env_preprocessor
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._lock = threading.Lock()
        self._device = torch.device(str(policy.config.device))

    def _predict(self, obs_t: dict) -> np.ndarray:
        """Apply full preprocessing pipeline and return (T, A) numpy chunk."""
        obs = self._env_preprocessor(obs_t)
        obs = self._preprocessor(obs)
        obs_gpu: dict = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
        with torch.no_grad():
            chunk: torch.Tensor = self._policy.predict_action_chunk(obs_gpu)  # (B, T, A)

        # Postprocessor expects (B, A); apply it treating each timestep as a batch element.
        # For linear transforms (unnormalize) this is identical to applying it to (B, T, A).
        batch, n_steps, action_dim = chunk.shape
        chunk_2d = chunk.reshape(batch * n_steps, action_dim)  # (B*T, A)
        chunk_2d = self._postprocessor(chunk_2d)  # (B*T, A)
        # Return only the first env's chunk — batch_size=1 per container.
        return chunk_2d[:n_steps].cpu().numpy()  # (T, A)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_host_ip() -> str:
    """Return the IP that containers can use to reach the host."""
    if platform.system() in ("Darwin", "Windows"):
        return "host.docker.internal"
    return "172.17.0.1"  # Linux Docker bridge default gateway


def _resolve_image(cfg: EvalPipelineConfig) -> str:
    """Return the Docker image name to use for the env containers."""
    if cfg.eval.docker.image:
        return cfg.eval.docker.image
    return f"lerobot-benchmark-{cfg.env.type}"


def _env_argv() -> list[str]:
    """Extract --env.* args from sys.argv to forward verbatim to the worker."""
    return [arg for arg in sys.argv[1:] if arg.startswith("--env.")]


def _spawn_container(
    *,
    image: str,
    instance_id: int,
    instance_count: int,
    server_address: str,
    n_episodes: int,
    seed: int,
    output_dir: Path,
    docker_cfg: Any,
    env_argv: list[str],
) -> subprocess.Popen:
    output_dir.mkdir(parents=True, exist_ok=True)
    container_results = "/results"

    cmd: list[str] = ["docker", "run", "--rm"]
    if docker_cfg.gpus:
        cmd += [f"--gpus={docker_cfg.gpus}"]
    cmd += [f"--shm-size={docker_cfg.shm_size}"]
    cmd += ["-v", f"{output_dir.resolve()}:{container_results}"]
    # Allow containers on Linux to resolve host.docker.internal.
    cmd += ["--add-host=host.docker.internal:host-gateway"]
    cmd.append(image)

    cmd += [
        "lerobot-eval-worker",
        *env_argv,
        f"--server_address={server_address}",
        f"--n_episodes={n_episodes}",
        f"--seed={seed}",
        f"--instance_id={instance_id}",
        f"--instance_count={instance_count}",
        f"--output_path={container_results}/worker_{instance_id}.json",
    ]

    logger.info(
        "Spawning container %d/%d: %s",
        instance_id + 1,
        instance_count,
        " ".join(cmd),
    )
    return subprocess.Popen(cmd)  # nosec B603 B607


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_eval_in_docker(cfg: EvalPipelineConfig) -> None:
    """Run eval with env in Docker containers and policy on the host GPU.

    Writes ``eval_info.json`` to ``cfg.output_dir``.  Called by
    ``lerobot_eval._run_eval_worker`` when ``eval.runtime == "docker"``.
    """
    # Import here to avoid circular import at module level.
    from lerobot.scripts.lerobot_eval import _aggregate_eval_from_per_task

    start_t = time.time()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docker_cfg = cfg.eval.docker

    # Optionally pull the image before starting.
    image = _resolve_image(cfg)
    if docker_cfg.pull:
        logger.info("Pulling Docker image: %s", image)
        subprocess.run(["docker", "pull", image], check=True)  # nosec B603 B607

    # ── Load policy + all preprocessors on the host GPU ──────────────────
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor_overrides: dict = {
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env,
        policy_cfg=cfg.policy,
    )

    # ── Start HTTP inference server(s) ────────────────────────────────────
    n_policy_servers = cfg.eval.policy_servers
    base_port = docker_cfg.port
    host_ip = _get_host_ip()
    instance_count = cfg.eval.instance_count
    env_argv = _env_argv()

    servers: list[_InferenceServer] = []
    for s_idx in range(n_policy_servers):
        port = base_port + s_idx
        if s_idx > 0:
            policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
            policy.eval()
        srv = _InferenceServer(
            ("0.0.0.0", port),  # nosec B104
            policy=policy,
            env_preprocessor=env_preprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        servers.append(srv)
        logger.info("Policy inference server %d/%d running on port %d", s_idx + 1, n_policy_servers, port)

    # ── Spawn containers (round-robin across policy servers) ──────────────
    container_dirs: list[Path] = []
    procs: list[subprocess.Popen] = []
    try:
        for i in range(instance_count):
            assigned_port = base_port + (i % n_policy_servers)
            server_address = f"{host_ip}:{assigned_port}"
            shard_dir = output_dir / "shards" / str(i)
            container_dirs.append(shard_dir)
            proc = _spawn_container(
                image=image,
                instance_id=i,
                instance_count=instance_count,
                server_address=server_address,
                n_episodes=cfg.eval.n_episodes,
                seed=cfg.seed,
                output_dir=shard_dir,
                docker_cfg=docker_cfg,
                env_argv=env_argv,
            )
            procs.append(proc)

        failed: list[tuple[int, int]] = []
        for i, proc in enumerate(procs):
            rc = proc.wait()
            if rc != 0:
                failed.append((i, rc))
                logger.error("Container %d/%d exited with code %d", i + 1, instance_count, rc)
        if failed:
            raise RuntimeError(f"Docker eval containers failed (instance_id, exit_code): {failed}")

    finally:
        for srv in servers:
            srv.shutdown()

    # ── Collect and merge per-task results ───────────────────────────────
    per_task: list[dict] = []
    for i, shard_dir in enumerate(container_dirs):
        result_file = shard_dir / f"worker_{i}.json"
        with open(result_file) as f:
            shard_data: dict = json.load(f)
        per_task.extend(shard_data.get("per_task", []))

    per_task.sort(key=lambda x: (x["task_group"], x["task_id"]))

    info = _aggregate_eval_from_per_task(per_task, total_eval_s=time.time() - start_t)
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info("Docker eval complete. Results: %s/eval_info.json", output_dir)
