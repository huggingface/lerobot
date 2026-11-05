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

"""SO-101 Bench: SO-101 pick-and-place in Isaac Sim 5.1 / Isaac Lab 2.3.

SO-101 Bench (https://github.com/5hadytru/so101_bench) runs on Isaac Sim 5.1, which needs Python 3.11, while
LeRobot needs Python >= 3.12. The simulator therefore runs in its own process: this integration starts the
SO-101 Bench checkout's ``integrations/lerobot/lerobot_env_so101_bench/sim_server.py`` under ``isaaclab.sh``
and talks to it over a local ZMQ socket (msgpack). Isaac Sim allows one app per process and takes about a
minute to start, so every episode of a process shares one simulator, reconfigured per episode.

Every task is one fixed episode (objects, layout, instruction, appearance). Suites are the benchmark's
task-family x object-split cells (``seen/bin_1``, ..., ``unseen_unseen_class/move``) and task ids are trial
ids. Observations and actions are in LeRobot SO-101 follower units (body joints in [-100, 100], gripper in
[0, 100]), with 480x640 ``overhead`` and ``wrist`` cameras.

Install: Isaac Sim 5.1, Isaac Lab 2.3 and an SO-101 Bench checkout (see docs/source/so101_bench.mdx).
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .utils import canonical_observation

__all__ = ["canonical_observation", "conventions", "env_fns"]

logger = logging.getLogger(__name__)

# Relative to the SO-101 Bench checkout.
SERVER_SCRIPT = Path("integrations/lerobot/lerobot_env_so101_bench/sim_server.py")
EPISODE_SETS = {
    "val_set": ("tasks/val_set.jsonl", "tasks/layouts/val_set_layouts.jsonl"),
    "mmrv_eval": ("tasks/mmrv_eval.jsonl", "tasks/layouts/mmrv_eval_layouts.jsonl"),
}
CAMERAS = ("overhead", "wrist")
JOINTS = tuple(
    f"{joint}.pos"
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
)
# LeRobot SO-101 follower units: body joints in [-100, 100], gripper in [0, 100].
ACTION_LOW = np.array([-100, -100, -100, -100, -100, 0], dtype=np.float32)
ACTION_HIGH = np.array([100, 100, 100, 100, 100, 100], dtype=np.float32)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def parse_trial_ids(spec: str | None) -> list[int] | None:
    """``"0-9,20"`` -> ``[0, ..., 9, 20]``."""
    if not spec:
        return None
    ids: list[int] = []
    for part in str(spec).split(","):
        start, _, end = part.strip().partition("-")
        ids += list(range(int(start), int(end or start) + 1))
    return ids


def parse_cameras(cameras: str) -> list[str]:
    """``"overhead,wrist_init"`` -> ``["overhead", "wrist_init"]``, checking each name."""
    names = [c.strip() for c in cameras.split(",") if c.strip()]
    for name in names:
        if name.removesuffix("_init") not in CAMERAS:
            raise ValueError(
                f"Unknown SO-101 Bench camera {name!r}; use overhead, wrist, overhead_init, wrist_init."
            )
    if not names:
        raise ValueError("SO-101 Bench needs at least one camera.")
    return names


def episode_files(root: Path, episode_set: str, layouts: str | None) -> tuple[Path, Path]:
    """Return the tasks and layouts JSONL of a named episode set, or of a tasks JSONL path."""
    if episode_set in EPISODE_SETS:
        tasks, default_layouts = EPISODE_SETS[episode_set]
        return root / tasks, root / (layouts or default_layouts)
    if not layouts:
        raise ValueError("A custom SO-101 Bench tasks JSONL needs --env.layouts=<layouts JSONL>.")
    return Path(episode_set).expanduser().resolve(), Path(layouts).expanduser().resolve()


class SO101BenchSimClient:
    """One Isaac Sim process per LeRobot process, shared by every episode's ``gym.Env``.

    ``lerobot-eval`` builds an env to read the env's features before it loads the policy, and loading a large
    policy can briefly take most of a 32 GB machine's RAM. So the constructor only starts the server long
    enough to read the episode list and stops it again; the simulator proper starts on the first request,
    after the policy is loaded.
    """

    def __init__(
        self,
        *,
        so101_bench_root: Path,
        isaaclab_sh: Path,
        isaac_venv: Path | None,
        episodes_jsonl: Path,
        layouts_jsonl: Path,
        trial_ids: list[int] | None,
        init_frames: bool,
        max_joint_step_deg: float,
        initial_hold_time_s: float,
        seed: int,
        port: int,
        log_path: Path,
        startup_timeout_s: float = 600.0,
        step_timeout_s: float = 300.0,
    ):
        import msgpack  # noqa: F401
        import zmq  # noqa: F401

        server_script = so101_bench_root / SERVER_SCRIPT
        if not server_script.is_file():
            raise FileNotFoundError(
                f"{server_script} not found: set so101_bench_root to an SO-101 Bench checkout."
            )
        if not isaaclab_sh.is_file():
            raise FileNotFoundError(f"{isaaclab_sh} not found: set isaaclab_sh to Isaac Lab's isaaclab.sh.")
        self.port = port or _free_port()
        command = [
            str(isaaclab_sh), "-p", str(server_script), "--headless",
            "--episodes_jsonl", str(episodes_jsonl), "--episode_layouts_jsonl", str(layouts_jsonl),
            "--port", str(self.port), "--seed", str(seed),
            "--max_joint_step_deg", str(max_joint_step_deg), "--initial_hold_time_s", str(initial_hold_time_s),
        ]  # fmt: skip
        if trial_ids:
            command += ["--trial_ids", ",".join(map(str, trial_ids))]
        if init_frames:
            command.append("--init_frames")
        # isaaclab.sh picks its interpreter from the active environment: point it at the Isaac venv
        # and drop LeRobot's (Python >= 3.12) environment.
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
        }
        if isaac_venv is not None:
            env["VIRTUAL_ENV"] = str(isaac_venv)
            env["PATH"] = f"{isaac_venv / 'bin'}:{env.get('PATH', '')}"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._command, self._env, self._cwd = command, env, so101_bench_root
        self._startup_timeout_s, self._step_timeout_s = startup_timeout_s, step_timeout_s
        self.log_path = log_path
        self.process: subprocess.Popen | None = None
        atexit.register(self.close)
        self.info = self._start()
        logger.info(
            "SO-101 Bench: %d episodes; stopping the simulator until the first episode",
            len(self.info["episodes"]),
        )
        self.close()

    def _start(self) -> dict:
        """Start the server and return its ``info`` reply once it answers."""
        import msgpack
        import zmq

        self._log = open(self.log_path, "ab")  # noqa: SIM115
        logger.info("Starting the SO-101 Bench simulator (log: %s)", self.log_path)
        self.process = subprocess.Popen(
            self._command,
            cwd=self._cwd,
            env=self._env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.socket: Any = zmq.Context.instance().socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://127.0.0.1:{self.port}")
        deadline = time.monotonic() + self._startup_timeout_s
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket.send(msgpack.packb({"cmd": "info"}, use_bin_type=True))
        while True:
            try:
                info = msgpack.unpackb(self.socket.recv(), raw=False)
                break
            except zmq.Again:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"SO-101 Bench simulator exited ({self.process.returncode}); see {self.log_path}"
                    ) from None
                if time.monotonic() > deadline:
                    self.close()
                    raise TimeoutError(
                        f"SO-101 Bench simulator did not start in {self._startup_timeout_s:.0f} s; "
                        f"see {self.log_path}"
                    ) from None
        self.socket.setsockopt(zmq.RCVTIMEO, int(self._step_timeout_s * 1000))
        logger.info("SO-101 Bench simulator ready")
        return info

    def request(self, payload: dict) -> dict:
        import msgpack
        import zmq

        if self.process is None:
            self._start()
        elif self.process.poll() is not None:
            raise RuntimeError(
                f"SO-101 Bench simulator exited ({self.process.returncode}); see {self.log_path}"
            )
        self.socket.send(msgpack.packb(payload, use_bin_type=True))
        try:
            reply = msgpack.unpackb(self.socket.recv(), raw=False)
        except zmq.Again as exc:
            raise TimeoutError(f"SO-101 Bench simulator stopped responding; see {self.log_path}") from exc
        if "error" in reply:
            raise RuntimeError(f"SO-101 Bench simulator: {reply['error']} (see {self.log_path})")
        return reply

    def decode_images(self, reply: dict) -> dict[str, np.ndarray]:
        shapes = {**self.info["cameras"], **{f"{k}_init": v for k, v in self.info["cameras"].items()}}
        return {
            name: np.frombuffer(data, dtype=np.uint8).reshape(*shapes[name], 3)
            for name, data in reply["images"].items()
        }

    def close(self) -> None:
        if self.process is None:
            return
        import msgpack
        import zmq

        if self.process.poll() is None:
            try:
                self.socket.setsockopt(zmq.RCVTIMEO, 10000)
                self.socket.send(msgpack.packb({"cmd": "close"}, use_bin_type=True))
                self.socket.recv()
                self.process.wait(timeout=60)
            except Exception as err:  # best effort; the process group is killed below
                logger.warning("SO-101 Bench simulator did not close cleanly (%s); killing it", err)
        if self.process.poll() is None:
            os.killpg(self.process.pid, 15)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, 9)
        self.socket.close()
        self._log.close()
        self.process = None


class SO101BenchEnv(gym.Env):
    """One fixed SO-101 Bench episode (objects, layout, instruction, appearance)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, client: SO101BenchSimClient, episode: dict, cameras: list[str]):
        super().__init__()
        self.client, self.episode, self.cameras = client, episode, cameras
        self.trial_id = int(episode["trial_id"])
        self.task = f"{episode['split']}/{episode['family']}"
        self.task_description = episode["instruction"]
        self._max_episode_steps = int(episode["max_steps"])
        self.metadata = {**self.metadata, "render_fps": client.info["fps"]}
        shapes = client.info["cameras"]
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        name: spaces.Box(0, 255, (*shapes[name.removesuffix("_init")], 3), dtype=np.uint8)
                        for name in cameras
                    }
                ),
                "agent_pos": spaces.Box(-np.inf, np.inf, (len(JOINTS),), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(ACTION_LOW, ACTION_HIGH, dtype=np.float32)
        self._last_frame: np.ndarray | None = None

    def _format(self, reply: dict) -> dict:
        images = self.client.decode_images(reply)
        self._last_frame = images[self.cameras[0]]
        return {
            "pixels": {name: images[name] for name in self.cameras},
            "agent_pos": np.asarray(reply["state"], dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        reply = self.client.request({"cmd": "reset", "trial_id": self.trial_id})
        return self._format(reply), {"is_success": False}

    def step(self, action: np.ndarray):
        reply = self.client.request({"cmd": "step", "action": np.asarray(action, dtype=np.float32).tolist()})
        info = {"is_success": bool(reply["is_success"]), "end_reason": reply["end_reason"]}
        return (
            self._format(reply),
            float(reply["reward"]),
            bool(reply["terminated"]),
            bool(reply["truncated"]),
            info,
        )

    def render(self) -> np.ndarray | None:
        return self._last_frame

    def close(self) -> None:
        # Each task's env is closed after its episodes; the shared simulator stays up until the process
        # exits (the client registers its own shutdown).
        pass


# One simulator per process and per settings, shared by every task's envs.
_CLIENTS: dict[str, SO101BenchSimClient] = {}


def _client(**settings: Any) -> SO101BenchSimClient:
    key = repr(sorted(settings.items()))
    if key not in _CLIENTS:
        _CLIENTS[key] = SO101BenchSimClient(**settings)
    return _CLIENTS[key]


def env_fns(
    task: str = "val_set",
    n_envs: int = 1,
    *,
    so101_bench_root: str | None = None,
    isaaclab_sh: str | None = None,
    isaac_venv: str | None = None,
    layouts: str | None = None,
    episodes: str | None = None,
    cameras: str = "overhead,wrist",
    seed: int = 0,
    port: int = 0,
    max_joint_step_deg: float = 15.0,
    initial_hold_time_s: float = 1.5,
    sim_log: str | None = None,
) -> dict[str, dict[int, list[Callable[[], SO101BenchEnv]]]]:
    """Return one env factory per episode, as ``{"<split>/<family>": {trial_id: [factory]}}``.

    `task` is an episode set of the SO-101 Bench checkout (``val_set``, ``mmrv_eval``) or a tasks JSONL path,
    with `layouts` then naming its layouts JSONL. `episodes` selects trial ids, e.g. ``"0-9,20"``.
    """
    if n_envs != 1:
        raise ValueError(
            "SO-101 Bench runs its episodes one at a time in a single simulator; use --eval.batch_size=1."
        )
    if not so101_bench_root or not isaaclab_sh:
        raise ValueError(
            "SO-101 Bench needs an SO-101 Bench checkout and Isaac Lab: set --env.so101_bench_root and "
            "--env.isaaclab_sh (or SO101_BENCH_ROOT / ISAACLAB_SH); see docs/source/so101_bench.mdx."
        )
    camera_list = parse_cameras(cameras)
    root = Path(so101_bench_root).expanduser()
    tasks_jsonl, layouts_jsonl = episode_files(root, task, layouts)
    client = _client(
        so101_bench_root=root,
        isaaclab_sh=Path(isaaclab_sh).expanduser(),
        isaac_venv=Path(isaac_venv).expanduser() if isaac_venv else None,
        episodes_jsonl=tasks_jsonl,
        layouts_jsonl=layouts_jsonl,
        trial_ids=parse_trial_ids(episodes),
        init_frames=any(c.endswith("_init") for c in camera_list),
        max_joint_step_deg=max_joint_step_deg,
        initial_hold_time_s=initial_hold_time_s,
        seed=seed,
        port=port,
        log_path=Path(sim_log) if sim_log else root / "outputs/lerobot_sim_server.log",
    )
    out: dict[str, dict[int, list[Callable[[], SO101BenchEnv]]]] = defaultdict(dict)
    for episode in client.info["episodes"]:
        suite = f"{episode['split']}/{episode['family']}"
        out[suite][int(episode["trial_id"])] = [partial(SO101BenchEnv, client, episode, camera_list)]
    return dict(out)


def conventions(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return SO-101 Bench's fps, control mode, joint names and robot type."""
    return {
        "fps": 30,
        "control_mode": "position",
        "state_names": JOINTS,
        "action_names": JOINTS,
        "robot_type": "so101_follower",
    }
