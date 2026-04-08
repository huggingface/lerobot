"""Tests for the process-isolated environment wrapper.

These tests verify that the ProcessIsolatedVectorEnv proxy correctly forwards
environment operations to a subprocess and that the subprocess runs without
GPU access.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from lerobot.envs.process_isolated import (
    EnvProxy,
    _collect_env_metadata,
    close_process_isolated_envs,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


class _DummyEnv(gym.Env):
    """Minimal gym env for testing without simulation dependencies."""

    metadata = {"render_fps": 10}

    def __init__(self, task_desc: str = "test task"):
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Dict(
                    {
                        "image": gym.spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        self.task_description = task_desc
        self._max_episode_steps = 10
        self._step_count = 0

    def reset(self, seed=None, options=None):
        self._step_count = 0
        obs = {"pixels": {"image": np.zeros((64, 64, 3), dtype=np.uint8)}}
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = {"pixels": {"image": np.ones((64, 64, 3), dtype=np.uint8)}}
        reward = 0.0
        terminated = self._step_count >= self._max_episode_steps
        truncated = False
        info = {}
        if terminated:
            info["is_success"] = True
        return obs, reward, terminated, truncated, info

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)


def _make_dummy_vec_env(n_envs: int = 1) -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv(
        [lambda: _DummyEnv() for _ in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


# ── Unit tests for metadata collection ───────────────────────────────────────


class TestCollectEnvMetadata:
    def test_collects_num_envs(self):
        vec_env = _make_dummy_vec_env(n_envs=2)
        meta = _collect_env_metadata(vec_env)
        assert meta["num_envs"] == 2
        vec_env.close()

    def test_detects_task_description(self):
        vec_env = _make_dummy_vec_env(n_envs=1)
        meta = _collect_env_metadata(vec_env)
        assert meta["has_task_description"] is True
        assert meta["task_descriptions"] == ["test task"]
        vec_env.close()

    def test_collects_max_episode_steps(self):
        vec_env = _make_dummy_vec_env(n_envs=1)
        meta = _collect_env_metadata(vec_env)
        assert meta["max_episode_steps"] == 10
        vec_env.close()

    def test_collects_render_fps(self):
        vec_env = _make_dummy_vec_env(n_envs=1)
        meta = _collect_env_metadata(vec_env)
        assert meta["render_fps"] == 10
        vec_env.close()


# ── Unit tests for EnvProxy ──────────────────────────────────────────────────


class TestEnvProxy:
    """Tests for proxy interface without spawning a real subprocess."""

    def _make_proxy(self, num_envs: int = 1) -> tuple[EnvProxy, mp.connection.Connection]:
        """Create a proxy with a mock connection."""
        import threading

        parent_conn, child_conn = mp.Pipe()
        metadata = {
            "num_envs": num_envs,
            "has_task_description": True,
            "has_task": False,
            "task_descriptions": [f"task_{i}" for i in range(num_envs)],
            "max_episode_steps": 10,
            "render_fps": 30,
        }
        proxy = EnvProxy(
            conn=parent_conn,
            lock=threading.Lock(),
            suite_name="test_suite",
            task_id=0,
            metadata=metadata,
            process=MagicMock(),
        )
        return proxy, child_conn

    def test_num_envs(self):
        proxy, child = self._make_proxy(num_envs=3)
        assert proxy.num_envs == 3
        child.close()

    def test_envs_proxy_len(self):
        proxy, child = self._make_proxy(num_envs=2)
        assert len(proxy.envs) == 2
        child.close()

    def test_envs_proxy_task_description(self):
        proxy, child = self._make_proxy(num_envs=2)
        assert proxy.envs[0].task_description == "task_0"
        assert proxy.envs[1].task_description == "task_1"
        child.close()

    def test_unwrapped_metadata(self):
        proxy, child = self._make_proxy()
        assert proxy.unwrapped.metadata["render_fps"] == 30
        child.close()

    def test_call_cached_max_episode_steps(self):
        """call('_max_episode_steps') should use cached value without IPC."""
        proxy, child = self._make_proxy()
        result = proxy.call("_max_episode_steps")
        assert result == [10]
        # child_conn should have nothing to receive (no IPC happened).
        assert not child.poll(timeout=0.01)
        child.close()

    def test_call_cached_task_description(self):
        proxy, child = self._make_proxy(num_envs=2)
        result = proxy.call("task_description")
        assert result == ["task_0", "task_1"]
        assert not child.poll(timeout=0.01)
        child.close()


# ── Integration test with real subprocess ────────────────────────────────────


class TestSubprocessIntegration:
    """Tests that exercise the full subprocess lifecycle."""

    @pytest.fixture()
    def _server(self):
        """Start a subprocess server with a dummy env and yield the connection."""
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()

        # We can't use _DummyEnv directly because spawn context requires picklable targets.
        # Instead, test with a registered gym env.
        process = ctx.Process(
            target=_subprocess_with_cartpole,
            args=(child_conn,),
            daemon=True,
        )
        process.start()
        child_conn.close()

        msg = parent_conn.recv()
        assert msg[0] == "ready", f"Server failed: {msg}"

        yield parent_conn, process, msg[1]

        try:
            parent_conn.send(("close_all", {}))
            parent_conn.recv()
        except (EOFError, BrokenPipeError):
            pass
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()

    def test_server_reports_metadata(self, _server):
        conn, process, metadata = _server
        assert "CartPole-v1" in metadata
        cart_meta = metadata["CartPole-v1"]["0"]
        assert cart_meta["num_envs"] == 1
        assert cart_meta["render_fps"] > 0

    def test_reset_returns_observation(self, _server):
        conn, process, metadata = _server
        conn.send(("reset", {"suite": "CartPole-v1", "task_id": 0}))
        status, (obs, info) = conn.recv()
        assert status == "ok"
        assert isinstance(obs, np.ndarray)

    def test_step_returns_tuple(self, _server):
        conn, process, metadata = _server
        # Reset first.
        conn.send(("reset", {"suite": "CartPole-v1", "task_id": 0}))
        conn.recv()
        # CartPole has Discrete(2) action space; VectorEnv expects shape (n_envs,).
        action = np.array([0])
        conn.send(("step", {"suite": "CartPole-v1", "task_id": 0, "action": action}))
        status, result = conn.recv()
        assert status == "ok", f"Subprocess error: {result}"
        assert isinstance(result, tuple)
        # VectorEnv.step returns (obs, reward, terminated, truncated, info).
        assert len(result) == 5


def _subprocess_with_cartpole(conn):
    """Subprocess target that creates a CartPole env for testing."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    vec_env = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1")],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    # Import inside subprocess since this runs in a spawn context.
    from lerobot.envs.process_isolated import (
        _collect_env_metadata as collect_meta,
        _serve_requests as serve,
    )

    metadata = {"CartPole-v1": {"0": collect_meta(vec_env)}}
    conn.send(("ready", metadata))

    try:
        serve(conn, {"CartPole-v1": {0: vec_env}})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        vec_env.close()
        conn.close()


# ── Test for close_process_isolated_envs ─────────────────────────────────────


class TestCloseProcessIsolatedEnvs:
    def test_closes_gracefully(self):
        import threading

        parent_conn, child_conn = mp.Pipe()
        mock_process = MagicMock()
        # Process is still alive, so close should call join.
        mock_process.is_alive.return_value = True

        proxy = EnvProxy(
            conn=parent_conn,
            lock=threading.Lock(),
            suite_name="test",
            task_id=0,
            metadata={
                "num_envs": 1,
                "has_task_description": False,
                "has_task": False,
                "task_descriptions": [],
                "max_episode_steps": 10,
                "render_fps": 30,
            },
            process=mock_process,
        )

        # Simulate the server responding to close_all.
        import threading

        def _respond():
            cmd, _ = child_conn.recv()
            assert cmd == "close_all"
            child_conn.send(("ok", None))

        t = threading.Thread(target=_respond)
        t.start()

        close_process_isolated_envs({"test": {0: proxy}})
        t.join(timeout=2)
        mock_process.join.assert_called_once()
