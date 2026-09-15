"""Tests for the process-isolated environment wrapper.

These tests verify that the ProcessIsolatedVectorEnv proxy correctly forwards
environment operations to a subprocess and that the subprocess runs without
GPU access.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from lerobot.envs import process_isolated
from lerobot.envs.process_isolated import (
    EnvProxy,
    _collect_env_metadata,
    _env_server_main,
    _serve_requests,
    close_process_isolated_envs,
    make_env_in_subprocess,
)
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, check_env_attributes_and_types

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


def test_environment_server_process_is_not_daemonic(monkeypatch):
    context = MagicMock()
    parent_conn = MagicMock()
    child_conn = MagicMock()
    process = MagicMock()
    context.Pipe.return_value = (parent_conn, child_conn)
    context.Process.return_value = process
    parent_conn.poll.return_value = True
    parent_conn.recv.return_value = (
        "ready",
        {"test_suite": {"0": {"num_envs": 1, "render_fps": 30}}},
    )
    monkeypatch.setattr(mp, "get_context", MagicMock(return_value=context))

    envs = make_env_in_subprocess(MagicMock(), use_async_envs=True)

    assert list(envs["test_suite"]) == [0]
    assert context.Process.call_args.kwargs.get("daemon", False) is False


def test_empty_metadata_terminates_environment_server(monkeypatch):
    context = MagicMock()
    parent_conn = MagicMock()
    child_conn = MagicMock()
    process = MagicMock()
    context.Pipe.return_value = (parent_conn, child_conn)
    context.Process.return_value = process
    parent_conn.poll.return_value = True
    parent_conn.recv.return_value = ("ready", {})
    process.is_alive.return_value = True
    process.terminate.side_effect = lambda: process.is_alive.configure_mock(return_value=False)
    monkeypatch.setattr(mp, "get_context", MagicMock(return_value=context))

    with pytest.raises(RuntimeError, match="returned no environments"):
        make_env_in_subprocess(MagicMock())

    parent_conn.close.assert_called_once_with()
    process.terminate.assert_called_once_with()
    process.join.assert_called_once_with(timeout=5)


def test_startup_timeout_terminates_environment_server(monkeypatch):
    context = MagicMock()
    parent_conn = MagicMock()
    child_conn = MagicMock()
    process = MagicMock()
    context.Pipe.return_value = (parent_conn, child_conn)
    context.Process.return_value = process
    parent_conn.poll.return_value = False
    process.exitcode = None
    process.is_alive.return_value = True
    process.terminate.side_effect = lambda: process.is_alive.configure_mock(return_value=False)
    monkeypatch.setattr(mp, "get_context", MagicMock(return_value=context))

    with pytest.raises(RuntimeError, match="did not respond"):
        make_env_in_subprocess(MagicMock(), use_async_envs=True)

    parent_conn.close.assert_called_once_with()
    process.terminate.assert_called_once_with()
    process.join.assert_called_once_with(timeout=5)


def test_startup_interrupt_terminates_environment_server(monkeypatch):
    context = MagicMock()
    parent_conn = MagicMock()
    child_conn = MagicMock()
    process = MagicMock()
    context.Pipe.return_value = (parent_conn, child_conn)
    context.Process.return_value = process
    parent_conn.poll.side_effect = KeyboardInterrupt
    process.is_alive.return_value = True
    process.terminate.side_effect = lambda: process.is_alive.configure_mock(return_value=False)
    monkeypatch.setattr(mp, "get_context", MagicMock(return_value=context))

    with pytest.raises(KeyboardInterrupt):
        make_env_in_subprocess(MagicMock())

    parent_conn.close.assert_called_once_with()
    process.terminate.assert_called_once_with()
    process.join.assert_called_once_with(timeout=5)


def test_startup_recv_interrupt_terminates_environment_server(monkeypatch):
    context = MagicMock()
    parent_conn = MagicMock()
    child_conn = MagicMock()
    process = MagicMock()
    context.Pipe.return_value = (parent_conn, child_conn)
    context.Process.return_value = process
    parent_conn.poll.return_value = True
    parent_conn.recv.side_effect = KeyboardInterrupt
    process.is_alive.return_value = True
    process.terminate.side_effect = lambda: process.is_alive.configure_mock(return_value=False)
    monkeypatch.setattr(mp, "get_context", MagicMock(return_value=context))

    with pytest.raises(KeyboardInterrupt):
        make_env_in_subprocess(MagicMock())

    parent_conn.close.assert_called_once_with()
    process.terminate.assert_called_once_with()
    process.join.assert_called_once_with(timeout=5)


def test_atexit_registration_interrupt_terminates_environment_server(monkeypatch):
    context = MagicMock()
    parent_conn = MagicMock()
    child_conn = MagicMock()
    process = MagicMock()
    context.Pipe.return_value = (parent_conn, child_conn)
    context.Process.return_value = process
    parent_conn.poll.return_value = True
    parent_conn.recv.return_value = (
        "ready",
        {"test_suite": {"0": {"num_envs": 1, "render_fps": 30}}},
    )
    process.is_alive.return_value = True
    process.terminate.side_effect = lambda: process.is_alive.configure_mock(return_value=False)
    monkeypatch.setattr(mp, "get_context", MagicMock(return_value=context))
    monkeypatch.setattr(process_isolated.atexit, "register", MagicMock(side_effect=KeyboardInterrupt))

    with pytest.raises(KeyboardInterrupt):
        make_env_in_subprocess(MagicMock())

    parent_conn.close.assert_called_once_with()
    process.terminate.assert_called_once_with()
    process.join.assert_called_once_with(timeout=5)


def test_process_start_failure_closes_both_pipe_ends(monkeypatch):
    context = MagicMock()
    parent_conn = MagicMock()
    child_conn = MagicMock()
    process = MagicMock()
    context.Pipe.return_value = (parent_conn, child_conn)
    context.Process.return_value = process
    process.start.side_effect = RuntimeError("spawn failed")
    monkeypatch.setattr(mp, "get_context", MagicMock(return_value=context))

    with pytest.raises(RuntimeError, match="spawn failed"):
        make_env_in_subprocess(MagicMock())

    parent_conn.close.assert_called_once_with()
    child_conn.close.assert_called_once_with()


class TestCollectEnvMetadata:
    def test_collects_num_envs(self):
        vec_env = _make_dummy_vec_env(n_envs=2)
        meta = _collect_env_metadata(vec_env)
        assert meta["num_envs"] == 2
        vec_env.close()

    def test_does_not_start_vector_env_workers(self):
        vec_env = MagicMock()
        vec_env.num_envs = 2
        vec_env.unwrapped = SimpleNamespace(metadata={"render_fps": 20})

        metadata = _collect_env_metadata(vec_env)

        assert metadata == {"num_envs": 2, "render_fps": 20}
        vec_env.call.assert_not_called()

    def test_collects_render_fps(self):
        vec_env = _make_dummy_vec_env(n_envs=1)
        meta = _collect_env_metadata(vec_env)
        assert meta["render_fps"] == 10
        vec_env.close()


def test_env_server_main_reports_ready_serves_requests_and_closes(monkeypatch):
    vec_env = _make_dummy_vec_env()
    make_env = MagicMock(return_value={"test_suite": {0: vec_env}})
    monkeypatch.setattr("lerobot.envs.factory.make_env", make_env)
    parent_conn, child_conn = mp.Pipe()
    server = threading.Thread(
        target=_env_server_main,
        args=(child_conn, MagicMock(), 1, False, False),
    )
    server.start()

    try:
        status, metadata = parent_conn.recv()
        assert status == "ready"
        assert metadata == {"test_suite": {"0": {"num_envs": 1, "render_fps": 10}}}

        parent_conn.send(("reset", {"suite": "test_suite", "task_id": 0}))
        assert parent_conn.recv()[0] == "ok"
        parent_conn.send(("close_all", {}))
        assert parent_conn.recv() == ("ok", None)
    finally:
        parent_conn.close()
        server.join(timeout=2)

    assert not server.is_alive()
    make_env.assert_called_once()


def test_env_server_main_reports_setup_error(monkeypatch):
    monkeypatch.setattr(
        "lerobot.envs.factory.make_env",
        MagicMock(side_effect=RuntimeError("setup failed")),
    )
    parent_conn, child_conn = mp.Pipe()

    _env_server_main(child_conn, MagicMock(), 1, False, False)

    status, error = parent_conn.recv()
    parent_conn.close()
    assert status == "error"
    assert "setup failed" in error


# ── Unit tests for EnvProxy ──────────────────────────────────────────────────


class TestEnvProxy:
    """Tests for proxy interface without spawning a real subprocess."""

    def _make_proxy(
        self,
        num_envs: int = 1,
        **metadata_overrides,
    ) -> tuple[EnvProxy, mp.connection.Connection]:
        """Create a proxy with a mock connection."""
        parent_conn, child_conn = mp.Pipe()
        metadata = {
            "num_envs": num_envs,
            "has_task_description": True,
            "has_task": False,
            "task_descriptions": [f"task_{i}" for i in range(num_envs)],
            "max_episode_steps": 10,
            "render_fps": 30,
        } | metadata_overrides
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

    def test_envs_proxy_reads_live_task_description(self):
        conn = MagicMock()
        conn.recv.side_effect = [("ok", ["first"]), ("ok", ["second"])]
        proxy = EnvProxy(
            conn=conn,
            lock=threading.Lock(),
            suite_name="test_suite",
            task_id=0,
            metadata={"num_envs": 1, "render_fps": 30},
            process=MagicMock(),
        )

        assert proxy.envs[0].task_description == "first"
        assert proxy.envs[0].task_description == "second"
        assert conn.send.call_count == 2

    def test_unwrapped_metadata(self):
        proxy, child = self._make_proxy()
        assert proxy.unwrapped.metadata["render_fps"] == 30
        child.close()

    @pytest.mark.parametrize("method", ["call", "get_attr"])
    def test_proxy_preserves_remote_attribute_error(self, method):
        conn = MagicMock()
        conn.recv.return_value = ("attribute_error", "task_description")
        proxy = EnvProxy(
            conn=conn,
            lock=threading.Lock(),
            suite_name="test_suite",
            task_id=0,
            metadata={"num_envs": 1, "render_fps": 30},
            process=MagicMock(),
        )

        with pytest.raises(AttributeError, match="task_description"):
            getattr(proxy, method)("task_description")

    def test_proxy_passes_current_main_attribute_check(self):
        conn = MagicMock()
        conn.recv.side_effect = [("ok", ["description"]), ("ok", ["task"])]
        proxy = EnvProxy(
            conn=conn,
            lock=threading.Lock(),
            suite_name="test_suite",
            task_id=0,
            metadata={"num_envs": 1, "render_fps": 30},
            process=MagicMock(),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            check_env_attributes_and_types(proxy)

        assert not caught

    def test_reset_sends_options_through_ipc(self):
        conn = MagicMock()
        conn.recv.return_value = ("ok", ({"observation": np.array([1])}, {}))
        proxy = EnvProxy(
            conn=conn,
            lock=threading.Lock(),
            suite_name="test_suite",
            task_id=3,
            metadata={
                "num_envs": 1,
                "has_task_description": False,
                "has_task": False,
                "task_descriptions": [],
                "max_episode_steps": 10,
                "render_fps": 30,
            },
            process=MagicMock(),
        )
        options = {NEW_ROLLOUT_OPTION: True}

        proxy.reset(seed=range(4, 5), options=options)

        conn.send.assert_called_once_with(
            (
                "reset",
                {
                    "suite": "test_suite",
                    "task_id": 3,
                    "seed": [4],
                    "options": options,
                },
            )
        )


def test_server_forwards_reset_options_to_vector_env():
    parent_conn, child_conn = mp.Pipe()
    vec_env = MagicMock()
    vec_env.reset.return_value = ({"observation": np.array([1])}, {})
    server = threading.Thread(
        target=_serve_requests,
        args=(child_conn, {"test_suite": {3: vec_env}}),
        daemon=True,
    )
    server.start()
    options = {NEW_ROLLOUT_OPTION: True}

    parent_conn.send(
        (
            "reset",
            {
                "suite": "test_suite",
                "task_id": 3,
                "seed": [4],
                "options": options,
            },
        )
    )
    assert parent_conn.recv()[0] == "ok"
    parent_conn.send(("close_all", {}))
    assert parent_conn.recv() == ("ok", None)
    server.join(timeout=2)

    vec_env.reset.assert_called_once_with(seed=[4], options=options)
    parent_conn.close()
    child_conn.close()


def test_server_forwards_call_and_get_attr():
    parent_conn, child_conn = mp.Pipe()
    vec_env = MagicMock()
    vec_env.call.return_value = ("called",)
    vec_env.get_attr.return_value = ("attribute",)
    server = threading.Thread(
        target=_serve_requests,
        args=(child_conn, {"test_suite": {3: vec_env}}),
        daemon=True,
    )
    server.start()

    parent_conn.send(
        (
            "call",
            {
                "suite": "test_suite",
                "task_id": 3,
                "method": "method_name",
                "args": (1,),
                "kwargs": {"enabled": True},
            },
        )
    )
    assert parent_conn.recv() == ("ok", ["called"])
    parent_conn.send(("get_attr", {"suite": "test_suite", "task_id": 3, "name": "attribute_name"}))
    assert parent_conn.recv() == ("ok", ["attribute"])
    parent_conn.send(("close_all", {}))
    assert parent_conn.recv() == ("ok", None)
    server.join(timeout=2)

    vec_env.call.assert_called_once_with("method_name", 1, enabled=True)
    vec_env.get_attr.assert_called_once_with("attribute_name")
    assert not server.is_alive()
    parent_conn.close()
    child_conn.close()


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

    def test_non_daemonic_server_can_run_async_vector_env(self):
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(target=_subprocess_with_async_cartpole, args=(child_conn,))
        process.start()
        child_conn.close()

        try:
            assert parent_conn.poll(timeout=30)
            status, result = parent_conn.recv()
            process.join(timeout=10)

            assert status == "ok", result
            assert result == 2
            assert process.exitcode == 0
        finally:
            parent_conn.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


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


def _make_cartpole():
    return gym.make("CartPole-v1")


def _subprocess_with_async_cartpole(conn):
    vec_env = None
    try:
        vec_env = gym.vector.AsyncVectorEnv([_make_cartpole, _make_cartpole], context="spawn")
        observation, _ = vec_env.reset(seed=[1, 2])
        conn.send(("ok", len(observation)))
    except Exception:
        import traceback

        conn.send(("error", traceback.format_exc()))
    finally:
        if vec_env is not None:
            vec_env.close()
        conn.close()


# ── Test for close_process_isolated_envs ─────────────────────────────────────


class TestCloseProcessIsolatedEnvs:
    def test_closes_gracefully(self):
        parent_conn, child_conn = mp.Pipe()
        mock_process = MagicMock()
        # Process is still alive, so close should call join.
        mock_process.is_alive.return_value = True
        mock_process.terminate.side_effect = lambda: mock_process.is_alive.configure_mock(return_value=False)

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
        def _respond():
            cmd, _ = child_conn.recv()
            assert cmd == "close_all"
            child_conn.send(("ok", None))

        t = threading.Thread(target=_respond)
        t.start()

        close_process_isolated_envs({"test": {0: proxy}})
        t.join(timeout=2)
        assert not t.is_alive()
        mock_process.terminate.assert_called_once_with()
        assert mock_process.join.call_count == 2

    def test_close_timeout_terminates_and_reaps_server(self):
        conn = MagicMock()
        conn.poll.return_value = False
        process = MagicMock()
        process.is_alive.return_value = True
        process.terminate.side_effect = lambda: process.is_alive.configure_mock(return_value=False)
        proxy = EnvProxy(
            conn=conn,
            lock=threading.Lock(),
            suite_name="test",
            task_id=0,
            metadata={"num_envs": 1, "render_fps": 30},
            process=process,
        )

        close_process_isolated_envs({"test": {0: proxy}})

        conn.poll.assert_called_once_with(timeout=10)
        conn.recv.assert_not_called()
        conn.close.assert_called_once_with()
        process.terminate.assert_called_once_with()
        assert process.join.call_count == 2

    def test_close_returns_when_shared_ipc_lock_is_busy(self, monkeypatch):
        conn = MagicMock()
        lock = threading.Lock()
        lock.acquire()
        process = MagicMock()
        proxy = EnvProxy(
            conn=conn,
            lock=lock,
            suite_name="test",
            task_id=0,
            metadata={"num_envs": 1, "render_fps": 30},
            process=process,
        )
        monkeypatch.setattr(process_isolated, "_SUBPROCESS_SHUTDOWN_TIMEOUT_S", 0.01)

        try:
            proxy._close_server_connection()
        finally:
            lock.release()

        conn.send.assert_not_called()
        conn.close.assert_called_once_with()
