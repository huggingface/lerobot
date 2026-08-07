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

"""Tests for the interactive rollout session (--interactive=true)."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import time
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout import (  # noqa: E402
    InteractiveCommand,
    InteractiveSession,
    LinkedEvent,
    StdinCommandListener,
    parse_command,
)


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@contextlib.contextmanager
def _pipe_stream():
    """A held-open pipe so the session's stdin listener never sees EOF."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r")
    writer = os.fdopen(write_fd, "w")
    try:
        yield reader, writer
    finally:
        with contextlib.suppress(OSError, ValueError):
            writer.close()
        with contextlib.suppress(OSError, ValueError):
            reader.close()


# ---------------------------------------------------------------------------
# Command parser
# ---------------------------------------------------------------------------


def test_parse_command_basic():
    cmd = parse_command("/start")
    assert cmd is not None
    assert cmd.name == "start"
    assert cmd.args == ""


def test_parse_command_case_whitespace_and_args():
    cmd = parse_command("  /SubTask Grab the red cube  ")
    assert cmd is not None
    assert cmd.name == "subtask"
    assert cmd.args == "Grab the red cube"


def test_parse_command_tab_separated():
    assert parse_command("/subtask\tgrab the cube") == InteractiveCommand(
        name="subtask", args="grab the cube"
    )


def test_parse_command_non_commands():
    assert parse_command("hello robot") is None
    assert parse_command("") is None
    assert parse_command("   ") is None
    assert parse_command("/") is None
    assert parse_command("/ start") is None


# ---------------------------------------------------------------------------
# LinkedEvent
# ---------------------------------------------------------------------------


def test_linked_event_local_flag():
    parent = Event()
    event = LinkedEvent(parent)
    assert not event.is_set()

    event.set()
    assert event.is_set()
    assert not parent.is_set()

    event.clear()
    assert not event.is_set()


def test_linked_event_reflects_parent():
    parent = Event()
    event = LinkedEvent(parent)
    parent.set()
    assert event.is_set()
    # Clearing the local flag never masks the parent.
    event.clear()
    assert event.is_set()


def test_linked_event_wait():
    parent = Event()
    event = LinkedEvent(parent)
    assert event.wait(timeout=0.05) is False

    parent.set()
    assert event.wait(timeout=0.05) is True

    parent.clear()
    event.set()
    assert event.wait(timeout=0.05) is True


def test_linked_event_wait_wakes_on_parent_set():
    parent = Event()
    event = LinkedEvent(parent)
    Thread(target=lambda: (time.sleep(0.05), parent.set()), daemon=True).start()
    assert event.wait(timeout=2.0) is True


# ---------------------------------------------------------------------------
# StdinCommandListener
# ---------------------------------------------------------------------------


def test_stdin_listener_reads_lines_and_eof():
    lines: list[str] = []
    eof = Event()

    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(lines.append, on_eof=eof.set, stream=reader)
        listener.start()
        writer.write("/start\n")
        writer.write("   \n")  # blank lines are skipped
        writer.write("/help\n")
        writer.flush()
        assert _wait_for(lambda: len(lines) == 2)
        assert lines == ["/start", "/help"]

        writer.close()
        assert _wait_for(eof.is_set)
        listener.stop()


def test_stdin_listener_handler_errors_do_not_kill_reader():
    lines: list[str] = []

    def flaky(line: str) -> None:
        if line == "/boom":
            raise RuntimeError("boom")
        lines.append(line)

    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(flaky, stream=reader)
        listener.start()
        writer.write("/boom\n/start\n")
        writer.flush()
        assert _wait_for(lambda: lines == ["/start"])
        listener.stop()


def test_stdin_listener_blocking_fallback():
    """Streams without a file descriptor (e.g. StringIO) use the blocking readline path."""
    lines: list[str] = []
    eof = Event()
    listener = StdinCommandListener(lines.append, on_eof=eof.set, stream=io.StringIO("/start\n\n/help\n"))
    assert not listener._use_select
    listener.start()
    assert _wait_for(eof.is_set)
    assert lines == ["/start", "/help"]
    listener.stop()


# ---------------------------------------------------------------------------
# InteractiveSession
# ---------------------------------------------------------------------------


def _make_session(input_stream, run_behavior=None):
    """Build a session around a mock strategy and a minimal fake context."""
    parent = Event()
    stop_event = LinkedEvent(parent)
    engine = MagicMock()
    engine.failed = False
    engine.failure_traceback = None
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(play_sounds=False),
            shutdown_event=stop_event,
        ),
        policy=SimpleNamespace(inference=engine),
        hardware=SimpleNamespace(initial_position={"joint.pos": 0.0}),
    )

    strategy = MagicMock()
    run_started = Event()

    def default_run(c):
        run_started.set()
        while not c.runtime.shutdown_event.is_set():
            time.sleep(0.005)

    strategy.run.side_effect = run_behavior or default_run

    session = InteractiveSession(strategy, ctx, input_stream=input_stream)
    return session, strategy, engine, parent, run_started


def _start_session_thread(session) -> Thread:
    thread = Thread(target=session.run, daemon=True)
    thread.start()
    return thread


def test_session_requires_linked_event():
    ctx = SimpleNamespace(runtime=SimpleNamespace(shutdown_event=Event()))
    with pytest.raises(TypeError, match="LinkedEvent"):
        InteractiveSession(MagicMock(), ctx)


def test_session_start_reset_restart_stop_flow():
    with _pipe_stream() as (reader, _writer):
        session, strategy, engine, _parent, run_started = _make_session(reader)
        thread = _start_session_thread(session)

        # Idle until /start: the strategy loop must not run on its own.
        time.sleep(0.05)
        strategy.run.assert_not_called()

        session._handle_line("/start")
        assert _wait_for(run_started.is_set)
        assert strategy.reset_control_state.call_count == 1

        # /reset ends the segment, pauses the engine, and returns to the initial position.
        session._handle_line("/reset")
        assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)
        assert engine.pause.call_count >= 1
        assert thread.is_alive()

        # /start again runs a fresh segment with freshly reset control state.
        run_started.clear()
        session._handle_line("/start")
        assert _wait_for(run_started.is_set)
        assert strategy.run.call_count == 2
        assert strategy.reset_control_state.call_count == 2

        # /stop ends the session; teardown stays with the caller (the CLI script).
        session._handle_line("/stop")
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        strategy.teardown.assert_not_called()


def test_session_stop_while_idle():
    with _pipe_stream() as (reader, _writer):
        session, strategy, _engine, _parent, _run_started = _make_session(reader)
        thread = _start_session_thread(session)
        session._handle_line("/stop")
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        strategy.run.assert_not_called()


def test_session_reset_while_idle_returns_to_initial_position():
    with _pipe_stream() as (reader, _writer):
        session, strategy, _engine, _parent, _run_started = _make_session(reader)
        thread = _start_session_thread(session)
        session._handle_line("/reset")
        assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)
        assert thread.is_alive()
        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_start_while_running_is_rejected():
    with _pipe_stream() as (reader, _writer):
        session, strategy, _engine, _parent, run_started = _make_session(reader)
        thread = _start_session_thread(session)

        session._handle_line("/start")
        assert _wait_for(run_started.is_set)
        session._handle_line("/start")
        time.sleep(0.05)
        assert strategy.run.call_count == 1

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_reset_cancels_pending_start():
    """Last command wins: a queued /start must not fire after a later /reset."""
    with _pipe_stream() as (reader, _writer):
        session, strategy, _engine, _parent, _run_started = _make_session(reader)
        # Queue both commands before the session loop starts servicing them.
        session._handle_line("/start")
        session._handle_line("/reset")
        thread = _start_session_thread(session)

        assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)
        time.sleep(0.05)
        strategy.run.assert_not_called()

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_stop_cancels_pending_start():
    with _pipe_stream() as (reader, _writer):
        session, strategy, _engine, _parent, _run_started = _make_session(reader)
        session._handle_line("/start")
        session._handle_line("/stop")
        thread = _start_session_thread(session)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        strategy.run.assert_not_called()


def test_session_exits_on_parent_shutdown():
    with _pipe_stream() as (reader, _writer):
        session, _strategy, _engine, parent, run_started = _make_session(reader)
        thread = _start_session_thread(session)
        session._handle_line("/start")
        assert _wait_for(run_started.is_set)

        parent.set()  # SIGINT/SIGTERM path
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def test_session_stops_on_engine_failure(capsys):
    def failing_run(c):
        # Mimic the RTC thread's fatal-error path: flag the failure, capture
        # the traceback, and set the engine's shutdown event (the LinkedEvent).
        c.policy.inference.failed = True
        c.policy.inference.failure_traceback = "RuntimeError: boom-traceback"
        c.runtime.shutdown_event.set()

    with _pipe_stream() as (reader, _writer):
        session, strategy, _engine, _parent, _run_started = _make_session(reader, run_behavior=failing_run)
        thread = _start_session_thread(session)
        session._handle_line("/start")
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        # A failed engine ends the session instead of returning to idle, and
        # the captured traceback is surfaced despite the muted console logs.
        strategy.return_to_initial_position.assert_not_called()
        assert "boom-traceback" in capsys.readouterr().out


def test_session_stops_on_engine_failure_while_idle():
    """A fatal engine error while idle ends the session instead of being masked by /start."""
    with _pipe_stream() as (reader, _writer):
        session, strategy, engine, _parent, _run_started = _make_session(reader)
        thread = _start_session_thread(session)
        time.sleep(0.05)
        engine.failed = True
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        strategy.run.assert_not_called()


def test_session_returns_to_idle_when_run_ends_naturally():
    def finite_run(c):
        return None  # e.g. --duration elapsed

    with _pipe_stream() as (reader, _writer):
        session, strategy, _engine, _parent, _run_started = _make_session(reader, run_behavior=finite_run)
        thread = _start_session_thread(session)

        session._handle_line("/start")
        assert _wait_for(lambda: strategy.run.call_count == 1)
        time.sleep(0.05)
        assert thread.is_alive()  # back to idle, not shut down

        # The session accepts another /start after a natural end.
        session._handle_line("/start")
        assert _wait_for(lambda: strategy.run.call_count == 2)

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_unknown_input_does_not_start(capsys):
    with _pipe_stream() as (reader, _writer):
        session, strategy, _engine, _parent, _run_started = _make_session(reader)
        thread = _start_session_thread(session)

        session._handle_line("/frobnicate")
        session._handle_line("hello robot")
        session._handle_line("/help")
        time.sleep(0.05)
        strategy.run.assert_not_called()

        out = capsys.readouterr().out
        assert "/frobnicate" in out
        assert "commands start with '/'" in out

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_eof_stops_session():
    with _pipe_stream() as (reader, writer):
        session, strategy, _engine, _parent, _run_started = _make_session(reader)
        thread = _start_session_thread(session)
        writer.close()  # EOF on the command stream
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        strategy.run.assert_not_called()


def test_session_commands_via_stream():
    """End-to-end: commands flow through the pipe and the listener thread."""
    with _pipe_stream() as (reader, writer):
        session, strategy, _engine, _parent, run_started = _make_session(reader)
        thread = _start_session_thread(session)

        writer.write("/start\n")
        writer.flush()
        assert _wait_for(run_started.is_set)

        writer.write("/stop\n")
        writer.flush()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert strategy.run.call_count == 1


def test_session_mutes_console_logging_and_restores_on_exit():
    import warnings

    root = logging.getLogger()
    console_handler = logging.StreamHandler(io.StringIO())
    console_handler.setLevel(logging.INFO)
    root.addHandler(console_handler)
    # Libraries like transformers attach their own console handler with
    # propagate=False; those must be muted too.
    lib_logger = logging.getLogger("test_interactive_fake_lib")
    lib_logger.propagate = False
    lib_handler = logging.StreamHandler(io.StringIO())
    lib_handler.setLevel(logging.WARNING)
    lib_logger.addHandler(lib_handler)
    n_warning_filters = len(warnings.filters)
    try:
        with _pipe_stream() as (reader, _writer):
            session, _strategy, _engine, _parent, _run_started = _make_session(reader)
            thread = _start_session_thread(session)
            assert _wait_for(
                lambda: console_handler.level == logging.CRITICAL + 1
                and lib_handler.level == logging.CRITICAL + 1
            )
            session._handle_line("/stop")
            thread.join(timeout=2.0)
        assert console_handler.level == logging.INFO
        assert lib_handler.level == logging.WARNING
        assert len(warnings.filters) == n_warning_filters
    finally:
        root.removeHandler(console_handler)
        lib_logger.removeHandler(lib_handler)


def test_session_does_not_mute_file_log_handlers(tmp_path):
    root = logging.getLogger()
    file_handler = logging.FileHandler(tmp_path / "session.log")
    file_handler.setLevel(logging.INFO)
    root.addHandler(file_handler)
    try:
        with _pipe_stream() as (reader, _writer):
            session, _strategy, _engine, _parent, run_started = _make_session(reader)
            thread = _start_session_thread(session)
            session._handle_line("/start")
            assert _wait_for(run_started.is_set)
            assert file_handler.level == logging.INFO
            session._handle_line("/stop")
            thread.join(timeout=2.0)
    finally:
        root.removeHandler(file_handler)
        file_handler.close()


def test_session_drives_real_base_strategy():
    """End-to-end with a real BaseStrategy control loop (only hardware/engine mocked)."""
    from lerobot.rollout import BaseStrategy, BaseStrategyConfig

    parent = Event()
    stop_event = LinkedEvent(parent)
    engine = MagicMock()
    engine.failed = False
    engine.get_action.return_value = None  # no action ready; the loop still ticks

    robot = MagicMock()
    robot.get_observation.return_value = {"joint.pos": 0.0}

    def identity(x):
        return x

    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                play_sounds=False,
                fps=100.0,
                duration=0.0,
                use_torch_compile=False,
                interpolation_multiplier=1,
                display_data=False,
            ),
            shutdown_event=stop_event,
        ),
        policy=SimpleNamespace(inference=engine),
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=None, initial_position={"joint.pos": 0.0}),
        processors=SimpleNamespace(
            teleop_action_processor=identity,
            robot_action_processor=identity,
            robot_observation_processor=identity,
        ),
        data=SimpleNamespace(dataset=None, dataset_features={}, hw_features={}, ordered_action_keys=[]),
    )

    strategy = BaseStrategy(BaseStrategyConfig())
    strategy.setup(ctx)
    strategy.return_to_initial_position = MagicMock()  # skip the 3s hardware sweep

    with _pipe_stream() as (reader, _writer):
        session = InteractiveSession(strategy, ctx, input_stream=reader)
        thread = _start_session_thread(session)

        session._handle_line("/start")
        assert _wait_for(lambda: engine.resume.called)
        assert _wait_for(lambda: robot.get_observation.call_count >= 3)

        session._handle_line("/reset")
        assert _wait_for(lambda: strategy.return_to_initial_position.called)
        assert engine.pause.called
        assert thread.is_alive()

        session._handle_line("/start")
        assert _wait_for(lambda: engine.resume.call_count >= 2)

        session._handle_line("/stop")
        thread.join(timeout=2.0)
        assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_interactive_requires_base_strategy():
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import RolloutConfig, SentryStrategyConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match="--interactive=true currently supports only"):
        RolloutConfig(
            robot=MockRobotConfig(),
            strategy=SentryStrategyConfig(),
            dataset=DatasetRecordConfig(repo_id="user/rollout_test", single_task="test"),
            interactive=True,
        )
