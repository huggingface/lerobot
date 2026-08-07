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
from threading import Event, Thread, current_thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs import TextKind  # noqa: E402
from lerobot.rollout import (  # noqa: E402
    InferenceEngine,
    InteractiveCommand,
    InteractiveSession,
    LinkedEvent,
    StdinCommandListener,
    TextQueryRequest,
    TextQueryWorker,
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


def test_text_query_worker_stop_is_bounded_and_suppresses_callback():
    query_started = Event()
    release_query = Event()
    answers = []

    def answer(request):
        query_started.set()
        assert release_query.wait(timeout=2.0)
        return request.question

    worker = TextQueryWorker(
        answer=answer,
        on_answer=lambda request, response: answers.append((request, response)),
        on_error=lambda request, exc: pytest.fail(f"unexpected error for {request}: {exc}"),
    )
    worker.start()
    assert worker.submit(TextQueryRequest("question", {}))
    assert _wait_for(query_started.is_set)

    assert worker.stop(timeout_s=0.01) is False
    active_thread = worker._thread
    worker.start()
    assert worker._thread is active_thread
    release_query.set()
    assert worker.stop(timeout_s=2.0) is True
    assert worker._thread is None
    assert answers == []


# ---------------------------------------------------------------------------
# InteractiveSession
# ---------------------------------------------------------------------------


class _FakeEngine(InferenceEngine):
    """Real task-holder semantics with mocked lifecycle methods.

    Subclassing the ABC (instead of using a bare MagicMock) means the
    session tests exercise the actual ``set_task``/``task`` plumbing.
    ``failed``/``failure_traceback`` shadow the base properties as plain
    class attributes so tests can assign them.
    """

    failed = False
    failure_traceback = None

    # Declared here to satisfy the ABC; the instances below shadow them.
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...
    def get_action(self, obs_frame=None): ...

    def __init__(self, task: str = "pick up the cube", policy=None) -> None:
        super().__init__(task=task, policy=policy)
        self.start = MagicMock()
        self.stop = MagicMock()
        self.reset = MagicMock()
        self.pause = MagicMock()
        self.resume = MagicMock()
        self.notify_observation = MagicMock()
        self.get_action = MagicMock(return_value=None)


def _make_session(input_stream, run_behavior=None, policy=None):
    """Build a session around a mock strategy and a minimal fake context."""
    parent = Event()
    stop_event = LinkedEvent(parent)
    engine = _FakeEngine(policy=policy)
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
                lambda: (
                    console_handler.level == logging.CRITICAL + 1
                    and lib_handler.level == logging.CRITICAL + 1
                )
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
    engine = _FakeEngine()
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


def test_session_subtask_sets_and_reports_task(capsys):
    with _pipe_stream() as (reader, _writer):
        session, _strategy, engine, _parent, _run_started = _make_session(reader)
        thread = _start_session_thread(session)

        session._handle_line("/subtask grab the red cube")
        assert engine.task == "grab the red cube"

        # No argument reports the current task without changing it.
        session._handle_line("/subtask")
        assert engine.task == "grab the red cube"

        # Re-issuing the same task is reported as a no-op.
        session._handle_line("/subtask grab the red cube")
        out = capsys.readouterr().out
        assert "Current task: 'grab the red cube'" in out
        assert "Task unchanged: 'grab the red cube'" in out

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_subtask_strips_quotes_and_works_while_running():
    with _pipe_stream() as (reader, _writer):
        session, _strategy, engine, _parent, run_started = _make_session(reader)
        thread = _start_session_thread(session)

        session._handle_line("/start")
        assert _wait_for(run_started.is_set)

        # Switching mid-run must not interrupt the control loop.
        session._handle_line('/subtask "fold the towel"')
        assert engine.task == "fold the towel"
        time.sleep(0.05)
        assert session._running.is_set()

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_subtask_after_reset_is_not_clobbered():
    """A /subtask issued right after /reset must win — both writes are ordered by command order."""
    with _pipe_stream() as (reader, writer):
        session, strategy, engine, _parent, run_started = _make_session(reader)
        thread = _start_session_thread(session)

        writer.write("/start\n")
        writer.flush()
        assert _wait_for(run_started.is_set)

        # Arrives as one chunk, so both handlers run back-to-back on the
        # listener thread while the segment is still unwinding.
        writer.write("/reset\n/subtask put the cube in the box\n")
        writer.flush()
        assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)
        time.sleep(0.1)
        assert engine.task == "put the cube in the box"

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_reset_restores_initial_task():
    with _pipe_stream() as (reader, _writer):
        session, strategy, engine, _parent, _run_started = _make_session(reader)
        initial = engine.task
        thread = _start_session_thread(session)

        session._handle_line("/subtask fold the towel")
        assert engine.task == "fold the towel"

        session._handle_line("/reset")
        assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)
        assert engine.task == initial

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_reset_invalidates_text_observation(capsys):
    policy = MagicMock()
    policy.supports_text_generation.return_value = True
    with _pipe_stream() as (reader, _writer):
        session, _strategy, engine, _parent, _run_started = _make_session(reader, policy=policy)
        engine._publish_text_observation({"observation.state": "before-reset"})
        thread = _start_session_thread(session)

        session._handle_line("/reset")
        # Simulate an action inference that started before /reset but reaches
        # publication after the command handler invalidated the old scene.
        engine._publish_text_observation({"observation.state": "late-before-reset"})
        assert engine.snapshot_text_observation() is None
        session._handle_line("/ask what can you see?")

        assert "rollout is not running" in capsys.readouterr().out
        policy.generate_text.assert_not_called()

        engine._reset_text_observation()
        engine._publish_text_observation({"observation.state": "after-restart"})
        assert engine.snapshot_text_observation()["observation.state"] == "after-restart"
        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_ask_reports_usage_unsupported_policy_and_missing_observation(capsys):
    with _pipe_stream() as (reader, _writer):
        session, _strategy, _engine, _parent, _run_started = _make_session(reader)
        thread = _start_session_thread(session)

        session._handle_line("/ask")
        session._handle_line("/ask what can you see?")
        out = capsys.readouterr().out
        assert "Usage: /ask <question>" in out
        assert "does not support /ask" in out

        session._handle_line("/stop")
        thread.join(timeout=2.0)

    policy = MagicMock()
    policy.supports_text_generation.return_value = True
    with _pipe_stream() as (reader, _writer):
        session, _strategy, _engine, _parent, run_started = _make_session(reader, policy=policy)
        thread = _start_session_thread(session)

        session._handle_line("/ask what can you see?")
        assert "rollout is not running" in capsys.readouterr().out

        session._handle_line("/start")
        assert _wait_for(run_started.is_set)
        session._handle_line("/ask what can you see?")
        assert "No policy observation is available yet" in capsys.readouterr().out
        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_ask_runs_in_background_without_stopping_rollout(capsys):
    query_started = Event()
    release_query = Event()
    calls = []

    class TextPolicy:
        @staticmethod
        def supports_text_generation() -> bool:
            return True

        @staticmethod
        def generate_text(observation, *, kind, user_text):
            calls.append((observation, kind, user_text, current_thread().name))
            query_started.set()
            assert release_query.wait(timeout=2.0)
            return "The red cube is beside the bowl."

    with _pipe_stream() as (reader, _writer):
        session, strategy, engine, _parent, run_started = _make_session(reader, policy=TextPolicy())
        engine._publish_text_observation({"observation.state": "snapshot", "task": "stale task"})
        engine.set_task("current task")
        thread = _start_session_thread(session)
        session._handle_line("/start")
        assert _wait_for(run_started.is_set)

        session._handle_line('/ask "Where is the red cube?"')
        assert _wait_for(query_started.is_set)
        # The query worker is blocked in generation, but the rollout segment
        # and command listener remain live and the engine has not been paused.
        assert session._running.is_set()
        assert strategy.run.call_count == 1
        engine.pause.assert_not_called()

        session._handle_line("/ask another question")
        assert "already being answered" in capsys.readouterr().out

        release_query.set()
        assert _wait_for(lambda: not session._text_query.busy)
        out = capsys.readouterr().out
        assert "[policy] The red cube is beside the bowl." in out
        assert len(calls) == 1
        observation, kind, user_text, thread_name = calls[0]
        assert observation["observation.state"] == "snapshot"
        assert observation["task"] == "current task"
        assert kind is TextKind.VQA
        assert user_text == "Where is the red cube?"
        assert thread_name == "InteractiveTextQuery"

        session._handle_line("/stop")
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def test_session_ask_failure_is_nonfatal(capsys):
    class FailingTextPolicy:
        @staticmethod
        def supports_text_generation() -> bool:
            return True

        @staticmethod
        def generate_text(observation, *, kind, user_text):
            del observation, kind, user_text
            raise RuntimeError("decoder failed")

    with _pipe_stream() as (reader, _writer):
        session, _strategy, engine, _parent, run_started = _make_session(reader, policy=FailingTextPolicy())
        engine._publish_text_observation({"observation.state": "snapshot"})
        thread = _start_session_thread(session)
        session._handle_line("/start")
        assert _wait_for(run_started.is_set)
        session._handle_line("/ask what can you see?")
        assert _wait_for(lambda: not session._text_query.busy)
        assert "Policy question failed (RuntimeError): decoder failed" in capsys.readouterr().out
        assert session._running.is_set()

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_does_not_restart_while_text_query_owns_policy(capsys):
    query_started = Event()
    release_query = Event()

    class TextPolicy:
        @staticmethod
        def supports_text_generation() -> bool:
            return True

        @staticmethod
        def generate_text(observation, *, kind, user_text):
            del observation, kind, user_text
            query_started.set()
            assert release_query.wait(timeout=2.0)
            return "answer"

    with _pipe_stream() as (reader, _writer):
        session, strategy, engine, _parent, run_started = _make_session(reader, policy=TextPolicy())
        engine._publish_text_observation({"observation.state": "snapshot"})
        thread = _start_session_thread(session)
        session._handle_line("/start")
        assert _wait_for(run_started.is_set)
        session._handle_line("/ask question")
        assert _wait_for(query_started.is_set)

        session._handle_line("/reset")
        assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)
        session._handle_line("/start")
        assert "question is still finishing" in capsys.readouterr().out
        assert strategy.run.call_count == 1

        release_query.set()
        assert _wait_for(lambda: not session._text_query.busy)
        run_started.clear()
        session._handle_line("/start")
        assert _wait_for(run_started.is_set)
        assert strategy.run.call_count == 2

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_stop_suppresses_late_text_answer(capsys):
    query_started = Event()
    release_query = Event()

    class TextPolicy:
        @staticmethod
        def supports_text_generation() -> bool:
            return True

        @staticmethod
        def generate_text(observation, *, kind, user_text):
            del observation, kind, user_text
            query_started.set()
            assert release_query.wait(timeout=2.0)
            return "late answer"

    with _pipe_stream() as (reader, _writer):
        session, _strategy, engine, _parent, run_started = _make_session(reader, policy=TextPolicy())
        engine._publish_text_observation({"observation.state": "snapshot"})
        thread = _start_session_thread(session)
        session._handle_line("/start")
        assert _wait_for(run_started.is_set)
        session._handle_line("/ask question")
        assert _wait_for(query_started.is_set)
        capsys.readouterr()

        session._handle_line("/stop")
        # Shutdown waits for the model call so teardown cannot race its GPU use.
        time.sleep(0.05)
        assert thread.is_alive()
        release_query.set()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert "[policy] late answer" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# InferenceEngine task holder (the /subtask plumbing)
# ---------------------------------------------------------------------------


def test_engine_task_holder_tracks_changes():
    engine = _FakeEngine("pick up the cube")
    assert engine.task == "pick up the cube"
    # No change yet: nothing for the inference thread to flush.
    assert engine._take_task() == ("pick up the cube", False)

    assert engine.set_task("fold the towel") is True
    assert engine.task == "fold the towel"
    # The change edge is delivered once, then consumed.
    assert engine._take_task() == ("fold the towel", True)
    assert engine._take_task() == ("fold the towel", False)

    # Setting the same value is a no-op and raises no edge.
    assert engine.set_task("fold the towel") is False
    assert engine._take_task() == ("fold the towel", False)


def test_engine_discard_task_change():
    engine = _FakeEngine("a")
    engine.set_task("b")
    engine._discard_task_change()
    assert engine._take_task() == ("b", False)


def test_engine_text_query_has_priority_without_blocking_action_loop():
    query_started = Event()
    release_query = Event()

    class TextPolicy:
        @staticmethod
        def supports_text_generation() -> bool:
            return True

        @staticmethod
        def generate_text(observation, *, kind, user_text):
            del observation, kind, user_text
            query_started.set()
            assert release_query.wait(timeout=2.0)
            return "answer"

    engine = _FakeEngine(policy=TextPolicy())
    query_thread = Thread(
        target=lambda: engine.generate_text({}, kind=TextKind.VQA, user_text="question"),
        daemon=True,
    )
    query_thread.start()
    assert _wait_for(query_started.is_set)

    # Action inference probes the gate without waiting for text decoding.
    assert engine._try_begin_action_inference() is False
    release_query.set()
    query_thread.join(timeout=2.0)
    assert not query_thread.is_alive()

    assert engine._try_begin_action_inference() is True
    engine._end_action_inference()


def test_rtc_engine_does_not_race_action_inference_with_text_query():
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference import RTCInferenceEngine

    query_started = Event()
    release_query = Event()

    def generate_text(observation, *, kind, user_text):
        del observation, kind, user_text
        query_started.set()
        assert release_query.wait(timeout=2.0)
        return "answer"

    policy = MagicMock()
    policy.supports_text_generation.return_value = True
    policy.generate_text.side_effect = generate_text
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=SimpleNamespace(steps=[]),
        postprocessor=SimpleNamespace(steps=[]),
        robot_wrapper=SimpleNamespace(action_features={}, robot_type="test"),
        rtc_config=RTCConfig(),
        hw_features={},
        task="test",
        fps=30,
        device="cpu",
    )
    query_thread = Thread(
        target=lambda: engine.generate_text({}, kind=TextKind.VQA, user_text="question"),
        daemon=True,
    )
    query_thread.start()
    assert _wait_for(query_started.is_set)

    engine.start()
    try:
        engine.notify_observation({"joint.pos": 0.0})
        engine.resume()
        time.sleep(0.05)
        # The RTC worker remains responsive but does not enter the same policy
        # while language decoding owns it.
        assert engine._rtc_thread is not None and engine._rtc_thread.is_alive()
        policy.predict_action_chunk.assert_not_called()
    finally:
        engine.stop()
        release_query.set()
        query_thread.join(timeout=2.0)
    assert not query_thread.is_alive()


def test_sync_engine_uses_new_task_and_flushes_precomputed_actions():
    """A /subtask switch must reach the policy and drop stale queued actions."""
    import torch

    from lerobot.rollout.inference import SyncInferenceEngine

    policy = MagicMock()
    policy.config.use_amp = False
    policy.select_action.return_value = torch.zeros(1, 2)

    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=lambda obs: obs,
        postprocessor=lambda action: action,
        dataset_features={
            "action": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
        },
        ordered_action_keys=["j1.pos", "j2.pos"],
        task="pick up the cube",
        device="cpu",
        robot_type="mock",
    )

    engine.get_action({"observation.state": np.zeros(1, dtype=np.float32)})
    assert policy.drop_queued_actions.call_count == 0
    assert policy.select_action.call_args[0][0]["task"] == "pick up the cube"
    assert engine.snapshot_text_observation()["task"] == "pick up the cube"

    engine.set_task("fold the towel")
    engine.get_action({"observation.state": np.zeros(1, dtype=np.float32)})
    # Precomputed chunk actions are dropped so the new task applies immediately,
    # without the wider episode reset (which would perturb observation history).
    assert policy.drop_queued_actions.call_count == 1
    assert policy.reset.call_count == 0
    assert policy.select_action.call_args[0][0]["task"] == "fold the towel"

    # Only the first call after a switch flushes.
    engine.get_action({"observation.state": np.zeros(1, dtype=np.float32)})
    assert policy.drop_queued_actions.call_count == 1


def test_drop_queued_actions_clears_both_queue_conventions():
    """PreTrainedPolicy.drop_queued_actions covers both action-queue idioms in the repo."""
    from collections import deque

    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.utils.constants import ACTION

    # PreTrainedPolicy's metaclass demands a config_class, so exercise the
    # method against stand-ins carrying each queue idiom.
    flush = PreTrainedPolicy.drop_queued_actions

    queues_policy = SimpleNamespace(  # smolvla / diffusion / vqbet / wall_x style
        _queues={ACTION: deque([1, 2, 3]), "observation.state": deque([9])}
    )
    flush(queues_policy)
    assert len(queues_policy._queues[ACTION]) == 0
    # Other episode state is intentionally left alone.
    assert len(queues_policy._queues["observation.state"]) == 1

    action_queue_policy = SimpleNamespace(_action_queue=deque([1, 2, 3]))  # act / pi0 / groot style
    flush(action_queue_policy)
    assert len(action_queue_policy._action_queue) == 0

    # Queue-less policies inherit a no-op.
    flush(SimpleNamespace())


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
