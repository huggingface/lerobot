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

"""Tests for interactive rollout control: the programmatic RolloutController
and the stdin-driven InteractiveSession (--interactive=true)."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import time
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout import (  # noqa: E402
    AskResult,
    InferenceEngine,
    InteractiveCommand,
    InteractiveSession,
    LinkedEvent,
    PolicyQuery,
    QueryAnswer,
    QueryKind,
    RolloutController,
    RolloutEvent,
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
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeEngine(InferenceEngine):
    """Real task-holder semantics with mocked lifecycle methods.

    Subclassing the ABC (instead of using a bare MagicMock) means these
    tests exercise the actual ``set_task``/``task`` plumbing.
    ``failed``/``failure_traceback`` shadow the base properties as plain
    class attributes so tests can assign them.
    """

    failed = False
    failure_traceback = None
    # Plain attributes shadowing the base properties, so tests can flip them.
    supports_text_queries = True
    # Answer inline on the caller's thread like the sync backend, so tests
    # drive the base pump_query's real poll/service/deliver path.
    control_thread_owns_policy = True

    # Declared here to satisfy the ABC; the instances below shadow them.
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...
    def get_action(self, obs_frame=None): ...

    def __init__(self, task: str = "pick up the cube") -> None:
        super().__init__(task=task)
        self.start = MagicMock()
        self.stop = MagicMock()
        self.reset = MagicMock()
        self.pause = MagicMock()
        self.resume = MagicMock()
        self.notify_observation = MagicMock()
        self.get_action = MagicMock(return_value=None)
        self.text_error: Exception | None = None
        self.seen_query_obs: list = []
        self.seen_queries: list = []
        self.subtasks_planned = 0

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        self.seen_queries.append((query.kind, query.text))
        self.seen_query_obs.append(obs_processed)
        if self.text_error is not None:
            raise self.text_error
        if query.kind is QueryKind.NEXT_SUBTASK:
            # A stand-in decomposer: one subtask per query, advancing.
            self.subtasks_planned += 1
            return f"subtask {self.subtasks_planned} of {query.text}"
        return f"answer: {query.text}"


def _make_ctx(run_behavior=None):
    """A mock strategy plus the minimal fake context the controller needs."""
    parent = Event()
    stop_event = LinkedEvent(parent)
    engine = _FakeEngine()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(play_sounds=False, autosteer_interval_s=0.0),
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
    return ctx, strategy, engine, parent, run_started


# ---------------------------------------------------------------------------
# RolloutController (the programmatic API)
# ---------------------------------------------------------------------------


def _event_recorder(events: list, answers: list | None = None):
    """Adapter for the controller's ``(event, payload)`` callback."""

    def record(event, payload=None):
        events.append(event)
        if answers is not None and payload is not None:
            answers.append(payload)

    return record


def _make_controller(run_behavior=None, answers=None):
    ctx, strategy, engine, parent, run_started = _make_ctx(run_behavior)
    events: list[RolloutEvent] = []
    controller = RolloutController(strategy, ctx, on_event=_event_recorder(events, answers))
    return controller, events, strategy, engine, parent, run_started


def _serve_thread(controller) -> Thread:
    thread = Thread(target=controller.serve, daemon=True)
    thread.start()
    return thread


def test_controller_requires_linked_event():
    ctx = SimpleNamespace(runtime=SimpleNamespace(shutdown_event=Event()))
    with pytest.raises(TypeError, match="LinkedEvent"):
        RolloutController(MagicMock(), ctx)


def test_controller_rejects_one_shot_strategies():
    """The library path enforces the same restartable-run() contract as the CLI.

    Wrapping a one-shot strategy would finalize the dataset on the first
    segment and record into it again on the second.
    """
    from lerobot.rollout import EpisodicStrategyConfig

    ctx, strategy, _engine, _parent, _run_started = _make_ctx()
    strategy.config = EpisodicStrategyConfig()
    with pytest.raises(ValueError, match="supports_interactive"):
        RolloutController(strategy, ctx)


def test_controller_start_reset_stop_flow_and_events():
    controller, events, strategy, engine, _parent, run_started = _make_controller()
    thread = _serve_thread(controller)

    # Idle until start(): the strategy loop must not run on its own.
    time.sleep(0.05)
    strategy.run.assert_not_called()
    assert not controller.running

    assert controller.start() is True
    assert _wait_for(run_started.is_set)
    assert controller.running
    assert strategy.reset_control_state.call_count == 1
    assert RolloutEvent.SEGMENT_STARTED in events

    # reset() ends the segment, pauses the engine, returns the robot home.
    controller.reset()
    assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)
    assert engine.pause.call_count >= 1
    assert thread.is_alive()
    assert _wait_for(lambda: RolloutEvent.RESET_DONE in events)
    assert RolloutEvent.RESET_STARTED in events

    # start() again runs a fresh segment with freshly reset control state.
    run_started.clear()
    assert controller.start() is True
    assert _wait_for(run_started.is_set)
    assert strategy.run.call_count == 2
    assert strategy.reset_control_state.call_count == 2

    # stop() ends serve(); teardown stays with the caller.
    controller.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    strategy.teardown.assert_not_called()
    assert events[-1] is RolloutEvent.STOPPED


def test_controller_start_rejected_during_segment_startup():
    """A start() racing the segment startup must be rejected, not queued.

    Otherwise the re-armed request survives the whole segment and the robot
    would start again, uncommanded, when the segment ends on its own.
    """
    ctx, strategy, _engine, _parent, run_started = _make_ctx()
    in_startup = Event()
    startup_gate = Event()

    def slow_reset_control_state():
        in_startup.set()
        startup_gate.wait(timeout=2.0)

    strategy.reset_control_state.side_effect = slow_reset_control_state
    controller = RolloutController(strategy, ctx)
    thread = _serve_thread(controller)

    assert controller.start() is True
    assert _wait_for(in_startup.is_set)
    # The serve thread is inside reset_control_state: the segment counts as
    # running for concurrent callers even though strategy.run hasn't begun.
    assert controller.start() is False
    startup_gate.set()
    assert _wait_for(run_started.is_set)

    # Ending the segment must not trigger a phantom second segment.
    controller.reset()
    assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)
    time.sleep(0.05)
    assert strategy.run.call_count == 1

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_start_returns_false_while_running():
    controller, _events, strategy, _engine, _parent, run_started = _make_controller()
    thread = _serve_thread(controller)

    assert controller.start() is True
    assert _wait_for(run_started.is_set)
    assert controller.start() is False
    time.sleep(0.05)
    assert strategy.run.call_count == 1

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_set_task_and_reset_restores_launch_task():
    controller, _events, strategy, engine, _parent, _run_started = _make_controller()
    thread = _serve_thread(controller)
    initial = controller.initial_task

    # reset() with the launch task still in place reports no restore.
    assert controller.reset() is False
    assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)

    assert controller.set_task("fold the towel") is True
    assert controller.task == "fold the towel"
    assert engine.task == "fold the towel"
    assert controller.set_task("fold the towel") is False

    assert controller.reset() is True  # the task had been changed
    assert controller.task == initial

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_is_one_shot_and_refuses_commands_after_stop():
    """Commands after serve() has returned must be refused, not acknowledged."""
    controller, _events, strategy, engine, _parent, _run_started = _make_controller()
    thread = _serve_thread(controller)
    controller.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert controller.stopped

    initial = engine.task
    assert controller.start() is False
    assert controller.reset() is False
    assert controller.set_task("fold the towel") is False
    assert engine.task == initial  # the refused set_task touched nothing
    assert controller.ask("what do you see?") is AskResult.NOT_RUNNING
    time.sleep(0.05)
    strategy.run.assert_not_called()

    with pytest.raises(RuntimeError, match="one-shot"):
        controller.serve()


def test_controller_strategy_failure_reaches_the_failure_surface():
    """A strategy.run() exception must not masquerade as a deliberate stop.

    Robot I/O and recording failures used to unwind through serve() with a
    clean STOPPED event, controller.failed False, and the traceback lost to
    the thread excepthook.
    """

    def exploding_run(c):
        raise RuntimeError("robot io broke")

    controller, events, _strategy, _engine, _parent, _run_started = _make_controller(exploding_run)
    thread = _serve_thread(controller)
    controller.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()  # serve() ended cleanly instead of dying with the exception

    assert RolloutEvent.STRATEGY_FAILED in events
    assert RolloutEvent.SEGMENT_ENDED not in events
    assert events[-1] is RolloutEvent.STOPPED
    assert controller.failed
    assert "robot io broke" in controller.failure_traceback


def test_controller_failed_return_move_emits_reset_failed():
    """RESET_DONE promises the robot is home; a failed move must not claim it."""
    controller, events, strategy, _engine, _parent, _run_started = _make_controller()
    strategy.return_to_initial_position.return_value = False  # the move errored partway
    thread = _serve_thread(controller)

    controller.reset()
    assert _wait_for(lambda: RolloutEvent.RESET_FAILED in events)
    assert RolloutEvent.RESET_DONE not in events

    controller.stop()
    thread.join(timeout=2.0)


def test_segment_startup_aborts_when_engine_failed_in_the_guard_window():
    """A fatal failure landing between serve()'s failed-check and segment
    startup must not start a segment against a dead engine.

    The engine signals failure by setting the shutdown event — in
    interactive mode the LinkedEvent's *local* flag, exactly the flag
    segment startup clears — so the startup guard must re-check
    engine.failed itself.
    """
    controller, _events, strategy, engine, _parent, _run_started = _make_controller()
    engine.failed = True
    controller._running.set()  # as serve() does under the control lock
    controller._run_segment()
    strategy.run.assert_not_called()
    assert not controller.running


def test_return_to_initial_position_reports_completion():
    """The now-public primitive tells its caller whether the move finished."""
    from lerobot.rollout.strategies import RolloutStrategy

    hw = SimpleNamespace(robot_wrapper=MagicMock(), initial_position={"joint.pos": 1.0})
    hw.robot_wrapper.get_observation.return_value = {"joint.pos": 0.0}
    assert RolloutStrategy.return_to_initial_position(hw, duration_s=0.02, fps=50) is True

    hw.robot_wrapper.get_observation.side_effect = OSError("serial port died")
    assert RolloutStrategy.return_to_initial_position(hw, duration_s=0.02, fps=50) is False


def test_controller_engine_failure_emits_event():
    def failing_run(c):
        c.policy.inference.failed = True
        c.policy.inference.failure_traceback = "RuntimeError: boom-traceback"
        c.runtime.shutdown_event.set()

    controller, events, strategy, _engine, _parent, _run_started = _make_controller(failing_run)
    thread = _serve_thread(controller)
    controller.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    strategy.return_to_initial_position.assert_not_called()
    assert RolloutEvent.ENGINE_FAILED in events
    assert controller.failed
    assert controller.failure_traceback == "RuntimeError: boom-traceback"


def test_controller_segment_ended_event_on_natural_end():
    def finite_run(c):
        return None  # e.g. --duration elapsed

    controller, events, strategy, _engine, _parent, _run_started = _make_controller(finite_run)
    thread = _serve_thread(controller)

    controller.start()
    assert _wait_for(lambda: RolloutEvent.SEGMENT_ENDED in events)
    assert thread.is_alive()  # back to idle, not shut down
    assert not controller.running

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_reset_skipped_without_initial_position():
    ctx, strategy, _engine, _parent, _run_started = _make_ctx()
    ctx.hardware.initial_position = {}
    events: list[RolloutEvent] = []
    controller = RolloutController(strategy, ctx, on_event=_event_recorder(events))
    thread = _serve_thread(controller)

    controller.reset()
    assert _wait_for(lambda: RolloutEvent.RESET_SKIPPED in events)
    strategy.return_to_initial_position.assert_not_called()

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_callback_errors_do_not_kill_serve():
    ctx, strategy, _engine, _parent, run_started = _make_ctx()

    def broken_observer(event, payload=None):
        raise RuntimeError("observer boom")

    controller = RolloutController(strategy, ctx, on_event=broken_observer)
    thread = _serve_thread(controller)

    controller.start()
    assert _wait_for(run_started.is_set)
    controller.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_controller_works_without_event_callback():
    ctx, strategy, _engine, _parent, _run_started = _make_ctx()
    controller = RolloutController(strategy, ctx)
    thread = _serve_thread(controller)
    controller.start()
    controller.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# InteractiveSession (the stdin CLI front-end)
# ---------------------------------------------------------------------------


def _make_session(input_stream, run_behavior=None):
    """Build a session around a mock strategy and a minimal fake context."""
    ctx, strategy, engine, parent, run_started = _make_ctx(run_behavior)
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


def test_session_mutes_logs_below_warning_and_restores_on_exit():
    import warnings

    # Libraries like transformers attach their own console handler with
    # propagate=False; the process-wide logging.disable gate covers those too.
    lib_stream = io.StringIO()
    lib_logger = logging.getLogger("test_interactive_fake_lib")
    lib_logger.propagate = False
    lib_logger.setLevel(logging.DEBUG)
    lib_handler = logging.StreamHandler(lib_stream)
    lib_handler.setLevel(logging.INFO)
    lib_logger.addHandler(lib_handler)
    n_warning_filters = len(warnings.filters)
    try:
        with _pipe_stream() as (reader, _writer):
            session, _strategy, _engine, _parent, _run_started = _make_session(reader)
            thread = _start_session_thread(session)
            assert _wait_for(lambda: logging.root.manager.disable == logging.INFO)

            lib_logger.info("muted-info")
            lib_logger.warning("visible-warning")  # e.g. control loop missing its FPS target
            lib_logger.error("visible-error")  # errors must surface mid-session

            session._handle_line("/stop")
            thread.join(timeout=2.0)

        output = lib_stream.getvalue()
        assert "muted-info" not in output
        assert "visible-warning" in output
        assert "visible-error" in output

        # Everything is restored once the session ends.
        assert logging.root.manager.disable == logging.NOTSET
        assert len(warnings.filters) == n_warning_filters
        lib_logger.info("post-session-info")
        assert "post-session-info" in lib_stream.getvalue()
    finally:
        lib_logger.removeHandler(lib_handler)


def test_session_restores_preexisting_disable_level():
    """An embedding application's own logging.disable level survives the session."""
    logging.disable(logging.DEBUG)
    try:
        with _pipe_stream() as (reader, _writer):
            session, _strategy, _engine, _parent, _run_started = _make_session(reader)
            thread = _start_session_thread(session)
            assert _wait_for(lambda: logging.root.manager.disable == logging.INFO)
            session._handle_line("/stop")
            thread.join(timeout=2.0)
        assert logging.root.manager.disable == logging.DEBUG
    finally:
        logging.disable(logging.NOTSET)


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
                autosteer_interval_s=0.0,
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
        assert session.controller.running

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
    # The discard also re-primes the dispatched-task marker: queues were just
    # cleared, so the next dispatched action can only come from the new task.
    assert engine.dispatched_task == "b"


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

    first = engine.get_action({"observation.state": np.zeros(1, dtype=np.float32)})
    assert first is not None
    assert engine.dispatched_task == "pick up the cube"
    assert policy.drop_queued_actions.call_count == 0
    assert policy.select_action.call_args[0][0]["task"] == "pick up the cube"

    engine.set_task("fold the towel")
    switched = engine.get_action({"observation.state": np.zeros(1, dtype=np.float32)})
    assert switched is not None
    assert engine.dispatched_task == "fold the towel"
    # Precomputed chunk actions are dropped so the new task applies immediately,
    # without the wider episode reset (which would perturb observation history).
    assert policy.drop_queued_actions.call_count == 1
    assert policy.reset.call_count == 0
    assert policy.select_action.call_args[0][0]["task"] == "fold the towel"

    # Only the first call after a switch flushes.
    engine.get_action({"observation.state": np.zeros(1, dtype=np.float32)})
    assert policy.drop_queued_actions.call_count == 1


def test_sync_action_keeps_task_snapshot_when_command_changes_during_inference():
    """An in-flight A action must not be relabeled when /subtask changes the requested task to B."""
    import torch

    from lerobot.rollout.inference import SyncInferenceEngine

    policy = MagicMock()
    policy.config.use_amp = False
    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=lambda obs: obs,
        postprocessor=lambda action: action,
        dataset_features={
            "action": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
        },
        ordered_action_keys=["j1.pos", "j2.pos"],
        task="task A",
        device="cpu",
        robot_type="mock",
    )

    def change_task_during_inference(_observation):
        engine.set_task("task B")
        return torch.zeros(1, 2)

    policy.select_action.side_effect = change_task_during_inference
    result = engine.get_action({"observation.state": np.zeros(1, dtype=np.float32)})

    assert result is not None
    assert engine.dispatched_task == "task A"
    assert engine.task == "task B"


def test_drop_queued_actions_clears_both_queue_conventions():
    """PreTrainedPolicy.drop_queued_actions covers both action-queue idioms in the repo."""
    from collections import deque

    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.utils.constants import ACTION

    # Subclassing PreTrainedPolicy demands a config_class (enforced in its
    # __init_subclass__), so exercise the method against stand-ins carrying
    # each queue idiom.
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
# Sentry strategy: restartable run() segments + dispatched-action task labels
# ---------------------------------------------------------------------------


def _make_sentry(monkeypatch):
    from lerobot.rollout import SentryStrategy, SentryStrategyConfig

    # Keep setup independent of camera features and never rotate episodes
    # mid-test; frame assembly is not under test here.
    monkeypatch.setattr("lerobot.rollout.strategies.sentry.estimate_max_episode_seconds", lambda *a, **k: 1e9)
    monkeypatch.setattr(
        "lerobot.rollout.strategies.sentry.send_next_action", lambda *a, **k: {"joint.pos": 0.5}
    )
    monkeypatch.setattr("lerobot.rollout.strategies.sentry.build_dataset_frame", lambda *a, **k: {})

    engine = _FakeEngine()
    dataset = MagicMock()
    robot = MagicMock()
    robot.get_observation.return_value = {"joint.pos": 0.0}
    stop_event = Event()

    def identity(x):
        return x

    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                play_sounds=False,
                fps=200.0,
                duration=0.0,
                use_torch_compile=False,
                interpolation_multiplier=1,
                display_data=False,
                return_to_initial_position=False,
                task="pick up the cube",
                dataset=SimpleNamespace(push_to_hub=False, tags=None, private=False),
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
        data=SimpleNamespace(dataset=dataset, dataset_features={}, hw_features={}, ordered_action_keys=[]),
    )

    strategy = SentryStrategy(SentryStrategyConfig())
    strategy.setup(ctx)
    return strategy, ctx, dataset, engine, stop_event


def test_sentry_run_is_restartable_and_finalizes_only_in_teardown(monkeypatch):
    strategy, ctx, dataset, _engine, stop_event = _make_sentry(monkeypatch)

    # Segment 1.
    thread = Thread(target=strategy.run, args=(ctx,), daemon=True)
    thread.start()
    assert _wait_for(lambda: dataset.add_frame.call_count >= 2)
    stop_event.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()

    # The segment saved its partial episode but left the dataset open.
    assert dataset.save_episode.call_count == 1
    dataset.finalize.assert_not_called()

    # Segment 2: run() is restartable on the same instance.
    stop_event.clear()
    frames_before = dataset.add_frame.call_count
    thread = Thread(target=strategy.run, args=(ctx,), daemon=True)
    thread.start()
    assert _wait_for(lambda: dataset.add_frame.call_count >= frames_before + 2)
    stop_event.set()
    thread.join(timeout=2.0)
    assert dataset.save_episode.call_count == 2
    dataset.finalize.assert_not_called()

    # Only teardown finalizes.
    strategy.teardown(ctx)
    dataset.finalize.assert_called_once()


def test_sentry_labels_frames_with_dispatched_action_task(monkeypatch):
    strategy, ctx, dataset, engine, stop_event = _make_sentry(monkeypatch)

    thread = Thread(target=strategy.run, args=(ctx,), daemon=True)
    thread.start()
    assert _wait_for(lambda: dataset.add_frame.call_count >= 2)

    frames_before_switch = dataset.add_frame.call_count
    # The requested task changes immediately, but frames keep the old label
    # until an action generated under the new instruction is dispatched —
    # the engine updates ``dispatched_task`` on that pop, simulated below.
    engine.set_task("fold the towel")
    assert _wait_for(lambda: dataset.add_frame.call_count >= frames_before_switch + 2)
    tasks = [call.args[0]["task"] for call in dataset.add_frame.call_args_list]
    assert tasks[-1] == "pick up the cube"

    engine._set_dispatched_task("fold the towel")
    assert _wait_for(
        lambda: any(call.args[0]["task"] == "fold the towel" for call in dataset.add_frame.call_args_list)
    )
    stop_event.set()
    thread.join(timeout=2.0)

    tasks = [call.args[0]["task"] for call in dataset.add_frame.call_args_list]
    assert tasks[0] == "pick up the cube"
    assert tasks[-1] == "fold the towel"

    strategy.teardown(ctx)


def test_sentry_tail_save_failure_poisons_dataset_and_reraises(monkeypatch):
    """A failed save_episode may have committed rows already; sentry must fail loudly.

    Swallow-and-continue would let the next segment reuse the same episode
    index with overlapping row indices and upload the corruption to the Hub.
    """
    strategy, ctx, dataset, _engine, stop_event = _make_sentry(monkeypatch)
    dataset.save_episode.side_effect = ValueError("disk full mid-write")
    stop_event.set()  # the segment ends on its first tick; the tail save fails

    with pytest.raises(ValueError, match="disk full mid-write"):
        strategy.run(ctx)

    # Cleanup goes through the public API: cancel the in-flight streaming
    # encode and discard the half-mutated buffer.
    dataset.clear_episode_buffer.assert_called_once_with(delete_images=False)

    # No further segments on a poisoned dataset...
    with pytest.raises(RuntimeError, match="partially committed"):
        strategy.run(ctx)

    # ...and no Hub push of the corruption; teardown still closes the dataset.
    ctx.runtime.cfg.dataset.push_to_hub = True
    strategy._needs_push.set()
    strategy.teardown(ctx)
    dataset.finalize.assert_called_once()
    dataset.push_to_hub.assert_not_called()


def test_sentry_empty_tail_segment_skips_the_save(monkeypatch):
    """A segment that recorded nothing must not raise (or log) through the tail save."""
    strategy, ctx, dataset, _engine, stop_event = _make_sentry(monkeypatch)
    dataset.has_pending_frames.return_value = False
    stop_event.set()  # end immediately: /start followed by /reset during warmup

    strategy.run(ctx)

    dataset.save_episode.assert_not_called()
    dataset.clear_episode_buffer.assert_not_called()


def test_sentry_tail_saves_count_toward_upload_cadence(monkeypatch):
    """Interactive segments are typically shorter than one rotation, so the
    segment-end save is the only save — it must advance the upload cadence."""
    strategy, ctx, dataset, _engine, stop_event = _make_sentry(monkeypatch)
    strategy.config.upload_every_n_episodes = 2
    pushes = []
    monkeypatch.setattr(strategy, "_background_push", lambda dataset, cfg: pushes.append(1))

    for _ in range(2):
        stop_event.clear()
        frames_before = dataset.add_frame.call_count
        thread = Thread(target=strategy.run, args=(ctx,), daemon=True)
        thread.start()
        assert _wait_for(lambda: dataset.add_frame.call_count > frames_before)
        stop_event.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    assert dataset.save_episode.call_count == 2
    assert pushes == [1]  # the second tail save crossed the threshold
    assert strategy._episodes_since_push == 0  # counter reset after the push


def test_sentry_warns_before_blocking_on_inflight_push(monkeypatch, caplog):
    """/reset//stop during a Hub upload must say why the robot is frozen.

    The WARNING level is deliberate: it pierces the interactive session's
    log muting.
    """
    strategy, ctx, dataset, _engine, stop_event = _make_sentry(monkeypatch)
    pending = MagicMock()
    pending.done.return_value = False
    strategy._pending_push = pending
    stop_event.set()

    with caplog.at_level(logging.WARNING, logger="lerobot.rollout.strategies.sentry"):
        strategy.run(ctx)

    assert any("in-flight Hub upload" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Text queries (/vqa)
# ---------------------------------------------------------------------------


def _pumping_run(started: Event, obs: dict | None = None):
    """A fake control loop that pumps the engine's query channel every tick."""
    tick_obs = obs if obs is not None else {"joint.pos": 0.0}

    def run(ctx):
        started.set()
        while not ctx.runtime.shutdown_event.is_set():
            ctx.policy.inference.pump_query(tick_obs)
            time.sleep(0.005)

    return run


def test_engine_query_channel_holds_one_question():
    engine = _FakeEngine()
    assert not engine.has_pending_query

    assert engine.ask("what do you see?") is True
    assert engine.has_pending_query
    # A second question must not silently displace the unanswered one.
    assert engine.ask("and now?") is False

    dropped = engine.drop_pending_query()
    assert (dropped.kind, dropped.text) == (QueryKind.VQA, "what do you see?")
    assert not engine.has_pending_query
    assert engine.drop_pending_query() is None


def test_engine_answers_question_and_delivers_once():
    engine = _FakeEngine()
    delivered = []
    engine.set_answer_observer(delivered.append)

    engine.ask("is the cube in the box?")
    engine.pump_query({"joint.pos": 1.0})

    assert len(delivered) == 1
    assert delivered[0].ok
    assert delivered[0].question == "is the cube in the box?"
    assert delivered[0].answer == "answer: is the cube in the box?"
    # The answer was generated from the observation handed to pump_query.
    assert engine.seen_query_obs == [{"joint.pos": 1.0}]

    # The slot is drained, so a second pump delivers nothing.
    engine.pump_query({"joint.pos": 1.0})
    assert len(delivered) == 1


def test_engine_query_without_observation_stays_queued():
    """The controller's idle poll pumps with no observation; the question waits."""
    engine = _FakeEngine()
    delivered = []
    engine.set_answer_observer(delivered.append)

    engine.ask("what do you see?")
    engine.pump_query(None)

    assert engine.has_pending_query
    assert delivered == []
    assert engine.seen_query_obs == []


def test_engine_query_failure_becomes_an_error_answer():
    """A policy that cannot answer must not take down the calling loop."""
    engine = _FakeEngine()
    engine.text_error = RuntimeError("no text head")
    delivered = []
    engine.set_answer_observer(delivered.append)

    engine.ask("what do you see?")
    engine.pump_query({"joint.pos": 0.0})

    assert len(delivered) == 1
    assert not delivered[0].ok
    assert "RuntimeError: no text head" in delivered[0].error


@pytest.mark.parametrize("bad_output", [None, 42, "", "   "])
def test_engine_rejects_invalid_generate_text_output(bad_output):
    """A contract-violating generate_text (None, tensor, empty) becomes an error answer.

    Without central validation, str() coercion would turn a forgotten return
    into the live task 'None' — steering the robot and labeling frames with it.
    """
    engine = _FakeEngine()
    engine._generate_text = lambda obs_processed, query: bad_output
    delivered = []
    engine.set_answer_observer(delivered.append)

    engine.ask("what do you see?")
    engine.pump_query({"joint.pos": 0.0})

    assert len(delivered) == 1
    assert not delivered[0].ok
    assert "TypeError" in delivered[0].error
    assert engine.task == "pick up the cube"  # nothing steered the robot


def test_engine_pump_query_reports_inline_service():
    """pump_query returns True only for the tick that generated inline."""
    engine = _FakeEngine()
    assert engine.pump_query({"joint.pos": 0.0}) is False  # nothing queued

    engine.ask("what do you see?")
    assert engine.pump_query({"joint.pos": 0.0}) is True  # generated inline
    assert engine.pump_query({"joint.pos": 0.0}) is False  # drained

    # Async backends never serve on the control thread, so they never
    # report an inline generation.
    engine.control_thread_owns_policy = False
    engine.ask("and now?")
    assert engine.pump_query({"joint.pos": 0.0}) is False
    assert engine.has_pending_query


def test_engine_undelivered_answers_queue_up_instead_of_being_overwritten():
    """Two answers landing between pumps must both reach the observer.

    On an async backend the query slot frees at claim time, so a second /vqa
    can be accepted and answered while the control thread is not pumping
    (e.g. during torch.compile warmup).  A single-slot answer channel would
    silently drop the first answer.
    """
    engine = _FakeEngine()
    engine.control_thread_owns_policy = False
    delivered = []
    engine.set_answer_observer(delivered.append)

    assert engine.ask("first?") is True
    engine._service_query({"joint.pos": 0.0})  # the async thread answers it
    assert engine.ask("second?") is True
    engine._service_query({"joint.pos": 0.0})

    # The next pump delivers both, oldest first.
    engine.pump_query({"joint.pos": 0.0})
    assert [a.question for a in delivered] == ["first?", "second?"]
    assert all(a.ok for a in delivered)


def test_policy_text_generation_contract_defaults_off():
    """The base policy declares the text-head contract as opt-in.

    ``supports_text_generation`` is what the rollout stack consults before
    accepting /vqa or /autosteer, and the default ``generate_text`` must
    fail loudly (a policy that flips the flag without implementing the
    method is a bug, not a silent no-op).
    """
    from lerobot.policies.pretrained import PreTrainedPolicy

    class _NoTextHead:  # a stand-in ``self``; the base methods touch nothing else
        pass

    assert PreTrainedPolicy.supports_text_generation(_NoTextHead()) is False
    with pytest.raises(NotImplementedError, match="has no text head"):
        PreTrainedPolicy.generate_text(_NoTextHead(), {})


@pytest.mark.parametrize("kind", [QueryKind.VQA, QueryKind.NEXT_SUBTASK])
def test_query_reaches_the_preprocessor_as_complementary_data(kind):
    """The query travels in the batch, not as generate_text arguments.

    ``batch_to_transition`` forwards only an allowlisted set of keys to
    complementary data, so an unregistered key would be silently dropped and
    the policy's preprocessor step would never see it.
    """
    from lerobot.lerobot_types import TransitionKey
    from lerobot.processor.converters import batch_to_transition
    from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT

    query = PolicyQuery(kind=kind, text="is the cube in the box?")
    batch = InferenceEngine._mark_query({"observation.state": np.zeros(2), "task": "tidy"}, query)
    complementary = batch_to_transition(batch)[TransitionKey.COMPLEMENTARY_DATA.value]

    assert complementary[QUERY_KIND] == kind.value
    assert complementary[QUERY_TEXT] == "is the cube in the box?"
    # They land beside the task, so a single ComplementaryDataProcessorStep
    # can read the kind and rewrite the query text into this policy's prompt
    # format before generate_text consumes it.
    assert complementary["task"] == "tidy"


def test_controller_ask_rejected_while_idle():
    """No segment means no observation stream, so there is nothing to answer from."""
    controller, _events, _strategy, engine, _parent, _run_started = _make_controller()
    thread = _serve_thread(controller)

    assert controller.ask("what do you see?") is AskResult.NOT_RUNNING
    assert not engine.has_pending_query

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_refuses_queries_for_a_policy_without_text_head():
    """UNSUPPORTED is decided up front, even mid-run — not one tick later."""
    started = Event()
    controller, _events, _strategy, engine, _parent, _run_started = _make_controller(
        run_behavior=_pumping_run(started)
    )
    engine.supports_text_queries = False
    thread = _serve_thread(controller)

    controller.start()
    assert _wait_for(started.is_set)

    assert controller.ask("what do you see?") is AskResult.UNSUPPORTED
    assert not engine.has_pending_query
    assert controller.autosteer("tidy the table") is AskResult.UNSUPPORTED
    assert engine.autosteer_goal is None

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_ask_answers_via_event():
    started = Event()
    answers = []
    controller, events, _strategy, _engine, _parent, _run_started = _make_controller(
        run_behavior=_pumping_run(started), answers=answers
    )
    thread = _serve_thread(controller)

    controller.start()
    assert _wait_for(started.is_set)

    assert controller.ask("is the cube in the box?") is AskResult.QUEUED
    assert _wait_for(lambda: bool(answers))

    assert RolloutEvent.QUERY_ANSWERED in events
    assert answers[0].ok
    assert answers[0].question == "is the cube in the box?"
    assert answers[0].answer == "answer: is the cube in the box?"

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_ask_rejects_a_second_question_while_one_is_pending():
    # The default run behaviour never pumps, so the first question stays queued.
    controller, _events, _strategy, _engine, _parent, run_started = _make_controller()
    thread = _serve_thread(controller)

    controller.start()
    assert _wait_for(run_started.is_set)

    assert controller.ask("first?") is AskResult.QUEUED
    assert controller.ask("second?") is AskResult.BUSY

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_drops_pending_question_when_segment_ends():
    """Left in the slot it would be answered against a different scene later."""
    answers = []
    controller, _events, _strategy, engine, _parent, run_started = _make_controller(answers=answers)
    thread = _serve_thread(controller)

    controller.start()
    assert _wait_for(run_started.is_set)
    assert controller.ask("what do you see?") is AskResult.QUEUED

    controller.reset()
    assert _wait_for(lambda: bool(answers))

    assert not answers[0].ok
    assert answers[0].question == "what do you see?"
    assert "run ended" in answers[0].error
    assert not engine.has_pending_query

    # A fresh segment must not resurrect it.
    run_started.clear()
    controller.start()
    assert _wait_for(run_started.is_set)
    assert not engine.has_pending_query

    controller.stop()
    thread.join(timeout=2.0)


def test_session_vqa_usage_and_idle_rejection(capsys):
    with _pipe_stream() as (reader, _writer):
        session, _strategy, engine, _parent, _run_started = _make_session(reader)
        thread = _start_session_thread(session)

        session._handle_line("/vqa")
        session._handle_line("/vqa what do you see?")

        out = capsys.readouterr().out
        assert "Usage: /vqa <question>" in out
        assert "Not running" in out
        assert not engine.has_pending_query

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_vqa_answers_from_a_real_base_strategy_loop(capsys):
    """End-to-end: /vqa is answered from the live observation of a running loop."""
    from lerobot.rollout import BaseStrategy, BaseStrategyConfig

    parent = Event()
    stop_event = LinkedEvent(parent)
    engine = _FakeEngine()

    robot = MagicMock()
    robot.get_observation.return_value = {"joint.pos": 0.42}

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
                autosteer_interval_s=0.0,
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
        assert _wait_for(lambda: robot.get_observation.call_count >= 3)

        session._handle_line("/vqa is the cube in the box?")
        assert _wait_for(lambda: bool(engine.seen_query_obs))
        # The answer is delivered by the same pump call that produced it; let
        # a couple more ticks run so the print has definitely landed.
        ticks = robot.get_observation.call_count
        assert _wait_for(lambda: robot.get_observation.call_count > ticks + 2)

        out = capsys.readouterr().out
        assert "Asked: 'is the cube in the box?'" in out
        assert "Q: is the cube in the box?" in out
        assert "A: answer: is the cube in the box?" in out
        # Answered from the observation the running control loop just read.
        assert engine.seen_query_obs[0] == {"joint.pos": 0.42}

        session._handle_line("/stop")
        thread.join(timeout=2.0)
        assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Autosteer (policy-driven subtask sequencing)
# ---------------------------------------------------------------------------


def test_engine_autosteer_queries_immediately_then_waits_for_the_interval():
    engine = _FakeEngine()
    engine.start_autosteer("tidy the table", interval_s=60.0)

    # The first subtask is requested on the very next tick, not a full
    # interval later.
    engine.pump_query({"joint.pos": 0.0})
    assert engine.task == "subtask 1 of tidy the table"

    # Then nothing until the interval elapses, however many loops run.
    for _ in range(10):
        engine.pump_query({"joint.pos": 0.0})
    assert engine.task == "subtask 1 of tidy the table"

    assert engine.seen_queries == [(QueryKind.NEXT_SUBTASK, "tidy the table")]


def test_engine_autosteer_requeries_once_the_interval_elapses():
    engine = _FakeEngine()
    engine.start_autosteer("tidy the table", interval_s=0.1)

    engine.pump_query({"joint.pos": 0.0})
    assert engine.task == "subtask 1 of tidy the table"

    engine.pump_query({"joint.pos": 0.0})
    assert engine.task == "subtask 1 of tidy the table"

    time.sleep(0.15)
    engine.pump_query({"joint.pos": 0.0})
    assert engine.task == "subtask 2 of tidy the table"


def test_engine_autosteer_interval_is_measured_from_when_a_subtask_lands():
    """A generate slower than the interval must not queue the next back-to-back.

    On the sync backend generation blocks the control loop, so timing the
    interval from the *query* would leave the robot no time to act on the
    subtask it was just given.
    """
    engine = _FakeEngine()
    slow_generate = 0.15

    original = engine._generate_text

    def slow(obs_processed, query):
        time.sleep(slow_generate)
        return original(obs_processed, query)

    engine._generate_text = slow
    engine.start_autosteer("tidy the table", interval_s=0.1)

    engine.pump_query({"joint.pos": 0.0})  # generation alone outlasts the interval
    assert engine.task == "subtask 1 of tidy the table"

    # The clock started when the subtask landed, so the next query is not due.
    engine.pump_query({"joint.pos": 0.0})
    assert len(engine.seen_queries) == 1


def test_engine_autosteer_does_not_count_idle_pumps():
    """The controller's idle poll passes no observation; it is not a control loop."""
    engine = _FakeEngine()
    engine.start_autosteer("tidy the table", interval_s=0.0)

    for _ in range(5):
        engine.pump_query(None)

    assert engine.seen_queries == []
    assert engine.task == "pick up the cube"


def test_engine_autosteer_success_is_published_after_being_applied():
    """A picked subtask is applied to the task first, then announced."""
    engine = _FakeEngine()
    delivered = []
    engine.set_answer_observer(delivered.append)

    engine.start_autosteer("tidy the table", interval_s=0.0)
    engine.pump_query({"joint.pos": 0.0})

    assert engine.task == "subtask 1 of tidy the table"
    assert len(delivered) == 1
    assert delivered[0].kind is QueryKind.NEXT_SUBTASK
    assert delivered[0].ok
    assert delivered[0].answer == "subtask 1 of tidy the table"
    assert delivered[0].question == "tidy the table"


def test_engine_autosteer_stops_and_reports_on_failure():
    """A sequencer that cannot plan must stop, not re-fail every interval."""
    engine = _FakeEngine()
    engine.text_error = RuntimeError("no planner")
    delivered = []
    engine.set_answer_observer(delivered.append)

    engine.start_autosteer("tidy the table", interval_s=0.0)
    engine.pump_query({"joint.pos": 0.0})

    assert engine.autosteer_goal is None
    assert len(delivered) == 1
    assert delivered[0].kind is QueryKind.NEXT_SUBTASK
    assert not delivered[0].ok
    assert "RuntimeError: no planner" in delivered[0].error

    # Stopped means stopped: no further queries.
    engine.pump_query({"joint.pos": 0.0})
    assert len(engine.seen_queries) == 1


def test_engine_autosteer_retries_when_the_operator_query_is_in_flight():
    """A pending /vqa must not cost the sequencer its turn."""
    engine = _FakeEngine()
    engine.start_autosteer("tidy the table", interval_s=0.0)

    # An operator question occupies the single slot before the tick lands.
    assert engine.ask("what do you see?") is True
    engine._poll_autosteer({"joint.pos": 0.0})
    # The slot still holds the operator's question, not a subtask query.
    assert engine.has_pending_query
    engine.pump_query({"joint.pos": 0.0})
    assert engine.seen_queries == [(QueryKind.VQA, "what do you see?")]

    # The very next tick retries the missed subtask query.
    engine.pump_query({"joint.pos": 0.0})
    assert engine.task == "subtask 1 of tidy the table"


def test_engine_autosteer_does_not_double_queue_while_generation_is_in_flight():
    """Async backends free the query slot at claim time; the sequencer must not requeue.

    Without in-flight tracking, every control tick during the (seconds-long)
    generation would queue a duplicate NEXT_SUBTASK, so a second generation
    replaces the just-applied subtask after only generation-latency of motion.
    """
    engine = _FakeEngine()
    engine.control_thread_owns_policy = False  # RTC-style backend
    engine.start_autosteer("tidy the table", interval_s=0.0)

    engine.pump_query({"joint.pos": 0.0})  # the sequencer queues its turn
    claimed = engine._take_query()  # the async thread claims it and starts generating
    assert claimed is not None and claimed.kind is QueryKind.NEXT_SUBTASK

    # Control ticks during the in-flight generation must not queue a duplicate.
    for _ in range(5):
        engine.pump_query({"joint.pos": 0.0})
    assert not engine.has_pending_query

    # Once the answer lands, the channel frees and the next turn queues again.
    engine._publish_answer(QueryAnswer(question=claimed.text, answer="subtask", kind=claimed.kind))
    engine.pump_query({"joint.pos": 0.0})
    assert engine.has_pending_query


@pytest.mark.parametrize(
    "stop_mid_generation",
    [
        lambda engine: engine.stop_autosteer(),  # /autosteer off (or /reset, or segment end)
        lambda engine: (engine.stop_autosteer(), engine.set_task("operator override")),  # /subtask
    ],
    ids=["autosteer_off", "subtask_takeover"],
)
def test_engine_discards_subtask_generated_after_sequencer_stopped(stop_mid_generation):
    """A stale in-flight subtask must not overwrite the operator's newer intent.

    The generation runs lock-free for seconds on async backends; a /reset,
    /subtask, or /autosteer off landing in that window used to be silently
    overwritten when the generation finished — and announced as applied.
    """
    engine = _FakeEngine()
    delivered = []
    engine.set_answer_observer(delivered.append)

    original = engine._generate_text

    def generate_then_stop(obs_processed, query):
        text = original(obs_processed, query)
        stop_mid_generation(engine)  # the operator's command lands mid-generation
        return text

    engine._generate_text = generate_then_stop
    engine.start_autosteer("tidy the table", interval_s=0.0)
    engine.pump_query({"joint.pos": 0.0})

    # The stale subtask was neither applied nor announced.
    assert not engine.task.startswith("subtask ")
    assert delivered == []
    # The channel is free again (no stuck in-flight flag).
    assert engine.ask("what do you see?") is True


def test_controller_segment_end_drops_stale_autosteer_answer_but_delivers_vqa():
    """An undelivered sequencer answer must not be announced after reset/stop.

    The idle serve loop pumps to deliver answers that landed just as the
    segment ended — that is meant for operator questions, not for turns of a
    sequencer the segment end already stopped.
    """
    answers = []
    controller, _events, _strategy, engine, _parent, run_started = _make_controller(answers=answers)
    thread = _serve_thread(controller)

    controller.start()
    assert _wait_for(run_started.is_set)
    assert controller.autosteer("tidy the table") is AskResult.QUEUED

    # The async backend publishes a sequencer answer and a VQA answer that
    # the (never-pumping) segment leaves undelivered.
    engine._publish_answer(
        QueryAnswer(question="tidy the table", answer="subtask 1", kind=QueryKind.NEXT_SUBTASK)
    )
    engine._publish_answer(QueryAnswer(question="is the cube in the box?", answer="yes", kind=QueryKind.VQA))

    controller.reset()
    # The idle pump delivers the operator's answer but not the stale turn.
    assert _wait_for(lambda: any(a.kind is QueryKind.VQA for a in answers))
    assert not any(a.kind is QueryKind.NEXT_SUBTASK for a in answers)

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_autosteer_rejected_while_idle():
    controller, _events, _strategy, engine, _parent, _run_started = _make_controller()
    thread = _serve_thread(controller)

    assert controller.autosteer("tidy the table") is AskResult.NOT_RUNNING
    assert engine.autosteer_goal is None

    controller.stop()
    thread.join(timeout=2.0)


def test_controller_autosteer_stopped_by_reset_set_task_and_segment_end():
    started = Event()
    controller, _events, _strategy, engine, _parent, _run_started = _make_controller(
        run_behavior=_pumping_run(started)
    )
    thread = _serve_thread(controller)
    controller.start()
    assert _wait_for(started.is_set)

    # The sequencer drives the task on its own.
    assert controller.autosteer("tidy the table") is AskResult.QUEUED
    assert _wait_for(lambda: controller.task.startswith("subtask "))

    # Taking the wheel by hand stops it, and the hand-set task sticks.
    controller.set_task("put the cube down")
    assert engine.autosteer_goal is None
    time.sleep(0.05)
    assert controller.task == "put the cube down"

    # A segment end stops it too: restarting resets the policy, and the
    # plan's progress lives there.
    assert controller.autosteer("tidy the table") is AskResult.QUEUED
    assert _wait_for(lambda: engine.autosteer_goal is not None)
    controller.reset()
    assert _wait_for(lambda: engine.autosteer_goal is None)
    assert controller.task == controller.initial_task

    controller.stop()
    thread.join(timeout=2.0)


def test_session_autosteer_reports_status_and_hands_control_back(capsys):
    started = Event()
    with _pipe_stream() as (reader, _writer):
        session, _strategy, engine, _parent, _run_started = _make_session(
            reader, run_behavior=_pumping_run(started)
        )
        thread = _start_session_thread(session)

        session._handle_line("/autosteer tidy the table")
        assert "Not running" in capsys.readouterr().out

        session._handle_line("/start")
        assert _wait_for(started.is_set)

        session._handle_line("/autosteer tidy the table")
        assert _wait_for(lambda: session.controller.task.startswith("subtask "))
        chat = capsys.readouterr().out
        assert "Autosteer on — goal 'tidy the table'" in chat

        # Each picked subtask is announced in the chat (the announcement is
        # delivered by the pump that applied it, so poll a few more ticks).
        def _subtask_announced():
            nonlocal chat
            chat += capsys.readouterr().out
            return "Autosteer subtask: 'subtask " in chat

        assert _wait_for(_subtask_announced)

        session._handle_line("/autosteer")
        assert "Autosteer on — goal 'tidy the table'." in capsys.readouterr().out

        session._handle_line("/autosteer off")
        out = capsys.readouterr().out
        assert "Autosteer off (was 'tidy the table')" in out
        assert engine.autosteer_goal is None

        # The last subtask stays in effect once the sequencer is off.
        steady = session.controller.task
        time.sleep(0.05)
        assert session.controller.task == steady

        session._handle_line("/stop")
        thread.join(timeout=2.0)


def test_session_subtask_takes_over_from_autosteer(capsys):
    started = Event()
    with _pipe_stream() as (reader, _writer):
        session, _strategy, engine, _parent, _run_started = _make_session(
            reader, run_behavior=_pumping_run(started)
        )
        thread = _start_session_thread(session)

        session._handle_line("/start")
        assert _wait_for(started.is_set)
        session._handle_line("/autosteer tidy the table")
        assert _wait_for(lambda: session.controller.task.startswith("subtask "))
        capsys.readouterr()

        session._handle_line("/subtask put the cube down")
        out = capsys.readouterr().out
        assert "Autosteer off (was 'tidy the table')" in out
        assert engine.autosteer_goal is None
        assert session.controller.task == "put the cube down"

        session._handle_line("/stop")
        thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_interactive_rejects_keyboard_bound_strategies():
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import HighlightStrategyConfig, RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match="--interactive=true supports"):
        RolloutConfig(
            robot=MockRobotConfig(),
            strategy=HighlightStrategyConfig(),
            dataset=DatasetRecordConfig(repo_id="user/rollout_test", single_task="test"),
            interactive=True,
        )


def test_interactive_allows_sentry():
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import RolloutConfig, SentryStrategyConfig
    from tests.mocks.mock_robot import MockRobotConfig

    cfg = RolloutConfig(
        robot=MockRobotConfig(),
        strategy=SentryStrategyConfig(),
        dataset=DatasetRecordConfig(repo_id="user/rollout_test", single_task="test"),
        policy=SimpleNamespace(device="cpu"),  # stands in for a PreTrainedConfig
        interactive=True,
    )
    assert cfg.interactive is True
    assert cfg.dataset.streaming_encoding is True  # sentry forces streaming
