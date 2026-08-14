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
from threading import Event, Thread, current_thread
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout import (  # noqa: E402
    AskResult,
    BaseStrategyConfig,
    InferenceEngine,
    InteractiveSession,
    LinkedEvent,
    QueryAnswer,
    QueryKind,
    RolloutController,
    RolloutEvent,
    RolloutStrategy,
)

# Front-end and engine internals, imported from their defining modules.
from lerobot.rollout.inference import PolicyQuery  # noqa: E402
from lerobot.rollout.interactive import InteractiveCommand, parse_command  # noqa: E402


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


def _join_session(thread: Thread) -> None:
    """Join a session/serve/strategy thread; a hung one leaks process-wide log muting."""
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "thread did not exit — a hung session leaks process-wide log muting"


# --- Command parser ---


def test_parse_command():
    cmd = parse_command("/start")
    assert cmd is not None
    assert (cmd.name, cmd.args) == ("start", "")

    # Case-folded, stripped, and tab- or space-separated args all land the same way.
    assert parse_command("  /SubTask Grab the red cube  ") == InteractiveCommand(
        name="subtask", args="Grab the red cube"
    )
    assert parse_command("/subtask\tgrab the cube") == InteractiveCommand(
        name="subtask", args="grab the cube"
    )

    for line in ("hello robot", "", "   ", "/", "/ start"):
        assert parse_command(line) is None


# --- LinkedEvent ---


def test_linked_event_local_and_parent_flags():
    parent = Event()
    event = LinkedEvent(parent)
    assert not event.is_set()

    event.set()
    assert event.is_set()
    assert not parent.is_set()  # the local flag never leaks upward
    event.clear()
    assert not event.is_set()

    parent.set()
    assert event.is_set()
    event.clear()
    assert event.is_set()  # clearing the local flag never masks the parent


def test_linked_event_wait():
    parent = Event()
    event = LinkedEvent(parent)
    assert event.wait(timeout=0.05) is False

    parent.set()
    assert event.wait(timeout=0.05) is True

    parent.clear()
    event.set()
    assert event.wait(timeout=0.05) is True


# --- Shared fakes ---


class _FakeEngine(InferenceEngine):
    """Real task-holder and query-channel semantics with mocked lifecycle methods."""

    failed = False
    failure_traceback = None
    # Plain attributes shadowing the base properties, so tests can flip them.
    supports_text_queries = True
    # Answer inline on the caller's thread like the sync backend, exercising pump_query.
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


def _loop_ctx(engine, robot=None, stop_event=None, **cfg_overrides):
    """The rollout context a real control loop needs, with identity processors."""

    def identity(x):
        return x

    cfg = {
        "play_sounds": False,
        "fps": 100.0,
        "duration": 0.0,
        "use_torch_compile": False,
        "interpolation_multiplier": 1,
        "display_data": False,
        "autosteer_interval_s": 0.0,
    }
    cfg.update(cfg_overrides)
    return SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(**cfg),
            shutdown_event=stop_event if stop_event is not None else LinkedEvent(Event()),
            cadence_report=None,
        ),
        policy=SimpleNamespace(inference=engine),
        hardware=SimpleNamespace(
            robot_wrapper=robot if robot is not None else MagicMock(),
            teleop=None,
            initial_position={"joint.pos": 0.0},
        ),
        processors=SimpleNamespace(
            teleop_action_processor=identity,
            robot_action_processor=identity,
            robot_observation_processor=identity,
        ),
        data=SimpleNamespace(dataset=None, dataset_features={}, hw_features={}, ordered_action_keys=[]),
    )


def _make_ctx(run_behavior=None):
    """A mock strategy plus the fake context the controller needs."""
    parent = Event()
    engine = _FakeEngine()
    ctx = _loop_ctx(engine, stop_event=LinkedEvent(parent))

    # Autospec so a strategy signature drift fails here, not on real hardware during /reset.
    strategy = create_autospec(RolloutStrategy, instance=True)
    strategy.config = BaseStrategyConfig()  # the controller enforces supports_interactive on it
    run_started = Event()

    def default_run(c):
        run_started.set()
        while not c.runtime.shutdown_event.is_set():
            time.sleep(0.005)

    strategy.run.side_effect = run_behavior or default_run
    return ctx, strategy, engine, parent, run_started


# --- RolloutController (the programmatic API) ---


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
    """Wrapping a one-shot strategy would finalize the dataset on the first segment."""
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

    # A start() while a segment runs is refused, not queued.
    assert controller.start() is False

    # So is a second question while the first is unanswered (this run never pumps).
    assert controller.ask("what do you see?") is AskResult.QUEUED
    assert controller.ask("and now?") is AskResult.BUSY

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
    _join_session(thread)
    strategy.teardown.assert_not_called()
    assert events[-1] is RolloutEvent.STOPPED


def test_controller_last_command_wins_over_a_pending_start():
    """Robot safety: a queued start must never fire after a later reset or stop."""
    controller, _events, strategy, _engine, _parent, _run_started = _make_controller()
    assert controller.start() is True  # queued; no serve thread has consumed it yet
    controller.reset()
    thread = _serve_thread(controller)
    assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)
    time.sleep(0.05)
    strategy.run.assert_not_called()
    controller.stop()
    _join_session(thread)

    controller, _events, strategy, _engine, _parent, _run_started = _make_controller()
    controller.start()
    controller.stop()
    assert not controller._start_requested.is_set()  # serve() breaks before it could fire
    _join_session(_serve_thread(controller))
    strategy.run.assert_not_called()

    # A start the serve loop already consumed still aborts when a failure raced it.
    controller, _events, strategy, engine, _parent, _run_started = _make_controller()
    engine.failed = True
    controller._running.set()  # as serve() does, under the control lock
    controller._run_segment()
    strategy.run.assert_not_called()
    assert not controller.running


def test_controller_is_one_shot_and_refuses_commands_after_stop():
    controller, _events, strategy, engine, _parent, _run_started = _make_controller()
    thread = _serve_thread(controller)
    controller.stop()
    _join_session(thread)
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
    """A strategy.run() exception must not masquerade as a deliberate stop."""

    def exploding_run(c):
        raise RuntimeError("robot io broke")

    controller, events, _strategy, _engine, _parent, _run_started = _make_controller(exploding_run)
    thread = _serve_thread(controller)
    controller.start()
    _join_session(thread)  # serve() ended cleanly instead of dying with the exception

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
    _join_session(thread)


def test_controller_engine_failure_emits_event():
    def failing_run(c):
        c.policy.inference.failed = True
        c.policy.inference.failure_traceback = "RuntimeError: boom-traceback"
        c.runtime.shutdown_event.set()

    controller, events, strategy, _engine, _parent, _run_started = _make_controller(failing_run)
    thread = _serve_thread(controller)
    controller.start()
    _join_session(thread)
    strategy.return_to_initial_position.assert_not_called()
    assert RolloutEvent.ENGINE_FAILED in events
    assert controller.failed
    assert controller.failure_traceback == "RuntimeError: boom-traceback"


def test_controller_segment_ended_event_on_natural_end():
    def finite_run(c):
        return None  # e.g. --duration elapsed

    controller, events, strategy, _engine, parent, _run_started = _make_controller(finite_run)
    thread = _serve_thread(controller)

    controller.start()
    assert _wait_for(lambda: RolloutEvent.SEGMENT_ENDED in events)
    assert thread.is_alive()  # back to idle, not shut down
    assert not controller.running

    parent.set()  # Ctrl-C on the parent run: serve() exits like a non-interactive run
    _join_session(thread)
    assert controller.stopped


# --- InteractiveSession (the stdin CLI front-end) ---


def _make_session(input_stream, run_behavior=None):
    ctx, strategy, engine, parent, run_started = _make_ctx(run_behavior)
    session = InteractiveSession(strategy, ctx, input_stream=input_stream)
    return session, strategy, engine, parent, run_started


def _start_session_thread(session) -> Thread:
    thread = Thread(target=session.run, daemon=True)
    thread.start()
    return thread


def test_session_flow_over_the_command_stream():
    """/start, /reset, /start, /stop through the pipe and the stdin listener thread."""
    with _pipe_stream() as (reader, writer):
        session, strategy, engine, _parent, run_started = _make_session(reader)
        thread = _start_session_thread(session)

        def send(line):
            writer.write(f"{line}\n")
            writer.flush()

        # Idle until /start: the strategy loop must not run on its own.
        time.sleep(0.05)
        strategy.run.assert_not_called()

        send("/start")
        assert _wait_for(run_started.is_set)
        assert strategy.reset_control_state.call_count == 1

        send("/reset")
        assert _wait_for(lambda: strategy.return_to_initial_position.call_count == 1)
        assert engine.pause.call_count >= 1
        assert thread.is_alive()

        run_started.clear()
        send("/start")
        assert _wait_for(run_started.is_set)
        assert strategy.run.call_count == 2
        assert strategy.reset_control_state.call_count == 2

        # /stop ends the session; teardown stays with the caller (the CLI script).
        send("/stop")
        _join_session(thread)
        strategy.teardown.assert_not_called()


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
        _join_session(thread)


def test_session_eof_stops_session():
    with _pipe_stream() as (reader, writer):
        session, strategy, _engine, _parent, _run_started = _make_session(reader)
        thread = _start_session_thread(session)
        writer.close()  # EOF on the command stream
        _join_session(thread)
        strategy.run.assert_not_called()


def test_session_mutes_logs_below_error_and_restores_on_exit():
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
    logging.disable(logging.DEBUG)  # an embedding application's own gate
    try:
        with _pipe_stream() as (reader, _writer):
            session, _strategy, _engine, _parent, _run_started = _make_session(reader)
            thread = _start_session_thread(session)
            assert _wait_for(lambda: logging.root.manager.disable == logging.WARNING)

            lib_logger.info("muted-info")
            lib_logger.warning("muted-warning")
            lib_logger.error("visible-error")  # errors must surface mid-session

            session._handle_line("/stop")
            _join_session(thread)

        output = lib_stream.getvalue()
        assert "muted-info" not in output
        assert "muted-warning" not in output
        assert "visible-error" in output

        # The application's own disable level and the warning filters come back.
        assert logging.root.manager.disable == logging.DEBUG
        assert len(warnings.filters) == n_warning_filters
        lib_logger.info("post-session-info")
        assert "post-session-info" in lib_stream.getvalue()
    finally:
        lib_logger.removeHandler(lib_handler)
        # An assertion failure mid-test must not leak the process-wide gate.
        logging.disable(logging.NOTSET)


def test_session_drives_real_base_strategy_and_answers_vqa(capsys):
    """End-to-end with a real BaseStrategy control loop (only hardware/engine mocked)."""
    from lerobot.rollout import BaseStrategy, BaseStrategyConfig

    engine = _FakeEngine()
    engine.get_action.return_value = None  # no action ready; the loop still ticks
    robot = MagicMock()
    robot.get_observation.return_value = {"joint.pos": 0.42}
    ctx = _loop_ctx(engine, robot)

    strategy = BaseStrategy(BaseStrategyConfig())
    strategy.setup(ctx)
    strategy.return_to_initial_position = MagicMock()  # skip the 3s hardware sweep

    with _pipe_stream() as (reader, _writer):
        session = InteractiveSession(strategy, ctx, input_stream=reader)
        thread = _start_session_thread(session)

        session._handle_line("/start")
        assert _wait_for(lambda: engine.resume.called)
        assert _wait_for(lambda: robot.get_observation.call_count >= 3)

        session._handle_line("/vqa is the cube in the box?")
        assert _wait_for(lambda: bool(engine.seen_query_obs))
        # The answer is delivered by the pump that produced it: let a few more ticks run.
        ticks = robot.get_observation.call_count
        assert _wait_for(lambda: robot.get_observation.call_count > ticks + 2)
        out = capsys.readouterr().out
        assert "Asked: 'is the cube in the box?'" in out
        assert "A: answer: is the cube in the box?" in out
        # Answered from the observation the running control loop just read.
        assert engine.seen_query_obs[0] == {"joint.pos": 0.42}

        session._handle_line("/reset")
        assert _wait_for(lambda: strategy.return_to_initial_position.called)
        assert engine.pause.called
        assert thread.is_alive()
        # The per-segment timer reports through the session's sink, despite the muted logs.
        assert "Cadence summary — whole run" in capsys.readouterr().out

        session._handle_line("/start")
        assert _wait_for(lambda: engine.resume.call_count >= 2)

        session._handle_line("/stop")
        _join_session(thread)

    assert ctx.runtime.cadence_report is None  # the sink is scoped to the session


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

        session._handle_line('/subtask "fold the towel"')  # quotes are stripped
        assert engine.task == "fold the towel"

        session._handle_line("/stop")
        _join_session(thread)


# --- InferenceEngine task holder (the /subtask plumbing) ---


def test_sync_engine_uses_new_task_and_flushes_precomputed_actions():
    """A /subtask switch must reach the policy and drop stale queued actions."""
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


def test_drop_queued_actions_clears_the_queues_dict_convention():
    """The /subtask fast switch relies on the smolvla-style ``_queues`` dict too, not only
    on act/pi0's ``_action_queue`` (covered against a real policy in the policy tests)."""
    from collections import deque

    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.utils.constants import ACTION

    policy = SimpleNamespace(  # smolvla / diffusion / vqbet / wall_x style
        _action_queue_attrs=PreTrainedPolicy._action_queue_attrs,
        _queues={ACTION: deque([1, 2, 3]), "observation.state": deque([9])},
    )
    PreTrainedPolicy.drop_queued_actions(policy)
    assert len(policy._queues[ACTION]) == 0
    assert len(policy._queues["observation.state"]) == 1  # other episode state is left alone


# --- Sentry strategy: restartable run() segments + dispatched-action task labels ---


def _make_sentry(monkeypatch):
    from lerobot.rollout import SentryStrategy, SentryStrategyConfig

    # Keep setup independent of camera features and never rotate episodes mid-test.
    monkeypatch.setattr("lerobot.rollout.strategies.sentry.estimate_max_episode_seconds", lambda *a, **k: 1e9)

    def _fake_send(obs_processed, obs_raw, ctx, interpolator, timer=None):
        # Recording is gated on ``interpolator.emitted_policy_action``, so push an action too.
        if interpolator.needs_new_action():
            interpolator.add(torch.zeros(1))
        interpolator.get()
        return {"joint.pos": 0.5}

    monkeypatch.setattr("lerobot.rollout.strategies.sentry.send_next_action", _fake_send)
    monkeypatch.setattr("lerobot.rollout.strategies.sentry.build_dataset_frame", lambda *a, **k: {})

    engine = _FakeEngine()
    dataset = MagicMock()
    robot = MagicMock()
    robot.get_observation.return_value = {"joint.pos": 0.0}
    stop_event = Event()

    ctx = _loop_ctx(
        engine,
        robot,
        stop_event=stop_event,
        fps=200.0,
        return_to_initial_position=False,
        task="pick up the cube",
        dataset=SimpleNamespace(push_to_hub=False, tags=None, private=False),
    )
    ctx.data.dataset = dataset

    strategy = SentryStrategy(SentryStrategyConfig())
    strategy.setup(ctx)
    return strategy, ctx, dataset, engine, stop_event


def test_sentry_run_is_restartable_and_finalizes_only_in_teardown(monkeypatch):
    strategy, ctx, dataset, _engine, stop_event = _make_sentry(monkeypatch)
    strategy.config.upload_every_n_episodes = 2
    pushes = []
    monkeypatch.setattr(strategy, "_background_push", lambda dataset, cfg: pushes.append(1))

    thread = Thread(target=strategy.run, args=(ctx,), daemon=True)
    thread.start()
    assert _wait_for(lambda: dataset.add_frame.call_count >= 2)
    stop_event.set()
    _join_session(thread)

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
    _join_session(thread)
    assert dataset.save_episode.call_count == 2
    dataset.finalize.assert_not_called()
    assert pushes == [1]  # segment-end tail saves count toward the upload cadence

    # Only teardown finalizes.
    strategy.teardown(ctx)
    dataset.finalize.assert_called_once()


def test_sentry_labels_frames_with_dispatched_action_task(monkeypatch):
    strategy, ctx, dataset, engine, stop_event = _make_sentry(monkeypatch)

    thread = Thread(target=strategy.run, args=(ctx,), daemon=True)
    thread.start()
    assert _wait_for(lambda: dataset.add_frame.call_count >= 2)

    frames_before_switch = dataset.add_frame.call_count
    # Frames keep the old label until an action generated under the new one is dispatched.
    engine.set_task("fold the towel")
    assert _wait_for(lambda: dataset.add_frame.call_count >= frames_before_switch + 2)
    tasks = [call.args[0]["task"] for call in dataset.add_frame.call_args_list]
    assert tasks[-1] == "pick up the cube"

    engine._set_dispatched_task("fold the towel")
    assert _wait_for(
        lambda: any(call.args[0]["task"] == "fold the towel" for call in dataset.add_frame.call_args_list)
    )
    stop_event.set()
    _join_session(thread)

    tasks = [call.args[0]["task"] for call in dataset.add_frame.call_args_list]
    assert tasks[0] == "pick up the cube"
    assert tasks[-1] == "fold the towel"

    strategy.teardown(ctx)


def test_sentry_tail_save_failure_poisons_dataset_and_reraises(monkeypatch):
    """A failed save_episode may have committed rows already; sentry must fail loudly."""
    strategy, ctx, dataset, _engine, stop_event = _make_sentry(monkeypatch)
    dataset.save_episode.side_effect = ValueError("disk full mid-write")
    stop_event.set()  # the segment ends on its first tick; the tail save fails

    with pytest.raises(ValueError, match="disk full mid-write"):
        strategy.run(ctx)

    # Cancel the in-flight streaming encode and discard the half-mutated buffer.
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


def test_sentry_queued_push_rechecks_poison_before_uploading(monkeypatch):
    """A push that passed the submit-time check must re-check before uploading."""
    strategy, ctx, dataset, _engine, _stop_event = _make_sentry(monkeypatch)

    # Holding the episode lock parks the submitted push at its lock acquisition,
    # exactly like a still-running previous push would.
    with strategy._episode_lock:
        strategy._background_push(dataset, ctx.runtime.cfg)
        strategy._dataset_poisoned = True  # a save fails while the push waits
    strategy._push_executor.shutdown(wait=True)

    dataset.push_to_hub.assert_not_called()


def test_sentry_empty_tail_segment_skips_the_save(monkeypatch):
    """Saving an empty buffer would fail, and a failed save poisons the dataset for good."""
    strategy, ctx, dataset, _engine, stop_event = _make_sentry(monkeypatch)
    dataset.has_pending_frames.return_value = False
    stop_event.set()  # /start followed by /reset before a frame was recorded

    strategy.run(ctx)

    dataset.save_episode.assert_not_called()


# --- Text queries (/vqa) ---


def _pumping_run(started: Event):
    """A fake control loop that pumps the engine's query channel every tick."""

    def run(ctx):
        started.set()
        while not ctx.runtime.shutdown_event.is_set():
            ctx.policy.inference.pump_query({"joint.pos": 0.0})
            time.sleep(0.005)

    return run


def test_engine_query_channel_holds_one_question_and_answers_it_once():
    engine = _FakeEngine()
    delivered = []
    engine.set_answer_observer(delivered.append)
    assert not engine.has_pending_query

    assert engine.ask("what do you see?") is True
    assert engine.has_pending_query
    # A second question must not silently displace the unanswered one.
    assert engine.ask("and now?") is False

    # The controller's idle poll pumps with no observation: the question waits.
    engine.pump_query(None)
    assert engine.has_pending_query
    assert engine.seen_query_obs == []

    dropped = engine.drop_pending_query()
    assert (dropped.kind, dropped.text) == (QueryKind.VQA, "what do you see?")
    assert not engine.has_pending_query
    assert engine.drop_pending_query() is None

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
    """Without central validation, str() coercion would turn a forgotten return into
    the live task 'None' — steering the robot and labeling frames with it."""
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


def test_engine_undelivered_answers_queue_up_instead_of_being_overwritten():
    """Two answers landing between pumps must both reach the observer.

    On an async backend the query slot frees at claim time, so a second /vqa can be
    accepted and answered while the control thread is not pumping.
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


def test_query_reaches_the_preprocessor_as_complementary_data():
    """``batch_to_transition`` only forwards allowlisted keys, so an unregistered
    query key would be silently dropped before the policy's preprocessor step."""
    from lerobot.lerobot_types import TransitionKey
    from lerobot.processor.converters import batch_to_transition
    from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT

    query = PolicyQuery(kind=QueryKind.VQA, text="is the cube in the box?")
    batch = InferenceEngine._mark_query({"observation.state": np.zeros(2), "task": "tidy"}, query)
    complementary = batch_to_transition(batch)[TransitionKey.COMPLEMENTARY_DATA.value]

    assert complementary[QUERY_KIND] == QueryKind.VQA.value
    assert complementary[QUERY_TEXT] == "is the cube in the box?"
    assert complementary["task"] == "tidy"  # they land beside the task


def test_controller_ask_answers_via_event():
    started = Event()
    answers = []
    controller, events, _strategy, engine, _parent, _run_started = _make_controller(
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

    # A policy without a text head is refused up front, not one tick later.
    engine.supports_text_queries = False
    assert controller.ask("what do you see?") is AskResult.UNSUPPORTED
    assert controller.autosteer("tidy the table") is AskResult.UNSUPPORTED

    controller.stop()
    _join_session(thread)


def test_controller_segment_end_drops_stale_subtask_answer_and_reports_the_pending_question():
    """The idle pump must not announce a subtask for a sequencer the segment end stopped, and
    an operator question the segment never served must come back as an error."""
    answers = []
    controller, _events, _strategy, engine, _parent, run_started = _make_controller(answers=answers)
    thread = _serve_thread(controller)

    controller.start()
    assert _wait_for(run_started.is_set)
    # Answers that landed just as the segment ended, left undelivered (this run never pumps).
    engine._publish_answer(QueryAnswer(question="tidy up", answer="s1", kind=QueryKind.NEXT_SUBTASK))
    engine._publish_answer(QueryAnswer(question="is the cube in?", answer="yes", kind=QueryKind.VQA))
    assert controller.ask("what do you see?") is AskResult.QUEUED

    controller.reset()
    assert _wait_for(lambda: len(answers) >= 2)
    assert not any(a.kind is QueryKind.NEXT_SUBTASK for a in answers)
    assert not engine.has_pending_query
    unserved = next(a for a in answers if a.question == "what do you see?")
    assert not unserved.ok and "run ended" in unserved.error

    controller.stop()
    _join_session(thread)


# --- Autosteer (policy-driven subtask sequencing) ---


class _FakeClock:
    """Deterministic stand-in for the engine module's perf_counter."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def engine_clock(monkeypatch):
    """Replace the engine module's clock so interval tests cannot race real time."""
    clock = _FakeClock()
    monkeypatch.setattr("lerobot.rollout.inference.base.time", SimpleNamespace(perf_counter=clock))
    return clock


def test_engine_autosteer_interval_is_measured_from_when_a_subtask_lands(engine_clock):
    """On the sync backend generation blocks the control loop, so timing the interval
    from the *query* would leave the robot no time to act on the subtask."""
    engine = _FakeEngine()
    original = engine._generate_text

    def slow(obs_processed, query):
        engine_clock.advance(15.0)  # generation alone outlasts the 10 s interval
        return original(obs_processed, query)

    engine._generate_text = slow
    engine.start_autosteer("tidy the table", interval_s=10.0)

    engine.pump_query(None)  # the controller's idle poll: no observation, no turn
    assert not engine.has_pending_query

    # The first subtask is requested on the very next tick, not an interval later.
    engine.pump_query({"joint.pos": 0.0})
    assert engine.task == "subtask 1 of tidy the table"

    # The clock started when the subtask landed, so the next query is not due.
    engine_clock.advance(9.9)
    for _ in range(5):
        engine.pump_query({"joint.pos": 0.0})
    assert len(engine.seen_queries) == 1

    # After a full interval of (fake) robot motion, it is.
    engine_clock.advance(0.2)
    engine.pump_query({"joint.pos": 0.0})
    assert len(engine.seen_queries) == 2
    assert engine.task == "subtask 2 of tidy the table"


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

    engine.pump_query({"joint.pos": 0.0})  # stopped means stopped
    assert len(engine.seen_queries) == 1


def test_engine_autosteer_does_not_double_queue_while_generation_is_in_flight():
    """Async backends free the query slot at claim time, so without in-flight tracking every tick
    of a seconds-long generation would queue a duplicate turn."""
    engine = _FakeEngine()
    engine.control_thread_owns_policy = False  # RTC-style backend
    engine.start_autosteer("tidy the table", interval_s=0.0)

    engine.pump_query({"joint.pos": 0.0})  # the sequencer queues its turn
    claimed = engine._take_query()  # the async thread claims it and starts generating
    assert claimed is not None and claimed.kind is QueryKind.NEXT_SUBTASK

    engine.pump_query({"joint.pos": 0.0})  # a control tick during the in-flight generation
    assert not engine.has_pending_query

    # Once the answer lands the channel frees, and the next turn queues again.
    engine._publish_answer(QueryAnswer(question=claimed.text, answer="subtask", kind=claimed.kind))
    engine.pump_query({"joint.pos": 0.0})
    assert engine.has_pending_query


@pytest.mark.parametrize(
    "stop_mid_generation",
    [
        lambda engine: engine.stop_autosteer(),  # /autosteer off (or /reset, or segment end)
        lambda engine: (engine.stop_autosteer(), engine.set_task("operator override")),  # /subtask
        # The same liveness rule on the failure path: a stale turn's failure must
        # not be announced, and must still free the channel.
        lambda engine: (engine.stop_autosteer(), setattr(engine, "text_error", RuntimeError("late"))),
    ],
    ids=["autosteer_off", "subtask_takeover", "stale_turn_failure"],
)
def test_engine_discards_subtask_generated_after_sequencer_stopped(stop_mid_generation):
    """A stale in-flight subtask must not overwrite the operator's newer intent."""
    engine = _FakeEngine()
    delivered = []
    engine.set_answer_observer(delivered.append)

    original = engine._generate_text

    def generate_then_stop(obs_processed, query):
        stop_mid_generation(engine)  # the operator's command lands mid-generation
        return original(obs_processed, query)

    engine._generate_text = generate_then_stop
    engine.start_autosteer("tidy the table", interval_s=0.0)
    engine.pump_query({"joint.pos": 0.0})

    # The stale subtask was neither applied nor announced.
    assert not engine.task.startswith("subtask ")
    assert delivered == []
    # The channel is free again (no stuck in-flight flag).
    assert engine.ask("what do you see?") is True


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

    # A segment end stops it too: restarting resets the policy, where the plan lives.
    assert controller.autosteer("tidy the table") is AskResult.QUEUED
    assert _wait_for(lambda: engine.autosteer_goal is not None)
    controller.reset()
    assert _wait_for(lambda: engine.autosteer_goal is None)
    assert controller.task == controller.initial_task

    controller.stop()
    _join_session(thread)


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

        # Each picked subtask is announced in the chat (by the pump that applied it).
        def _subtask_announced():
            nonlocal chat
            chat += capsys.readouterr().out
            return "Autosteer subtask: 'subtask " in chat

        assert _wait_for(_subtask_announced)

        session._handle_line("/autosteer")  # bare: report the goal without changing it
        assert "Autosteer on — goal 'tidy the table'.\n" in capsys.readouterr().out

        session._handle_line("/autosteer off")
        assert "Autosteer off (was 'tidy the table')" in capsys.readouterr().out
        assert engine.autosteer_goal is None

        # The last subtask stays in effect once the sequencer is off.
        steady = session.controller.task
        time.sleep(0.05)
        assert session.controller.task == steady

        session._handle_line("/stop")
        _join_session(thread)


# --- RTC inference engine: the async half of the engine contract ---


class _IdentityPipeline:
    """Stand-in for PolicyProcessorPipeline: no steps, identity transform."""

    steps = ()

    def __call__(self, batch):
        return batch

    def reset(self):
        pass


class _StubChunkPolicy:
    """A chunk policy whose inferences the test releases one at a time, so the RTC
    background loop's timing is deterministic."""

    def __init__(self, chunk_len: int = 10, action_dim: int = 2):
        from threading import Semaphore

        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self._release = Semaphore(0)
        self.in_inference = Event()
        self.predicted_tasks: list[str] = []
        self.generate_thread_names: list[str] = []

    def allow_one_inference(self) -> None:
        self._release.release()

    def unblock(self) -> None:
        """Wake any predict parked on the gate so teardown never stalls."""
        self._release.release(3)

    def predict_action_chunk(self, batch, inference_delay=0, prev_chunk_left_over=None):
        self.in_inference.set()
        released = self._release.acquire(timeout=2.0)
        self.in_inference.clear()
        if not released:
            raise TimeoutError("test never released the inference gate")
        self.predicted_tasks.append(batch["task"][0])
        return torch.zeros(1, self.chunk_len, self.action_dim)

    def reset(self):
        pass

    def supports_text_generation(self):
        return True

    def generate_text(self, batch):
        from lerobot.utils.constants import QUERY_TEXT

        self.generate_thread_names.append(current_thread().name)
        return f"answer: {batch[QUERY_TEXT]}"


def _make_rtc_engine(rtc_queue_threshold: int = 30, chunk_len: int = 10):
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference import RTCInferenceEngine

    policy = _StubChunkPolicy(chunk_len=chunk_len)
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=_IdentityPipeline(),
        postprocessor=_IdentityPipeline(),
        robot_wrapper=SimpleNamespace(robot_type="mock", action_features={}),
        rtc_config=RTCConfig(enabled=True, execution_horizon=8, max_guidance_weight=1.0),
        dataset_features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
        },
        ordered_action_keys=["j1.pos", "j2.pos"],
        task="task A",
        fps=30.0,
        device="cpu",
        rtc_queue_threshold=rtc_queue_threshold,
    )
    return engine, policy


_RTC_OBS = {"j1.pos": 0.0, "j2.pos": 0.0}


@contextlib.contextmanager
def _running_rtc_engine(rtc_queue_threshold: int = 30, chunk_len: int = 10):
    engine, policy = _make_rtc_engine(rtc_queue_threshold=rtc_queue_threshold, chunk_len=chunk_len)
    engine.start()
    try:
        engine.resume()
        engine.notify_observation(dict(_RTC_OBS))
        yield engine, policy
    finally:
        policy.unblock()
        engine.stop()


def test_rtc_engine_dispatched_task_tracks_chunk_provenance_across_set_task():
    """``dispatched_task`` may only advance to the new instruction when an action from a
    chunk conditioned on it is popped — that is what frame provenance rests on."""
    with _running_rtc_engine() as (engine, policy):
        policy.allow_one_inference()  # first chunk, conditioned on "task A"
        assert _wait_for(lambda: len(policy.predicted_tasks) == 1)
        assert _wait_for(lambda: engine.get_action(None) is not None)
        assert engine.dispatched_task == "task A"

        # The loop is parked inside the next predict, with that chunk's task (still A)
        # already taken, so nothing new can merge until we allow it.
        assert _wait_for(policy.in_inference.is_set)
        assert engine.set_task("task B") is True

        # Leftovers still serve under the old label.
        assert engine.get_action(None) is not None
        assert engine.dispatched_task == "task A"

        policy.allow_one_inference()  # the in-flight chunk (still task A) lands
        policy.allow_one_inference()  # the next chunk is conditioned on task B
        assert _wait_for(lambda: len(policy.predicted_tasks) == 3)
        assert policy.predicted_tasks == ["task A", "task A", "task B"]

        def _dispatched_b():
            return engine.get_action(None) is not None and engine.dispatched_task == "task B"

        assert _wait_for(_dispatched_b)


def test_rtc_engine_reset_discards_chunk_from_inflight_inference(caplog):
    """Reset isolation: without it the first actions after the next /start jerk the
    robot toward the pre-reset pose."""
    with _running_rtc_engine() as (engine, policy):
        assert _wait_for(policy.in_inference.is_set)  # inference in flight

        engine.reset()  # bumps the epoch, clears queue and observation

        with caplog.at_level(logging.INFO, logger="lerobot.rollout.inference.rtc"):
            policy.allow_one_inference()  # the stale chunk completes now
            assert _wait_for(lambda: any("Discarding action chunk" in r.getMessage() for r in caplog.records))
        assert engine.action_queue.qsize() == 0
        assert engine.get_action(None) is None

        # The engine is not dead: a fresh observation produces a fresh chunk.
        engine.notify_observation(dict(_RTC_OBS))
        assert _wait_for(policy.in_inference.is_set)
        policy.allow_one_inference()
        assert _wait_for(lambda: engine.get_action(None) is not None)


def test_rtc_engine_answers_vqa_on_rtc_thread_delivered_by_control_pump():
    """The RTC thread generates the answer; only the control thread's pump delivers it."""
    # rtc_queue_threshold=-1 parks the chunk path, so the loop's only work is the query.
    with _running_rtc_engine(rtc_queue_threshold=-1) as (engine, policy):
        delivered = []
        engine.set_answer_observer(delivered.append)

        assert engine.supports_text_queries
        assert engine.ask("what do you see?") is True

        assert _wait_for(lambda: len(engine._ready_answers) > 0)
        assert policy.generate_thread_names == ["RTCInference"]
        assert delivered == []  # the generating thread never fires the observer itself

        engine.pump_query()
        assert len(delivered) == 1
        assert delivered[0].ok
        assert delivered[0].answer == "answer: what do you see?"


def test_rtc_engine_get_action_raises_on_unlabeled_action():
    """An unlabeled action would silently corrupt dispatched_task and the frame labels."""
    from lerobot.policies.rtc import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    engine, _policy = _make_rtc_engine()
    queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=8, max_guidance_weight=1.0))
    queue.merge(torch.zeros(4, 2), torch.zeros(4, 2), real_delay=0)  # no task label
    engine._action_queue = queue

    with pytest.raises(RuntimeError, match="task provenance"):
        engine.get_action(None)


# --- Config validation ---


def test_interactive_rejects_keyboard_bound_strategies_and_allows_sentry():
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import HighlightStrategyConfig, RolloutConfig, SentryStrategyConfig
    from tests.mocks.mock_robot import MockRobotConfig

    def make(strategy, **kwargs):
        return RolloutConfig(
            robot=MockRobotConfig(),
            strategy=strategy,
            dataset=DatasetRecordConfig(repo_id="user/rollout_test", single_task="test"),
            interactive=True,
            **kwargs,
        )

    with pytest.raises(ValueError, match="--interactive=true supports"):
        make(HighlightStrategyConfig())

    cfg = make(SentryStrategyConfig(), policy=SimpleNamespace(device="cpu"))
    assert cfg.interactive is True
    assert cfg.dataset.streaming_encoding is True  # sentry forces streaming


def test_autosteer_interval_bounds():
    """Negatives are a CLI error; 0 is the legitimate re-plan-every-tick edge."""
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match="autosteer_interval_s"):
        RolloutConfig(
            robot=MockRobotConfig(),
            policy=SimpleNamespace(device="cpu"),
            autosteer_interval_s=-1.0,
        )

    cfg = RolloutConfig(
        robot=MockRobotConfig(),
        policy=SimpleNamespace(device="cpu"),
        autosteer_interval_s=0.0,
    )
    assert cfg.autosteer_interval_s == 0.0
