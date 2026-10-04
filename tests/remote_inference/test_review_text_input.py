# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Invalid operator/generated text must not disturb an otherwise valid rollout."""

import io
import logging

import pytest

pytest.importorskip("datasets", reason="interactive rollout requires lerobot[dataset]")

from lerobot.inference import QueryKind
from lerobot.rollout.controller import AskResult, RolloutController
from lerobot.rollout.interactive import InteractiveSession
from tests import test_interactive_rollout as interactive_helpers
from tests.remote_inference import test_engine as engine_helpers

session = engine_helpers.session
capture = engine_helpers.capture
wait_for = engine_helpers.wait_for


def test_oversized_generated_subtask_stops_sequencer_without_fault_and_resumes_fresh(session):
    engine, client = session
    client.descriptor["limits"]["max_input_chars"] = 16
    client.text = "x" * 17
    engine.resume()
    engine.start_autosteer("goal", 10)
    engine.pump_query({})
    engine.start()
    assert not engine.dispatch_allowed()
    engine.acknowledge_hold()
    capture(engine)
    assert client.text_started.wait(2)
    client.text_release.set()
    assert wait_for(lambda: not engine._hold_requested)

    assert not engine.failed
    assert engine.autosteer_goal is None
    assert engine.task == "initial task"
    assert not engine._query_in_flight
    assert len(engine._ready_answers) == 1
    answer = engine._ready_answers[0]
    assert answer.kind is QueryKind.NEXT_SUBTASK
    assert answer.answer is None
    assert "Generated subtask" in answer.error
    assert "limit is 16" in answer.error
    assert not client.action_started.is_set(), "post-query motion requires a fresh observation"
    capture(engine)
    assert client.action_started.wait(2)
    assert wait_for(lambda: not engine.runtime.queue.empty())
    assert client.requests[-1].observation.task == "initial task"


def test_vqa_answer_may_exceed_instruction_limit(session):
    engine, client = session
    client.descriptor["limits"]["max_input_chars"] = 16
    client.text = "x" * 17
    engine_helpers.start_query(engine, client)
    client.text_release.set()
    assert wait_for(lambda: bool(engine._ready_answers))
    assert engine._ready_answers[0].answer == client.text
    assert not engine.failed


@pytest.mark.parametrize("command", ["vqa", "autosteer", "subtask"])
def test_oversized_operator_input_is_actionable_and_does_not_cancel_autosteer(
    session, command, capsys, caplog
):
    engine, client = session
    client.descriptor["limits"]["max_input_chars"] = 16
    ctx, strategy, *_ = interactive_helpers._make_ctx()
    ctx.policy.inference = engine
    ui = InteractiveSession(strategy, ctx, input_stream=io.StringIO())
    ui.controller._running.set()
    engine.start_autosteer("goal", 10)
    generation = engine.query_intent_generation
    ui._handle_line(f"/{command} {'x' * 17}")

    output = capsys.readouterr().out
    assert "rejected" in output
    assert "limit is 16" in output
    assert "Shorten it" in output
    assert "busy" not in output
    assert "Autosteer off" not in output
    assert engine.task == "initial task"
    assert engine.autosteer_goal == "goal"
    assert engine.query_intent_generation == generation
    assert not engine.has_pending_query
    assert not engine.failed
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_direct_remote_task_rejection_is_nonthrowing_and_leaves_state_untouched(session, caplog):
    engine, client = session
    client.descriptor["limits"]["max_input_chars"] = 16
    engine.start_autosteer("goal", 10)
    assert not engine.set_task("x" * 17)
    assert engine.task == "initial task"
    assert engine.task_version == 0
    assert engine.autosteer_goal == "goal"
    assert "Instruction rejected" in caplog.text


def test_remote_queries_honor_deployed_limit_above_local_default(session):
    engine, client = session
    client.descriptor["limits"]["max_input_chars"] = 5000
    assert engine.ask("x" * 5000)
    engine.drop_pending_query()
    engine.start_autosteer("x" * 5000, 10)
    assert engine.autosteer_goal == "x" * 5000
    assert engine.set_task("x" * 5000)


@pytest.mark.parametrize("text", ["", "   ", "x" * 4097])
@pytest.mark.parametrize("method", ["ask", "autosteer"])
def test_local_controller_rejects_invalid_text_before_changing_query_state(text, method):
    ctx, strategy, engine, *_ = interactive_helpers._make_ctx()
    controller = RolloutController(strategy, ctx)
    controller._running.set()
    engine.start_autosteer("goal", 10)
    assert getattr(controller, method)(text) is AskResult.INVALID
    assert controller.text_input_error(text)
    assert engine.autosteer_goal == "goal"
    assert not engine.has_pending_query


@pytest.mark.parametrize("method", ["ask", "autosteer"])
def test_remote_controller_checks_deployment_limit_before_touching_engine(session, method):
    engine, client = session
    client.descriptor["limits"]["max_input_chars"] = 16
    ctx, strategy, *_ = interactive_helpers._make_ctx()
    ctx.policy.inference = engine
    controller = RolloutController(strategy, ctx)
    controller._running.set()
    engine.start_autosteer("goal", 10)
    assert getattr(controller, method)("x" * 17) is AskResult.INVALID
    assert not controller.set_task("x" * 17)
    assert engine.autosteer_goal == "goal"
    assert engine.task == "initial task"
    assert not engine.has_pending_query
