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

"""Invalid operator/generated text must not disturb an otherwise valid rollout."""

import io
import logging

import pytest

pytest.importorskip("datasets", reason="interactive rollout requires lerobot[dataset]")

from lerobot.inference import QueryKind
from lerobot.rollout.interactive import InteractiveSession
from tests import interactive_helpers
from tests.remote_inference import helpers as engine_helpers
from tests.remote_inference.helpers import capture
from tests.utils import wait_for


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
    assert wait_for(lambda: not engine._hold.requested)

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
