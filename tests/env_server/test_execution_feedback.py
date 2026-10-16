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

"""Deterministic execution-boundary checks without a second networking fixture."""

import time
from queue import Empty
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("msgpack")

from lerobot.env_server.client import EnvClient
from lerobot.env_server.server import EnvServer, ServerConfig
from lerobot.sims.backend import Backend
from lerobot.transport.wire.codec import decode_message, encode_message
from lerobot.transport.wire.protocol import Envelope, ErrorCode, MessageType


@pytest.fixture
def executor():
    server = EnvServer(ServerConfig())
    channel = SimpleNamespace(
        get=lambda: (_ for _ in ()).throw(Empty()),
        dropped=0,
        close=lambda: None,
        undeclare=lambda: None,
    )
    server.transport = SimpleNamespace(
        subscribe_liveliness=lambda *a: channel,
        declare_token=lambda *a: channel,
    )
    probe = Backend(server.config.sim)
    try:
        server.descriptor = probe.descriptor()
    finally:
        probe.close()
    opened = server._open({"clock": "realtime", "seeds": [0]})
    session = server.sessions[opened["session"]]
    yield server, opened["session"], session
    if opened["session"] in server.sessions:
        server._close_session(opened["session"])


def submit(executor, request_id, operation="apply", **body):
    server, session_id, session = executor
    request = Envelope(
        MessageType.CONTROL,
        server.instance,
        session_id,
        session.generation,
        request_id,
        {"operation": operation, **body},
    )
    replies = []
    pending = SimpleNamespace(
        payload=encode_message(request),
        key=session.prefix + "/control",
        deadline=time.monotonic() + 10,
        expired=False,
        reply=replies.append,
        drop=lambda: None,
    )
    server._reply(pending)
    return replies


def tick(executor):
    executor[2].next_tick = 0
    executor[0]._tick()


def test_confirmation_pairs_applied_action_with_following_observation(executor):
    session = executor[2]
    before = session.backend.snapshot().obs["observation.state"].copy()
    replies = submit(executor, "command", actions=np.array([[0.1, 0]], np.float32))
    assert not replies  # Acceptance does not acknowledge execution.
    tick(executor)
    reply = decode_message(replies[0])
    assert reply.message_type is MessageType.ACK
    feedback = reply.body["execution"]
    assert feedback["request_id"] == "command" and feedback["advanced"].tolist() == [True]
    assert feedback["started_at"] <= feedback["completed_at"]
    np.testing.assert_allclose(
        reply.body["result"]["obs"]["observation.state"], before + np.array([[0.1, 0]], np.float32)
    )
    np.testing.assert_array_equal(feedback["applied"], [[np.float32(0.1), 0]])
    assert "result" not in feedback  # Images are serialized once.


def test_realtime_world_waits_for_first_command(executor):
    session = executor[2]
    before = session.sequence
    tick(executor)
    assert session.sequence == before  # The open/reset target is not a policy command.
    submit(executor, "command", actions=np.array([[0.1, 0]], np.float32))
    tick(executor)
    assert session.sequence == before + 1


@pytest.mark.parametrize("step_duration", [0.02, 0.08])
def test_realtime_period_includes_backend_work_without_catchup(executor, monkeypatch, step_duration):
    server, _, session = executor
    clock = SimpleNamespace(now=100.0)
    step = session.backend.step

    def timed_step(actions):
        result = step(actions)
        clock.now += step_duration
        return result

    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(session.backend, "step", timed_step)
    period = 1 / server.config.sim.fps
    session.commanded = True
    session.next_tick = 100.0
    server._tick()
    # On time: the next deadline follows the previous one, so poll latency does not drift.
    # Overrun: the next tick is due immediately.
    assert session.next_tick == pytest.approx(max(100.0 + period, 100.0 + step_duration))
    # A delayed worker runs one tick, rather than replaying a backlog of missed ticks.
    clock.now = 101.0
    before = session.sequence
    server._tick()
    assert session.sequence == before + 1
    assert session.next_tick == pytest.approx(101.0 + step_duration)
    server._tick()
    assert session.sequence == before + 2


@pytest.mark.parametrize("operation", ["step", "apply"])
@pytest.mark.parametrize(
    "actions",
    [np.zeros((1, 2), np.float64), np.zeros((1, 2), np.int32), [[0.0, 0.0]], np.zeros((1, 3), np.float32)],
)
def test_invalid_action_contract_is_rejected_before_execution(executor, operation, actions):
    session = executor[2]
    session.clock = "lockstep" if operation == "step" else "realtime"
    before = session.backend.snapshot()
    reply = decode_message(submit(executor, "invalid", operation, actions=actions)[0])
    assert reply.message_type is MessageType.ERROR
    assert reply.body["code"] == ErrorCode.INCOMPATIBLE
    assert session.command is None and session.fault is None
    np.testing.assert_array_equal(session.backend.snapshot().step, before.step)


def test_saturation_does_not_replace_accepted_command(executor):
    first = submit(executor, "first", actions=np.ones((1, 2), np.float32) * 0.1)
    second = submit(executor, "second", actions=np.ones((1, 2), np.float32) * 0.2)
    assert decode_message(second[0]).body["code"] == ErrorCode.BUSY
    tick(executor)
    assert decode_message(first[0]).body["execution"]["request_id"] == "first"


@pytest.mark.parametrize("operation", ["reset", "hold", "close"])
def test_control_cancels_unexecuted_commands(executor, operation):
    pending = submit(executor, "pending", actions=np.ones((1, 2), np.float32))
    submit(executor, "control", operation)
    assert decode_message(pending[0]).body["code"] == ErrorCode.STALE


def test_expired_command_is_not_executed(executor):
    before = executor[2].backend.snapshot().obs["observation.state"].copy()
    replies = submit(executor, "expired", actions=np.ones((1, 2), np.float32))
    executor[2].command.pending.expired = True  # The client's query deadline has passed.
    tick(executor)
    assert not replies  # Nobody is waiting for the reply any more.
    np.testing.assert_array_equal(executor[2].backend.snapshot().obs["observation.state"], before)


def test_terminal_world_is_not_reported_as_executing_again(executor):
    submit(executor, "terminal", actions=np.ones((1, 2), np.float32) * 2)
    tick(executor)
    terminal = executor[2].backend.snapshot()
    replies = submit(executor, "frozen", actions=np.ones((1, 2), np.float32))
    tick(executor)
    reply = decode_message(replies[0])
    assert reply.body["execution"]["advanced"].tolist() == [False]
    np.testing.assert_array_equal(
        reply.body["result"]["obs"]["observation.state"], terminal.obs["observation.state"]
    )


@pytest.mark.parametrize("field", ["request_id", "generation", "sequence"])
def test_client_rejects_stale_execution_feedback(executor, field):
    from lerobot.transport.wire.protocol import ProtocolError

    replies = submit(executor, "execution", actions=np.zeros((1, 2), np.float32))
    tick(executor)
    body = decode_message(replies[0]).body
    client = EnvClient("tcp/127.0.0.1:7448", "default")
    client.descriptor = executor[0].descriptor
    client.instance = executor[0].instance
    client.session = executor[1]
    client.num_envs = 1
    client.execution = executor[2].execution

    def query(key, payload, *args, **kwargs):
        request = decode_message(payload)
        body["execution"]["request_id"] = request.request_id
        body["execution"][field] = {"request_id": "other", "generation": 1, "sequence": 0}[field]
        return [encode_message(request.reply(MessageType.ACK, body))]

    client.transport = SimpleNamespace(query=query)
    with pytest.raises(ProtocolError) as error:
        client.request("apply", actions=np.zeros((1, 2), np.float32))
    assert error.value.code == ErrorCode.STALE
    assert client.failed
