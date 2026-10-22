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

"""Real loopback regressions for reply-handle ownership across session boundaries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

pytest.importorskip("zenoh")
pytest.importorskip("msgpack")
pytest.importorskip("datasets")

from lerobot.remote_inference.codec import decode_message, encode_message
from lerobot.remote_inference.protocol import (
    Envelope,
    ErrorCode,
    MessageType,
    deployment_prefix,
    instance_prefix,
    session_prefix,
)
from lerobot.transport.zenoh import ZenohConfig, ZenohTransport
from tests.remote_inference.helpers import control_request, open_request
from tests.utils import wait_for


def exchange(transport, key, request):
    replies = transport.query(key, encode_message(request), timeout=2)
    assert len(replies) == 1
    response = decode_message(replies[0])
    assert response.request_id == request.request_id
    return response


def test_transport_loss_cleans_up_and_readmits_without_restarting_server(remote_server):
    worker, config = remote_server
    key = instance_prefix(config.deployment, worker.instance_id) + "/open"
    disconnected = ZenohTransport(ZenohConfig(connect_endpoints=[config.endpoint])).open()

    try:
        first = exchange(disconnected, key, open_request(worker))
        assert first.message_type is MessageType.ACCEPTED
        disconnected.declare_token(
            session_prefix(config.deployment, worker.instance_id, first.session_id) + "/alive"
        )
        assert wait_for(lambda: worker.descriptor["session"]["client_present"] is True)
        # Close only the transport, as after a crashed process: no session CLOSE operation.
        disconnected.close()
        assert wait_for(lambda: worker.descriptor["session"]["client_present"] is False)
        with ZenohTransport(ZenohConfig(connect_endpoints=[config.endpoint])) as replacement:
            rejected = exchange(replacement, key, open_request(worker))
            assert rejected.message_type is MessageType.ERROR
            assert rejected.body["code"] == ErrorCode.BUSY
            # The exact 10-second boundary is covered with a controlled clock in test_session.
            worker.idle_timeout_s = 0.02
            assert wait_for(lambda: worker.descriptor["available"])
            worker.idle_timeout_s = 10.0
            accepted = exchange(replacement, key, open_request(worker))
            assert accepted.message_type is MessageType.ACCEPTED
            assert accepted.session_id != first.session_id
            assert worker.runner.policy.resets == 4
            control_key = (
                session_prefix(config.deployment, worker.instance_id, accepted.session_id) + "/control"
            )
            assert (
                exchange(
                    replacement, control_key, control_request(worker, accepted.session_id, 0, "close")
                ).message_type
                is MessageType.ACK
            )
    finally:
        disconnected.close()


def test_close_ack_and_new_acceptance_can_be_outstanding_together(remote_server, monkeypatch):
    worker, config = remote_server
    key = instance_prefix(config.deployment, worker.instance_id) + "/open"
    with ZenohTransport(ZenohConfig(connect_endpoints=[config.endpoint])) as client:
        first = exchange(client, key, open_request(worker))
        control_key = session_prefix(config.deployment, worker.instance_id, first.session_id) + "/control"
        replacement_request = open_request(worker)
        closed_state = Event()
        release_close = Event()
        original_execute = worker._execute
        original_submit = worker.submit

        def execute(message):
            response = original_execute(message)
            if message.message_type is MessageType.CONTROL and message.session_id == first.session_id:
                closed_state.set()
                assert release_close.wait(2), "test did not release completed close"
            return response

        def submit(message):
            future = original_submit(message)
            if message.request_id == replacement_request.request_id:
                # Hold this IO-pump turn until both replies exist. That makes the
                # close/open interleaving deterministic without mocking transport.
                release_close.set()
                future.result(2)
            return future

        monkeypatch.setattr(worker, "_execute", execute)
        monkeypatch.setattr(worker, "submit", submit)
        try:
            with ThreadPoolExecutor() as pool:
                closing = pool.submit(
                    exchange, client, control_key, control_request(worker, first.session_id, 0, "close")
                )
                assert closed_state.wait(2)
                replacement = pool.submit(exchange, client, key, replacement_request)
                assert closing.result(3).message_type is MessageType.ACK
                accepted = replacement.result(3)
                assert accepted.message_type is MessageType.ACCEPTED
                assert accepted.session_id != first.session_id
                assert worker.session_id == accepted.session_id
        finally:
            release_close.set()
        control_key = session_prefix(config.deployment, worker.instance_id, accepted.session_id) + "/control"
        assert (
            exchange(
                client, control_key, control_request(worker, accepted.session_id, 0, "close")
            ).message_type
            is MessageType.ACK
        )


def test_malformed_queries_release_reply_slots_immediately(remote_server):
    worker, config = remote_server
    key = deployment_prefix(config.deployment) + "/describe"
    with ZenohTransport(ZenohConfig(connect_endpoints=[config.endpoint])) as client:
        # More malformed envelopes than the queryable's four-slot capacity must
        # not deny valid callers for the query-handle lifetime (30 seconds).
        for _ in range(6):
            assert client.query(key, b"\xc1", timeout=1) == []
        descriptor = exchange(client, key, Envelope(MessageType.DESCRIBE, request_id=uuid4().hex))
        assert descriptor.message_type is MessageType.DESCRIPTOR
        assert descriptor.body["instance_id"] == worker.instance_id
