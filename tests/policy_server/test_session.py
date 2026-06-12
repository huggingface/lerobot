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

"""Unit tests for Session (latest-only mailbox, episode reset, close,
processor-step introspection) and SessionRegistry (thread-safe map)."""

from __future__ import annotations

import threading

from lerobot.policy_server.schema import MsgHeader
from lerobot.policy_server.session import Session, SessionRegistry
from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep
from tests.policy_server.conftest import TASK, MockPipeline, make_mock_processors


def make_session(
    client_uuid: str = "client-a",
    preprocessor: MockPipeline | None = None,
    postprocessor: MockPipeline | None = None,
    publisher=None,
) -> Session:
    default_pre, default_post = make_mock_processors()
    return Session(
        session_id=f"sess-{client_uuid}",
        client_uuid=client_uuid,
        task=TASK,
        robot_type="mock_robot",
        rtc_enabled=False,
        preprocessor=preprocessor if preprocessor is not None else default_pre,
        postprocessor=postprocessor if postprocessor is not None else default_post,
        action_publisher=publisher,
    )


# ---------------------------------------------------------------------------
# Mailbox: latest-only deposit / take
# ---------------------------------------------------------------------------


def test_deposit_then_take_returns_item():
    session = make_session()
    header = MsgHeader(seq_id=7)
    session.deposit(header, b"payload-7")

    item = session.take()
    assert item is not None
    assert item.header.seq_id == 7
    assert item.payload == b"payload-7"
    assert item.recv_mono > 0


def test_second_deposit_supersedes_and_take_returns_newer():
    session = make_session()
    session.deposit(MsgHeader(seq_id=1), b"old")
    session.deposit(MsgHeader(seq_id=2), b"new")

    assert session.stats.superseded == 1
    assert session.stats.superseded_since_reply == 1

    item = session.take()
    assert item is not None
    assert item.header.seq_id == 2
    assert item.payload == b"new"


def test_deposit_after_take_is_not_superseded():
    session = make_session()
    session.deposit(MsgHeader(seq_id=1), b"one")
    session.take()
    session.deposit(MsgHeader(seq_id=2), b"two")

    assert session.stats.superseded == 0
    assert session.stats.superseded_since_reply == 0


def test_take_clears_mailbox_second_take_is_none():
    session = make_session()
    session.deposit(MsgHeader(seq_id=1), b"one")

    assert session.take() is not None
    assert session.take() is None


def test_has_pending_transitions():
    session = make_session()
    assert not session.has_pending()

    session.deposit(MsgHeader(seq_id=1), b"one")
    assert session.has_pending()

    session.take()
    assert not session.has_pending()


def test_deposit_marks_alive_and_clears_token_drop():
    session = make_session()
    session.alive = False
    session.token_dropped_mono = 123.4
    before = session.last_seen_mono

    session.deposit(MsgHeader(seq_id=1), b"one")

    assert session.alive is True
    assert session.token_dropped_mono is None
    assert session.last_seen_mono >= before


# ---------------------------------------------------------------------------
# Episode boundary
# ---------------------------------------------------------------------------


def test_reset_episode_resets_pipelines_clears_mailbox_and_increments():
    pre, post = make_mock_processors()
    session = make_session(preprocessor=pre, postprocessor=post)
    session.deposit(MsgHeader(seq_id=1), b"stale")
    assert session.episode_id == 0

    session.reset_episode()

    assert not session.has_pending()
    assert pre.reset_count == 1
    assert post.reset_count == 1
    assert session.episode_id == 1

    session.reset_episode()
    assert session.episode_id == 2
    assert pre.reset_count == 2
    assert post.reset_count == 2


def test_reset_episode_with_explicit_id():
    session = make_session()
    session.reset_episode(episode_id=7)
    assert session.episode_id == 7


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class FakePublisher:
    def __init__(self, raise_on_undeclare: bool = False):
        self.undeclare_calls = 0
        self.raise_on_undeclare = raise_on_undeclare

    def undeclare(self):
        self.undeclare_calls += 1
        if self.raise_on_undeclare:
            raise RuntimeError("transport already closed")


def test_close_clears_mailbox_and_undeclares_publisher_exactly_once():
    publisher = FakePublisher()
    session = make_session(publisher=publisher)
    session.deposit(MsgHeader(seq_id=1), b"stale")

    session.close()

    assert not session.has_pending()
    assert publisher.undeclare_calls == 1
    assert session.action_publisher is None

    # Idempotent: a second close must not undeclare again.
    session.close()
    assert publisher.undeclare_calls == 1


def test_close_tolerates_undeclare_raising():
    publisher = FakePublisher(raise_on_undeclare=True)
    session = make_session(publisher=publisher)
    session.deposit(MsgHeader(seq_id=1), b"stale")

    session.close()  # must not raise

    assert publisher.undeclare_calls == 1
    assert not session.has_pending()
    assert session.action_publisher is None


def test_close_without_publisher_is_noop():
    session = make_session(publisher=None)
    session.close()  # must not raise
    assert session.action_publisher is None


# ---------------------------------------------------------------------------
# Processor-step introspection
# ---------------------------------------------------------------------------


def test_relative_and_normalizer_steps_detected():
    relative = RelativeActionsProcessorStep(enabled=True)
    normalizer = NormalizerProcessorStep(features={}, norm_map={})
    pre = MockPipeline(steps=[relative, normalizer])
    session = make_session(preprocessor=pre)

    assert session.relative_step is relative
    assert session.normalizer_step is normalizer


def test_disabled_relative_step_is_not_detected():
    relative = RelativeActionsProcessorStep(enabled=False)
    pre = MockPipeline(steps=[relative])
    session = make_session(preprocessor=pre)

    assert session.relative_step is None
    assert session.normalizer_step is None


def test_empty_pipeline_yields_no_introspected_steps():
    session = make_session()
    assert session.relative_step is None
    assert session.normalizer_step is None


# ---------------------------------------------------------------------------
# SessionRegistry
# ---------------------------------------------------------------------------


def test_registry_add_get_remove_len_snapshot():
    registry = SessionRegistry()
    assert len(registry) == 0
    assert registry.get("missing") is None
    assert registry.snapshot() == []

    session_a = make_session("uuid-a")
    session_b = make_session("uuid-b")
    assert registry.add(session_a) is None
    assert registry.add(session_b) is None

    assert len(registry) == 2
    assert registry.get("uuid-a") is session_a
    assert registry.get("uuid-b") is session_b
    assert set(registry.snapshot()) == {session_a, session_b}

    removed = registry.remove("uuid-a")
    assert removed is session_a
    assert len(registry) == 1
    assert registry.get("uuid-a") is None


def test_registry_remove_missing_returns_none():
    registry = SessionRegistry()
    assert registry.remove("never-added") is None


def test_registry_add_returns_displaced_same_uuid_session():
    registry = SessionRegistry()
    first = make_session("uuid-x")
    second = make_session("uuid-x")

    assert registry.add(first) is None
    displaced = registry.add(second)

    assert displaced is first
    assert registry.get("uuid-x") is second
    assert len(registry) == 1


def test_registry_thread_safety_smoke():
    registry = SessionRegistry()
    errors: list[Exception] = []

    def worker(prefix: str) -> None:
        try:
            for i in range(200):
                session = make_session(f"{prefix}-{i}")
                registry.add(session)
                assert registry.get(session.client_uuid) is session
                assert registry.remove(session.client_uuid) is session
        except Exception as exc:  # noqa: BLE001 — surfaced to the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(p,)) for p in ("alpha", "beta")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert errors == []
    assert len(registry) == 0
    assert registry.snapshot() == []
