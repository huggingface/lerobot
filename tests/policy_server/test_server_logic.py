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

"""Pure-logic PolicyServer tests (no zenoh transport).

Covers status snapshots, session open/reject/re-handshake, the
per-request inference path (determinism, RTC forwarding, echo fields,
supersession), episode boundaries in ``_serve_one``, warmup, and the
error/metrics accounting.
"""

import numpy as np
import pytest

pytest.importorskip("msgpack")

from lerobot.policy_server import codec  # noqa: E402
from lerobot.policy_server.schema import MsgHeader, ObservationMsg, SessionOpenMsg  # noqa: E402
from lerobot.policy_server.validation import PolicyClassification, ServingClass  # noqa: E402
from tests.policy_server.conftest import (  # noqa: E402
    ACTION_DIM,
    ACTION_NAMES,
    CHUNK_SIZE,
    IMG_H,
    IMG_W,
    MODEL_ID,
    STATE_DIM,
    TASK,
    MockChunkPolicy,
    make_logic_server,
    make_manifest,
)

CAMERA_KEY = "observation.images.front"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_open_msg(client_uuid: str = "client-a", **overrides) -> SessionOpenMsg:
    kwargs = {
        "client_uuid": client_uuid,
        "robot_type": "so101",
        "policy_type": "mockchunk",
        "fps": 30.0,
        "action_names": list(ACTION_NAMES),
        "camera_names": [CAMERA_KEY],
        "state_dim": STATE_DIM,
        "rtc_enabled": True,
        "task": TASK,
    }
    kwargs.update(overrides)
    return SessionOpenMsg(**kwargs)


def make_obs(**overrides) -> ObservationMsg:
    kwargs = {
        "state": np.arange(STATE_DIM, dtype=np.float32),
        "images": {CAMERA_KEY: np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)},
        "task": TASK,
        "jpeg_quality": 0,
    }
    kwargs.update(overrides)
    return ObservationMsg(**kwargs)


def open_session(server, client_uuid: str = "client-a", **overrides):
    ack = server._handle_session_open(make_open_msg(client_uuid=client_uuid, **overrides))
    assert ack.accepted, ack.error
    return server.registry.get(client_uuid), ack


def deposit(session, obs: ObservationMsg, header: MsgHeader | None = None) -> None:
    session.deposit(header or MsgHeader(episode_id=session.episode_id), codec.encode_observation(obs))


def make_exclusive_server(policy=None):
    return make_logic_server(
        manifest=make_manifest(serving_mode="exclusive"),
        policy=policy,
        classification=PolicyClassification(
            ServingClass.EXCLUSIVE, supports_rtc=False, needs_queue_population=False, reason="x"
        ),
    )


def expected_chunk(state: np.ndarray) -> np.ndarray:
    steps = np.arange(CHUNK_SIZE, dtype=np.float32)[:, None] * np.float32(0.01)
    return state[None, :ACTION_DIM] + steps


# ---------------------------------------------------------------------------
# status_snapshot
# ---------------------------------------------------------------------------


def test_status_snapshot_capabilities():
    server = make_logic_server()
    snap = server.status_snapshot()
    assert snap.model_repo == MODEL_ID
    assert snap.policy_type == "mockchunk"
    assert snap.action_names == ACTION_NAMES
    assert snap.state_dim == STATE_DIM
    assert snap.chunk_size == CHUNK_SIZE
    assert snap.expected_cameras == [CAMERA_KEY]
    assert snap.supports_rtc is True
    assert snap.warmed_up is True
    assert snap.serving_mode == "shared"
    assert snap.active_sessions == 0
    assert snap.max_sessions == 4  # make_manifest default


# ---------------------------------------------------------------------------
# _handle_session_open
# ---------------------------------------------------------------------------


def test_session_open_happy_path():
    server = make_logic_server()
    session, ack = open_session(server, "client-a")
    assert ack.accepted is True
    assert ack.error == ""
    assert ack.session_id == session.session_id
    assert ack.session_id != ""
    assert ack.model_repo == MODEL_ID
    assert ack.policy_type == "mockchunk"
    assert ack.action_names == ACTION_NAMES
    assert ack.expected_cameras == [CAMERA_KEY]
    assert ack.state_dim == STATE_DIM
    assert ack.chunk_size == CHUNK_SIZE
    assert ack.supports_rtc is True
    assert ack.serving_mode == "shared"
    assert ack.warmed_up is True
    assert len(server.registry) == 1
    assert session.rtc_enabled is True
    assert session.task == TASK
    assert server.metrics["sessions_opened_total"] == 1


def test_session_open_fresh_processor_pair_per_session():
    server = make_logic_server()
    assert len(server.factory_calls) == 0  # warmup_inferences=0: no warmup pair
    session_a, _ = open_session(server, "client-a")
    assert len(server.factory_calls) == 1
    session_b, _ = open_session(server, "client-b")
    assert len(server.factory_calls) == 2
    assert session_a.preprocessor is not session_b.preprocessor
    assert session_a.postprocessor is not session_b.postprocessor


def test_session_open_rejects_action_order_mismatch():
    server = make_logic_server()
    ack = server._handle_session_open(make_open_msg(action_names=list(reversed(ACTION_NAMES))))
    assert ack.accepted is False
    assert "action" in ack.error
    assert len(server.registry) == 0


def test_session_open_rejects_at_capacity():
    server = make_logic_server()
    for i in range(4):  # make_manifest max_sessions=4
        open_session(server, f"client-{i}")
    ack = server._handle_session_open(make_open_msg(client_uuid="client-overflow"))
    assert ack.accepted is False
    assert "sessions" in ack.error
    assert len(server.registry) == 4


def test_rehandshake_replaces_session_without_counting_against_capacity():
    server = make_logic_server()
    first_ack = None
    for i in range(4):
        _, ack = open_session(server, f"client-{i}")
        if i == 0:
            first_ack = ack
    # Server is full; the same client re-handshakes and must be accepted.
    session, ack = open_session(server, "client-0")
    assert ack.accepted is True
    assert len(server.registry) == 4
    assert ack.session_id != first_ack.session_id
    assert server.registry.get("client-0").session_id == session.session_id


def test_session_open_rtc_downgrade():
    server = make_logic_server(
        classification=PolicyClassification(
            ServingClass.SHARED, supports_rtc=False, needs_queue_population=False, reason="x"
        )
    )
    session, ack = open_session(server, "client-a", rtc_enabled=True)
    assert ack.accepted is True
    assert ack.supports_rtc is False
    assert session.rtc_enabled is False
    assert any("RTC" in w for w in ack.warnings)


# ---------------------------------------------------------------------------
# run_inference_request
# ---------------------------------------------------------------------------


def test_inference_deterministic_chunks():
    server = make_logic_server()
    session, _ = open_session(server)
    state = np.arange(STATE_DIM, dtype=np.float32)
    reply = server.run_inference_request(session, MsgHeader(), make_obs(state=state))
    assert reply.chunk_model.shape == (CHUNK_SIZE, ACTION_DIM)
    assert reply.chunk_robot.shape == (CHUNK_SIZE, ACTION_DIM)
    np.testing.assert_allclose(reply.chunk_model[0], state, rtol=0, atol=0)
    np.testing.assert_allclose(reply.chunk_model, expected_chunk(state), rtol=1e-6)
    np.testing.assert_allclose(reply.chunk_robot, 2.0 * reply.chunk_model, rtol=0, atol=0)


def test_inference_delay_forwarded_to_policy():
    server = make_logic_server()
    session, _ = open_session(server)
    policy = server._policy
    server.run_inference_request(session, MsgHeader(), make_obs(inference_delay_steps=3))
    assert policy.calls[-1]["inference_delay"] == 3
    assert policy.calls[-1]["prev_chunk_left_over"] is None


def test_prefix_model_forwarded_padded_to_execution_horizon():
    server = make_logic_server()
    session, _ = open_session(server)
    policy = server._policy
    prefix = (np.arange(3 * ACTION_DIM, dtype=np.float32).reshape(3, ACTION_DIM)) + 1.0
    server.run_inference_request(session, MsgHeader(), make_obs(prefix_model=prefix))
    received = policy.calls[-1]["prev_chunk_left_over"]
    assert received is not None
    horizon = server._manifest.rtc.execution_horizon  # 10 by default
    assert received.shape == (horizon, ACTION_DIM)
    np.testing.assert_allclose(received[:3].numpy(), prefix, rtol=0, atol=0)
    np.testing.assert_allclose(received[3:].numpy(), np.zeros((horizon - 3, ACTION_DIM)), rtol=0, atol=0)


def test_prefix_model_truncated_to_execution_horizon():
    server = make_logic_server()
    session, _ = open_session(server)
    policy = server._policy
    horizon = server._manifest.rtc.execution_horizon
    prefix = np.ones((horizon + 5, ACTION_DIM), dtype=np.float32)
    server.run_inference_request(session, MsgHeader(), make_obs(prefix_model=prefix))
    assert policy.calls[-1]["prev_chunk_left_over"].shape == (horizon, ACTION_DIM)


def test_reply_echo_fields():
    server = make_logic_server()
    session, _ = open_session(server)
    header = MsgHeader(seq_id=7, episode_id=2, client_mono_ns=123_456_789)
    reply = server.run_inference_request(session, header, make_obs())
    assert reply.seq_id_echo == 7
    assert reply.episode_id_echo == 2
    assert reply.client_mono_ns_echo == 123_456_789


def test_superseded_seqs_reported_then_reset():
    server = make_logic_server()
    session, _ = open_session(server)
    deposit(session, make_obs(), MsgHeader(seq_id=1))
    deposit(session, make_obs(), MsgHeader(seq_id=2))  # supersedes seq 1
    item = session.take()
    assert item.header.seq_id == 2  # latest-only mailbox
    reply = server.run_inference_request(session, item.header, codec.decode_observation(item.payload))
    assert reply.superseded_seqs == 1
    deposit(session, make_obs(), MsgHeader(seq_id=3))
    item = session.take()
    reply = server.run_inference_request(session, item.header, codec.decode_observation(item.payload))
    assert reply.superseded_seqs == 0


# ---------------------------------------------------------------------------
# _serve_one
# ---------------------------------------------------------------------------


def test_serve_one_episode_boundary_resets_session_pipelines():
    server = make_logic_server()
    session, _ = open_session(server)
    # Fresh sessions start at the -1 sentinel so their first request
    # always lands on the episode-boundary branch (mid-episode reconnects
    # can never inherit stale state).
    assert session.episode_id == -1
    deposit(session, make_obs(episode_start=True), MsgHeader(episode_id=1))
    server._serve_one(session)
    assert session.preprocessor.reset_count == 1
    assert session.postprocessor.reset_count == 1
    assert session.episode_id == 1
    # Shared mode never resets the policy itself.
    assert server._policy.reset_count == 0


def test_serve_one_no_boundary_no_reset():
    server = make_logic_server()
    session, _ = open_session(server)
    # First request always resets (the -1 sentinel) and syncs the episode.
    deposit(session, make_obs(), MsgHeader(episode_id=0))
    server._serve_one(session)
    assert session.preprocessor.reset_count == 1
    assert session.episode_id == 0
    # Same-episode follow-up: no further reset.
    deposit(session, make_obs(), MsgHeader(episode_id=0, seq_id=2))
    server._serve_one(session)
    assert session.preprocessor.reset_count == 1
    assert session.postprocessor.reset_count == 1


def test_serve_one_exclusive_mode_resets_policy_on_boundary():
    policy = MockChunkPolicy()
    server = make_exclusive_server(policy=policy)
    assert server.status_snapshot().serving_mode == "exclusive"
    assert server.status_snapshot().max_sessions == 1  # exclusive forces 1
    session, _ = open_session(server, rtc_enabled=False)
    # Exclusive session open already resets the policy to fresh state.
    base_resets = policy.reset_count
    assert base_resets >= 1
    deposit(session, make_obs(episode_start=True), MsgHeader(episode_id=1))
    server._serve_one(session)
    assert policy.reset_count == base_resets + 1
    assert session.episode_id == 1


def test_serve_one_inference_error_counted_not_propagated(monkeypatch):
    server = make_logic_server()
    session, _ = open_session(server)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(server._policy, "predict_action_chunk", boom)
    deposit(session, make_obs())
    server._serve_one(session)  # must not raise
    assert server.metrics["errors_total"] == 1
    assert server.metrics["requests_total"] == 0
    assert session.stats.errors == 1


def test_serve_one_increments_requests_total():
    server = make_logic_server()
    session, _ = open_session(server)
    for seq in (1, 2):
        deposit(session, make_obs(), MsgHeader(seq_id=seq))
        server._serve_one(session)
    assert server.metrics["requests_total"] == 2
    assert server.metrics["errors_total"] == 0
    assert session.stats.requests == 2


# ---------------------------------------------------------------------------
# Episode reset semantics (session-level, as used by _on_reset_query)
# ---------------------------------------------------------------------------


def test_session_reset_episode_clears_state():
    server = make_logic_server()
    session, _ = open_session(server)
    deposit(session, make_obs())
    assert session.has_pending()
    session.reset_episode(5)
    assert not session.has_pending()  # mailbox cleared
    assert session.episode_id == 5
    assert session.preprocessor.reset_count == 1
    assert session.postprocessor.reset_count == 1
    session.reset_episode()  # no explicit id: increments
    assert session.episode_id == 6


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def test_warmup_runs_before_any_session():
    policy = MockChunkPolicy()
    server = make_logic_server(manifest=make_manifest(warmup_inferences=2), policy=policy)
    assert len(policy.calls) == 2
    assert len(server.registry) == 0  # warmup session is not registered
    for call in policy.calls:
        assert tuple(call["state"].shape) == (1, STATE_DIM)
        assert float(call["state"].abs().sum()) == 0.0  # synthetic zeros
    assert server.status_snapshot().warmed_up is True


def test_synthetic_observation_matches_input_features():
    server = make_logic_server()
    obs = server._synthetic_observation()
    assert obs.state.shape == (STATE_DIM,)
    assert obs.state.dtype == np.float32
    assert set(obs.images) == {CAMERA_KEY}
    assert obs.images[CAMERA_KEY].shape == (IMG_H, IMG_W, 3)
    assert obs.images[CAMERA_KEY].dtype == np.uint8
    assert obs.jpeg_quality == 0
