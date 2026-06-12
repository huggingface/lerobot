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

"""Real zenoh peer-to-peer loopback tests: PolicyServer ↔ RemoteInferenceEngine.

The server listens on a fresh loopback TCP port per test; the engine
dials it directly (``mode=peer``, no router).  Mock policy values are
deterministic — chunk_robot[t, j] = 2 * (state[j] + 0.01 t) — so first
actions identify which client's observation produced them (the
per-session isolation regression).  Chaos tests kill/restart the server
mid-episode and assert the engine degrades and recovers without ever
raising on the control thread.
"""

import time
from threading import Event

import pytest
import torch

pytest.importorskip("msgpack")
zenoh = pytest.importorskip("zenoh")

from lerobot.policy_server.schema import MsgHeader, obs_key  # noqa: E402
from lerobot.policy_server.zenoh_utils import build_zenoh_config  # noqa: E402
from lerobot.rollout.inference.factory import FallbackMode  # noqa: E402
from lerobot.rollout.inference.remote import ClientState, RemoteInferenceEngine  # noqa: E402
from tests.policy_server.conftest import (  # noqa: E402
    ACTION_DIM,
    ACTION_NAMES,
    TASK,
    free_tcp_port,
    make_logic_server,
    make_loopback_manifest,
    make_remote_config,
    make_robot_obs,
)

_FPS = 30.0
_TICK_S = 1.0 / _FPS
# Settle time after server.start() for zenoh declarations to propagate.
_DECLARATION_SETTLE_S = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_until(predicate, timeout_s: float, interval_s: float = 0.05) -> bool:
    """Poll ``predicate`` until true or the deadline passes (never a fixed sleep)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return bool(predicate())


def _start_loopback_server(port: int, attempts: int = 3):
    """Start a fully-injected PolicyServer listening on tcp/127.0.0.1:<port>."""
    last_error: Exception | None = None
    for _ in range(attempts):
        server = make_logic_server(make_loopback_manifest(port))
        try:
            server.start()
        except Exception as e:  # noqa: BLE001 — e.g. lingering socket on restart
            last_error = e
            server.stop()
            time.sleep(0.5)
            continue
        time.sleep(_DECLARATION_SETTLE_S)
        return server
    raise last_error


def _make_engine(port: int, server, hw_features: dict, **config_overrides) -> RemoteInferenceEngine:
    return RemoteInferenceEngine(
        config=make_remote_config(port, **config_overrides),
        policy_config=server._policy_cfg,
        hw_features=hw_features,
        ordered_action_keys=list(ACTION_NAMES),
        task=TASK,
        fps=_FPS,
        robot_type="mock",
        shutdown_event=Event(),
    )


def _start_engine(engine: RemoteInferenceEngine, attempts: int = 4) -> None:
    """Engine start with handshake retries (declarations may still be settling)."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            engine.start()
            return
        except ConnectionError as e:
            last_error = e
            engine.stop()
            time.sleep(0.3)
    raise last_error


def _collect_actions(engine: RemoteInferenceEngine, n: int, timeout_s: float) -> list[torch.Tensor]:
    """Poll ``get_action`` at ~30 Hz until ``n`` actions arrive or the deadline passes."""
    actions: list[torch.Tensor] = []
    deadline = time.monotonic() + timeout_s
    while len(actions) < n and time.monotonic() < deadline:
        action = engine.get_action(None)
        if action is not None:
            actions.append(action)
        time.sleep(_TICK_S)
    return actions


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_end_to_end_chunks(hw_features):
    port = free_tcp_port()
    server = _start_loopback_server(port)
    engine = _make_engine(port, server, hw_features)
    try:
        _start_engine(engine)
        engine.notify_observation(make_robot_obs(2.0))

        actions = _collect_actions(engine, n=20, timeout_s=15.0)
        assert len(actions) >= 20

        # chunk_robot[t, j] = 2 * (2.0 + 0.1 j + 0.01 t); the queue head is
        # trimmed by the (small, loopback) delay → within 0.1 of the t=0 value.
        first = actions[0]
        assert first.shape == (ACTION_DIM,)
        for j in range(ACTION_DIM):
            expected = 2.0 * (2.0 + 0.1 * j)
            assert abs(first[j].item() - expected) < 0.1, f"joint {j}: {first[j].item()} vs {expected}"

        assert engine.state == ClientState.STREAMING
        assert engine.ready
        assert engine.failed is False
        assert engine.stats["merges"] >= 1
    finally:
        engine.stop()
        server.stop()


@pytest.mark.timeout(60)
def test_multi_client_no_cross_contamination(hw_features):
    port = free_tcp_port()
    server = _start_loopback_server(port)
    engine_a = _make_engine(port, server, hw_features, client_uuid="client-a")
    engine_b = _make_engine(port, server, hw_features, client_uuid="client-b")
    try:
        _start_engine(engine_a)
        _start_engine(engine_b)
        engine_a.notify_observation(make_robot_obs(2.0))
        engine_b.notify_observation(make_robot_obs(7.0))

        actions_a = _collect_actions(engine_a, n=1, timeout_s=10.0)
        actions_b = _collect_actions(engine_b, n=1, timeout_s=10.0)
        assert actions_a, "engine A produced no actions"
        assert actions_b, "engine B produced no actions"

        # Each engine's first action must reflect ITS OWN observation seed:
        # 2*(2.0) = 4.0 for A, 2*(7.0) = 14.0 for B (gap 10.0 ≫ tolerance).
        first_a = actions_a[0][0].item()
        first_b = actions_b[0][0].item()
        assert abs(first_a - 4.0) < 0.3, f"engine A got {first_a} (cross-contamination?)"
        assert abs(first_b - 14.0) < 0.3, f"engine B got {first_b} (cross-contamination?)"
    finally:
        engine_a.stop()
        engine_b.stop()
        server.stop()


@pytest.mark.timeout(60)
def test_reset_roundtrip(hw_features):
    port = free_tcp_port()
    server = _start_loopback_server(port)
    engine = _make_engine(port, server, hw_features)
    try:
        _start_engine(engine)
        engine.notify_observation(make_robot_obs(2.0))
        assert _collect_actions(engine, n=3, timeout_s=10.0), "no actions before reset"

        merges_before = engine.stats["merges"]
        engine.reset()
        engine.notify_observation(make_robot_obs(5.0))

        # New merges land after the reset (worker keeps cycling).
        assert _wait_until(lambda: engine.stats["merges"] > merges_before, timeout_s=10.0)
        # The queue refills with post-reset chunks.
        assert _collect_actions(engine, n=1, timeout_s=5.0), "queue did not refill after reset"

        # Server-side session advanced to the new episode (via the acked
        # reset query, or via the episode bump in the next obs header).
        def _episode_advanced() -> bool:
            sessions = server.registry.snapshot()
            return bool(sessions) and sessions[0].episode_id >= 1

        assert _wait_until(_episode_advanced, timeout_s=8.0), "server session episode_id never advanced"
        assert engine.failed is False
    finally:
        engine.stop()
        server.stop()


# ---------------------------------------------------------------------------
# Chaos: server death / restart
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_server_death_is_safe(hw_features):
    port = free_tcp_port()
    server = _start_loopback_server(port)
    engine = _make_engine(port, server, hw_features)
    try:
        _start_engine(engine)
        engine.notify_observation(make_robot_obs(2.0))
        assert _collect_actions(engine, n=5, timeout_s=10.0), "no actions before server death"

        server.stop()

        # Keep ticking at 30 Hz for ~2 s: get_action must never raise.
        results = []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            results.append(engine.get_action(None))
            time.sleep(_TICK_S)

        # The local queue drains and HOLD fallback yields None.
        assert all(result is None for result in results[-5:]), "queue never drained to HOLD fallback"
        # max_offline_s=8 not reached → not failed, in a degraded-but-alive state.
        assert engine.failed is False
        assert engine.state in {ClientState.DEGRADED, ClientState.STALLED, ClientState.RECONNECTING}
    finally:
        engine.stop()
        server.stop()


@pytest.mark.timeout(60)
def test_server_restart_recovery(hw_features):
    port = free_tcp_port()
    server = _start_loopback_server(port)
    engine = _make_engine(port, server, hw_features, max_offline_s=45.0)
    new_server = None
    try:
        _start_engine(engine)
        engine.notify_observation(make_robot_obs(2.0))
        assert _collect_actions(engine, n=3, timeout_s=10.0), "no actions before server death"

        server.stop()
        # Let the engine notice the death (liveliness drop / request timeout).
        _wait_until(lambda: engine.state != ClientState.STREAMING, timeout_s=5.0)

        new_server = _start_loopback_server(port)

        # Re-handshake: bounded by the engine backoff and zenoh's TCP
        # reconnect period — poll generously rather than sleeping.
        reconnected = _wait_until(lambda: engine.stats["reconnects"] >= 1, timeout_s=25.0, interval_s=0.1)
        assert reconnected, f"engine never re-handshook (state={engine.state})"

        engine.notify_observation(make_robot_obs(2.0))
        actions = _collect_actions(engine, n=3, timeout_s=10.0)
        assert len(actions) >= 3, "no actions after server restart"
        assert abs(actions[-1][0].item() - 4.0) < 0.3
        assert engine.failed is False
    finally:
        engine.stop()
        server.stop()
        if new_server is not None:
            new_server.stop()


# ---------------------------------------------------------------------------
# Robustness: unknown clients, fallback modes
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_unknown_client_dropped(hw_features):
    port = free_tcp_port()
    server = _start_loopback_server(port)
    intruder = None
    engine = None
    try:
        intruder = zenoh.open(build_zenoh_config(mode="peer", connect_endpoints=[f"tcp/127.0.0.1:{port}"]))
        key = obs_key(server.prefix, "intruder-uuid")
        header_bytes = MsgHeader().pack()  # valid header, garbage body, no session

        deadline = time.monotonic() + 8.0
        while server.metrics["dropped_unknown_client_total"] < 1 and time.monotonic() < deadline:
            intruder.put(key, b"\xde\xad\xbe\xef", attachment=header_bytes)
            time.sleep(0.1)
        assert server.metrics["dropped_unknown_client_total"] >= 1
        assert len(server.registry) == 0

        # The server stays healthy: a legitimate engine still works.
        engine = _make_engine(port, server, hw_features)
        _start_engine(engine)
        engine.notify_observation(make_robot_obs(2.0))
        actions = _collect_actions(engine, n=1, timeout_s=10.0)
        assert actions, "legitimate engine got no actions after garbage traffic"
        assert abs(actions[0][0].item() - 4.0) < 0.3
    finally:
        if engine is not None:
            engine.stop()
        if intruder is not None:
            intruder.close()
        server.stop()


@pytest.mark.timeout(60)
def test_fallback_zero(hw_features):
    port = free_tcp_port()
    server = _start_loopback_server(port)
    engine = _make_engine(port, server, hw_features, fallback=FallbackMode.ZERO)
    try:
        _start_engine(engine)
        engine.notify_observation(make_robot_obs(2.0))
        # With ZERO fallback an empty queue already yields zeros, so wait
        # for a *streamed* (nonzero ~4.0) action to prove chunks flowed.
        streamed = False
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            action = engine.get_action(None)
            if action is not None and torch.count_nonzero(action) > 0:
                streamed = True
                break
            time.sleep(_TICK_S)
        assert streamed, "no streamed (nonzero) actions before server death"

        server.stop()

        # Drain the local queue; once dry, ZERO fallback must return an
        # explicit zero command (never None) of the action dimension.
        saw_zero = False
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            action = engine.get_action(None)
            assert action is not None, "FallbackMode.ZERO returned None"
            if torch.count_nonzero(action) == 0:
                assert action.shape == (len(ACTION_NAMES),)
                saw_zero = True
                break
            time.sleep(_TICK_S)
        assert saw_zero, "queue never drained to the zero fallback"
        assert engine.failed is False
    finally:
        engine.stop()
        server.stop()
