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

"""Remote inference engine: network-decoupled policy inference over Zenoh.

The same architecture as :class:`RTCInferenceEngine` with the thread
boundary replaced by a network boundary.  The edge stays **weightless**
(no policy weights, no policy processors); a ``lerobot-policy-server``
runs the heavy half.  All chunk state — leftover prefixes, latency
tracking, delay computation — lives client-side in the existing
``ActionQueue``/``LatencyTracker`` machinery, so the server is stateless
per request and a server crash loses zero control state.

Threading model:
- **Main thread** (strategy loop): ``notify_observation`` writes a
  latest-only slot; ``get_action`` pops the local queue and applies the
  staleness bound + fallback ladder.  Never any I/O.
- **Network worker** (one daemon thread): self-clocked by
  ``buffer_time_s``, publishes one observation, awaits its chunk (or
  timeout), merges, repeats.  One-in-flight is a *correctness*
  requirement: ``idx_before``/prefix snapshots must serialize with
  merges.
- **Zenoh threads**: deposit-only callbacks (chunk → bounded queue,
  liveliness → event).

Clock iron rule: wall-clock instants never cross machines.  The header's
``client_mono_ns`` is opaque to the server and echoed back; the server
reports only durations.
"""

from __future__ import annotations

import contextlib
import logging
import math
import queue as queue_module
import time
import traceback
import uuid as uuid_module
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.policies.rtc import ActionQueue, LatencyTracker
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policy_server import codec
from lerobot.policy_server.schema import (
    MSG_TYPE_OBS,
    SCHEMA_VERSION,
    MsgHeader,
    ObservationMsg,
    ResetMsg,
    SessionAckMsg,
    SessionCloseMsg,
    SessionOpenMsg,
    action_key,
    client_alive_key,
    obs_key,
    reset_key,
    sanitize_key_segment,
    server_alive_key,
    service_prefix,
    session_key,
    status_key,
)
from lerobot.policy_server.zenoh_utils import build_zenoh_config, import_zenoh, obs_publisher_qos
from lerobot.utils.constants import OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine

if TYPE_CHECKING:
    from lerobot.configs.policies import PreTrainedConfig

    from .factory import RemoteInferenceConfig

logger = logging.getLogger(__name__)

_IDLE_SLEEP_S = 0.01
_MAX_CONSECUTIVE_WORKER_ERRORS = 10


class ClientState:
    """Fail-safe state machine states (see the design doc §9.2)."""

    CONNECTING = "CONNECTING"
    STREAMING = "STREAMING"
    DEGRADED = "DEGRADED"
    STALLED = "STALLED"
    RECONNECTING = "RECONNECTING"
    DEAD = "DEAD"


class RemoteInferenceEngine(InferenceEngine):
    """``--inference.type=remote``: weightless edge client of a policy server."""

    def __init__(
        self,
        config: RemoteInferenceConfig,
        policy_config: PreTrainedConfig,
        hw_features: dict,
        ordered_action_keys: list[str],
        task: str,
        fps: float,
        robot_type: str,
        rename_map: dict[str, str] | None = None,
        shutdown_event: Event | None = None,
    ) -> None:
        self._config = config
        self._policy_config = policy_config
        self._hw_features = hw_features
        self._ordered_action_keys = list(ordered_action_keys)
        self._task = task
        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._robot_type = robot_type
        self._rename_map = dict(rename_map or {})
        self._global_shutdown_event = shutdown_event

        self._client_uuid = sanitize_key_segment(config.client_uuid or uuid_module.uuid4().hex)
        model_id = config.service_model_id or getattr(policy_config, "pretrained_path", "") or "model"
        self._prefix = service_prefix(model_id, config.service_revision, config.service_task or task)

        # Latest-only observation slot (identical to rtc.py's _obs_holder).
        self._obs_holder: dict[str, Any] = {"obs": None}
        self._obs_lock = Lock()

        self._action_queue: ActionQueue | None = None
        self._latency_tracker = LatencyTracker()
        self._effective_rtc: RTCConfig = config.rtc

        # Replies deposited by the zenoh callback, consumed by the worker.
        self._reply_queue: queue_module.Queue[tuple[MsgHeader, bytes]] = queue_module.Queue(maxsize=4)

        self._zenoh = None
        self._obs_publisher = None
        self._declarations: list[Any] = []
        self._alive_token = None
        self._server_alive = Event()

        self._worker: Thread | None = None
        self._stop_event = Event()
        self._active = Event()
        self._dead = Event()
        self._session_ack: SessionAckMsg | None = None

        self.state = ClientState.CONNECTING
        self._state_lock = Lock()
        self._seq_id = 0
        self._epoch = 0
        self._episode_id = 0
        self._pending_reset = False

        # Staleness bookkeeping: client-monotonic send time of the
        # observation that produced the current queue contents.
        # _anchor_lock serializes {merge + anchor update} (worker),
        # {staleness clear} (control thread), and {reset clear} so a
        # stale chunk can never merge into a freshly-reset queue and the
        # safety path can never clear a just-merged one.
        self._anchor_lock = Lock()
        self._chunk_anchor_mono: float | None = None
        self._last_chunk_mono: float | None = None
        self._offline_since_mono: float | None = None
        self._last_action: torch.Tensor | None = None

        self.stats: dict[str, float] = {
            "requests": 0,
            "timeouts": 0,
            "merges": 0,
            "stale_drops": 0,
            "fallback_ticks": 0,
            "reconnects": 0,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """Session opened, capabilities validated, server warmed up."""
        ack = self._session_ack
        return ack is not None and ack.warmed_up and not self._dead.is_set()

    @property
    def failed(self) -> bool:
        return self._dead.is_set()

    @property
    def action_queue(self) -> ActionQueue | None:
        return self._action_queue

    def start(self) -> None:
        """Open transport, handshake, start the network worker.

        Raises on initial connection/validation failure so a bad
        deployment aborts before the robot moves (reconnect logic only
        guards established sessions).
        """
        zenoh = import_zenoh()
        cfg = self._config
        self._zenoh = zenoh.open(
            build_zenoh_config(
                mode=cfg.zenoh_mode,
                connect_endpoints=[cfg.connect_endpoint] if cfg.connect_endpoint else None,
                tls_root_ca_certificate=cfg.tls_ca,
                tls_connect_certificate=cfg.tls_cert,
                tls_connect_private_key=cfg.tls_key,
            )
        )

        try:
            ack = self._handshake(initial=True)
        except Exception:
            # Fail fast without leaking the transport session.
            with contextlib.suppress(Exception):
                self._zenoh.close()
            self._zenoh = None
            raise
        self._configure_from_ack(ack)

        handlers = zenoh.handlers
        self._declarations.append(
            self._zenoh.declare_subscriber(
                action_key(self._prefix, self._client_uuid), handlers.Callback(self._on_chunk)
            )
        )
        self._obs_publisher = self._zenoh.declare_publisher(
            obs_key(self._prefix, self._client_uuid), **obs_publisher_qos(zenoh)
        )
        self._declarations.append(self._obs_publisher)
        self._server_alive.set()
        self._declarations.append(
            self._zenoh.liveliness().declare_subscriber(
                server_alive_key(self._prefix), handlers.Callback(self._on_server_liveliness), history=True
            )
        )
        self._alive_token = self._zenoh.liveliness().declare_token(
            client_alive_key(self._prefix, self._client_uuid)
        )

        self._stop_event.clear()
        self._dead.clear()
        self._active.set()
        self._worker = Thread(target=self._worker_loop, daemon=True, name="RemoteInference")
        self._worker.start()
        logger.info(
            "Remote inference started: prefix=%s client=%s rtc=%s",
            self._prefix,
            self._client_uuid,
            self._effective_rtc.enabled,
        )

    def stop(self) -> None:
        logger.info("Stopping remote inference engine...")
        self._stop_event.set()
        self._active.clear()
        if self._worker is not None and self._worker.is_alive():
            # Worst case the worker is mid-handshake inside _enter_reconnect.
            join_timeout = max(3.0, self._config.handshake_timeout_s + self._config.request_timeout_s + 2.0)
            self._worker.join(timeout=join_timeout)
            if self._worker.is_alive():
                logger.warning("Remote inference worker did not join")
        self._worker = None

        if self._zenoh is not None:
            # Best-effort graceful close; the server also GCs on liveliness drop.
            with contextlib.suppress(Exception):
                self._control_query(
                    session_key(self._prefix),
                    codec.encode_session_close(
                        SessionCloseMsg(
                            client_uuid=self._client_uuid,
                            session_id=self._session_ack.session_id if self._session_ack else "",
                        )
                    ),
                    timeout=1.0,
                )
            if self._alive_token is not None:
                with contextlib.suppress(Exception):
                    self._alive_token.undeclare()
                self._alive_token = None
            for declaration in self._declarations:
                with contextlib.suppress(Exception):
                    declaration.undeclare()
            self._declarations.clear()
            self._obs_publisher = None
            with contextlib.suppress(Exception):
                self._zenoh.close()
            self._zenoh = None
        logger.info("Remote inference engine stopped")

    def pause(self) -> None:
        """Stop publishing observations; the local queue stays intact."""
        logger.info("Pausing remote inference (publishing stops, queue intact)")
        self._active.clear()

    def resume(self) -> None:
        logger.info("Resuming remote inference")
        self._active.set()

    def reset(self) -> None:
        """Episode boundary: clear local chunk state, notify the server.

        The acked reset query runs on the worker thread (never I/O on the
        control thread); thanks to per-request server statelessness a
        lost ack only costs a warning — the next observation announces
        the new episode in its header anyway.
        """
        logger.info("Resetting remote inference state (queue + episode)")
        with self._anchor_lock:
            if self._action_queue is not None:
                self._action_queue.clear()
            self._chunk_anchor_mono = None
            with self._state_lock:
                self._episode_id += 1
                self._pending_reset = True
        with self._obs_lock:
            # The previous episode's final frame must not seed the new
            # episode's first request.
            self._obs_holder["obs"] = None
        self._last_action = None
        # LatencyTracker intentionally survives reset: latency is
        # episode-invariant (parity with local RTC).

    # ------------------------------------------------------------------
    # Action production (main thread — never any I/O)
    # ------------------------------------------------------------------

    def notify_observation(self, obs: dict) -> None:
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        queue = self._action_queue
        if queue is None:
            return None

        # Staleness bound (sync safety): never execute an action whose
        # source observation is older than max_action_age_s.  The lock
        # makes the check-and-clear atomic with the worker's merge.
        with self._anchor_lock:
            anchor = self._chunk_anchor_mono
            if (
                anchor is not None
                and queue.qsize() > 0
                and time.monotonic() - anchor > self._config.max_action_age_s
            ):
                logger.warning(
                    "Dropping %d stale actions (older than %.1fs) — applying fallback",
                    queue.qsize(),
                    self._config.max_action_age_s,
                )
                self.stats["stale_drops"] += 1
                queue.clear()
                self._chunk_anchor_mono = None

        action = queue.get()
        if action is not None:
            self._last_action = action
            return action

        self._set_state(ClientState.STALLED if self.state == ClientState.DEGRADED else self.state)
        return self._fallback_action()

    def _fallback_action(self) -> torch.Tensor | None:
        from .factory import FallbackMode

        mode = self._config.fallback
        if mode == FallbackMode.REPEAT_LAST and self._last_action is not None:
            self.stats["fallback_ticks"] += 1
            return self._last_action.clone()
        if mode == FallbackMode.ZERO:
            # For velocity-controlled robots "send nothing" means "keep
            # last velocity" — an explicit zero command is the safe stop.
            self.stats["fallback_ticks"] += 1
            return torch.zeros(len(self._ordered_action_keys))
        return None  # HOLD: send_next_action tolerates None

    # ------------------------------------------------------------------
    # Handshake & control plane
    # ------------------------------------------------------------------

    def _handshake(self, initial: bool) -> SessionAckMsg:
        """status (pre-flight) + session open; raises on rejection."""
        cfg = self._config
        status_data = self._control_query(status_key(self._prefix), b"", timeout=cfg.handshake_timeout_s)
        if status_data is None:
            raise ConnectionError(
                f"No policy server answered status query at {status_key(self._prefix)!r} "
                f"via {cfg.connect_endpoint!r} (timeout {cfg.handshake_timeout_s}s)"
            )
        status = codec.decode_status(status_data)
        logger.info(
            "Server status: model=%s@%s policy=%s sessions=%d/%d warmed_up=%s",
            status.model_repo,
            status.model_revision,
            status.policy_type,
            status.active_sessions,
            status.max_sessions,
            status.warmed_up,
        )

        open_msg = SessionOpenMsg(
            client_uuid=self._client_uuid,
            robot_type=self._robot_type,
            policy_type=getattr(self._policy_config, "type", ""),
            fps=self._fps,
            action_names=self._ordered_action_keys,
            camera_names=self._wire_camera_names(),
            state_dim=self._state_dim(),
            schema_version=SCHEMA_VERSION,
            rtc_enabled=cfg.rtc.enabled,
            task=self._task,
            tags=cfg.tags,
        )
        ack_data = self._control_query(
            session_key(self._prefix), codec.encode_session_open(open_msg), timeout=cfg.request_timeout_s
        )
        if ack_data is None:
            raise ConnectionError("Session open query timed out")
        ack = codec.decode_session_ack(ack_data)
        if not ack.accepted:
            raise ConnectionError(f"Policy server rejected the session: {ack.error}")
        for warning in ack.warnings:
            logger.warning("Server warning: %s", warning)

        # Hard sync-safety contract: chunk columns map to motors by order.
        if ack.action_names and ack.action_names != self._ordered_action_keys:
            raise ValueError(
                "Action name/order mismatch between server policy and this robot.\n"
                f"  server: {ack.action_names}\n  client: {self._ordered_action_keys}"
            )
        if not initial and self._session_ack is not None:
            previous = self._session_ack
            if (ack.model_repo, ack.model_revision) != (previous.model_repo, previous.model_revision):
                raise ValueError(
                    f"Server model changed across reconnect "
                    f"({previous.model_repo}@{previous.model_revision} → "
                    f"{ack.model_repo}@{ack.model_revision}) — refusing to execute wrong-model chunks"
                )
        return ack

    def _configure_from_ack(self, ack: SessionAckMsg) -> None:
        rtc_requested = self._config.rtc.enabled
        rtc_effective = rtc_requested and ack.supports_rtc
        if rtc_requested and not rtc_effective:
            logger.warning("RTC downgraded to chunk-append (server does not support RTC)")
        if self._action_queue is not None and self._action_queue.cfg.enabled != rtc_effective:
            # The queue's merge semantics (replace vs append) were fixed at
            # session start; a server whose RTC capability changed across a
            # reconnect would corrupt them.
            raise ValueError(
                "Server RTC capability changed across reconnect "
                f"(queue merge mode {'replace' if self._action_queue.cfg.enabled else 'append'} "
                f"vs server RTC={rtc_effective}) — refusing to continue"
            )
        self._effective_rtc = RTCConfig(
            enabled=rtc_effective,
            prefix_attention_schedule=self._config.rtc.prefix_attention_schedule,
            max_guidance_weight=self._config.rtc.max_guidance_weight,
            execution_horizon=ack.rtc_execution_horizon or self._config.rtc.execution_horizon,
            debug=self._config.rtc.debug,
            debug_maxlen=self._config.rtc.debug_maxlen,
        )
        if self._action_queue is None:
            self._action_queue = ActionQueue(self._effective_rtc)
        self._session_ack = ack

    def _control_query(self, key: str, payload: bytes, timeout: float) -> bytes | None:
        """One request/reply on the control plane; None on timeout/no-server."""
        zenoh = import_zenoh()
        try:
            replies = self._zenoh.get(
                key,
                handler=zenoh.handlers.FifoChannel(4),
                payload=payload,
                timeout=timeout,
            )
            deadline = time.monotonic() + timeout + 0.5
            while time.monotonic() < deadline:
                try:
                    reply = replies.try_recv()
                except Exception:  # zenoh.ZError: channel closed (no queryable / finished)
                    return None
                if reply is None:
                    time.sleep(0.005)
                    continue
                if reply.ok is not None:
                    return reply.ok.payload.to_bytes()
                return None  # Reply.err (e.g. b"Timeout")
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("Control query %s failed: %s", key, e)
            return None

    # ------------------------------------------------------------------
    # Zenoh callbacks (deposit-only)
    # ------------------------------------------------------------------

    def _on_chunk(self, sample: Any) -> None:
        try:
            attachment = sample.attachment
            if attachment is None:
                return
            header = MsgHeader.unpack(attachment.to_bytes())
            item = (header, sample.payload.to_bytes())
            try:
                self._reply_queue.put_nowait(item)
            except queue_module.Full:
                # Drop oldest, keep newest.
                with contextlib.suppress(queue_module.Empty):
                    self._reply_queue.get_nowait()
                with contextlib.suppress(queue_module.Full):
                    self._reply_queue.put_nowait(item)
        except Exception as e:  # noqa: BLE001
            logger.error("chunk callback error: %s", e)

    def _on_server_liveliness(self, sample: Any) -> None:
        try:
            import zenoh

            if sample.kind == zenoh.SampleKind.DELETE:
                logger.warning("Server liveliness token dropped")
                self._server_alive.clear()
            else:
                self._server_alive.set()
        except Exception as e:  # noqa: BLE001
            logger.error("liveliness callback error: %s", e)

    # ------------------------------------------------------------------
    # Network worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        consecutive_errors = 0
        try:
            while not self._stop_event.is_set():
                if not self._active.is_set():
                    time.sleep(_IDLE_SLEEP_S)
                    continue
                try:
                    self._maybe_send_reset()

                    if not self._server_alive.is_set():
                        self._enter_reconnect("server liveliness dropped")
                        continue

                    queue = self._action_queue
                    if queue is not None and queue.qsize() * self._dt > self._config.buffer_time_s:
                        time.sleep(_IDLE_SLEEP_S)
                        continue

                    with self._obs_lock:
                        obs = self._obs_holder.get("obs")
                    if obs is None:
                        time.sleep(_IDLE_SLEEP_S)
                        continue

                    self._request_cycle(obs)
                    consecutive_errors = 0
                except ConnectionError as e:
                    # Raised by reconnect on hard contract violations.
                    raise e
                except Exception as e:  # noqa: BLE001 — transient worker errors retry
                    consecutive_errors += 1
                    logger.error(
                        "Remote inference worker error (%d/%d): %s",
                        consecutive_errors,
                        _MAX_CONSECUTIVE_WORKER_ERRORS,
                        e,
                    )
                    logger.debug(traceback.format_exc())
                    if consecutive_errors >= _MAX_CONSECUTIVE_WORKER_ERRORS:
                        raise
                    time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            logger.error("Fatal error in remote inference worker: %s", e)
            logger.error(traceback.format_exc())
            self._go_dead(str(e))

    def _request_cycle(self, obs: dict) -> None:
        """Publish one observation and merge its chunk (one-in-flight)."""
        cfg = self._config
        queue = self._action_queue

        obs_frame = build_dataset_frame(self._hw_features, obs, prefix=OBS_STR)
        if self._rename_map:
            obs_frame = {self._rename_map.get(k, k): v for k, v in obs_frame.items()}

        state = obs_frame.pop(OBS_STATE, None)
        images = {k: v for k, v in obs_frame.items() if isinstance(v, np.ndarray) and v.ndim == 3}

        with self._state_lock:
            self._seq_id += 1
            seq_id = self._seq_id
            episode_id = self._episode_id
            epoch = self._epoch

        # Snapshot RTC state (must precede the publish; merge validates
        # against idx_before).
        idx_before = queue.get_action_index()
        prefix_model: np.ndarray | None = None
        prefix_robot: np.ndarray | None = None
        delay_steps = 0
        if self._effective_rtc.enabled:
            horizon = self._effective_rtc.execution_horizon
            left_over = queue.get_left_over()
            if left_over is not None and left_over.numel():
                prefix_model = left_over[:horizon].to(torch.float32).numpy()
            processed_left_over = queue.get_processed_left_over()
            if processed_left_over is not None and processed_left_over.numel():
                prefix_robot = processed_left_over[:horizon].to(torch.float32).numpy()
            max_latency = self._latency_tracker.max() if len(self._latency_tracker) else 0.0
            delay_steps = math.ceil(max_latency / self._dt) if max_latency else 0

        # A reset/reconnect between the counter snapshot and the prefix
        # snapshot would pair a new episode id with old-episode prefixes
        # — skip the cycle instead.
        with self._state_lock:
            if (self._episode_id, self._epoch) != (episode_id, epoch):
                return

        header = MsgHeader(
            schema_version=SCHEMA_VERSION,
            msg_type=MSG_TYPE_OBS,
            seq_id=seq_id,
            episode_id=episode_id,
            client_mono_ns=time.monotonic_ns(),
            session_epoch=epoch,
        )
        msg = ObservationMsg(
            state=state,
            images=images,
            task=self._task,
            inference_delay_steps=delay_steps,
            prefix_model=prefix_model,
            prefix_robot=prefix_robot,
            episode_start=(queue.qsize() == 0 and idx_before == 0 and self._chunk_anchor_mono is None),
            jpeg_quality=cfg.jpeg_quality,
        )

        t_send = time.perf_counter()
        self._obs_publisher.put(codec.encode_observation(msg), attachment=header.pack())
        self.stats["requests"] += 1

        reply = self._await_chunk(seq_id, episode_id, epoch, timeout=cfg.request_timeout_s)
        if reply is None:
            self.stats["timeouts"] += 1
            self._on_request_timeout()
            return

        chunk = codec.decode_action_chunk(reply)
        if chunk.chunk_model is None or chunk.chunk_robot is None:
            # A persistently malformed server must still trip the
            # degradation ladder, not stall in nominal state.
            logger.warning("Chunk for seq=%d had empty tensors — dropping", seq_id)
            self.stats["timeouts"] += 1
            self._on_request_timeout()
            return

        latency = time.perf_counter() - t_send
        real_delay = math.ceil(latency / self._dt)
        with self._anchor_lock:
            # reset() takes the same lock before clearing: either the
            # reset fully precedes this merge (episode changed → drop the
            # stale chunk) or the merge completes first (and the reset
            # then clears it) — a stale chunk can never survive a reset.
            with self._state_lock:
                if (self._episode_id, self._epoch) != (episode_id, epoch):
                    logger.debug("Dropping chunk seq=%d: episode/epoch changed mid-flight", seq_id)
                    return
            queue.merge(
                torch.from_numpy(np.ascontiguousarray(chunk.chunk_model)),
                torch.from_numpy(np.ascontiguousarray(chunk.chunk_robot)),
                real_delay,
                idx_before,
            )
            self._chunk_anchor_mono = time.monotonic() - latency  # ≈ when the source obs was sent
        self._latency_tracker.add(latency)
        self._last_chunk_mono = time.monotonic()
        self._offline_since_mono = None
        self.stats["merges"] += 1
        self._set_state(ClientState.STREAMING)
        logger.debug(
            "merge: seq=%d latency=%.0fms delay=%d queue=%d server(inf=%.0fms wait=%.0fms load=%.2f)",
            seq_id,
            latency * 1e3,
            real_delay,
            queue.qsize(),
            chunk.inference_ms,
            chunk.queue_wait_ms,
            chunk.server_load,
        )

    def _await_chunk(self, seq_id: int, episode_id: int, epoch: int, timeout: float) -> bytes | None:
        """Wait for the chunk answering the latest outstanding request.

        Stale replies (older seq/episode/epoch) are dropped — under
        one-in-flight a late chunk can only ever answer an older request.
        """
        deadline = time.monotonic() + timeout
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                header, payload = self._reply_queue.get(timeout=min(remaining, 0.1))
            except queue_module.Empty:
                continue
            if header.session_epoch != epoch or header.episode_id != episode_id:
                continue  # stale epoch/episode (reset or reconnect happened)
            if header.seq_id != seq_id:
                continue  # late reply to a superseded request
            return payload
        return None

    def _maybe_send_reset(self) -> None:
        with self._state_lock:
            pending, episode_id = self._pending_reset, self._episode_id
            self._pending_reset = False
        if pending and self._zenoh is not None:
            ack_data = self._control_query(
                reset_key(self._prefix, self._client_uuid),
                codec.encode_reset(ResetMsg(client_uuid=self._client_uuid, episode_id=episode_id)),
                timeout=1.0,
            )
            if ack_data is None:
                # Harmless: the server is stateless per request and the next
                # observation header announces the new episode anyway.
                logger.warning("Reset ack not received (continuing — header carries the episode bump)")

    # ------------------------------------------------------------------
    # Degradation / reconnect / death
    # ------------------------------------------------------------------

    def _on_request_timeout(self) -> None:
        if self._stop_event.is_set():
            # _await_chunk aborted by a normal stop(), not by the network.
            return
        now = time.monotonic()
        if self._offline_since_mono is None:
            self._offline_since_mono = now
        offline_for = now - self._offline_since_mono

        queue = self._action_queue
        if queue is not None and queue.qsize() > 0:
            self._set_state(ClientState.DEGRADED)
        else:
            self._set_state(ClientState.STALLED)

        last = self._last_chunk_mono or 0.0
        if (now - last if last else offline_for) >= self._config.degraded_after_s:
            logger.warning(
                "No chunk for %.1fs (queue=%d) — %s",
                offline_for,
                queue.qsize() if queue else 0,
                self.state,
            )
        if offline_for > self._config.max_offline_s:
            self._go_dead(f"offline for {offline_for:.0f}s (> max_offline_s)")
            return
        if not self._server_alive.is_set() or offline_for >= 2 * self._config.request_timeout_s:
            self._enter_reconnect(f"request timeouts for {offline_for:.0f}s")

    def _enter_reconnect(self, reason: str) -> None:
        """Backoff + re-handshake loop. Hard contract violations → DEAD."""
        self._set_state(ClientState.RECONNECTING)
        logger.warning("Reconnecting: %s", reason)
        if self._offline_since_mono is None:
            self._offline_since_mono = time.monotonic()
        backoff = self._config.reconnect_initial_backoff_s
        while not self._stop_event.is_set():
            if not self._active.is_set():
                # Paused (e.g. DAgger human correction): keep trying to
                # reconnect, but a pause must never burn the offline budget
                # into a mid-correction shutdown.
                self._offline_since_mono = time.monotonic()
            offline_for = time.monotonic() - self._offline_since_mono
            if offline_for > self._config.max_offline_s:
                self._go_dead(f"offline for {offline_for:.0f}s (> max_offline_s)")
                return
            self._stop_event.wait(timeout=backoff)
            if self._stop_event.is_set():
                return
            backoff = min(backoff * 2, self._config.reconnect_max_backoff_s)
            try:
                with self._state_lock:
                    self._epoch += 1
                ack = self._handshake(initial=False)
                self._configure_from_ack(ack)
            except ValueError as e:
                # Capability/schema/model mismatch: never execute wrong-model chunks.
                self._go_dead(str(e))
                return
            except Exception as e:  # noqa: BLE001 — server still down, keep trying
                logger.info("Re-handshake failed (%s) — retrying in %.1fs", e, backoff)
                continue
            # A successful handshake is proof of life even if the liveliness
            # PUT was missed or hasn't been delivered yet.
            self._server_alive.set()
            # The offline budget is only reset by the next successful merge:
            # a server that handshakes but never delivers chunks must still
            # run out of budget and go DEAD.
            self.stats["reconnects"] += 1
            self._set_state(ClientState.STREAMING)
            logger.info("Reconnected (epoch=%d, session=%s)", self._epoch, ack.session_id)
            return

    def _go_dead(self, reason: str) -> None:
        if self._dead.is_set():
            return
        logger.error("Remote inference DEAD: %s", reason)
        self._set_state(ClientState.DEAD)
        self._dead.set()
        if self._global_shutdown_event is not None:
            self._global_shutdown_event.set()

    def _set_state(self, new_state: str) -> None:
        if new_state != self.state:
            logger.info("Client state: %s → %s", self.state, new_state)
            self.state = new_state

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _wire_camera_names(self) -> list[str]:
        names = [
            key for key, feature in self._hw_features.items() if feature.get("dtype") in ("image", "video")
        ]
        return [self._rename_map.get(name, name) for name in names]

    def _state_dim(self) -> int:
        state_feature = self._hw_features.get(OBS_STATE)
        if state_feature and state_feature.get("names"):
            return len(state_feature["names"])
        return 0
