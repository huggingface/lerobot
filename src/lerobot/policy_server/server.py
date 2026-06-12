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

"""``lerobot-policy-server``: multi-client GPU inference over Zenoh.

One process serves one pre-warmed (model, revision, dtype, device) to up
to ``max_sessions`` robot clients.  The process is **stateless per
request**: clients ship RTC prefixes and a delay hint with every
observation, so a server crash loses zero control state and reconnects
are trivial.

Concurrency model (pure threads — zenoh-python has no asyncio API):

    zenoh subscriber (.../*/obs)          inference worker (1 thread, owns GPU)
      deposit-only callback:                loop:
      session.deposit(header, body)  ──►      pick next session with pending obs (RR)
      (per-client latest-only slot)           decode → per-session preprocess
                                              predict_action_chunk(delay, prefix)
    control queryables (status/session/       per-session postprocess → encode
      reset): validate, mutate session        publisher.put(.../<uuid>/action)
      registry, reply inline

The single worker thread serializes GPU access; newest-wins mailboxes
mean overload degrades into longer cycle times (larger but correct
client delays), never into queue buildup.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import logging
import threading
import time
import traceback
import uuid as uuid_module
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.relative import reanchor_relative_rtc_prefix
from lerobot.policies.utils import populate_queues, prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from . import codec
from .manifest import PolicyServerManifest
from .scheduler import RoundRobinScheduler, Scheduler
from .schema import (
    SCHEMA_VERSION,
    ActionChunkMsg,
    MsgHeader,
    ObservationMsg,
    ResetAckMsg,
    SessionAckMsg,
    SessionCloseMsg,
    SessionOpenMsg,
    StatusMsg,
    action_key,
    client_alive_key,
    client_alive_wildcard,
    client_uuid_from_key,
    obs_wildcard,
    reset_wildcard,
    server_alive_key,
    service_prefix,
    session_key,
    status_key,
)
from .session import Session, SessionRegistry
from .validation import (
    PolicyClassification,
    classify_policy,
    resolve_serving_mode,
    validate_session_open,
)
from .zenoh_utils import action_publisher_qos, build_zenoh_config, import_zenoh

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("lerobot.policy_server.audit")

# Grace period after a client liveliness token drops before its session
# is garbage-collected (rides out router blips and reconnects).
_LIVELINESS_GC_GRACE_S = 5.0
# Worker idle wait between work-event checks (also paces the GC sweep).
_WORKER_IDLE_WAIT_S = 0.05


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or truncate RTC prefix actions to a fixed length (mirrors rtc.py)."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


class PolicyServer:
    """Zenoh policy server: control-plane queryables + data-plane pub/sub + one GPU worker."""

    def __init__(
        self,
        manifest: PolicyServerManifest,
        *,
        policy: PreTrainedPolicy | None = None,
        policy_cfg: PreTrainedConfig | None = None,
        processor_factory: Callable[[], tuple[Any, Any]] | None = None,
        classification: PolicyClassification | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        """``policy``/``policy_cfg``/``processor_factory``/``classification``
        are injection points for tests; production loads everything from
        the manifest via :meth:`load_policy`.
        """
        self._manifest = manifest
        self._device = torch.device(manifest.model.device)
        self._policy = policy
        self._policy_cfg = policy_cfg
        self._processor_factory = processor_factory
        self._classification = classification
        self._scheduler = scheduler or RoundRobinScheduler()

        self._serving_mode: str = ""
        self._max_sessions: int = manifest.max_sessions
        self._rtc_active = False
        self._warmed_up = False

        self.registry = SessionRegistry()
        self._registry_lock = threading.Lock()  # serializes open/close/GC decisions
        # Serializes inference against episode resets: in exclusive mode a
        # reset (policy.reset(), pipeline reset) arriving on a queryable
        # thread mid-predict would corrupt the in-flight request's state.
        self._inference_lock = threading.Lock()

        self._zenoh = None
        self._declarations: list[Any] = []
        self._alive_token = None

        self._work = threading.Event()
        self._shutdown = threading.Event()
        self._worker: threading.Thread | None = None
        self._health_server: http.server.ThreadingHTTPServer | None = None

        self._unknown_clients_warned: set[str] = set()
        self._capture_count = 0

        self.metrics: dict[str, float] = {
            "requests_total": 0,
            "errors_total": 0,
            "superseded_total": 0,
            "dropped_unknown_client_total": 0,
            "sessions_opened_total": 0,
            "sessions_closed_total": 0,
        }
        self._metrics_lock = threading.Lock()

        task_slug_source = manifest.service_name or manifest.default_task or "default"
        self.prefix = service_prefix(manifest.model.repo_or_path, manifest.model.revision, task_slug_source)

    # ------------------------------------------------------------------
    # Loading & warmup
    # ------------------------------------------------------------------

    def load_policy(self) -> None:
        """Load config + weights, apply RTC settings, classify, warm up."""
        manifest = self._manifest
        if self._policy is None:
            logger.info(
                "Loading policy from '%s' (revision=%s)...",
                manifest.model.repo_or_path,
                manifest.model.revision,
            )
            policy_cfg = PreTrainedConfig.from_pretrained(manifest.model.repo_or_path)
            policy_cfg.pretrained_path = manifest.model.repo_or_path
            policy_class = get_policy_class(policy_cfg.type)
            policy = policy_class.from_pretrained(manifest.model.repo_or_path, config=policy_cfg)
            self._policy = policy
            self._policy_cfg = policy_cfg
        elif self._policy_cfg is None:
            self._policy_cfg = self._policy.config

        if self._classification is None:
            self._classification = classify_policy(self._policy)
        logger.info("Policy classification: %s", self._classification.reason)

        self._serving_mode, self._max_sessions = resolve_serving_mode(self._classification, manifest)
        logger.info("Serving mode: %s (max_sessions=%d)", self._serving_mode, self._max_sessions)

        # RTC is a per-process decision: init_rtc_processor mutates the
        # shared policy instance.
        self._rtc_active = (
            manifest.rtc.enabled
            and self._classification.supports_rtc
            and hasattr(self._policy.config, "rtc_config")
        )
        if self._rtc_active:
            self._policy.config.rtc_config = manifest.rtc
            if hasattr(self._policy, "init_rtc_processor"):
                self._policy.init_rtc_processor()
            logger.info("RTC active (execution_horizon=%d)", manifest.rtc.execution_horizon)

        if manifest.model.dtype:
            self._policy = self._policy.to(getattr(torch, manifest.model.dtype))
        self._policy = self._policy.to(self._device)
        self._policy.eval()

        if not self.action_names:
            logger.warning(
                "Policy config has no action_feature_names: the action-order contract "
                "cannot be enforced at session open. Clients are trusted to match training order."
            )

        if manifest.warmup_inferences > 0:
            self._warmup(manifest.warmup_inferences)
        self._warmed_up = True

    def make_session_processors(self) -> tuple[Any, Any]:
        """Build a fresh per-session pre/post pipeline pair.

        The rename step is forced to identity: clients apply their
        rename map before encoding, so the wire format is canonical
        policy-feature keys across heterogeneous robots.
        """
        if self._processor_factory is not None:
            return self._processor_factory()
        return make_pre_post_processors(
            policy_cfg=self._policy_cfg,
            pretrained_path=self._policy_cfg.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self._device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )

    def _warmup(self, n: int) -> None:
        """Run dummy forwards through the full request path (covers compile/caches)."""
        logger.info("Warmup: %d inferences...", n)
        obs = self._synthetic_observation()
        preprocessor, postprocessor = self.make_session_processors()
        session = Session(
            session_id="warmup",
            client_uuid="warmup",
            task=self._manifest.default_task,
            robot_type="",
            rtc_enabled=self._rtc_active,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        reply = self.run_inference_request(session, MsgHeader(), obs)
        if self._rtc_active and reply.chunk_model is not None and n > 1:
            # Exercise the prefix-conditioned path too so its compile/cache
            # cost isn't paid by the first real RTC request.
            action_dim = reply.chunk_model.shape[-1]
            horizon = self._manifest.rtc.execution_horizon
            obs.prefix_model = np.zeros((horizon, action_dim), dtype=np.float32)
            obs.prefix_robot = np.zeros((horizon, action_dim), dtype=np.float32)
            obs.inference_delay_steps = 1
        for _ in range(n - 1):
            self.run_inference_request(session, MsgHeader(), obs)
        session.close()
        # Stateful policies must not carry warmup observations into real sessions.
        if self._serving_mode == "exclusive":
            self._policy.reset()
        logger.info("Warmup complete")

    def _synthetic_observation(self) -> ObservationMsg:
        cfg = self._policy_cfg
        state_dim = self.state_dim or 1
        images = {}
        for key, feature in cfg.input_features.items():
            if feature.type == FeatureType.VISUAL:
                channels, height, width = feature.shape
                images[key] = np.zeros((height, width, channels), dtype=np.uint8)
        return ObservationMsg(
            state=np.zeros(state_dim, dtype=np.float32),
            images=images,
            task=self._manifest.default_task,
            jpeg_quality=0,
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def action_names(self) -> list[str]:
        names = getattr(self._policy_cfg, "action_feature_names", None)
        return list(names) if names else []

    @property
    def state_dim(self) -> int:
        cfg = self._policy_cfg
        for key, feature in getattr(cfg, "input_features", {}).items():
            if key == OBS_STATE or feature.type == FeatureType.STATE:
                return int(feature.shape[0])
        return 0

    @property
    def chunk_size(self) -> int:
        cfg = self._policy_cfg
        for attr in ("chunk_size", "n_action_steps", "horizon"):
            value = getattr(cfg, attr, None)
            if value:
                return int(value)
        return 0

    def status_snapshot(self) -> StatusMsg:
        cfg = self._policy_cfg
        expected_cameras = [
            key
            for key, feature in getattr(cfg, "input_features", {}).items()
            if feature.type == FeatureType.VISUAL
        ]
        return StatusMsg(
            model_repo=self._manifest.model.repo_or_path,
            model_revision=self._manifest.model.revision,
            policy_type=getattr(cfg, "type", "") or getattr(self._policy, "name", ""),
            action_names=self.action_names,
            expected_cameras=expected_cameras,
            state_dim=self.state_dim,
            chunk_size=self.chunk_size,
            trained_fps=self._manifest.trained_fps,
            supports_rtc=self._rtc_active,
            rtc_execution_horizon=self._manifest.rtc.execution_horizon if self._rtc_active else 0,
            serving_mode=self._serving_mode,
            warmed_up=self._warmed_up,
            active_sessions=len(self.registry),
            max_sessions=self._max_sessions,
        )

    @property
    def server_load(self) -> float:
        return len(self.registry) / max(1, self._max_sessions)

    # ------------------------------------------------------------------
    # The per-request inference path (pure: no zenoh — parity-testable)
    # ------------------------------------------------------------------

    def run_inference_request(
        self, session: Session, header: MsgHeader, obs: ObservationMsg
    ) -> ActionChunkMsg:
        """Mirror of the local RTC loop's compute step (rtc.py), minus the queue merge."""
        t0 = time.perf_counter()

        obs_np: dict[str, np.ndarray] = {}
        if obs.state is not None:
            obs_np[OBS_STATE] = np.asarray(obs.state, dtype=np.float32)
        for name, img in obs.images.items():
            obs_np[name] = img

        task = obs.task or session.task or self._manifest.default_task
        batch = prepare_observation_for_inference(obs_np, self._device, task, session.robot_type)
        batch["task"] = [task]

        preprocessed = session.preprocessor(batch)

        use_rtc = self._rtc_active and session.rtc_enabled
        if use_rtc:
            delay = max(0, int(obs.inference_delay_steps))
            prev_actions: torch.Tensor | None = None
            if obs.prefix_model is not None and obs.prefix_model.size:
                prev_actions = torch.from_numpy(np.ascontiguousarray(obs.prefix_model)).to(self._device)

            if prev_actions is not None and session.relative_step is not None:
                # Re-anchor the absolute leftover tail against the state
                # cached by THIS request's preprocess (mirrors rtc.py).
                raw_state = session.relative_step.get_cached_state()
                prefix_robot = obs.prefix_robot
                if raw_state is not None and prefix_robot is not None and prefix_robot.size:
                    prev_actions = reanchor_relative_rtc_prefix(
                        prev_actions_absolute=torch.from_numpy(np.ascontiguousarray(prefix_robot)),
                        current_state=raw_state,
                        relative_step=session.relative_step,
                        normalizer_step=session.normalizer_step,
                        policy_device=self._device,
                    )

            if prev_actions is not None:
                prev_actions = _normalize_prev_actions_length(
                    prev_actions, target_steps=self._manifest.rtc.execution_horizon
                )

            actions = self._policy.predict_action_chunk(
                preprocessed, inference_delay=delay, prev_chunk_left_over=prev_actions
            )
        else:
            if self._classification is not None and self._classification.needs_queue_population:
                preprocessed = self._populate_select_queues(preprocessed)
            actions = self._policy.predict_action_chunk(preprocessed)

        original = actions.squeeze(0).clone()
        processed = session.postprocessor(actions).squeeze(0)
        inference_ms = (time.perf_counter() - t0) * 1e3

        session.stats.requests += 1
        session.stats.last_inference_ms = inference_ms
        superseded = session.take_superseded()

        return ActionChunkMsg(
            seq_id_echo=header.seq_id,
            client_mono_ns_echo=header.client_mono_ns,
            episode_id_echo=header.episode_id,
            chunk_model=original.detach().to("cpu", torch.float32).numpy(),
            chunk_robot=processed.detach().to("cpu", torch.float32).numpy(),
            inference_ms=inference_ms,
            superseded_seqs=superseded,
            server_load=self.server_load,
        )

    def _populate_select_queues(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Exclusive-mode shim for select_action-fed policies (diffusion family).

        Mirrors ``DiffusionPolicy.select_action``: stack camera features
        into OBS_IMAGES, then populate the policy's observation queues so
        ``predict_action_chunk`` sees the same history it would locally.
        """
        policy = self._policy
        batch = {k: v for k, v in batch.items() if k != ACTION}
        if getattr(policy.config, "image_features", None):
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)
        policy._queues = populate_queues(policy._queues, batch)
        return batch

    # ------------------------------------------------------------------
    # Zenoh wiring
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open zenoh, declare the service surface, start worker + health threads."""
        if self._policy is None or not self._warmed_up:
            self.load_policy()

        zenoh = import_zenoh()
        spec = self._manifest.zenoh
        self._zenoh = zenoh.open(
            build_zenoh_config(
                mode=spec.mode,
                connect_endpoints=spec.connect_endpoints,
                listen_endpoints=spec.listen_endpoints,
                tls_root_ca_certificate=spec.tls_root_ca_certificate,
                tls_connect_certificate=spec.tls_connect_certificate,
                tls_connect_private_key=spec.tls_connect_private_key,
                extra_config_json5=spec.extra_config_json5,
            )
        )
        handlers = zenoh.handlers

        # Data plane: wildcard subscriber, deposit-only callback.
        self._declarations.append(
            self._zenoh.declare_subscriber(obs_wildcard(self.prefix), handlers.Callback(self._on_obs))
        )
        # Control plane: queryables reply inline (low rate).
        self._declarations.append(
            self._zenoh.declare_queryable(status_key(self.prefix), handlers.Callback(self._on_status_query))
        )
        self._declarations.append(
            self._zenoh.declare_queryable(session_key(self.prefix), handlers.Callback(self._on_session_query))
        )
        self._declarations.append(
            self._zenoh.declare_queryable(
                reset_wildcard(self.prefix), handlers.Callback(self._on_reset_query)
            )
        )
        # Presence: watch client tokens; publish our own.
        self._declarations.append(
            self._zenoh.liveliness().declare_subscriber(
                client_alive_wildcard(self.prefix), handlers.Callback(self._on_liveliness), history=True
            )
        )
        self._alive_token = self._zenoh.liveliness().declare_token(server_alive_key(self.prefix))

        self._shutdown.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="PolicyServerWorker")
        self._worker.start()

        if self._manifest.health_port:
            self._start_health_server(self._manifest.health_port)

        logger.info(
            "Policy server up: prefix=%s mode=%s max_sessions=%d rtc=%s",
            self.prefix,
            self._serving_mode,
            self._max_sessions,
            self._rtc_active,
        )

    def serve_forever(self) -> None:
        try:
            while not self._shutdown.is_set():
                self._shutdown.wait(timeout=0.5)
        except KeyboardInterrupt:
            logger.info("Interrupted — draining")
        finally:
            self.stop()

    def stop(self) -> None:
        """Drain: drop the liveliness token first (clients ride their buffers
        through the drain), finish the in-flight inference, then close."""
        if self._alive_token is not None:
            with contextlib.suppress(Exception):
                self._alive_token.undeclare()
            self._alive_token = None

        self._shutdown.set()
        self._work.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=10.0)
            if self._worker.is_alive():
                logger.warning("Inference worker did not join within 10s")
        self._worker = None

        # Undeclare the control/data surface BEFORE closing sessions so a
        # late session open cannot be accepted by a server that has
        # already drained its worker.
        for declaration in self._declarations:
            with contextlib.suppress(Exception):
                declaration.undeclare()
        self._declarations.clear()

        for session in self.registry.snapshot():
            self._close_session(session, reason="server shutdown")

        if self._zenoh is not None:
            with contextlib.suppress(Exception):
                self._zenoh.close()
            self._zenoh = None

        if self._health_server is not None:
            self._health_server.shutdown()
            self._health_server = None
        logger.info("Policy server stopped")

    # ------------------------------------------------------------------
    # Zenoh callbacks (deposit-only on the data plane)
    # ------------------------------------------------------------------

    def _on_obs(self, sample: Any) -> None:
        try:
            attachment = sample.attachment
            if attachment is None:
                return
            header = MsgHeader.unpack(attachment.to_bytes())
            if header.schema_version != SCHEMA_VERSION:
                return
            client_uuid = client_uuid_from_key(str(sample.key_expr))
            session = self.registry.get(client_uuid)
            if session is None:
                self._bump("dropped_unknown_client_total")
                # Bounded: garbage publishers must not grow this set (or
                # the log) without limit.
                if (
                    client_uuid not in self._unknown_clients_warned
                    and len(self._unknown_clients_warned) < 256
                ):
                    self._unknown_clients_warned.add(client_uuid)
                    logger.warning(
                        "Observation from unknown client '%s' (no session) — dropping", client_uuid
                    )
                return
            session.deposit(header, sample.payload.to_bytes())
            self._work.set()
        except Exception as e:  # noqa: BLE001 — a malformed sample must never kill the subscriber
            logger.error("obs callback error: %s", e)

    def _on_liveliness(self, sample: Any) -> None:
        try:
            import zenoh

            client_uuid = client_uuid_from_key(str(sample.key_expr))
            session = self.registry.get(client_uuid)
            if session is None:
                return
            if sample.kind == zenoh.SampleKind.DELETE:
                session.alive = False
                session.token_dropped_mono = time.monotonic()
                logger.info(
                    "Client '%s' liveliness dropped — GC in %.0fs", client_uuid, _LIVELINESS_GC_GRACE_S
                )
            else:
                session.alive = True
                session.token_dropped_mono = None
        except Exception as e:  # noqa: BLE001
            logger.error("liveliness callback error: %s", e)

    # ------------------------------------------------------------------
    # Control-plane queryables
    # ------------------------------------------------------------------

    def _on_status_query(self, query: Any) -> None:
        try:
            query.reply(status_key(self.prefix), codec.encode_status(self.status_snapshot()))
        except Exception as e:  # noqa: BLE001
            logger.error("status query error: %s", e)

    def _on_session_query(self, query: Any) -> None:
        try:
            payload = query.payload.to_bytes() if query.payload is not None else b""
            op = codec.decode_raw(payload).get("op", "open") if payload else "open"
            if op == "close":
                self._handle_session_close(codec.decode_session_close(payload))
                query.reply(session_key(self.prefix), codec.encode_reset_ack(ResetAckMsg(ok=True)))
                return
            ack = self._handle_session_open(codec.decode_session_open(payload))
            query.reply(session_key(self.prefix), codec.encode_session_ack(ack))
        except Exception as e:  # noqa: BLE001
            logger.error("session query error: %s\n%s", e, traceback.format_exc())
            with contextlib.suppress(Exception):
                query.reply(
                    session_key(self.prefix),
                    codec.encode_session_ack(SessionAckMsg(accepted=False, error=f"server error: {e}")),
                )

    def _on_reset_query(self, query: Any) -> None:
        try:
            payload = query.payload.to_bytes() if query.payload is not None else b""
            msg = codec.decode_reset(payload)
            session = self.registry.get(msg.client_uuid)
            if session is None:
                ack = ResetAckMsg(ok=False, error=f"unknown client '{msg.client_uuid}'")
            else:
                # Serialize with the worker: resetting pipelines/policy
                # mid-predict would corrupt the in-flight request.
                with self._inference_lock:
                    session.reset_episode(msg.episode_id)
                    if self._serving_mode == "exclusive":
                        # Safe: max_sessions=1, the policy belongs to this client.
                        self._policy.reset()
                ack = ResetAckMsg(ok=True)
            query.reply(str(query.key_expr), codec.encode_reset_ack(ack))
        except Exception as e:  # noqa: BLE001
            logger.error("reset query error: %s", e)

    def _handle_session_open(self, msg: SessionOpenMsg) -> SessionAckMsg:
        capabilities = self.status_snapshot()
        with self._registry_lock:
            # A re-handshake from a known client replaces its session and
            # does not count against capacity.
            existing = self.registry.get(msg.client_uuid)
            active = len(self.registry) - (1 if existing else 0)
            result = validate_session_open(msg, capabilities, self._manifest, active)
            if not result.ok:
                logger.warning("Session rejected for '%s': %s", msg.client_uuid, result.error)
                return SessionAckMsg(accepted=False, error=result.error, server_load=self.server_load)

            preprocessor, postprocessor = self.make_session_processors()
            session = Session(
                session_id=uuid_module.uuid4().hex,
                client_uuid=msg.client_uuid,
                task=msg.task or self._manifest.default_task,
                robot_type=msg.robot_type,
                rtc_enabled=msg.rtc_enabled and not result.rtc_downgraded,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                action_publisher=self._declare_action_publisher(msg.client_uuid),
            )
            if session.relative_step is not None and session.relative_step.action_names is None:
                session.relative_step.action_names = self.action_names or list(msg.action_names)
            # Sentinel: the FIRST observation of a fresh session always
            # triggers the episode-boundary branch in _serve_one, so a
            # mid-episode reconnect can never inherit stale state.
            session.episode_id = -1
            displaced = self.registry.add(session)
        if displaced is not None:
            displaced.close()
            self._bump("sessions_closed_total")
            logger.info("Client '%s' re-handshake: previous session replaced", msg.client_uuid)
        if self._serving_mode == "exclusive":
            # A new exclusive session must start from fresh policy state.
            with self._inference_lock:
                self._policy.reset()
        self._bump("sessions_opened_total")
        self._unknown_clients_warned.discard(msg.client_uuid)
        logger.info(
            "Session opened: client=%s session=%s task=%r rtc=%s (%d/%d)",
            msg.client_uuid,
            session.session_id,
            session.task,
            session.rtc_enabled,
            len(self.registry),
            self._max_sessions,
        )
        return SessionAckMsg(
            accepted=True,
            warnings=result.warnings,
            session_id=session.session_id,
            model_repo=capabilities.model_repo,
            model_revision=capabilities.model_revision,
            policy_type=capabilities.policy_type,
            action_names=capabilities.action_names,
            expected_cameras=capabilities.expected_cameras,
            state_dim=capabilities.state_dim,
            chunk_size=capabilities.chunk_size,
            trained_fps=capabilities.trained_fps,
            supports_rtc=capabilities.supports_rtc and session.rtc_enabled,
            rtc_execution_horizon=capabilities.rtc_execution_horizon,
            serving_mode=capabilities.serving_mode,
            warmed_up=capabilities.warmed_up,
            server_load=self.server_load,
        )

    def _declare_action_publisher(self, client_uuid: str) -> Any:
        if self._zenoh is None:  # pure-logic tests run without transport
            return None
        zenoh = import_zenoh()
        return self._zenoh.declare_publisher(
            action_key(self.prefix, client_uuid), **action_publisher_qos(zenoh)
        )

    def _handle_session_close(self, msg: SessionCloseMsg) -> None:
        session = self.registry.get(msg.client_uuid)
        if session is not None and (not msg.session_id or msg.session_id == session.session_id):
            self._close_session(session, reason="client close")

    def _close_session(self, session: Session, reason: str) -> None:
        # Identity-checked removal: never tear down a same-uuid session
        # that replaced this one via a re-handshake.
        removed = self.registry.remove(session.client_uuid, expected=session)
        if removed is not None:
            removed.close()
            self._bump("sessions_closed_total")
            logger.info("Session closed: client=%s (%s)", session.client_uuid, reason)

    # ------------------------------------------------------------------
    # Inference worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        last_gc = time.monotonic()
        while not self._shutdown.is_set():
            ready = [s for s in self.registry.snapshot() if s.has_pending()]
            if not ready:
                self._work.wait(timeout=_WORKER_IDLE_WAIT_S)
                self._work.clear()
            else:
                for session in self._scheduler.select(ready):
                    self._serve_one(session)

            now = time.monotonic()
            if now - last_gc > 1.0:
                last_gc = now
                self._gc_sessions(now)

    def _serve_one(self, session: Session) -> None:
        item = session.take()
        if item is None:
            return
        queue_wait_ms = (time.monotonic() - item.recv_mono) * 1e3
        outcome = "ok"
        try:
            obs = codec.decode_observation(item.payload)
            self._capture("req", item.payload)

            with self._inference_lock:
                # Belt-and-braces episode ordering: the first observation of
                # an episode also announces the boundary (one-in-flight makes
                # the reset query race-free, but a lost ack must not desync
                # us; fresh sessions start at the -1 sentinel so their first
                # request always lands here).
                if obs.episode_start or item.header.episode_id != session.episode_id:
                    session.preprocessor.reset()
                    session.postprocessor.reset()
                    session.episode_id = item.header.episode_id
                    if self._serving_mode == "exclusive":
                        self._policy.reset()

                reply = self.run_inference_request(session, item.header, obs)
            reply.queue_wait_ms = queue_wait_ms
            session.stats.last_queue_wait_ms = queue_wait_ms

            body = codec.encode_action_chunk(reply)
            self._capture("rep", body)
            # Local ref: a re-handshake can null session.action_publisher
            # between the check and the put.
            publisher = session.action_publisher
            if publisher is not None:
                reply_header = MsgHeader(
                    schema_version=SCHEMA_VERSION,
                    msg_type=2,  # MSG_TYPE_CHUNK
                    seq_id=item.header.seq_id,
                    episode_id=item.header.episode_id,
                    client_mono_ns=item.header.client_mono_ns,
                    session_epoch=item.header.session_epoch,
                )
                publisher.put(body, attachment=reply_header.pack())
            self._bump("requests_total")
            self._bump("superseded_total", reply.superseded_seqs)
        except Exception as e:  # noqa: BLE001 — one bad request must not kill the worker
            outcome = f"error: {e}"
            session.stats.errors += 1
            self._bump("errors_total")
            logger.error(
                "Inference error for client '%s' seq=%d: %s\n%s",
                session.client_uuid,
                item.header.seq_id,
                e,
                traceback.format_exc(),
            )
        finally:
            audit_logger.info(
                json.dumps(
                    {
                        "session_id": session.session_id,
                        "client_uuid": session.client_uuid,
                        "seq_id": item.header.seq_id,
                        "episode_id": item.header.episode_id,
                        "queue_wait_ms": round(queue_wait_ms, 3),
                        "inference_ms": round(session.stats.last_inference_ms, 3),
                        "superseded": session.stats.superseded,
                        "outcome": outcome,
                    }
                )
            )

    def _gc_sessions(self, now: float) -> None:
        for session in self.registry.snapshot():
            if (
                session.token_dropped_mono is not None
                and now - session.token_dropped_mono > _LIVELINESS_GC_GRACE_S
            ):
                if self._client_token_alive(session.client_uuid):
                    # The DELETE was a late echo of a previous incarnation
                    # (the token key is per client, not per epoch) — the
                    # client re-declared and is alive.
                    session.token_dropped_mono = None
                    session.alive = True
                    continue
                self._close_session(session, reason="liveliness token dropped")
            elif now - session.last_seen_mono > self._manifest.session_idle_timeout_s:
                self._close_session(session, reason="idle timeout")

    def _client_token_alive(self, client_uuid: str) -> bool:
        """Confirm a client's liveliness token via an explicit get (GC double-check)."""
        if self._zenoh is None:
            return False
        try:
            zenoh = import_zenoh()
            replies = self._zenoh.liveliness().get(
                client_alive_key(self.prefix, client_uuid),
                handler=zenoh.handlers.FifoChannel(4),
                timeout=0.5,
            )
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    reply = replies.try_recv()
                except Exception:  # channel closed: no token found  # noqa: BLE001
                    return False
                if reply is None:
                    time.sleep(0.01)
                    continue
                if reply.ok is not None:
                    return True
            return False
        except Exception:  # noqa: BLE001 — treat transport trouble as "not alive"
            return False

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _bump(self, key: str, amount: float = 1) -> None:
        with self._metrics_lock:
            self.metrics[key] = self.metrics.get(key, 0) + amount

    def _capture(self, kind: str, data: bytes) -> None:
        capture_dir = self._manifest.debug.capture_dir
        if not capture_dir:
            return
        try:
            directory = Path(capture_dir)
            directory.mkdir(parents=True, exist_ok=True)
            index = self._capture_count % max(1, self._manifest.debug.capture_max)
            (directory / f"{kind}_{index:05d}.bin").write_bytes(data)
            if kind == "rep":
                self._capture_count += 1
        except OSError as e:
            logger.warning("debug capture failed: %s", e)

    def _start_health_server(self, port: int) -> None:
        server_ref = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 — http.server API
                if self.path == "/healthz":
                    worker = server_ref._worker  # local ref: stop() may null it mid-read
                    healthy = worker is not None and worker.is_alive()
                    self.send_response(200 if healthy else 503)
                    self.end_headers()
                    self.wfile.write(b"ok" if healthy else b"worker dead")
                elif self.path == "/metrics":
                    with server_ref._metrics_lock:
                        counters = dict(server_ref.metrics)
                    counters["active_sessions"] = len(server_ref.registry)
                    counters["server_load"] = server_ref.server_load
                    body = "".join(
                        f"lerobot_policy_server_{name} {value}\n" for name, value in sorted(counters.items())
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args: Any) -> None:  # silence per-request logging
                pass

        self._health_server = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)  # nosec B104
        threading.Thread(target=self._health_server.serve_forever, daemon=True, name="HealthHTTP").start()
        logger.info("Health/metrics on :%d (/healthz, /metrics)", port)
