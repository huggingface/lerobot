# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Remote executor. This module never imports a model class or loads weights."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, fields
from queue import Empty
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import torch

from lerobot.inference.contracts import (
    ActionChunk,
    ActionProvenance,
    ExecutionMode,
    FeatureSpec,
    ObservationSnapshot,
    PolicyCapabilities,
)
from lerobot.inference.execution import ChunkRequest
from lerobot.transport.zenoh import (
    BoundedSubscriber,
    PresenceToken,
    QueryCancelled,
    ZenohConfig,
    ZenohTransport,
)

from .build_info import SOFTWARE_BUILD
from .chunk_contract import (
    RTC_MODEL_SPACE,
    chunk_settings,
    required_chunk_capabilities,
    validate_chunk_contract,
)
from .codec import RGBImage, decode_message, encode_message, peek_envelope
from .protocol import (
    PROTOCOL_VERSION,
    AdmissionDeniedError,
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
    deployment_prefix,
    instance_prefix,
    session_prefix,
)

if TYPE_CHECKING:
    from lerobot.rollout.inference.factory import RemoteInferenceConfig

logger = logging.getLogger(__name__)


class RequestCancelled(Exception):  # noqa: N818
    """Local intent invalidation; says nothing about completion of server computation."""


def parse_capabilities(value: dict[str, Any]) -> PolicyCapabilities:
    """Validate a descriptor's ordered feature schemas and execution capabilities."""
    value = dict(value)
    value["features"] = tuple(FeatureSpec(**feature) for feature in value["features"])
    value["action_feature"] = FeatureSpec(**value["action_feature"])
    value["modes"] = tuple(ExecutionMode(mode) for mode in value["modes"])
    return PolicyCapabilities(**value)


def _compare_feature(actual: FeatureSpec, expected: FeatureSpec) -> None:
    """Identify the first incompatible field without dumping unrelated schemas."""
    for field in fields(FeatureSpec):
        actual_value, expected_value = getattr(actual, field.name), getattr(expected, field.name)
        if actual_value != expected_value:
            raise ProtocolError(
                ErrorCode.INCOMPATIBLE,
                f"Feature {expected.name!r} field {field.name!r} differs: "
                f"client={actual_value!r}, server={expected_value!r}",
            )


class RemoteClient:
    """Setup and worker-only bounded exchanges; no automatic inference replay."""

    def __init__(self, transport: ZenohTransport, config: RemoteInferenceConfig, descriptor: dict) -> None:
        """Attach an already connected deployment without loading any local policy."""
        self.transport = transport
        self.config = config
        self.descriptor = descriptor
        self.capabilities = parse_capabilities(descriptor["capabilities"])
        self.chunk_settings = chunk_settings(
            config.chunk_merge, config.blend_steps, config.blend_weight, config.blend_components
        )
        self.blend_indices: tuple[int, ...] = ()
        self.instance_id = descriptor["instance_id"]
        self.artifact_identity = descriptor["artifact_identity"]
        self.session_id = ""
        self.generation = 0
        self._request_sequence = 0
        self._instance_key = instance_prefix(config.deployment, self.instance_id)
        self._presence = transport.subscribe_liveliness(self._instance_key + "/alive")
        self._present = True
        self._actions: BoundedSubscriber[bytes] | None = None
        self._language: BoundedSubscriber[bytes] | None = None
        self._token: PresenceToken | None = None
        self._closed = False
        self._key = ""

    @classmethod
    def connect(cls, config: RemoteInferenceConfig) -> RemoteClient:
        """Discover exactly one ready deployment within the handshake deadline."""
        logger.info("Remote client software=%s protocol=%s", asdict(SOFTWARE_BUILD), PROTOCOL_VERSION)
        transport = ZenohTransport(
            ZenohConfig(
                mode=config.zenoh_mode,
                connect_endpoints=[config.endpoint],
                config_file=config.zenoh_config_path,
                open_timeout_s=config.handshake_timeout_s,
            )
        )
        try:
            try:
                transport.open()
            except Exception as exc:
                raise ConnectionError(
                    f"Cannot open remote connection to {config.endpoint!r} "
                    f"for deployment {config.deployment!r}: {exc}. "
                    "Check that the server/router is running and the endpoint, network access "
                    "and Zenoh security configuration are correct."
                ) from exc
            request = Envelope(MessageType.DESCRIBE, request_id=uuid4().hex)
            try:
                replies = transport.query(
                    deployment_prefix(config.deployment) + "/describe",
                    encode_message(request),
                    config.handshake_timeout_s,
                )
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Deployment {config.deployment!r} did not answer discovery at {config.endpoint!r} "
                    f"within {config.handshake_timeout_s:g} s. Check server readiness, "
                    "--inference.deployment and the Zenoh routing/permissions."
                ) from exc
            descriptors: dict[str, dict] = {}
            for payload in replies:
                response = decode_message(payload)
                if (
                    response.message_type is not MessageType.DESCRIPTOR
                    or response.request_id != request.request_id
                ):
                    raise ProtocolError(ErrorCode.MALFORMED, "Invalid deployment descriptor response")
                descriptor = response.body
                if (
                    descriptor.get("instance_id") != response.instance_id
                    or descriptor.get("deployment") != config.deployment
                    or not descriptor.get("ready")
                ):
                    raise ProtocolError(ErrorCode.INCOMPATIBLE, "Deployment is not ready or identity differs")
                if config.instance is None or response.instance_id == config.instance:
                    descriptors[response.instance_id] = descriptor
            if not descriptors:
                raise ProtocolError(
                    ErrorCode.INCOMPATIBLE,
                    f"No ready instance for deployment {config.deployment!r} at {config.endpoint!r} "
                    f"matches --inference.instance={config.instance!r}; "
                    "check server readiness, deployment name and instance selection",
                )
            if len(descriptors) != 1:
                raise ProtocolError(
                    ErrorCode.INCOMPATIBLE,
                    "Expected exactly one ready instance; select --inference.instance if ambiguous",
                )
            descriptor = next(iter(descriptors.values()))
            logger.info(
                "Remote server deployment=%s instance=%s software=%r protocol=%s",
                config.deployment,
                descriptor["instance_id"],
                descriptor.get("software", "unavailable (peer does not report its loaded build)"),
                PROTOCOL_VERSION,
            )
            if config.expected_artifact and config.expected_artifact != descriptor.get("artifact_identity"):
                raise ProtocolError(
                    ErrorCode.INCOMPATIBLE, "Expected artifact differs from deployed checkpoint"
                )
            return cls(transport, config, descriptor)
        except Exception:
            transport.close()
            raise

    def _query(
        self,
        key: str,
        request: Envelope,
        timeout: float,
        expected: MessageType,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> Envelope:
        try:
            replies = (
                self.transport.query(key, encode_message(request), timeout)
                if cancelled is None
                else self.transport.query(key, encode_message(request), timeout, cancelled=cancelled)
            )
        except QueryCancelled as exc:
            if not self.present:
                raise ConnectionError("Server presence was lost during control acknowledgement") from exc
            raise RequestCancelled(str(exc)) from exc
        if len(replies) != 1:
            raise TimeoutError("Expected one control reply within its deadline")
        response = decode_message(replies[0])
        self._correlate(response, request, session=expected is not MessageType.ACCEPTED)
        self._raise_error(response)
        if response.message_type is not expected:
            raise ProtocolError(ErrorCode.MALFORMED, "Unexpected control reply type")
        return response

    @staticmethod
    def _raise_error(response: Envelope) -> None:
        if response.message_type is MessageType.ERROR:
            try:
                code = ErrorCode(response.body["code"])
            except (KeyError, ValueError, TypeError) as exc:
                raise ProtocolError(ErrorCode.MALFORMED, "Invalid error code") from exc
            details = response.body.get("details")
            if details is not None and not isinstance(details, dict):
                raise ProtocolError(ErrorCode.MALFORMED, "Invalid error details")
            message = response.body.get("message", "Remote inference error")
            if not isinstance(message, str):
                raise ProtocolError(ErrorCode.MALFORMED, "Invalid error message")
            raise ProtocolError(code, message, details=details)

    @staticmethod
    def _correlate(response: Envelope, request: Envelope, *, session: bool = True) -> None:
        if (
            response.instance_id != request.instance_id
            or response.request_id != request.request_id
            or response.generation != request.generation
            or (session and response.session_id != request.session_id)
        ):
            raise ProtocolError(ErrorCode.STALE, "Reply context differs from the active request")

    def admit(
        self,
        *,
        features: tuple[FeatureSpec, ...],
        action_feature: FeatureSpec,
        semantics: str,
        action_interval: float,
        mode: str,
    ) -> None:
        """Validate the robot contract and acquire one exclusive session."""
        caps = self.capabilities
        names, expected_names = (
            tuple(feature.name for feature in features),
            tuple(feature.name for feature in caps.features),
        )
        if names != expected_names:
            raise ProtocolError(
                ErrorCode.INCOMPATIBLE,
                f"Observation feature names/order differ: client={names!r}, server={expected_names!r}. "
                "Check robot cameras, --rename_map and the server feature contract",
            )
        for feature, expected in zip(features, caps.features, strict=True):
            _compare_feature(feature, expected)
        _compare_feature(action_feature, caps.action_feature)
        if semantics != self.descriptor["semantics"]:
            raise ProtocolError(
                ErrorCode.INCOMPATIBLE,
                f"Semantic convention differs: client={semantics!r}, server={self.descriptor['semantics']!r}",
            )
        if not math.isclose(action_interval, caps.action_interval, rel_tol=1e-6):
            raise ProtocolError(
                ErrorCode.INCOMPATIBLE,
                f"Policy action interval differs: client={action_interval:g} s, "
                f"server={caps.action_interval:g} s. Use --fps={1 / caps.action_interval:g}",
            )
        if mode not in caps.modes:
            raise ProtocolError(
                ErrorCode.INCOMPATIBLE,
                f"Execution mode {mode!r} is not supported; "
                f"server modes={tuple(str(mode) for mode in caps.modes)!r}",
            )
        required = required_chunk_capabilities(self.chunk_settings)
        if required:
            if mode != ExecutionMode.CHUNK:
                raise ProtocolError(ErrorCode.INCOMPATIBLE, "Aligned merge requires chunk execution")
            advertised = self.descriptor.get("execution_contracts", [])
            if not isinstance(advertised, list) or any(name not in advertised for name in required):
                raise ProtocolError(
                    ErrorCode.UNSUPPORTED,
                    "Server does not support the requested chunk alignment/blending contract; update the server",
                )
            try:
                self.blend_indices = validate_chunk_contract(
                    self.chunk_settings, caps, self.descriptor.get("blendable_components", [])
                )
            except ValueError as exc:
                raise ProtocolError(ErrorCode.INCOMPATIBLE, str(exc)) from exc
        if mode != ExecutionMode.CHUNK and caps.model_action_dim not in (None, caps.action_feature.shape[0]):
            if RTC_MODEL_SPACE not in self.descriptor.get("execution_contracts", []):
                raise ProtocolError(
                    ErrorCode.UNSUPPORTED, "Server lacks the distinct RTC model-space contract"
                )
            required.append(RTC_MODEL_SPACE)
        request = Envelope(
            MessageType.OPEN,
            self.instance_id,
            request_id=uuid4().hex,
            body={
                "expected_artifact": self.config.expected_artifact or self.artifact_identity,
                "features": [asdict(feature) for feature in features],
                "action_feature": asdict(action_feature),
                "semantics": semantics,
                "action_interval": action_interval,
                "mode": mode,
                "encoding": self.config.encoding,
                **(
                    {"chunk_settings": self.chunk_settings, "required_capabilities": required}
                    if required
                    else {}
                ),
            },
        )
        try:
            accepted = self._query(
                self._instance_key + "/open", request, self.config.handshake_timeout_s, MessageType.ACCEPTED
            )
        except ProtocolError as exc:
            if exc.code is not ErrorCode.BUSY:
                raise
            raise AdmissionDeniedError(self.config.deployment, str(exc), details=exc.details) from exc
        if required:
            try:
                accepted_settings = accepted.body.get("chunk_settings")
                if not isinstance(accepted_settings, dict):
                    raise ValueError("Missing accepted chunk settings")
                validate_chunk_contract(
                    accepted_settings, caps, accepted.body.get("blendable_components", [])
                )
            except ValueError as exc:
                raise ProtocolError(
                    ErrorCode.INCOMPATIBLE, "Admission changed the advertised contract"
                ) from exc
        if (
            accepted.body.get("artifact_identity") != self.artifact_identity
            or parse_capabilities(accepted.body["capabilities"]) != caps
            or accepted.body.get("mode") != mode
            or (required and accepted.body.get("chunk_settings") != self.chunk_settings)
        ):
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Admission changed the advertised contract")
        self.session_id = accepted.session_id
        if not self.session_id:
            raise ProtocolError(ErrorCode.MALFORMED, "Server did not allocate a session")
        self._key = session_prefix(self.config.deployment, self.instance_id, self.session_id)
        self._actions = self.transport.subscribe(self._key + "/act", capacity=4)
        self._language = self.transport.subscribe(self._key + "/language/result", capacity=4)
        self._token = self.transport.declare_token(self._key + "/alive")
        # This acknowledged control also proves session endpoints are ready before
        # the first lossy/nonblocking data publication.
        self.control("reset", 0)
        self.transport.wait_for_subscriber(self._key + "/obs", self.config.handshake_timeout_s)
        self.transport.wait_for_subscriber(self._key + "/language/request", self.config.handshake_timeout_s)
        logger.info(
            "Remote session ready: deployment=%s instance=%s session=%s mode=%s merge=%s "
            "action_rate=%.1f Hz horizon=%.3fs; contract verified",
            self.config.deployment,
            self.instance_id,
            self.session_id,
            mode,
            self.config.chunk_merge,
            1 / caps.action_interval,
            caps.execution_steps * caps.action_interval,
        )
        logger.debug(
            "Remote admission artifact=%s limits=%s chunk_settings=%s blend_indices=%s",
            self.artifact_identity,
            accepted.body.get("limits"),
            self.chunk_settings,
            self.blend_indices,
        )

    @property
    def present(self) -> bool:
        """Report server presence, latching any observed loss until a new client is created."""
        while True:
            try:
                event = self._presence.get(timeout=0)
            except Empty:
                break
            # Presence never grants recovery after a loss.
            if not event.alive:
                self._present = False
        if self._presence.dropped:
            self._present = False
        return self._present

    def control(
        self, operation: str, generation: int, *, cancelled: Callable[[], bool] | None = None
    ) -> None:
        """Wait for a worker-applied control acknowledgement on the network thread."""
        request = Envelope(
            MessageType.CONTROL,
            self.instance_id,
            self.session_id,
            generation,
            uuid4().hex,
            {"operation": operation},
        )
        limits = self.descriptor.get("limits", {})
        timeout = self.config.handshake_timeout_s + max(
            limits.get("action_deadline_s", 0), limits.get("language_deadline_s", 0)
        )
        if operation == "close":
            timeout = min(self.config.handshake_timeout_s, 1.0)
        response = self._query(
            self._key + "/control",
            request,
            timeout,
            MessageType.ACK,
            cancelled=lambda: (cancelled is not None and cancelled()) or not self.present,
        )
        if (
            response.body.get("operation") != operation
            or response.body.get("applied_generation") != generation
        ):
            raise ProtocolError(ErrorCode.STALE, "Control was not applied at the requested generation")
        self.generation = generation

    def _observation_body(self, observation: ObservationSnapshot) -> dict:
        self._request_sequence += 1
        rgb = {feature.name for feature in self.capabilities.features if feature.kind == "rgb"}
        return {
            "sequence": self._request_sequence,
            "artifact_identity": self.artifact_identity,
            "observation_id": observation.observation_id,
            "capture_time": observation.capture_time,
            "task": observation.task,
            "task_version": observation.task_version,
            "features": {
                name: RGBImage(value, self.config.encoding, self.config.jpeg_quality)
                if name in rgb
                else value
                for name, value in observation.features.items()
            },
        }

    def _exchange(
        self, request: Envelope, suffix: str, channel: Any, timeout: float, cancelled: Callable[[], bool]
    ) -> Envelope:
        if cancelled():
            raise RequestCancelled()
        if not self.present:
            raise ConnectionError("Server presence lost; session cannot recover")
        started = time.monotonic()
        payload = encode_message(request)  # included in the same client deadline
        if cancelled():
            raise RequestCancelled()
        self.transport.publish(self._key + suffix, payload)
        while time.monotonic() - started <= timeout:
            if cancelled():
                raise RequestCancelled()
            if channel.oversized or channel.dropped:
                raise ProtocolError(
                    ErrorCode.MALFORMED, "Inference reply channel overflow or oversized message"
                )
            try:
                payload = channel.get(timeout=min(0.02, timeout))
            except Empty:
                continue
            # Obsolete traffic has no authority to refresh deadlines or permission.
            header = peek_envelope(payload)
            try:
                self._correlate(header, request)
            except ProtocolError:
                continue
            response = decode_message(payload)
            self._raise_error(response)
            return response
        raise TimeoutError("Remote inference request deadline exceeded")

    def infer(self, request: ChunkRequest, *, cancelled: Callable[[], bool] = lambda: False) -> ActionChunk:
        """Exchange one action request and validate its identity, execution context and values."""
        body = self._observation_body(request.observation)
        model_continuation = request.continuation.model_actions
        canonical_continuation = request.continuation.canonical_actions
        body.update(
            {
                "delay": request.delay,
                "model_continuation": model_continuation
                if model_continuation is not None and len(model_continuation)
                else None,
                "canonical_continuation": canonical_continuation
                if canonical_continuation is not None and len(canonical_continuation)
                else None,
                "cursor": request.continuation.cursor,
            }
        )
        if self.chunk_settings["chunk_merge"] == "aligned":
            body["observation_cursor"] = request.observation.action_cursor
        envelope = Envelope(
            MessageType.OBSERVATION,
            self.instance_id,
            self.session_id,
            request.generation,
            request.request_id,
            body,
        )
        response = self._exchange(envelope, "/obs", self._actions, self.config.action_timeout_s, cancelled)
        self._validate_context(response, body, MessageType.ACTION)
        actions = response.body.get("canonical_actions")
        model = response.body.get("model_actions")
        caps = self.capabilities
        steps = caps.execution_steps if request.mode is ExecutionMode.CHUNK else caps.prediction_steps
        if (
            not isinstance(actions, np.ndarray)
            or actions.shape != (steps, *caps.action_feature.shape)
            or actions.dtype.name != "float32"
            or not np.isfinite(actions).all()
            or response.body.get("execution_steps") != steps
        ):
            raise ProtocolError(ErrorCode.MALFORMED, "Invalid canonical action chunk")
        if request.mode is not ExecutionMode.CHUNK and (
            not isinstance(model, np.ndarray)
            or model.shape != (steps, caps.model_action_dim or caps.action_feature.shape[0])
            or model.dtype.name != "float32"
            or not np.isfinite(model).all()
        ):
            raise ProtocolError(ErrorCode.MALFORMED, "Missing or invalid RTC model continuation")
        provenance = ActionProvenance(
            request.observation.capture_time,
            request.observation.task,
            request.observation.task_version,
            request.observation.observation_id,
            request.request_id,
            request.generation,
            self.session_id,
            self.instance_id,
            self.artifact_identity,
        )
        return ActionChunk(
            None if model is None else torch.from_numpy(model),
            torch.from_numpy(actions),
            provenance,
            steps,
            response.body.get("server_durations", {}),
        )

    def _validate_context(self, response: Envelope, body: dict, expected: MessageType) -> None:
        context_keys = ["artifact_identity", "observation_id", "capture_time", "task", "task_version"]
        if expected is MessageType.ACTION and self.chunk_settings["chunk_merge"] == "aligned":
            context_keys.extend(["observation_cursor", "cursor"])
            if any(type(response.body.get(key)) is not int for key in ("observation_cursor", "cursor")):
                raise ProtocolError(
                    ErrorCode.MALFORMED, "Result action cursor context differs from its request"
                )
        if response.message_type is not expected or any(
            response.body.get(key) != body[key] for key in context_keys
        ):
            raise ProtocolError(ErrorCode.MALFORMED, "Result execution context differs from its request")

    def query_language(
        self,
        observation: ObservationSnapshot,
        *,
        kind: str,
        text: str,
        intent_generation: int,
        generation: int,
        cancelled: Callable[[], bool],
    ) -> str:
        """Run one bounded text query on the same exclusive policy worker."""
        if not self.capabilities.language:
            raise ProtocolError(ErrorCode.UNSUPPORTED, "Deployment has no text capability")
        body = self._observation_body(observation)
        body.update({"kind": kind, "text": text, "intent_generation": intent_generation})
        request = Envelope(
            MessageType.LANGUAGE_REQUEST, self.instance_id, self.session_id, generation, uuid4().hex, body
        )
        result = self._exchange(
            request, "/language/request", self._language, self.config.language_timeout_s, cancelled
        )
        self._validate_context(result, body, MessageType.LANGUAGE_RESULT)
        answer = result.body.get("answer")
        if (
            result.body.get("intent_generation") != intent_generation
            or result.body.get("kind") != kind
            or not isinstance(answer, str)
            or not answer.strip()
            or len(answer) > self.descriptor["limits"]["max_output_chars"]
        ):
            raise ProtocolError(ErrorCode.MALFORMED, "Invalid language answer context or length")
        return answer

    def close(self) -> bool:
        """Close the admitted session and transport; never reconnect or replay work."""
        if self._closed:
            return False
        self._closed = True
        try:
            if self.session_id and self.present:
                self.control("close", self.generation)
                return True
            return False
        finally:
            self.transport.close()
