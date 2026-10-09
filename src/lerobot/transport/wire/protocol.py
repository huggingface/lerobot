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

"""Shared versioned envelopes and generic protocol errors (no Python-object serialization)."""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

PROTOCOL_VERSION = 1
_MAX_IDENTIFIER = 128
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class MessageType(StrEnum):
    """The operation tags understood by protocol major one."""

    DESCRIBE = "describe"
    DESCRIPTOR = "descriptor"
    OPEN = "open"
    ACCEPTED = "accepted"
    CONTROL = "control"
    ACK = "ack"
    OBSERVATION = "observation"
    ACTION = "action"
    LANGUAGE_REQUEST = "language_request"
    LANGUAGE_RESULT = "language_result"
    ERROR = "error"


class ErrorCode(StrEnum):
    """Stable error classes; details are diagnostic, never executable instructions."""

    UNSUPPORTED = "unsupported"
    MALFORMED = "malformed"
    BUSY = "busy"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    PROTOCOL = "protocol"


class ProtocolError(ValueError):
    """A bounded, structured failure suitable for returning to a peer."""

    def __init__(self, code: ErrorCode, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Attach the machine-readable error category."""
        super().__init__(message)
        self.code = code
        self.details = details


def validate_segment(value: str, label: str = "identifier") -> str:
    """Validate a bounded single segment before composing a Zenoh key."""
    if not isinstance(value, str) or not _SEGMENT.fullmatch(value):
        raise ProtocolError(ErrorCode.MALFORMED, f"Invalid {label}: expected a single safe key segment")
    return value


@dataclass(frozen=True)
class Envelope:
    """Every data message has a correlation identity; negotiation permits empty instance/session IDs."""

    message_type: MessageType | str
    instance_id: str = ""
    session_id: str = ""
    generation: int = 0
    request_id: str = ""
    body: dict[str, Any] = field(default_factory=dict)
    version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        """Validate correlation fields before decoding potentially large bodies."""
        if type(self.version) is not int or self.version != PROTOCOL_VERSION:
            raise ProtocolError(ErrorCode.PROTOCOL, f"Unsupported protocol version: {self.version!r}")
        try:
            object.__setattr__(self, "message_type", MessageType(self.message_type))
        except (ValueError, TypeError) as exc:
            raise ProtocolError(ErrorCode.MALFORMED, "Unknown message type") from exc
        if type(self.generation) is not int or not 0 <= self.generation < 2**63:
            raise ProtocolError(ErrorCode.MALFORMED, "Invalid execution generation")
        for label in ("instance_id", "session_id", "request_id"):
            value = getattr(self, label)
            if not isinstance(value, str) or len(value) > _MAX_IDENTIFIER:
                raise ProtocolError(ErrorCode.MALFORMED, f"Invalid {label}")
            if value:
                validate_segment(value, label)
        if not isinstance(self.body, dict) or any(not isinstance(key, str) for key in self.body):
            raise ProtocolError(ErrorCode.MALFORMED, "Message body must be a string-keyed map")
        if self.message_type in {
            MessageType.OBSERVATION,
            MessageType.ACTION,
            MessageType.LANGUAGE_REQUEST,
            MessageType.LANGUAGE_RESULT,
            MessageType.CONTROL,
            MessageType.ACK,
        } and not all((self.instance_id, self.session_id, self.request_id)):
            raise ProtocolError(
                ErrorCode.MALFORMED, "Session messages require instance, session and request IDs"
            )

    def reply(self, message_type: MessageType, body: dict[str, Any]) -> "Envelope":
        """Create a reply retaining the complete request correlation identity."""
        return Envelope(
            message_type, self.instance_id, self.session_id, self.generation, self.request_id, body
        )

    def error(self, code: ErrorCode, message: str, *, details: dict[str, Any] | None = None) -> "Envelope":
        """Create a correlated error with bounded diagnostic text."""
        body: dict[str, Any] = {"code": str(code), "message": message[:2048]}
        if details is not None:
            body["details"] = details
        return self.reply(MessageType.ERROR, body)


def correlate_reply(
    response: Envelope, request: Envelope, *, check_instance: bool = True, check_session: bool = True
) -> None:
    """Reject replies outside the request identity, allowing negotiated peer identifiers."""
    if (
        response.request_id != request.request_id
        or response.generation != request.generation
        or (check_instance and response.instance_id != request.instance_id)
        or (check_session and response.session_id != request.session_id)
    ):
        raise ProtocolError(ErrorCode.STALE, "Reply context differs from the active request")


def raise_peer_error(response: Envelope) -> None:
    """Validate and propagate structured peer errors without service-specific interpretation."""
    if response.message_type is not MessageType.ERROR:
        return
    try:
        code = ErrorCode(response.body["code"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ProtocolError(ErrorCode.MALFORMED, "Invalid error code") from exc
    details = response.body.get("details")
    if details is not None and not isinstance(details, dict):
        raise ProtocolError(ErrorCode.MALFORMED, "Invalid error details")
    message = response.body.get("message", "Peer error")
    if not isinstance(message, str):
        raise ProtocolError(ErrorCode.MALFORMED, "Invalid error message")
    raise ProtocolError(code, message, details=details)


def validate_reply(
    response: Envelope,
    request: Envelope,
    expected: MessageType,
    *,
    check_instance: bool = True,
    check_session: bool = True,
) -> None:
    """Check correlation before peer errors and the successful operation's reply type."""
    correlate_reply(response, request, check_instance=check_instance, check_session=check_session)
    raise_peer_error(response)
    if response.message_type is not expected:
        raise ProtocolError(ErrorCode.MALFORMED, "Unexpected reply type")
