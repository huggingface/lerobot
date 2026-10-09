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

"""Inference addressing and admission errors over the shared wire contract."""

import math
from typing import Any

from lerobot.transport.wire.protocol import (
    PROTOCOL_VERSION,
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
    validate_segment,
)

__all__ = [
    "PROTOCOL_VERSION",
    "IDENTITY_KEYS",
    "Envelope",
    "ErrorCode",
    "MessageType",
    "ProtocolError",
    "AdmissionDeniedError",
    "validate_segment",
    "deployment_prefix",
    "instance_prefix",
    "session_prefix",
]

IDENTITY_KEYS = ("artifact_identity", "observation_id", "capture_time", "task", "task_version")


class AdmissionDeniedError(ProtocolError):
    """An expected BUSY reply specifically to an attempt to open a session.

    The exception text explains the contention to the operator without promising worker
    completion, so an uncaught failure reads correctly; ``diagnostic`` keeps the server's message.
    """

    def __init__(self, deployment: str, diagnostic: str, *, details: dict[str, Any] | None = None) -> None:
        """Retain the rejected deployment and the server's structured diagnostic."""
        super().__init__(ErrorCode.BUSY, _explain_admission_denial(deployment, details), details=details)
        self.deployment = deployment
        self.diagnostic = diagnostic


def _explain_admission_denial(deployment: str, details: dict[str, Any] | None) -> str:
    """Blocker names are the ones ``SessionWorker`` reports in the BUSY reply details."""
    details = details or {}
    blocker = details.get("admission_blocker")
    if not isinstance(blocker, str):
        blocker = None
    remaining = details.get("absence_grace_remaining_s")
    if blocker in {"absence_grace", "awaiting_initial_presence"}:
        reason = (
            "the previous client is absent"
            if blocker == "absence_grace"
            else "the previous client is still establishing its presence"
        )
        if (
            isinstance(remaining, (float, int))
            and not isinstance(remaining, bool)
            and math.isfinite(remaining)
            and remaining >= 0
        ):
            reason += f"; about {remaining:.1f} s of cleanup grace remain (worker cleanup may take longer)"
        else:
            reason += "; waiting for its cleanup grace and worker cleanup"
    elif blocker == "active_session":
        reason = "another client owns the deployment; stop that client before retrying"
    elif blocker == "unfinished_inference":
        reason = "waiting for an unfinished model call and session cleanup; completion time is unknown"
    elif blocker in {"worker_cleanup_pending", "cleanup_queue_full"}:
        reason = "waiting for worker session cleanup; completion time is unknown"
    else:
        reason = "the deployment is busy; see server logs for the session owner or pending cleanup"
    return f"Remote admission denied for deployment {deployment!r}: {reason}."


def deployment_prefix(deployment: str) -> str:
    """Return a validated deployment address."""
    return f"lerobot/inference/v{PROTOCOL_VERSION}/deployments/{validate_segment(deployment, 'deployment')}"


def instance_prefix(deployment: str, instance: str) -> str:
    """Return the address of one selected server boot."""
    return f"{deployment_prefix(deployment)}/instances/{validate_segment(instance, 'instance')}"


def session_prefix(deployment: str, instance: str, session: str) -> str:
    """Return the address of an admitted session."""
    return f"{instance_prefix(deployment, instance)}/sessions/{validate_segment(session, 'session')}"
