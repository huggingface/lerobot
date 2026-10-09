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

"""Explicit, non-retrying simulator RPC client."""

import uuid
from threading import RLock
from typing import Any

from lerobot.transport.wire.client import query_reply
from lerobot.transport.wire.features import feature_mismatch
from lerobot.transport.wire.protocol import Envelope, ErrorCode, MessageType, ProtocolError
from lerobot.transport.zenoh import PresenceToken, ZenohConfig, ZenohTransport

from .contracts import EnvDescriptor, ExecutionFeedback, StepResult, validate_result
from .protocol import deployment_prefix, instance_prefix, session_prefix


class EnvClient:
    """Own one explicitly negotiated simulator session without replaying mutations."""

    def __init__(self, endpoint: str, deployment: str = "default", timeout_s: float = 120):
        """Initialize owned resources without advancing simulation time."""
        self.transport = ZenohTransport(ZenohConfig(connect_endpoints=[endpoint]))
        self.deployment = deployment
        self.timeout_s = timeout_s
        self.instance = ""
        self.session = ""
        self.generation = 0
        self.token: PresenceToken | None = None
        self.descriptor: EnvDescriptor | None = None
        self.lock = RLock()
        self.failed = False
        self.num_envs = 0
        self.execution: ExecutionFeedback | None = None

    def describe(self) -> EnvDescriptor:
        """Discover the simulator boot and canonical schema without opening a world session."""
        with self.lock:
            self.transport.open()
            body = self._query(deployment_prefix(self.deployment) + "/describe", MessageType.DESCRIBE, {})
            self.instance = body["instance"]
            self.descriptor = EnvDescriptor.from_dict(body["descriptor"])
            return self.descriptor

    @property
    def prefix(self) -> str:
        """Return the address of the currently owned world session."""
        if not self.session:
            raise RuntimeError("Simulator session is not open")
        return session_prefix(self.deployment, self.instance, self.session)

    def open(
        self,
        num_envs: int,
        clock: str,
        task_group: str | None = None,
        task_id: int = 0,
        seeds: list[int | None] | None = None,
    ) -> StepResult:
        """Admit an exclusive environment batch with the requested clock."""
        with self.lock:
            if self.session:
                raise RuntimeError("Simulator session already open")
            if self.descriptor is None:
                self.describe()
            body = self._query(
                instance_prefix(self.deployment, self.instance) + "/open",
                MessageType.OPEN,
                {
                    "num_envs": num_envs,
                    "clock": clock,
                    "task_group": task_group,
                    "task_id": task_id,
                    "seeds": seeds,
                },
            )
            assert self.descriptor is not None
            opened_descriptor = EnvDescriptor.from_dict(body["descriptor"])
            if (
                tuple(f.name for f in opened_descriptor.features)
                != tuple(f.name for f in self.descriptor.features)
                or any(
                    feature_mismatch(actual, expected)
                    for actual, expected in zip(
                        opened_descriptor.features, self.descriptor.features, strict=True
                    )
                )
                or feature_mismatch(opened_descriptor.action_feature, self.descriptor.action_feature)
                or opened_descriptor.semantics != self.descriptor.semantics
            ):
                self.transport.close()
                raise ProtocolError(
                    ErrorCode.INCOMPATIBLE, "Selected task differs from the negotiated simulator schema"
                )
            self.descriptor = opened_descriptor
            self.session = body["session"]
            self.generation = body["generation"]
            self.token = self.transport.declare_token(self.prefix + "/client_alive")
            self.num_envs = num_envs
            result = StepResult.from_dict(body["result"])
            validate_result(result, self.descriptor, num_envs)
            return result

    def request(self, operation: str, **body: Any) -> dict[str, Any]:
        """Issue one correlated world operation and fail on an ambiguous result."""
        with self.lock:
            if self.failed:
                raise RuntimeError("Simulator session failed; close and reopen before continuing")
            key = self.prefix + (f"/{operation}" if operation in {"reset", "step"} else "/control")
            if operation not in {"reset", "step"}:
                body["operation"] = operation
            if operation == "apply":
                body["timeout_s"] = self.timeout_s
            try:
                result = self._query(key, MessageType.CONTROL, body)
                if "result" in result:
                    assert self.descriptor is not None
                    validate_result(StepResult.from_dict(result["result"]), self.descriptor, self.num_envs)
                if "generation" in result:
                    self.generation = result["generation"]
                return result
            except Exception:
                # A timed-out mutation may have executed. Never continue on an ambiguous world.
                self.failed = True
                raise

    def _query(self, key: str, kind: MessageType, body: dict[str, Any]) -> dict[str, Any]:
        message = Envelope(kind, self.instance, self.session, self.generation, uuid.uuid4().hex, body)
        response = query_reply(
            self.transport,
            key,
            message,
            self.timeout_s,
            MessageType.ACK if message.session_id else MessageType.ACCEPTED,
            max_replies=1,
        )
        if "execution" in response.body:
            assert self.descriptor is not None
            execution = ExecutionFeedback.from_dict(
                response.body["execution"], StepResult.from_dict(response.body["result"])
            )
            execution.validate(self.descriptor, self.num_envs)
            if execution.generation != message.generation or (
                body.get("operation", key.rsplit("/", 1)[-1]) in {"apply", "step"}
                and execution.request_id != message.request_id
            ):
                raise ProtocolError(ErrorCode.STALE, "Execution differs from the active command")
            if self.execution is not None and execution.sequence < self.execution.sequence:
                raise ProtocolError(ErrorCode.STALE, "Execution sequence moved backwards")
            self.execution = execution
        elif "generation" in response.body:
            self.execution = None
        return response.body

    def close(self) -> None:
        """Release owned simulator resources and transport declarations."""
        with self.lock:
            try:
                if self.session and not self.failed:
                    self.request("close")
            finally:
                self.transport.close()
                self.token = None
                self.session = ""
                self.instance = ""
                self.descriptor = None
                self.failed = False
                self.execution = None
