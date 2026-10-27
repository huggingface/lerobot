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

"""Client of a simulator server, and the vector environment it backs."""

import uuid
from threading import RLock
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space

from lerobot.sims.types import EnvDescriptor, StepResult
from lerobot.transport.wire.client import query_reply
from lerobot.transport.wire.protocol import Envelope, ErrorCode, MessageType, ProtocolError
from lerobot.transport.zenoh import PresenceToken, ZenohConfig, ZenohTransport
from lerobot.utils.feature_spec import feature_mismatch

from .contracts import (
    ANY_DEPLOYMENT_DESCRIBE,
    decode_descriptor,
    decode_step_result,
    deployment_prefix,
    instance_prefix,
    session_prefix,
    validate_result,
)


class EnvClient:
    """One session with a simulator server; operations are never retried."""

    def __init__(
        self,
        endpoint: str,
        deployment: str | None = None,
        timeout_s: float = 120,
        setup_timeout_s: float | None = None,
        encoding: str = "raw",
        jpeg_quality: int = 90,
    ):
        """Configure the connection; nothing is opened before ``describe`` or ``open``.

        ``timeout_s`` bounds per-tick operations; ``setup_timeout_s`` (default: the same)
        bounds describe/open/reset, which may build or rebuild native simulators. Without a
        ``deployment``, the client uses whichever deployment the endpoint serves. ``encoding="jpeg"``
        asks the server to send camera frames as JPEG at ``jpeg_quality``, a bandwidth for fidelity tradeoff.
        """
        if encoding not in {"raw", "jpeg"} or not 1 <= jpeg_quality <= 100:
            raise ValueError("Image encoding must be raw or jpeg, with jpeg_quality in [1, 100]")
        self.encoding = encoding
        self.jpeg_quality = jpeg_quality
        self.transport = ZenohTransport(ZenohConfig(connect_endpoints=[endpoint]))
        self.deployment = deployment or ""
        self.timeout_s = timeout_s
        self.setup_timeout_s = max(timeout_s, setup_timeout_s or timeout_s)
        self.instance = ""
        self.session = ""
        self.generation = 0
        self.token: PresenceToken | None = None
        self.descriptor: EnvDescriptor | None = None
        self.lock = RLock()
        self.failed = False
        self.num_envs = 0

    def describe(self) -> EnvDescriptor:
        """Discover the simulator boot and canonical schema without opening an environment session."""
        with self.lock:
            self.transport.open()
            body = self._query(
                deployment_prefix(self.deployment) + "/describe"
                if self.deployment
                else ANY_DEPLOYMENT_DESCRIBE,
                MessageType.DESCRIBE,
                {},
                self.setup_timeout_s,
            )
            self.deployment = body["deployment"]
            self.instance = body["instance"]
            self.descriptor = decode_descriptor(body["descriptor"])
            return self.descriptor

    @property
    def prefix(self) -> str:
        """Return the address of the currently owned environment session."""
        if not self.session:
            raise RuntimeError("Simulator session is not open")
        return session_prefix(self.deployment, self.instance, self.session)

    def open(
        self,
        num_envs: int,
        task_group: str | None = None,
        task_id: int = 0,
        seeds: list[int | None] | None = None,
        use_async_envs: bool = False,
    ) -> StepResult:
        """Admit an explicitly stepped environment batch.

        ``use_async_envs`` asks the server to run each environment in its own process.
        """
        with self.lock:
            if self.session:
                raise RuntimeError("Simulator session already open")
            descriptor = self.descriptor or self.describe()
            body = self._query(
                instance_prefix(self.deployment, self.instance) + "/open",
                MessageType.OPEN,
                {
                    "num_envs": num_envs,
                    "task_group": task_group,
                    "task_id": task_id,
                    "seeds": seeds,
                    "use_async_envs": use_async_envs,
                    "encoding": self.encoding,
                    "jpeg_quality": self.jpeg_quality,
                },
                self.setup_timeout_s,
            )
            opened_descriptor = decode_descriptor(body["descriptor"])
            if (
                tuple(f.name for f in opened_descriptor.features)
                != tuple(f.name for f in descriptor.features)
                or any(
                    feature_mismatch(actual, expected)
                    for actual, expected in zip(opened_descriptor.features, descriptor.features, strict=True)
                )
                or feature_mismatch(opened_descriptor.action_feature, descriptor.action_feature)
                or opened_descriptor.semantics != descriptor.semantics
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
            result = decode_step_result(body["result"])
            validate_result(result, self.descriptor, num_envs)
            return result

    def request(self, operation: str, **body: Any) -> dict[str, Any]:
        """Issue one correlated environment operation and fail on an ambiguous result."""
        with self.lock:
            if self.failed:
                raise RuntimeError("Simulator session failed; close and reopen before continuing")
            descriptor = self.descriptor
            if not self.session or descriptor is None:
                raise RuntimeError("Simulator session is not open")
            key = self.prefix + (f"/{operation}" if operation in {"reset", "step"} else "/control")
            if operation not in {"reset", "step"}:
                body["operation"] = operation
            try:
                timeout = self.setup_timeout_s if operation == "reset" else self.timeout_s
                result = self._query(key, MessageType.CONTROL, body, timeout)
                if "result" in result:
                    validate_result(decode_step_result(result["result"]), descriptor, self.num_envs)
                if "generation" in result:
                    self.generation = result["generation"]
                return result
            except Exception:
                # A timed-out mutation may have executed. Never continue with ambiguous environment state.
                self.failed = True
                raise

    def _query(
        self, key: str, kind: MessageType, body: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        message = Envelope(kind, self.instance, self.session, self.generation, uuid.uuid4().hex, body)
        response = query_reply(
            self.transport,
            key,
            message,
            self.timeout_s if timeout is None else timeout,
            MessageType.ACK if message.session_id else MessageType.ACCEPTED,
            max_replies=1,
            # Mutations are never retried, so a reply dropped under congestion is fatal: block instead.
            congestion_control="block",
        )
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


class RemoteVectorEnv(gym.vector.VectorEnv):
    """``num_envs`` served environments of one task, stepped explicitly."""

    canonical_observations = True

    def __init__(
        self,
        client: EnvClient,
        num_envs: int = 1,
        task_group: str | None = None,
        task_id: int = 0,
        descriptor: EnvDescriptor | None = None,
        use_async_envs: bool = False,
    ):
        """Describe the served task; its session opens on the first reset or step."""
        self.client = client
        self.use_async_envs = use_async_envs
        self.descriptor = descriptor or self.client.describe()
        self.num_envs = num_envs
        self.task_group = task_group or next(iter(self.descriptor.tasks))
        self.task_id = task_id
        self.metadata = {
            "render_fps": self.descriptor.fps,
            "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
        }
        self.render_mode = "rgb_array"
        self.single_observation_space = gym.spaces.Dict(
            {
                f.name: gym.spaces.Box(0, 255, f.shape, np.uint8)
                if f.kind == "rgb"
                else gym.spaces.Box(-np.inf, np.inf, f.shape, np.dtype(f.dtype))
                for f in self.descriptor.features
            }
        )
        self.single_action_space = gym.spaces.Box(
            -np.inf, np.inf, self.descriptor.action_feature.shape, np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self._latest: StepResult | None = None
        self.closed = False

    @property
    def latest(self) -> StepResult:
        """The result of the last reset or step."""
        if self._latest is None:
            raise RuntimeError("Reset before reading the environment")
        return self._latest

    def _open_session(self, seeds=None):
        # Not `_ensure`: lerobot-eval prefetches through `_ensure`, and opening resets the environments.
        if not self.client.session:
            self._latest = self.client.open(
                self.num_envs, self.task_group, self.task_id, seeds, self.use_async_envs
            )
            self.descriptor = self.client.descriptor or self.descriptor
            return True
        return False

    def _result(self, body):
        self._latest = decode_step_result(body["result"])
        return self._latest

    def reset(self, *, seed=None, options=None):
        """Reset every served environment with explicit seeds."""
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        if not self._open_session(seed):
            self._result(self.client.request("reset", seeds=seed))
        return self.latest.obs, self._info(self.latest)

    def step(self, actions):
        """Advance every unfinished served environment once."""
        self._open_session()
        result = self._result(self.client.request("step", actions=np.asarray(actions, dtype=np.float32)))
        return (
            result.obs,
            result.reward,
            result.terminated,
            result.truncated,
            self._info(result),
        )

    @staticmethod
    def _info(result: StepResult) -> dict[str, object]:
        return {
            "is_success": result.is_success,
            "task": result.task,
            "elapsed_steps": result.elapsed_steps,
            "sim_time": result.sim_time,
        }

    def call(self, name, *args, **kwargs):
        """Return a supported per-environment attribute."""
        if name == "_max_episode_steps":
            return (self.descriptor.max_episode_steps,) * self.num_envs
        if name == "task":
            return (self.task_group,) * self.num_envs
        self._open_session()
        if name == "task_description":
            return self.latest.task
        if name == "render":
            return tuple(self.client.request("render")["frames"])
        raise AttributeError(f"Simulator vector attribute is not exposed: {name}")

    def get_attr(self, name):
        """Return a supported per-environment attribute."""
        return self.call(name)

    def render(self):
        """Return the latest RGB frame of every served environment."""
        return self.call("render")

    def close_extras(self, **kwargs):
        """Close the served session."""
        self.client.close()

    def close(self, **kwargs):
        """Close the served session."""
        self.client.close()
        self.closed = True
