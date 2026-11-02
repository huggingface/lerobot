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

"""Client for the env server."""

import logging
import uuid
from threading import RLock
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space

from lerobot.sim.types import EnvDescriptor, StepResult
from lerobot.transport.protocol.client import query_reply
from lerobot.transport.protocol.codec import CodecLimits
from lerobot.transport.protocol.envelope import Envelope, ErrorCode, MessageType, ProtocolError
from lerobot.transport.zenoh import ZenohConfig, ZenohTransport

from .protocol import (
    ANY_DEPLOYMENT_DESCRIBE,
    DEFAULT_MAX_PAYLOAD_BYTES,
    decode_descriptor,
    decode_step_result,
    deployment_prefix,
    instance_prefix,
    session_prefix,
    validate_result,
)

logger = logging.getLogger(__name__)


class EnvClient:
    """One session with an env server. Requests are never retried."""

    def __init__(
        self,
        endpoint: str,
        deployment: str | None = None,
        timeout_s: float = 120,
        setup_timeout_s: float | None = None,
        encoding: Literal["raw", "jpeg"] = "raw",
        jpeg_quality: int = 90,
        handshake_timeout_s: float = 10.0,
        zenoh_config_path: str | None = None,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ):
        """Store the settings; nothing connects until `describe` or `open`.

        Args:
            endpoint (`str`):
                Zenoh endpoint of the server, e.g. `tcp/127.0.0.1:7448`.
            deployment (`str`, *optional*):
                Name the server serves its simulator under, e.g. `libero`. `None` takes the one the endpoint serves.
            timeout_s (`float`, *optional*, defaults to `120`):
                Time limit for each request, such as a step.
            setup_timeout_s (`float`, *optional*):
                Time limit for open and reset, which may build simulators. Never below `timeout_s`.
            encoding (`str`, *optional*, defaults to `"raw"`):
                How camera frames are sent: `"raw"` or `"jpeg"`.
            jpeg_quality (`int`, *optional*, defaults to `90`):
                JPEG quality, from 1 to 100.
            handshake_timeout_s (`float`, *optional*, defaults to `10.0`):
                Time limit for connecting and `describe`, so a missing server fails fast.
            zenoh_config_path (`str`, *optional*):
                Zenoh JSON5 file with TLS and access-control settings.
            max_payload_bytes (`int`, *optional*, defaults to `DEFAULT_MAX_PAYLOAD_BYTES`):
                Largest reply in bytes. The server refuses a batch whose observations exceed it.
        """
        self.encoding = encoding
        self.jpeg_quality = jpeg_quality
        self.endpoint = endpoint
        self.transport = ZenohTransport(
            ZenohConfig(
                connect_endpoints=[endpoint],
                config_file=zenoh_config_path,
                max_payload_bytes=max_payload_bytes,
                open_timeout_s=handshake_timeout_s,
            )
        )
        self.limits = CodecLimits(
            max_payload_bytes=max_payload_bytes,
            max_tensor_bytes=max_payload_bytes,
            max_decoded_bytes=2 * max_payload_bytes,
        )
        self.handshake_timeout_s = handshake_timeout_s
        self.deployment = deployment or ""
        self.timeout_s = timeout_s
        self.setup_timeout_s = max(timeout_s, setup_timeout_s or timeout_s)
        self.instance_id = ""
        self.session_id = ""
        self.generation = 0
        self.descriptor: EnvDescriptor | None = None
        self._lock = RLock()
        self.failed = False
        self.num_envs = 0

    def describe(self) -> EnvDescriptor:
        """Return the env the server runs (features, fps, tasks), without building any simulator."""
        with self._lock:
            try:
                self.transport.open()
            except Exception as error:  # Zenoh reports an unreachable endpoint with its own error type.
                raise ConnectionError(
                    f"Could not connect to an env server at {self.endpoint} ({error}): start one with "
                    "lerobot-env-server, or check the endpoint"
                ) from error
            if self.deployment:
                try:
                    body = self._query(
                        deployment_prefix(self.deployment) + "/describe",
                        MessageType.DESCRIBE,
                        {},
                        self.handshake_timeout_s,
                        MessageType.DESCRIPTOR,
                    )
                except TimeoutError:
                    # The endpoint serves another deployment, or the server finished starting after the query.
                    body = self._describe_any()
                    if body["deployment"] != self.deployment:
                        raise ProtocolError(
                            ErrorCode.INCOMPATIBLE,
                            f"The env server at {self.endpoint} serves deployment {body['deployment']!r}, not "
                            f"{self.deployment!r}: connect with --env.server.deployment={body['deployment']} "
                            f"(--robot.env.server.deployment for a sim robot), or start a server for "
                            f"{self.deployment!r}",
                        ) from None
            else:
                body = self._query(
                    ANY_DEPLOYMENT_DESCRIBE,
                    MessageType.DESCRIBE,
                    {},
                    self.handshake_timeout_s,
                    MessageType.DESCRIPTOR,
                )
            self.deployment = body["deployment"]
            self.instance_id = body["instance_id"]
            self.descriptor = decode_descriptor(body["descriptor"])
            return self.descriptor

    def _describe_any(self) -> dict[str, Any]:
        """Describe whichever deployment the endpoint serves; a timeout if no server answers."""
        try:
            return self._query(
                ANY_DEPLOYMENT_DESCRIBE,
                MessageType.DESCRIBE,
                {},
                self.handshake_timeout_s,
                MessageType.DESCRIPTOR,
            )
        except (TimeoutError, ProtocolError):
            raise TimeoutError(
                f"No env server answered at {self.endpoint} within {self.handshake_timeout_s:g} s"
            ) from None

    @property
    def prefix(self) -> str:
        """Key prefix of the open session."""
        if not self.session_id:
            raise RuntimeError("Env server session is not open")
        return session_prefix(self.deployment, self.instance_id, self.session_id)

    def open(
        self,
        num_envs: int,
        task_group: str | None = None,
        task_id: int | None = None,
        seeds: list[int | None] | None = None,
        use_async_envs: bool = False,
    ) -> StepResult:
        """Open a session of `num_envs` envs on one task and return the result of its first reset."""
        with self._lock:
            if self.session_id:
                raise RuntimeError("Env server session already open")
            descriptor = self.descriptor or self.describe()
            body = self._query(
                instance_prefix(self.deployment, self.instance_id) + "/open",
                MessageType.OPEN,
                {
                    "num_envs": num_envs,
                    "task_group": task_group,
                    "task_id": task_id,
                    "seeds": seeds,
                    "use_async_envs": use_async_envs,
                    "encoding": self.encoding,
                    "jpeg_quality": self.jpeg_quality,
                    "max_payload_bytes": self.limits.max_payload_bytes,
                },
                self.setup_timeout_s,
                MessageType.ACCEPTED,
            )
            opened_descriptor = decode_descriptor(body["descriptor"])
            if not opened_descriptor.same_schema(descriptor):
                self.transport.close()
                raise ProtocolError(
                    ErrorCode.INCOMPATIBLE,
                    "The selected task has different features than the server described",
                )
            self.descriptor = opened_descriptor
            self.session_id = body["session_id"]
            self.generation = body["generation"]
            self.transport.declare_token(self.prefix + "/client_alive")
            self.num_envs = num_envs
            try:
                result = decode_step_result(body["result"])
                validate_result(result, self.descriptor, num_envs)
            except Exception:
                self.failed = True
                raise
            return result

    def request(self, operation: Literal["reset", "step"], **body: Any) -> StepResult:
        """Send one operation and return its checked result. A failure fails the session: it may have run."""
        with self._lock:
            if self.failed:
                raise RuntimeError("Env server session failed; close and reopen it before continuing")
            descriptor = self.descriptor
            if not self.session_id or descriptor is None:
                raise RuntimeError("Env server session is not open")
            key = f"{self.prefix}/{operation}"
            try:
                timeout = self.setup_timeout_s if operation == "reset" else self.timeout_s
                reply = self._query(key, MessageType.CONTROL, body, timeout, MessageType.ACK)
                result = decode_step_result(reply["result"])
                validate_result(result, descriptor, self.num_envs)
                self.generation = reply.get("generation", self.generation)
                return result
            except Exception:
                self.failed = True
                raise

    def close(self) -> None:
        """Close the session and the connection. If the server doesn't answer, log a warning instead of raising."""
        with self._lock:
            try:
                if self.session_id and not self.failed:
                    # Short limit: if this is lost, the server still frees the session when the connection closes.
                    self._query(
                        f"{self.prefix}/close",
                        MessageType.CONTROL,
                        {},
                        self.handshake_timeout_s,
                        MessageType.ACK,
                    )
            except Exception as error:
                logger.warning("Could not close the env server session cleanly: %s", error)
            finally:
                self.transport.close()
                self.session_id = ""
                self.instance_id = ""
                self.descriptor = None
                self.failed = False

    def _query(
        self, key: str, kind: MessageType, body: dict[str, Any], timeout: float, expected: MessageType
    ) -> dict[str, Any]:
        message = Envelope(kind, self.instance_id, self.session_id, self.generation, uuid.uuid4().hex, body)
        response = query_reply(
            self.transport,
            key,
            message,
            timeout,
            expected,
            limits=self.limits,
            max_replies=1,
            # Requests are never retried, so a reply dropped under congestion would fail the session.
            congestion_control="block",
        )
        return response.body


class RemoteVectorEnv(gym.vector.VectorEnv):
    """Vector env of `num_envs` envs on one task, run by an env server, without autoreset."""

    def __init__(
        self,
        client: EnvClient,
        num_envs: int = 1,
        task_group: str | None = None,
        task_id: int | None = None,
        use_async_envs: bool = False,
        descriptor: EnvDescriptor | None = None,
    ):
        """Fetch the descriptor unless one is given; the session opens on the first `reset`.

        Args:
            client (`EnvClient`):
                Client of the env server that runs the envs.
            num_envs (`int`, *optional*, defaults to `1`):
                Number of envs in the batch.
            task_group (`str`, *optional*):
                Suite to run, e.g. `libero_spatial`. Defaults to the first one the server lists.
            task_id (`int`, *optional*):
                Task within the suite. Defaults to its first task.
            use_async_envs (`bool`, *optional*, defaults to `False`):
                Run each env in its own worker process on the server.
            descriptor (`EnvDescriptor`, *optional*):
                The server's descriptor, if already fetched; saves a `describe` call.
        """
        self.client = client
        self.use_async_envs = use_async_envs
        self.descriptor = descriptor or self.client.describe()
        self.num_envs = num_envs
        self.task_group = task_group or next(iter(self.descriptor.tasks))
        tasks = self.descriptor.tasks.get(self.task_group, [])
        # Default to the group's first task, like a local env.
        self.task_id: int = next(iter(tasks), 0) if task_id is None else task_id
        if self.task_id not in tasks:
            raise ValueError(f"Unknown task: {self.task_group}/{self.task_id}")
        self.metadata = {
            "render_fps": self.descriptor.fps,
            "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
            "canonical_observations": True,
        }
        self.render_mode = "rgb_array"
        self.single_observation_space = self.descriptor.observation_space()
        self.single_action_space = self.descriptor.action_space()
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self._latest: StepResult | None = None

    @property
    def latest(self) -> StepResult:
        """The result of the last reset or step."""
        if self._latest is None:
            raise RuntimeError("Reset before reading the environment")
        return self._latest

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset every env, opening the session on first use; env `i` gets seed `seed + i`."""
        if options:
            raise ValueError("This environment does not define reset options")
        if isinstance(seed, int):
            seeds: list[int | None] | None = [seed + i for i in range(self.num_envs)]
        else:
            seeds = list(seed) if seed is not None else None
        if self.client.session_id:
            self._latest = self.client.request("reset", seeds=seeds)
        else:
            self._latest = self.client.open(
                self.num_envs, self.task_group, self.task_id, seeds, self.use_async_envs
            )
            self.descriptor = self.client.descriptor or self.descriptor
        return self.latest.obs, self.latest.info()

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step every unfinished env once."""
        if not self.client.session_id:
            raise RuntimeError("Reset before stepping")
        self._latest = self.client.request("step", actions=np.asarray(actions, dtype=np.float32))
        result = self.latest
        return result.obs, result.reward, result.terminated, result.truncated, result.info()

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Return a supported per-environment attribute."""
        if args or kwargs:
            raise TypeError("Environment attributes do not take arguments")
        if name == "_max_episode_steps":
            return (self.descriptor.max_episode_steps,) * self.num_envs
        if name == "task":
            return (self.task_group,) * self.num_envs
        if name == "task_description":
            return self.latest.task
        if name == "render":
            return self.render()
        raise AttributeError(f"Environment attribute is not exposed: {name}")

    def get_attr(self, name: str) -> tuple[Any, ...]:
        """Return a supported per-environment attribute."""
        return self.call(name)

    def render(self) -> tuple[np.ndarray, ...]:
        """Return the latest RGB frame of every env, including envs whose episode has ended."""
        return self.latest.frames()

    def close_extras(self, **kwargs: Any) -> None:
        """Close the served session."""
        self.client.close()
