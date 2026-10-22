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

"""Drive the shared simulator session synchronously, without transport or an owner thread."""

import uuid
from threading import RLock
from typing import Any

from lerobot.sims.backend import Backend, BackendConfig

from .contracts import EnvDescriptor, StepResult
from .session import SimSession, check_open


class InProcessEnvClient:
    """Run simulator calls on the caller's thread with the server's episode semantics."""

    def __init__(self, sim: BackendConfig, descriptor: EnvDescriptor | None = None) -> None:
        """Store configuration without constructing or retaining native resources."""
        self.sim = sim
        self.session = ""
        self.generation = 0
        self.descriptor = descriptor
        self.failed = False
        self.num_envs = 0
        self.lock = RLock()
        self._world: SimSession | None = None

    def describe(self) -> EnvDescriptor:
        """Cache the schema and close any environment constructed to discover it."""
        with self.lock:
            if self.descriptor is None:
                probe = Backend(self.sim)
                try:
                    self.descriptor = probe.descriptor()
                finally:
                    probe.close()
            return self.descriptor

    def open(
        self,
        num_envs: int,
        clock: str,
        task_group: str | None = None,
        task_id: int = 0,
        seeds: list[int | None] | None = None,
        async_worlds: bool = False,
    ) -> StepResult:
        """Create one session on this thread, or native worker processes when requested."""
        with self.lock:
            if self.session:
                raise RuntimeError("Environment session already open")
            descriptor = self.describe()
            check_open(descriptor, num_envs, clock, num_envs)
            self._world = SimSession(
                self.sim, descriptor, num_envs, clock, task_group, task_id, seeds, async_worlds
            )
            self.descriptor = self._world.descriptor
            self.session = uuid.uuid4().hex
            self.generation = 0
            self.num_envs = num_envs
            return self._world.opened

    def request(self, operation: str, **body: Any) -> dict[str, Any]:
        """Run the same operation as the server, without serializing an RPC."""
        with self.lock:
            if self.failed:
                raise RuntimeError("Environment session failed; close and reopen before continuing")
            if self._world is None:
                raise RuntimeError("Environment session is not open")
            try:
                if operation == "close":
                    self.close()
                    return {}
                reply = self._world.handle(operation, body)
                if "generation" in reply:
                    self.generation = reply["generation"]
                return reply
            except Exception:
                self.failed = True
                raise

    def close(self) -> None:
        """Release the session; cached metadata owns no simulator resources."""
        with self.lock:
            world, self._world = self._world, None
            self.session = ""
            self.failed = False
            if world is not None:
                world.close()
