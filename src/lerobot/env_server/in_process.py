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

"""Drive an environment in this process, with the env server's semantics and no transport."""

import contextlib
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from queue import Empty, Queue
from threading import Event, RLock, Thread
from typing import Any, TypeVar

import numpy as np

from lerobot.sims.backend import Backend, BackendConfig
from lerobot.transport.wire.protocol import ErrorCode, ProtocolError

from .contracts import EnvDescriptor, ExecutionFeedback, StepResult
from .session import Command, SimSession, check_open

T = TypeVar("T")


class _Command(Command):
    def __init__(self, request_id: str):
        super().__init__(request_id, np.empty(0))
        self.reply: Future[dict[str, Any]] = Future()


class _Session(SimSession):
    def _command_done(self, command: Command, reply: dict[str, Any]) -> None:
        assert isinstance(command, _Command)
        command.reply.set_result(reply)

    def _command_failed(self, command: Command, code: ErrorCode, reason: str) -> None:
        assert isinstance(command, _Command)
        command.reply.set_exception(ProtocolError(code, reason))


class InProcessEnvClient:
    """Own one environment session in this process, as ``EnvClient`` owns one on a server.

    One owner thread runs every simulator call, as the server's worker does, so renderer contexts stay on
    the thread that created them, and it advances realtime worlds at the control rate while the caller
    computes its next action.
    """

    def __init__(
        self,
        sim: BackendConfig,
        command_timeout_s: float = 1,
        descriptor: EnvDescriptor | None = None,
    ):
        """Initialize owned resources without building the environment.

        A known ``descriptor`` spares building a world just to describe the environment.
        """
        self.sim = sim
        self.command_timeout_s = command_timeout_s
        self.session = ""
        self.generation = 0
        self.descriptor = descriptor
        self.execution: ExecutionFeedback | None = None
        self.failed = False
        self.num_envs = 0
        self.lock = RLock()
        self._world: _Session | None = None
        # The describe probe, reused as the first session's world when the batch matches.
        self._probe: Backend | None = None
        self._calls: Queue[Callable[[], None]] = Queue()
        self._stop = Event()
        self._thread: Thread | None = None

    def describe(self) -> EnvDescriptor:
        """Build the environment and describe its canonical schema without opening a session."""
        with self.lock:
            if self.descriptor is None:
                self._probe = self._run(lambda: Backend(self.sim))
                self.descriptor = self._run(self._probe.descriptor)
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
        """Admit an exclusive environment batch with the requested clock.

        ``async_worlds`` runs each world in its own process, so the batch steps in parallel.
        """
        with self.lock:
            if self.session:
                raise RuntimeError("Environment session already open")
            descriptor = self.describe()
            check_open(descriptor, num_envs, clock, num_envs)
            self._world = self._run(
                lambda: self._session(num_envs, clock, task_group, task_id, seeds, async_worlds)
            )
            self.descriptor = self._run(self._world.backend.descriptor)
            self.session = uuid.uuid4().hex
            self.generation = 0
            self.num_envs = num_envs
            return self._world.opened

    def _session(
        self,
        num_envs: int,
        clock: str,
        task_group: str | None,
        task_id: int,
        seeds: list[int | None] | None,
        async_worlds: bool,
    ) -> _Session:
        probe, self._probe = self._probe, None
        reuse = (
            probe is not None
            and not async_worlds
            and probe.num_envs == num_envs
            and task_group in (None, probe.group)
            and task_id == probe.task_id
        )
        if probe is not None and not reuse:
            probe.close()
        assert self.descriptor is not None
        backend = probe if reuse else Backend(self.sim, num_envs, task_group, task_id, async_worlds)
        return _Session(
            self.sim,
            self.descriptor,
            num_envs,
            clock,
            task_group,
            task_id,
            seeds,
            self.command_timeout_s,
            backend,
        )

    def request(self, operation: str, **body: Any) -> dict[str, Any]:
        """Run one world operation and fail permanently on any error, as ``EnvClient`` does."""
        with self.lock:
            if self.failed:
                raise RuntimeError("Environment session failed; close and reopen before continuing")
            world = self._world
            if world is None:
                raise RuntimeError("Environment session is not open")
            try:
                if operation == "close":
                    self._close_world()
                    return {}
                command = _Command(uuid.uuid4().hex) if operation == "apply" else None
                reply = self._run(lambda: world.handle(operation, body, command))
                if reply is None:
                    assert command is not None
                    # Answered by the owner thread after the tick that executes it.
                    reply = command.reply.result()
                if "execution" in reply:
                    self.execution = ExecutionFeedback.from_dict(
                        reply["execution"], StepResult.from_dict(reply["result"])
                    )
                elif "generation" in reply:
                    self.execution = None
                    self.generation = reply["generation"]
                return reply
            except Exception:
                # Never continue on a world whose state after the failure is unknown.
                self.failed = True
                raise

    def close(self) -> None:
        """Release the environment and stop the owner thread."""
        with self.lock:
            try:
                if self._thread is not None:
                    self._close_world()
                    probe, self._probe = self._probe, None
                    if probe is not None:
                        self._run(probe.close)
            finally:
                self._stop.set()
                if self._thread is not None:
                    self._thread.join()
                self._thread = None
                self._stop = Event()
                self.session = ""
                self.descriptor = None
                self.failed = False
                self.execution = None

    def _close_world(self) -> None:
        world, self._world = self._world, None
        self.session = ""
        if world is not None:
            self._run(world.close)

    def _run(self, call: Callable[[], T]) -> T:
        """Run ``call`` on the owner thread and return its result or raise its error."""
        if self._thread is None:
            self._thread = Thread(target=self._serve, name="lerobot-env", daemon=True)
            self._thread.start()
        result: Future[T] = Future()

        def run() -> None:
            try:
                result.set_result(call())
            except BaseException as exc:
                result.set_exception(exc)

        self._calls.put(run)
        return result.result()

    def _serve(self) -> None:
        stop = self._stop
        while not stop.is_set():
            with contextlib.suppress(Empty):
                self._calls.get(timeout=0.002)()
            world = self._world
            if world is not None:
                world.tick(time.monotonic())
