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

"""One admitted batch of worlds and its clock, independent of how requests reach it.

The env server and the in-process client both drive worlds through this class, so a world behaves
the same whether it runs in the policy's process or behind a server.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from lerobot.sims.adapters import hold_action
from lerobot.sims.backend import Backend, BackendConfig
from lerobot.transport.wire.features import validate_array
from lerobot.transport.wire.protocol import ErrorCode, ProtocolError

from .contracts import EnvDescriptor, ExecutionFeedback, StepResult, validate_result

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """One accepted realtime command, answered after the tick that executes it."""

    request_id: str
    actions: np.ndarray


def check_open(descriptor: EnvDescriptor, num_envs: Any, clock: str, max_envs: int) -> None:
    """Reject a batch size or clock the simulator cannot serve."""
    if type(num_envs) is not int or not 1 <= num_envs <= max_envs:
        raise ProtocolError(
            ErrorCode.INCOMPATIBLE, f"Batch size {num_envs!r} is not between 1 and {max_envs}"
        )
    if clock not in descriptor.clocks:
        raise ProtocolError(
            ErrorCode.INCOMPATIBLE, f"Clock {clock!r} is not one of {list(descriptor.clocks)}"
        )


class SimSession:
    """Own a batch of worlds, its lockstep or realtime clock and its realtime command.

    All methods run on one owner thread. Realtime commands are answered through ``_command_done`` and
    ``_command_failed``, which the transport overrides.
    """

    def __init__(
        self,
        sim: BackendConfig,
        descriptor: EnvDescriptor,
        num_envs: int,
        clock: str,
        task_group: str | None = None,
        task_id: int = 0,
        seeds: list[int | None] | None = None,
        command_timeout_s: float = 1,
        backend: Backend | None = None,
        async_worlds: bool = False,
    ):
        """Build (or adopt ``backend``) and reset the worlds; the first realtime tick waits for a command.

        ``async_worlds`` runs each world in its own process, so the batch steps in parallel.
        """
        self.descriptor = descriptor
        self.clock = clock
        self.command_timeout_s = command_timeout_s
        self.backend = backend or Backend(sim, num_envs, task_group, task_id, async_worlds)
        try:
            self.opened = self.backend.reset(seeds)
            validate_result(self.opened, self.backend.descriptor(), num_envs)
            self.target = self._initial_target(self.opened)
        except BaseException:
            self.backend.close()
            raise
        self.generation = 0
        self.next_tick = time.monotonic() + 1 / descriptor.fps
        self.last_command = time.monotonic()
        # Realtime worlds stay frozen until the first command; the open/reset target is not a policy command.
        self.commanded = False
        self.fault: str | None = None
        self.command: Command | None = None
        self.execution: ExecutionFeedback | None = None
        self.sequence = 0

    def handle(self, operation: str, body: dict[str, Any], command: Command | None = None) -> dict | None:
        """Run one world operation and return its reply, or None for a command answered after its tick.

        ``command`` carries an ``apply``'s actions and request identity once its body is validated.
        """
        if self.fault is not None:
            raise ProtocolError(ErrorCode.EXECUTION, self.fault)
        if operation == "reset":
            self.cancel_command("World reset before command execution")
            self.execution = None
            try:
                result = self.backend.reset(body.get("seeds"))
            except Exception as exc:
                self.fault = f"Simulator reset failed: {exc}"
                raise
            self.generation += 1
            self.commanded = False
            self.target = self._initial_target(result)
            self.last_command = time.monotonic()
            return {"generation": self.generation, "result": result.to_dict()}
        if operation in {"step", "apply"}:
            actions = self.actions(body.get("actions"))
        if operation == "step":
            if self.clock != "lockstep":
                raise ProtocolError(ErrorCode.UNSUPPORTED, "Step requires a lockstep session")
            try:
                # Lockstep replies are the transition itself; execution feedback serves realtime only.
                return {"result": self.backend.step(actions).to_dict()}
            except Exception as exc:
                self.fault = f"Simulator step failed: {exc}"
                raise
        if operation in {"apply", "hold"}:
            if self.clock != "realtime":
                raise ProtocolError(ErrorCode.UNSUPPORTED, "Commands require realtime mode")
            if operation == "apply":
                # Clients send one blocking command at a time; the request's own deadline bounds it.
                if command is None or self.command is not None:
                    raise ProtocolError(ErrorCode.BUSY, "A command is already pending execution")
                command.actions = actions.copy()
                self.command = command
                self.commanded = True
                return None
            self.cancel_command("Hold superseded the pending command")
            self.target = self._hold(self.target)
            self.commanded = True
            self.last_command = time.monotonic()
            return {"applied": self.target}
        if operation == "snapshot":
            reply: dict[str, Any] = {"result": self.backend.snapshot().to_dict()}
            if self.execution is not None:
                reply["execution"] = self.execution.to_dict()
            return reply
        if operation == "render":
            return {"frames": self.backend.render()}
        raise ProtocolError(ErrorCode.UNSUPPORTED, f"Unsupported operation: {operation}")

    def actions(self, value: Any) -> np.ndarray:
        """Validate a batch of commands against the action contract."""
        try:
            validate_array(value, self.descriptor.action_feature, leading_shape=(self.backend.num_envs,))
        except ValueError as exc:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Command shape, dtype or values invalid") from exc
        return cast(np.ndarray, value)

    def due(self, now: float) -> bool:
        """Whether a realtime tick should run now."""
        return self.clock == "realtime" and self.commanded and self.fault is None and now >= self.next_tick

    def tick(self, now: float) -> None:
        """Advance a due realtime world once, with the pending command or the held target."""
        if not self.due(now):
            return
        command, self.command = self.command, None
        try:
            if command is not None:
                self.target = command.actions
                self.last_command = now
            elif now - self.last_command > self.command_timeout_s:
                self.target = self._hold(self.target)
            before = self.backend.snapshot()
            applied = self.target.copy()
            started = time.monotonic()
            result = self.backend.step(applied)
            request_id = command.request_id if command else uuid.uuid4().hex
            reply = self._executed(request_id, applied, before, result, started)
            if self.descriptor.control == "delta":
                self.target = hold_action(self.target, "delta", self._retained)
            if command is not None:
                self._command_done(command, reply)
            # Schedule from the previous deadline so poll latency does not accumulate.
            # When execution overruns, resume at once without replaying missed ticks.
            self.next_tick = max(self.next_tick + 1 / self.descriptor.fps, time.monotonic())
        except Exception as exc:
            self.fault = f"Environment execution failed: {exc}"
            if command is not None:
                self._command_failed(command, ErrorCode.EXECUTION, self.fault)
            logger.exception("Environment session failed")

    def cancel_command(self, reason: str) -> None:
        """Answer an unexecuted command as stale."""
        command, self.command = self.command, None
        if command is not None:
            self._command_failed(command, ErrorCode.STALE, reason)

    def close(self) -> None:
        """Answer the pending command and release the worlds."""
        try:
            self.cancel_command("Environment session closed")
        finally:
            self.backend.close()

    def _command_done(self, command: Command, reply: dict[str, Any]) -> None:
        """Deliver an executed command's reply."""

    def _command_failed(self, command: Command, code: ErrorCode, reason: str) -> None:
        """Deliver a command's failure."""

    @property
    def _retained(self) -> tuple[int, ...]:
        return self.descriptor.gripper_indices + self.descriptor.command_retention_indices

    def _hold(self, target: np.ndarray) -> np.ndarray:
        return hold_action(target, self.descriptor.control, self._retained)

    def _initial_target(self, result: StepResult) -> np.ndarray:
        dim = self.descriptor.action_feature.shape[0]
        target: np.ndarray = np.zeros((self.backend.num_envs, dim), dtype=np.float32)
        if self.descriptor.control in {"position", "eef_pose"}:
            state = next((f for f in self.descriptor.features if f.name == "observation.state"), None)
            if state is None or not set(self.descriptor.action_feature.names).issubset(state.names):
                raise ValueError("Position control requires observed initial positions")
            target = result.obs[state.name][
                :, [state.names.index(n) for n in self.descriptor.action_feature.names]
            ].copy()
        return target

    def _executed(
        self, request_id: str, actions: np.ndarray, before: StepResult, result: StepResult, started: float
    ) -> dict[str, Any]:
        advanced = ~(before.terminated | before.truncated)
        applied = actions.copy()
        # No action reaches an absorbing world. Zero is a placeholder qualified by advanced=False.
        applied[~advanced] = 0
        self.sequence += 1
        feedback = ExecutionFeedback(
            request_id,
            self.generation,
            self.sequence,
            applied,
            advanced,
            started,
            time.monotonic(),
            result,
        )
        feedback.validate(self.descriptor, self.backend.num_envs)
        self.execution = feedback
        return {"result": result.to_dict(), "execution": feedback.to_dict()}
