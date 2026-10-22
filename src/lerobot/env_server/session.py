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

"""Explicitly stepped simulator sessions shared by local and served execution."""

from typing import Any, cast

import numpy as np

from lerobot.sims.backend import Backend, BackendConfig
from lerobot.transport.wire.features import validate_array
from lerobot.transport.wire.protocol import ErrorCode, ProtocolError

from .contracts import EnvDescriptor, validate_result


def check_open(descriptor: EnvDescriptor, num_envs: Any, clock: str, max_envs: int) -> None:
    """Reject unsupported batch sizes and clocks before creating native resources."""
    if type(num_envs) is not int or not 1 <= num_envs <= max_envs:
        raise ProtocolError(
            ErrorCode.INCOMPATIBLE, f"Batch size {num_envs!r} is not between 1 and {max_envs}"
        )
    if clock != "lockstep" or clock not in descriptor.clocks:
        raise ProtocolError(ErrorCode.INCOMPATIBLE, "Simulation supports the lockstep clock only")


class SimSession:
    """Own one explicitly stepped batch and fail closed after an execution error."""

    def __init__(
        self,
        sim: BackendConfig,
        descriptor: EnvDescriptor,
        num_envs: int,
        clock: str,
        task_group: str | None = None,
        task_id: int = 0,
        seeds: list[int | None] | None = None,
        async_worlds: bool = False,
    ) -> None:
        """Construct and reset the selected task without starting a background clock."""
        check_open(descriptor, num_envs, clock, num_envs)
        self.backend = Backend(sim, num_envs, task_group, task_id, async_worlds)
        try:
            self.descriptor = self.backend.descriptor()
            if (
                self.descriptor.features != descriptor.features
                or self.descriptor.action_feature != descriptor.action_feature
                or self.descriptor.semantics != descriptor.semantics
            ):
                raise ProtocolError(
                    ErrorCode.INCOMPATIBLE, "Selected task differs from the negotiated simulator schema"
                )
            self.opened = self.backend.reset(seeds)
            validate_result(self.opened, self.descriptor, num_envs)
        except BaseException:
            self.backend.close()
            raise
        self.generation = 0
        self.fault: str | None = None

    def handle(self, operation: str, body: dict[str, Any]) -> dict[str, Any]:
        """Reset, step or read cached state; observation reads never advance physics."""
        if self.fault is not None:
            raise ProtocolError(ErrorCode.EXECUTION, self.fault)
        if operation == "reset":
            try:
                result = self.backend.reset(body.get("seeds"))
                validate_result(result, self.descriptor, self.backend.num_envs)
            except Exception as exc:
                self.fault = f"Simulator reset failed: {exc}"
                raise
            self.generation += 1
            return {"generation": self.generation, "result": result.to_dict()}
        if operation == "step":
            actions = self.actions(body.get("actions"))
            try:
                result = self.backend.step(actions)
                validate_result(result, self.descriptor, self.backend.num_envs)
            except Exception as exc:
                self.fault = f"Simulator step failed: {exc}"
                raise
            return {"result": result.to_dict()}
        if operation == "snapshot":
            return {"result": self.backend.snapshot().to_dict()}
        if operation == "render":
            return {"frames": self.backend.render()}
        raise ProtocolError(ErrorCode.UNSUPPORTED, f"Unsupported operation: {operation}")

    def actions(self, value: Any) -> np.ndarray:
        """Validate commands before any environment is stepped."""
        try:
            validate_array(value, self.descriptor.action_feature, leading_shape=(self.backend.num_envs,))
        except ValueError as exc:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Command shape, dtype or values invalid") from exc
        return cast(np.ndarray, value)

    def close(self) -> None:
        """Release the native environments."""
        self.backend.close()
