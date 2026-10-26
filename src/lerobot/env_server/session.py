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

"""One served batch of explicitly stepped environments."""

from typing import Any, cast

import numpy as np

from lerobot.sims.types import EnvDescriptor, SimConfig, StepResult
from lerobot.sims.vector_env import VectorEnv
from lerobot.transport.wire.features import validate_array
from lerobot.transport.wire.protocol import ErrorCode, ProtocolError

from .contracts import encode_step_result, validate_result


def check_open(num_envs: Any, max_envs: int) -> None:
    """Reject a batch size the server cannot admit."""
    if type(num_envs) is not int or not 1 <= num_envs <= max_envs:
        raise ProtocolError(
            ErrorCode.INCOMPATIBLE, f"Batch size {num_envs!r} is not between 1 and {max_envs}"
        )


class EnvSession:
    """Own a served environment batch whose physics advances only through ``step``."""

    def __init__(
        self,
        sim: SimConfig,
        descriptor: EnvDescriptor,
        num_envs: int,
        task_group: str | None = None,
        task_id: int = 0,
        seeds: list[int | None] | None = None,
        use_async_envs: bool = False,
    ) -> None:
        """Build and reset the environment batch."""
        self.descriptor = descriptor
        self.env = VectorEnv(sim, num_envs, task_group, task_id, use_async_envs)
        self.generation = 0
        self.fault: str | None = None
        try:
            self.env.reset(seed=seeds)
            self.opened = self._latest()
            validate_result(self.opened, self.env.descriptor, num_envs)
        except BaseException:
            self.env.close()
            raise

    def _latest(self) -> StepResult:
        assert self.env.latest is not None
        return self.env.latest

    def handle(self, operation: str, body: dict[str, Any]) -> dict[str, Any]:
        """Reset, step, or render the owned environments."""
        if self.fault is not None:
            raise ProtocolError(ErrorCode.EXECUTION, self.fault)
        if operation == "reset":
            try:
                self.env.reset(seed=body.get("seeds"))
            except Exception as exc:
                self.fault = f"Simulator reset failed: {exc}"
                raise
            self.generation += 1
            return {"generation": self.generation, "result": encode_step_result(self._latest())}
        if operation == "step":
            try:
                self.env.step(self.actions(body.get("actions")))
            except Exception as exc:
                self.fault = f"Simulator step failed: {exc}"
                raise
            return {"result": encode_step_result(self._latest())}
        if operation == "render":
            return {"frames": self.env.render()}
        raise ProtocolError(ErrorCode.UNSUPPORTED, f"Unsupported operation: {operation}")

    def actions(self, value: Any) -> np.ndarray:
        """Validate a batch of actions against the environment contract."""
        try:
            validate_array(value, self.descriptor.action_feature, leading_shape=(self.env.num_envs,))
        except ValueError as exc:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Action shape, dtype or values invalid") from exc
        return cast(np.ndarray, value)

    def close(self) -> None:
        """Release the environments."""
        self.env.close()
