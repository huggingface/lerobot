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

"""Keys, validation and wire encoding at the environment-service boundary.

Keys carry the shared envelope version, so a wire bump moves both services together.
"""

from dataclasses import asdict, fields
from typing import Any

import numpy as np

from lerobot.sims.types import EnvDescriptor, FeatureSpec, StepResult
from lerobot.transport.wire.features import validate_array
from lerobot.transport.wire.protocol import PROTOCOL_VERSION, validate_segment


def deployment_prefix(deployment: str) -> str:
    """Return a validated simulator deployment key."""
    return f"lerobot/env/v{PROTOCOL_VERSION}/deployments/{validate_segment(deployment)}"


def any_deployment_describe() -> str:
    """Return the describe key every server answers, for clients that take whichever deployment is served."""
    return f"lerobot/env/v{PROTOCOL_VERSION}/describe"


def instance_prefix(deployment: str, instance: str) -> str:
    """Return a simulator boot key under its deployment."""
    return f"{deployment_prefix(deployment)}/instances/{validate_segment(instance)}"


def session_prefix(deployment: str, instance: str, session: str) -> str:
    """Return an admitted environment-session key."""
    return f"{instance_prefix(deployment, instance)}/sessions/{validate_segment(session)}"


def encode_descriptor(descriptor: EnvDescriptor) -> dict[str, Any]:
    """Encode environment metadata for transport."""
    return asdict(descriptor)


def decode_descriptor(data: dict[str, Any]) -> EnvDescriptor:
    """Decode environment metadata, ignoring fields added by newer servers."""
    known = {item.name for item in fields(EnvDescriptor)}
    values = {key: value for key, value in data.items() if key in known}
    values["features"] = tuple(FeatureSpec(**feature) for feature in values["features"])
    values["action_feature"] = FeatureSpec(**values["action_feature"])
    return EnvDescriptor(**values)


def encode_step_result(result: StepResult) -> dict[str, Any]:
    """Encode a simulation result for transport."""
    return {name: getattr(result, name) for name in result.__dataclass_fields__}


def decode_step_result(data: dict[str, Any]) -> StepResult:
    """Decode a transport result into simulation values."""
    # JPEG replies carry one frame per environment; stack them back into the batch.
    obs = {name: np.stack(value) if isinstance(value, list) else value for name, value in data["obs"].items()}
    return StepResult(**{**data, "obs": obs, "task": tuple(data["task"])})


def validate_result(result: StepResult, descriptor: EnvDescriptor, num_envs: int) -> None:
    """Reject transitions that disagree with the negotiated schema or batch size."""
    if set(result.obs) != {f.name for f in descriptor.features}:
        raise ValueError("Simulator observation keys differ from the negotiated schema")
    for feature in descriptor.features:
        validate_array(result.obs[feature.name], feature, leading_shape=(num_envs,))
    if len(result.task) != num_envs or any(not isinstance(task, str) for task in result.task):
        raise ValueError("Simulator task descriptions must match the batch")
    for name in ("reward", "terminated", "truncated", "is_success", "step"):
        array = getattr(result, name)
        if not isinstance(array, np.ndarray) or array.shape != (num_envs,):
            raise ValueError(f"Simulator transition batch mismatch: {name}")
        expected = (
            "bool"
            if name in {"terminated", "truncated", "is_success"}
            else "int64"
            if name == "step"
            else "float32"
        )
        if array.dtype.name != expected:
            raise ValueError(f"Simulator transition dtype mismatch: {name}")
    if (
        not np.isfinite(result.reward).all()
        or not np.isfinite(result.sim_time)
        or result.sim_time < 0
        or (result.step < 0).any()
    ):
        raise ValueError("Invalid simulator reward/time/step values")
