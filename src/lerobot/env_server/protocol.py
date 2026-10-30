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

"""Keys, encoding and result checks shared by the env server and its client.

Keys carry the protocol version, so a client and a server of different versions never connect.
"""

from dataclasses import asdict, fields
from typing import Any

import numpy as np

from lerobot.sim.types import EnvDescriptor, StepResult
from lerobot.transport.protocol.envelope import validate_segment
from lerobot.utils.feature_spec import FeatureSpec, validate_array

# The env server's own version, separate from the remote inference and envelope versions.
PROTOCOL_VERSION = 1


def deployment_prefix(deployment: str) -> str:
    """Return the key prefix of a deployment, the name a server serves its simulator under."""
    return f"lerobot/env/v{PROTOCOL_VERSION}/deployments/{validate_segment(deployment)}"


# Every server also answers describe here, for clients that do not name a deployment.
ANY_DEPLOYMENT_DESCRIBE = f"lerobot/env/v{PROTOCOL_VERSION}/describe"


def instance_prefix(deployment: str, instance: str) -> str:
    """Return the key prefix of one server process (instance) in a deployment."""
    return f"{deployment_prefix(deployment)}/instances/{validate_segment(instance)}"


def session_prefix(deployment: str, instance: str, session: str) -> str:
    """Return the key prefix of one session on a server process."""
    return f"{instance_prefix(deployment, instance)}/sessions/{validate_segment(session)}"


def encode_descriptor(descriptor: EnvDescriptor) -> dict[str, Any]:
    """Encode an env descriptor as a plain dict."""
    return asdict(descriptor)


def decode_descriptor(data: dict[str, Any]) -> EnvDescriptor:
    """Decode an env descriptor, ignoring fields added by newer servers."""
    known = {item.name for item in fields(EnvDescriptor)}
    values = {key: value for key, value in data.items() if key in known}
    values["features"] = tuple(FeatureSpec(**feature) for feature in values["features"])
    values["action_feature"] = FeatureSpec(**values["action_feature"])
    return EnvDescriptor(**values)


def encode_step_result(result: StepResult) -> dict[str, Any]:
    """Encode a step result as a dict of its fields."""
    # Not asdict(): it deep-copies every array, camera frames included.
    return {field.name: getattr(result, field.name) for field in fields(result)}


def decode_step_result(data: dict[str, Any]) -> StepResult:
    """Decode a step result from its wire dict."""
    # JPEG replies carry a list of frames, one per env; stack them into one batch array.
    obs = {name: np.stack(value) if isinstance(value, list) else value for name, value in data["obs"].items()}
    return StepResult(**{**data, "obs": obs, "task": tuple(data["task"])})


# Per-env fields of a result and their dtype.
_BATCH_DTYPES = {
    "reward": "float32",
    "terminated": "bool",
    "truncated": "bool",
    "is_success": "bool",
    "elapsed_steps": "int64",
}


def validate_result(result: StepResult, descriptor: EnvDescriptor, num_envs: int) -> None:
    """Raise ``ValueError`` if a result does not match the descriptor or the batch size."""
    if set(result.obs) != {f.name for f in descriptor.features}:
        raise ValueError("Simulator observation keys differ from its descriptor")
    for feature in descriptor.features:
        validate_array(result.obs[feature.name], feature, leading_shape=(num_envs,))
    if len(result.task) != num_envs:
        raise ValueError("Simulator task descriptions must match the batch")
    for name, dtype in _BATCH_DTYPES.items():
        array = getattr(result, name)
        if not isinstance(array, np.ndarray) or array.shape != (num_envs,) or array.dtype.name != dtype:
            raise ValueError(f"Simulator {name} must be {dtype}, one value per environment")
    if not np.isfinite(result.reward).all():
        raise ValueError("Simulator rewards must be finite")
