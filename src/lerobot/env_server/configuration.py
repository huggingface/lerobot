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

"""Read native simulator configuration with explicit CLI overrides."""

import copy
from pathlib import Path
from typing import Any

import yaml

# Serving an EnvHub repository. Settings left null come from the repository's optional LEROBOT_SIM dict in
# env.py, else from defaults (fps 30, position control, generic component names); --set overrides either.
ENVHUB_CONFIG: dict[str, Any] = {
    "deployment": "envhub",
    "sim": {
        "type": "envhub",
        "task": "",
        "fps": None,
        "control": None,
        "semantics": None,
        "kwargs": {"hub_path": "", "trust_remote_code": False},
        "state_names": None,
        "action_names": None,
        "gripper_indices": None,
    },
}


def load_config(path: str | Path, overrides: list[str]) -> dict[str, Any]:
    """Read a YAML configuration and override existing dotted keys; reject typos and malformed paths."""
    return apply_overrides(yaml.safe_load(Path(path).read_text()), overrides)


def envhub_config(repo: str, overrides: list[str], path: str | Path | None = None) -> dict[str, Any]:
    """The configuration serving an EnvHub repository (``user/repo[@revision]``), from ``path`` if given.

    Passing the repository is the explicit consent to run its ``env.py``, like ``trust_remote_code=True``.
    """
    data = yaml.safe_load(Path(path).read_text()) if path else copy.deepcopy(ENVHUB_CONFIG)
    serve = [f"sim.kwargs.hub_path={repo}", "sim.kwargs.trust_remote_code=true"]
    return apply_overrides(data, [*serve, *overrides])


def apply_overrides(data: Any, overrides: list[str]) -> dict[str, Any]:
    """Override existing dotted keys of a configuration mapping, in place."""
    if not isinstance(data, dict):
        raise ValueError("Simulator configuration must be a YAML mapping")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"Expected a dotted key=value override, got {override!r}")
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown simulator override path: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown simulator override key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    return data
